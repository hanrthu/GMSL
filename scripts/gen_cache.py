from Bio.PDB import PDBParser
from tqdm import tqdm
from openbabel import pybel
import re
import os
import pandas as pd
import pickle
import json
from tqdm.contrib.concurrent import process_map

atomic_num_dict = lambda x: {1: 'H', 2: 'HE', 3: 'LI', 4: 'BE', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 11: 'NA',
                   15: 'P', 16: 'S', 17: 'Cl', 20:'Ca', 25: 'MN', 26: 'FE', 30: 'ZN', 35: 'Br', 53: 'I', 80: 'Hg'}.get(x, 'Others')

amino_acids_dict = {"GLY": 0, "ALA": 1, "SER": 2, "PRO": 3, "VAL": 4, "THR": 5, "CYS": 6, "ILE": 7, "LEU": 8,
                  "ASN": 9, "ASP": 10, "GLN": 11, "LYS": 12, "GLU": 13, "MET": 14, "HIS": 15, "PHE": 16,
                  "ARG": 17, "TYR": 18, "TRP": 19}



class GenMultitask_Cache(object):
    def __init__(self, root_dir:str = None, split:str = 'train_all', label_dir:str = './datasets/MultiTask/uniformed_labels.json'):
        super().__init__()
        self.root_dir = root_dir
        self.cache_dir = os.path.join(root_dir, "{}.cache".format(split))
        file_dir = os.path.join(root_dir, split+'.txt')
        with open(file_dir, 'r') as f:
            self.files = f.readlines()
            self.files = [i.strip() for i in self.files]

        with open(label_dir, 'r') as f:
            self.labels = json.load(f)
        self.ec_root = './datasets/EnzymeCommission/all'
        self.go_root = './datasets/GeneOntology/all'
        self.lba_root = './datasets/PDBbind/refined-set'
        self.pp_root = './datasets/PDBbind/PP'
        self.remove_hoh=True
        self.remove_hydrogen=True

    def gen_df(self, coords, elements):
        assert len(coords) == len(elements)
        unified_elements = []
        xs, ys, zs = [coord[0] for coord in coords], [coord[1] for coord in coords], [coord[2] for coord in coords]
        for item in elements:
            if item in ['CL', 'Cl', 'Br', 'BR', 'AT', 'At', 'F', 'I']:
                element = 'Halogen'
            elif item in ['FE', 'ZN', 'MG', 'MN', 'K', 'LI', 'Ca', 'HG', 'NA']:
                element = 'Metal'
            elif item[0] in ['C', 'N', 'O', 'S', 'P', 'H']:
                element = item[0]
            else:
                element = item
            unified_elements.append(element)
        df = pd.DataFrame({'element': unified_elements, 'resname': 'LIG', 'x': xs, 'y': ys, 'z': zs})
        return df

    def find_structure(self, item):
        self.ec_files = os.listdir(self.ec_root)
        self.go_files = os.listdir(self.go_root)
        self.lba_files = os.listdir(self.lba_root)
        self.pp_files = os.listdir(self.pp_root)
        if '-' in item:
            if item+'.pdb' in self.ec_files:
                return os.path.join(self.ec_root, item+'.pdb'), -1
            elif item+'.pdb' in self.go_files:
                return os.path.join(self.go_root, item+'.pdb'), -1
        else:
            if item + '.ent.pdb' in self.pp_files:
                return os.path.join(self.pp_root, item+'.ent.pdb'), -1
            elif item in self.lba_files:
                protein_dir = os.path.join(self.lba_root, item, item + "_protein.pdb")
                ligand_dir = os.path.join(self.lba_root, item, item + '_ligand.mol2')
                return protein_dir, ligand_dir
        print(item)
        return -1, -1
    def generate_cache_files(self):
        p = PDBParser(QUIET=True)
        nmr_files = []
        wrong_number = []
        self.processed_complexes = []
        corrupted = []
        for score_idx, item in enumerate(tqdm(self.files)):
            structure_dir, ligand_dir = self.find_structure(item)
            if ligand_dir != -1:
                ligand = next(pybel.readfile('mol2', ligand_dir))
                ligand_coords = [atom.coords for atom in ligand]
                atom_map_lig = [atomic_num_dict(atom.atomicnum) for atom in ligand]
                ligand_df = self.gen_df(ligand_coords, atom_map_lig)
            else:
                ligand_df = None
            try:
                structure = p.get_structure(item, structure_dir)
            except:
                corrupted.append(item)
                continue
            # structure = p.get_structure(item, structure_dir)
            compound_info = structure.header['compound']
            protein_numbers = len(compound_info.items())

            if len(structure) > 1:
                nmr_files.append(item)
                continue
            if item not in self.labels:
                wrong_number.append(item)
                continue
            model = structure[0]
            chains = list(model.get_chains())
            pattern = re.compile(r'\d+H.')
            processed_complex = {'complex_id': item, 'num_proteins': protein_numbers, 'labels': self.labels[item],
                                 'atoms_protein': [], 'protein_seq': [], 'atoms_ligand': ligand_df}
            elements = []
            xs = []
            ys = []
            zs = []
            chain_ids = []
            protein_seq = []
            names = []
            resnames = []
            residues = []
            # chain = chains[0]
            curr_residue = 0
            for chain in chains:
                if chain.id == ' ':
                    continue
                for residue in chain.get_residues():
                    # 删除HOH原子
                    if self.remove_hoh and residue.get_resname() == 'HOH':
                        continue
                    for atom in residue:
                        # 删除氢原子
                        atom_id = atom.get_id()
                        if self.remove_hydrogen and residue.get_resname() in amino_acids_dict and (
                                atom.get_id().startswith('H') or pattern.match(atom.get_id()) != None):
                            continue
                        if residue.get_resname() in amino_acids_dict and (
                                atom_id.startswith('H') or pattern.match(atom.get_id()) != None):
                            element = 'H'
                        elif atom_id[0:2] in ['CL', 'Cl', 'Br', 'BR', 'AT', 'At']:
                            element = 'Halogen'
                        elif atom_id[0:2] in ['FE', 'ZN', 'MG', 'MN', 'K', 'LI', 'Ca', 'HG', 'NA']:
                            element = 'Metal'
                        elif atom_id[0] in ['F', 'I']:
                            element = 'Halogen'
                        elif atom_id[0] in ['C', 'N', 'O', 'S', 'P']:
                            element = atom_id[0]
                        else:
                            element = atom_id
                        names.append(atom_id)
                        residues.append(curr_residue)
                        elements.append(element)
                        chain_ids.append(chain.id)
                        resnames.append(residue.get_resname())
                        x, y, z = atom.get_vector()
                        xs.append(x)
                        ys.append(y)
                        zs.append(z)
                    protein_seq.append(residue.get_resname())
                    curr_residue += 1
            protein_df = pd.DataFrame(
                {'chain': chain_ids, 'residue': residues, 'resname': resnames, 'element': elements, 'name': names, 'x': xs,
                 'y': ys, 'z': zs})
            processed_complex['atoms_protein'] = protein_df
            processed_complex['protein_seq'] = protein_seq

            self.processed_complexes.append(processed_complex)

        print("Structure processed Done, dumping...")
        print("Structures with Wrong numbers:", len(wrong_number), wrong_number)
        print("Structures with NMR methods:", len(nmr_files), nmr_files)
        print("Corrupted:", len(corrupted), corrupted)
        # np.s
        pickle.dump(self.processed_complexes, open(self.cache_dir, 'wb'))

def main():
    splits = ['train_all', 'train', 'val', 'test']
    root_dir = './datasets/MultiTask_Resplit/'
    for split in splits:
        generator = GenMultitask_Cache(root_dir=root_dir, split=split)
        print(f'Processing {split} split...')
        generator.generate_cache_files()

if __name__ == '__main__':
    main()