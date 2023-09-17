from pathlib import Path
import re

from Bio.PDB import PDBParser, MMCIFParser, FastMMCIFParser
from Bio.PDB.Model import Model
from openbabel import pybel
import pandas as pd
from tqdm import tqdm

atomic_num_dict = lambda x: {1: 'H', 2: 'HE', 3: 'LI', 4: 'BE', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 11: 'NA',
                   15: 'P', 16: 'S', 17: 'Cl', 20:'Ca', 25: 'MN', 26: 'FE', 30: 'ZN', 35: 'Br', 53: 'I', 80: 'Hg'}.get(x, 'Others')

amino_acids_dict = {"GLY": 0, "ALA": 1, "SER": 2, "PRO": 3, "VAL": 4, "THR": 5, "CYS": 6, "ILE": 7, "LEU": 8,
                  "ASN": 9, "ASP": 10, "GLN": 11, "LYS": 12, "GLU": 13, "MET": 14, "HIS": 15, "PHE": 16,
                  "ARG": 17, "TYR": 18, "TRP": 19}

parser = PDBParser(QUIET=True)

def gen_df(coords, elements):
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

def process(pdb_id: str):
    # if ligand_path != -1:
    #     ligand = next(pybel.readfile('mol2', ligand_path))
    #     ligand_coords = [atom.coords for atom in ligand]
    #     atom_map_lig = [atomic_num_dict(atom.atomicnum) for atom in ligand]
    #     ligand_df = gen_df(ligand_coords, atom_map_lig)
    # else:
    #     ligand_df = None
    try:
        structure = parser.get_structure(pdb_id, Path('datasets') / 'pdb' / f'{pdb_id}.pdb')
    except Exception as e:
        # corrupt
        return
    # structure = p.get_structure(item, structure_dir)
    compound_info = structure.header['compound']
    protein_numbers = len(compound_info.items())

    if len(structure) > 1:
        # nmr_files.append(item)
        return
    # if item not in self.labels:
    #     wrong_number.append(item)
    #     return
    model: Model = structure[0]
    chains = list(model.get_chains())
    pattern = re.compile(r'\d+H.')
    # processed_complex = {
    #     'complex_id': pdb_id, 'num_proteins': protein_numbers, 'labels': self.labels[item],
    #     'atoms_protein': [], 'protein_seq': [], 'atoms_ligand': ligand_df
    # }
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
        {
            'chain': chain_ids, 'residue': residues, 'resname': resnames, 'element': elements, 'name': names,
            'x': xs, 'y': ys, 'z': zs
        }
    )
    processed_complex['atoms_protein'] = protein_df
    processed_complex['protein_seq'] = protein_seq

    self.processed_complexes.append(processed_complex)

def generate_cache_files(self):
    parser = PDBParser(QUIET=True)
    nmr_files = []
    wrong_number = []
    self.processed_complexes = []
    corrupted = []
    for score_idx, item in enumerate(tqdm(self.files)):
        pass

def main():
    # pdb_id = '1a1e'
    # model_pdb = parser.get_structure('pdb', Path('datasets') / 'pdb' / f'{pdb_id}.pdb')[0]
    # model_rs = parser.get_structure('refined-set', Path('datasets/PDBbind/refined-set/1a1e/1a1e_protein.pdb'))[0]
    # ligand = next(pybel.readfile('mol2', 'datasets/PDBbind/refined-set/1a1e/1a1e_ligand.mol2'))
    model_pdb = parser.get_structure('pdb', Path('datasets') / 'pdb' / f'1a3b.pdb')[0]
    model_pp = parser.get_structure('pp', 'datasets/PDBbind/PP/1a3b.ent.pdb')[0]
    print(233)

if __name__ == '__main__':
    main()
