i
mport os
import re
from tqdm import tqdm
import numpy as np
from Bio.PDB import PDBParser
import pickle
import pandas as pd
import openbabel
from openbabel import pybel
# ob_log_handler = pybel.ob.OBMessageHandler()
# ob_log_handler.SetOutputLevel(0)
pybel.ob.obErrorLog.SetOutputLevel(0)
atomic_num_dict = lambda x: {1: 'H', 2: 'HE', 3: 'LI', 4: 'BE', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 11: 'NA',
                   15: 'P', 16: 'S', 17: 'Cl', 20:'Ca', 25: 'MN', 26: 'FE', 30: 'ZN', 35: 'Br', 53: 'I', 80: 'Hg'}.get(x, 'Others')

def load_pk_data(data_path):
    res = dict()
    with open(data_path) as f:
        for line in f:
            if '#' in line:
                continue
            cont = line.strip().split()
            if len(cont) < 5:
                continue
            code, pk = cont[0], cont[3]
            res[code] = float(pk)
    return res

def random_split(dataset_size, split_ratio=0.9, seed=0, shuffle=True):
    """random splitter"""
    np.random.seed(seed)
    indices = list(range(dataset_size))
    if shuffle:
        np.random.shuffle(indices)
    split = int(split_ratio * dataset_size)
    train_idx, valid_idx = indices[:split], indices[split:]
    return train_idx, valid_idx

def gen_df(coords, elements):
    assert len(coords) == len(elements)
    unified_elements = []
    xs, ys, zs = [coord[0] for coord in coords], [coord[1] for coord in coords], [coord[2] for coord in coords]
    for item in elements:
        if item in ['CL', 'Cl', 'Br', 'BR', 'AT', 'At', 'F', 'I']:
            element = 'Halogen'
        elif item in ['FE', 'ZN', 'MG', 'MN', 'K', 'LI', 'Ca', 'Hg']:
            element = 'Metal'
        else:
            element = item
        unified_elements.append(element)
    df = pd.DataFrame({'element': unified_elements, 'x': xs, 'y': ys, 'z': zs})
    return df
def process_complexes(root_dir, ids, annot_res):
    # p = PDBParser(QUIET=True)
    # nmr_files = []
    # uneuqal_to_two = []
    processed_complexes = []
    print("Start processing PDBBind2016 files...")
    for item in tqdm(ids):
        if item =='3t0x':
            continue
        # print(item)
        ligand = next(pybel.readfile('mol2', os.path.join(root_dir, item, '{}_ligand.mol2'.format(item))))
        pocket = next(pybel.readfile('pdb', os.path.join(root_dir, item, '{}_pocket.pdb'.format(item))))
        protein = next(pybel.readfile('pdb', os.path.join(root_dir, item, '{}_protein.pdb'.format(item))))
        hoh_pocket = 0
        hoh_protein = 0
        with open(os.path.join(root_dir, item, '{}_pocket.pdb'.format(item)), 'r') as f:
            lines = f.readlines()
            for line in lines:
                if 'HETATM' in line.split()[0]:
                    hoh_pocket += 1
        with open(os.path.join(root_dir, item, '{}_protein.pdb'.format(item)), 'r') as f:
            lines = f.readlines()
            for line in lines:
                if 'HETATM' in line.split()[0]:
                    hoh_protein += 1
        # print(hoh_pocket, hoh_protein)
        score = annot_res[item]
        ligand_coords = [atom.coords for atom in ligand]
        atom_map_lig = [atomic_num_dict(atom.atomicnum) for atom in ligand]
        if hoh_pocket == 0:
            pocket_coords = [atom.coords for atom in pocket]
            atom_map_pocket = [atomic_num_dict(atom.atomicnum) for atom in pocket]
        else: 
            pocket_coords = [atom.coords for atom in pocket][:-hoh_pocket]
            atom_map_pocket = [atomic_num_dict(atom.atomicnum) for atom in pocket][:-hoh_pocket]
        if hoh_protein == 0:
            protein_coords = [atom.coords for atom in protein]
            atom_map_protein = [atomic_num_dict(atom.atomicnum) for atom in protein]
        else:
            protein_coords = [atom.coords for atom in protein][:-hoh_protein]
            atom_map_protein = [atomic_num_dict(atom.atomicnum) for atom in protein][:-hoh_protein]



        # print(atom_map_lig)
        # print(atom_map_pocket)
        atoms_ligand = gen_df(ligand_coords, atom_map_lig)
        atoms_pocket = gen_df(pocket_coords, atom_map_pocket)        
        atoms_protein = gen_df(protein_coords, atom_map_protein)
        
        processed_complex = {'id': item, 'scores':{'neglog_aff': score}, 'atoms_protein': atoms_protein, 'atoms_pocket': atoms_pocket, 'atoms_ligand': atoms_ligand}
        processed_complexes.append(processed_complex)
    return processed_complexes


if __name__ == '__main__':
    cache_root = './data/PDBbind/'
    root_dir = './data/PDBbind/refined-set/'
    test_dir = './data/PDBbind_2016/core-set/'
    annot_dir = './data/PDBbind/refined-set/index/INDEX_general_PL_data.2020'
    complex_ids = os.listdir(root_dir)
    test_ids = os.listdir(test_dir)
    train_val_ids = [x for x in complex_ids if x not in test_ids and x not in ['index', 'readme']]
    annot_res = load_pk_data(annot_dir)
    train_idx, valid_idx = random_split(len(train_val_ids), split_ratio=0.9, seed=2020, shuffle=True)
    train_ids = [train_val_ids[i] for i in train_idx]
    valid_ids = [train_val_ids[i] for i in valid_idx]
    train_complexes = process_complexes(root_dir, train_ids, annot_res)
    val_complexes = process_complexes(root_dir, valid_ids, annot_res)
    test_complexes = process_complexes(test_dir, test_ids, annot_res)
    pickle.dump(train_complexes, open(os.path.join(cache_root, "pdbbind2016_train.pkl"), 'wb'))
    pickle.dump(val_complexes, open(os.path.join(cache_root, "pdbbind2016_val.pkl"), 'wb'))
    pickle.dump(test_complexes, open(os.path.join(cache_root, "pdbbind2016_test.pkl"), 'wb'))
    # print(len(train_val_ids))