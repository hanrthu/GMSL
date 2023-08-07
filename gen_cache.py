import os
import re
import pickle
import json
from Bio.PDB import PDBParser
import pandas as pd
from tqdm import tqdm
def test_func(item:str,label:str,split:str="train",cache_dir:str="./datasets/EnzymeCommissionNew/train.cache",remove_hoh:bool=True,remove_hydrogen:bool=True):
    pdb_id,chain_id = item.split("-")
    pdb_file_path = "./datasets/EnzymeCommission/all/{}.pdb".format(item)
    p = PDBParser()
    if(os.path.exists(pdb_file_path)):
        file_is_chain = True
    else:
        pdb_file_path = "./datasets/ProtFunc/all/{}.pdb".format(pdb_id)
        if(os.path.exists(pdb_file_path)):
            file_is_chain = False
        else:
            print("cannot find pdb file for chain:",item)
           
    try:
        structure = p.get_structure(item, pdb_file_path)
    except:
        print("cannot parse structure for chain:",item)
    model = structure[0]
    chains = list(model.get_chains())   
    if(file_is_chain==True):
        chain = chains[0]  
    else:
        for chain in chains:
            if(chain.id==chain_id):
                break
    if(chain.id!=chain_id):
        print("cannot find chain:",item)
    chains = [chain]
    pattern = re.compile(r'\d+H.')
    elements = []
    xs = []
    ys = []
    zs = []
    chain_ids = []
    protein_seq = []
    names = []
    resnames = []
    processed_complexes = []
    # chain = chains[0]
    for chain in chains:
        if chain.id == ' ':
            continue
        for residue in chain.get_residues():
            # 删除HOH原子
            if remove_hoh and residue.get_resname() == 'HOH':
                continue
            protein_seq.append(residue.get_resname())
            for atom in residue:
                # 删除氢原子
                atom_id = atom.get_id()
                if remove_hydrogen and atom.get_id().startswith('H') or pattern.match(atom.get_id()) != None:
                    continue
                if atom_id.startswith('H') or pattern.match(atom.get_id()) != None:
                    element = 'H'
                elif atom_id[0:2] in ['CL', 'Cl', 'Br', 'BR', 'AT', 'At']:
                    element = 'Halogen'
                elif atom_id[0:2] in ['FE', 'ZN', 'MG', 'MN', 'K', 'LI']:
                    element = 'Metal'
                elif atom_id[0] in ['F', 'I']:
                    element = 'Halogen'
                elif atom_id[0] in ['C','N','O','S','P']:
                    element = atom_id[0]
                else:
                    element = atom_id
                names.append(atom_id)
                elements.append(element)
                chain_ids.append(chain.id)
                resnames.append(residue.get_resname())
                x, y, z = atom.get_vector()
                xs.append(x)
                ys.append(y)
                zs.append(z)
    protein_df = pd.DataFrame({'chain': chain_ids, 'resname': resnames, 'element': elements, 'name': names, 'x': xs, 'y': ys, 'z': zs})
    processed_complex = {'complex_id': item, 'num_proteins': 1, 'labels': label,
                                    'atoms_protein': protein_df, 'protein_seq': protein_seq, 'atoms_ligand':None}
    processed_complexes.append(processed_complex)
    print("processed_complex:",processed_complex)
def gen_cache_for_ec_new(labels:dict,split:str="train",cache_dir:str="./datasets/EnzymeCommissionNew/train.cache",remove_hoh:bool=True,remove_hydrogen:bool=False):
    file_path = "./datasets/EnzymeCommissionNew/nrPDB-EC_{}.txt".format(split)
    file = open(file_path, "r")
    success_count = 0
    processed_complexes = []
    for item in tqdm(file.readlines()):
        item = item.strip()
        # print("item is:",item)
        if(item not in labels):
            print("cannot find label for chain:",item)
            continue
        pdb_id,chain_id = item.split("-")
        pdb_file_path = "./datasets/EnzymeCommission/all/{}.pdb".format(item)
        p = PDBParser()
        if(os.path.exists(pdb_file_path)):
            file_is_chain = True
        else:
            pdb_file_path = "./datasets/ProtFunc/all/{}.pdb".format(pdb_id)
            if(os.path.exists(pdb_file_path)):
                file_is_chain = False
            else:
                print("cannot find pdb file for chain:",item)
           
        try:
            structure = p.get_structure(item, pdb_file_path)
        except:
            print("cannot parse structure for chain:",item)
        model = structure[0]
        chains = list(model.get_chains())   
        if(file_is_chain==True):
            chain = chains[0]  
        else:
            for chain in chains:
                if(chain.id==chain_id):
                    break
        if(chain.id!=chain_id):
            print("cannot find chain:",item)
        chains = [chain]
        pattern = re.compile(r'\d+H.')
        elements = []
        xs = []
        ys = []
        zs = []
        chain_ids = []
        protein_seq = []
        names = []
        resnames = []
    
        for chain in chains:
            if chain.id == ' ':
                continue
            for residue in chain.get_residues():
                # 删除HOH原子
                if remove_hoh and residue.get_resname() == 'HOH':
                    continue
                protein_seq.append(residue.get_resname())
                for atom in residue:
                    # 删除氢原子
                    atom_id = atom.get_id()
                    if remove_hydrogen and atom.get_id().startswith('H') or pattern.match(atom.get_id()) != None:
                        continue
                    if atom_id.startswith('H') or pattern.match(atom.get_id()) != None:
                        element = 'H'
                    elif atom_id[0:2] in ['CL', 'Cl', 'Br', 'BR', 'AT', 'At']:
                        element = 'Halogen'
                    elif atom_id[0:2] in ['FE', 'ZN', 'MG', 'MN', 'K', 'LI']:
                        element = 'Metal'
                    elif atom_id[0] in ['F', 'I']:
                        element = 'Halogen'
                    elif atom_id[0] in ['C','N','O','S','P']:
                        element = atom_id[0]
                    else:
                        element = atom_id
                    names.append(atom_id)
                    elements.append(element)
                    chain_ids.append(chain.id)
                    resnames.append(residue.get_resname())
                    x, y, z = atom.get_vector()
                    xs.append(x)
                    ys.append(y)
                    zs.append(z)
        protein_df = pd.DataFrame({'chain': chain_ids, 'resname': resnames, 'element': elements, 'name': names, 'x': xs, 'y': ys, 'z': zs})
        processed_complex = {'complex_id': item, 'num_proteins': 1, 'labels': labels[item],
                                    'atoms_protein': protein_df, 'protein_seq': protein_seq, 'atoms_ligand':None}
        processed_complexes.append(processed_complex)
        success_count += 1
    pickle.dump(processed_complexes, open(cache_dir, 'wb'))
    # processed_complexes = pickle.load(open(cache_dir, 'rb'))
    # print("processed_complexes:",processed_complexes)
    print("success_count:",success_count)
if __name__ == "__main__":
    # test_func("4QW3-K","fake label")
    with open("./datasets/EnzymeCommissionNew/uniformed_labels.json", 'r') as f:
        labels = json.load(f)  
    for split in ["train","test","val"]:
        gen_cache_for_ec_new(labels,split,"./datasets/EnzymeCommissionNew/{}.cache".format(split))