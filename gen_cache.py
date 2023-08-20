import os
import re
import pickle
import json
from Bio.PDB import PDBParser
import pandas as pd
from tqdm import tqdm
import h5py
import numpy as np
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
def gen_label_for_fold():
    base_dir = "./datasets/HomologyTAPE/"
    output_path = base_dir +"uniformed_labels.json"
    class_map = open(base_dir + "class_map.txt",'r')
    input_lines = class_map.readlines()
    label_dict = {}
    fold_class_dict = {} # key:a.1	value:0
    uniprot_dict = json.load(open('./output_info/Homology_uniprots.json','r'))
    for input_line in input_lines:
        fold_class,num = input_line.strip('\n').split("\t")
        fold_class_dict[fold_class] = int(num)
    for split_file_name in ['train.txt','test.txt','test_fold.txt','test_superfamily.txt','val.txt']:
        split_file = open(base_dir + split_file_name,'r')
        lines = split_file.readlines()
        for line in lines:
            line = line.strip('\n').split('\t')
            prot_name = line[0]
            label = line[-1]
            if(prot_name in uniprot_dict.keys() and len(uniprot_dict[prot_name])>0):
                label_dict[prot_name] = {'uniprots':uniprot_dict[prot_name],'ec':[-1],'go':[-1],'fold':[[fold_class_dict[label]]]}
    label_file = open(output_path,'w')
    json.dump(label_dict,label_file)
    return label_dict
def gen_cache_for_fold(labels,split:str="training",cache_dir:str="./datasets/HomologyTape/train.cache",remove_hoh:bool=True,remove_hydrogen:bool=False):
    base_dir = "./datasets/HomologyTAPE/"
    split_file = open(base_dir + split + ".txt")
    processed_complexes = []
    succ = 0
    for line in tqdm(split_file.readlines()):
        item = line.split('\t')[0]
        if(item not in labels.keys()):
            continue
        hdf5_file_path = base_dir + split + "/" + item + ".hdf5"
        if(not os.path.exists(hdf5_file_path)):
            # print("cannot find hdf5 file for ",item)
            continue
        # else:
        #     print("find item")
        h5File = h5py.File(hdf5_file_path,'r')
        auxAtomNames = h5File["atom_names"][()]
        auxAtomResNames = h5File["atom_residue_names"][()]
        auxAtomChainNames = h5File["atom_chain_names"][()]
        atomNames_ = np.array([curName.decode('utf8') for curName in auxAtomNames])
        atomResidueNames_ = np.array([curName.decode('utf8') for curName in auxAtomResNames])
        atomChainNames_ = np.array([curName.decode('utf8') for curName in auxAtomChainNames])
        atomPos_ = h5File["atom_pos"][()][0]
        atomResidueIds_ = h5File["atom_residue_id"][()]
        
        elements = []
        xs = []
        ys = []
        zs = []
        chain_ids = []
        protein_seq = []
        names = []
        resnames = []
        
        pattern = re.compile(r'\d+H.')
        
        atom_num = len(atomNames_.tolist())
        for idx in range(atom_num):
            
            if(atomChainNames_[idx]==' '):
                continue
            if remove_hoh and atomResidueNames_[idx] == 'HOH':
                continue
            atom_id = atomNames_[idx]
            if remove_hydrogen and atom_id.startswith('H') or pattern.match(atom_id) != None:
                continue
            if(idx==0 or atomResidueIds_[idx]!= atomResidueIds_[idx-1]):
                protein_seq.append(atomResidueNames_[idx])
            if atom_id.startswith('H') or pattern.match(atom_id) != None:
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
            chain_ids.append(item)
            resnames.append(atomResidueNames_[idx])
            xs.append(atomPos_[idx][0])
            ys.append(atomPos_[idx][1])
            zs.append(atomPos_[idx][2])
        protein_df = pd.DataFrame({'chain': chain_ids, 'resname': resnames, 'element': elements, 'name': names, 'x': xs, 'y': ys, 'z': zs})
        processed_complex = {'complex_id': item, 'num_proteins': 1, 'labels': labels[item],
                                    'atoms_protein': protein_df, 'protein_seq': protein_seq, 'atoms_ligand':None}
        processed_complexes.append(processed_complex)
        succ += 1
    pickle.dump(processed_complexes, open(cache_dir, 'wb'))
    print("succ times:",succ)
if __name__ == "__main__":
    # test_func("4QW3-K","fake label")
    # with open("./datasets/EnzymeCommissionNew/uniformed_labels.json", 'r') as f:
    #     labels = json.load(f)  
    # for split in ["train","test","val"]:
    #     gen_cache_for_ec_new(labels,split,"./datasets/EnzymeCommissionNew/{}.cache".format(split))
    labels = gen_label_for_fold()
    for split in ["train","test_fold","test","test_superfamily","val"]:
        gen_cache_for_fold(labels,split,"./datasets/HomologyTAPE/"+split+".cache")