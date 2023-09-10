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
def gen_protein_property_uniprots(json_dir: str, single=True):
    with open(json_dir, 'r') as f:
        info_dict = json.load(f)
    uniprot_dict = {}
    # print(len(info_dict))
    info_dict_new = {}
    for k, v in info_dict.items():
        if single:
            if len(v) != 1:
                continue
            else:
                uniprot_id = v[0]
                info_dict_new[k] = [uniprot_id]   
                # 只记录同一个uniprot id的所有pdb_id
                if uniprot_id not in uniprot_dict:     
                    uniprot_dict[uniprot_id] = k
                # else:
                #     uniprot_dict[uniprot_id].append(k)
        else:
            # Remove items whose uniprot ids are larger than 3 or equal zero.
            if len(v) == 0 or len(v) > 3:
                continue
            else:
                info_dict_new[k] = v
                for uniprot_id in v:
                    if uniprot_id not in uniprot_dict:
                        uniprot_dict[uniprot_id] = [k]
                    else:
                        uniprot_dict[uniprot_id].append(k)
    # print("Unitest:", len(uniprot_dict), len(info_dict_new))
    return uniprot_dict, info_dict_new

def gen_cache_for_multi_new_new():
    '''
    在multitask new的基础上，给每一个protein_df添加一列：domains(列表，因为每一个氨基酸可能属于多个域)
    修改labels
    '''
    

def gen_cache_for_multi_new_new():
    '''
    在multitask new的基础上，给每一个protein_df添加一列：domains(列表，因为每一个氨基酸可能属于多个域)
    修改labels
    '''
    src_complexes = []
    src_complexes_dict = dict()
    file_name_list = ['train','test','val',"train_all"]
    
    for cache_file_name in file_name_list:
        cache_file = open("./datasets/MultiTaskNew/" + cache_file_name + '.cache','rb')
        src_complexes.extend(pickle.load(cache_file))
        
    for a_complex in src_complexes:
        src_complexes_dict[a_complex['complex_id']] = a_complex
    
    tar_complexes = []
    new_labels = json.load(open('./datasets/MultiTaskNewNew/uniformed_labels.json'))
    domain_dict = json.load(open('./output_info/domain_info.json'))
    fold_uniprot_dict,fold_info_dict = gen_protein_property_uniprots('./output_info/Homology_uniprots.json',False)
    
    for file_name in file_name_list:
        f = open('./datasets/MultiTaskNewNew/'+file_name+'.txt')
        cache_dir = './datasets/MultiTaskNewNew/'+file_name +'.cache'
        for line in f.readlines():
            pdb_id = line.strip()
            pdb_id = "5d7j"
            try:
                a_complex = src_complexes_dict[pdb_id]
            except:
                print("{} not found in cache!".format(pdb_id))
                continue
            new_label = new_labels[pdb_id]
            
            protein_df = a_complex['atoms_protein']
            chain_ids = list(protein_df['chain'])
            domain_ids = [[]]*len(chain_ids)
            for idx,fold_uniprot_label in enumerate(new_label['fold']):
                
                if isinstance(fold_uniprot_label,int):
                    pass
                else:
                    uniprot_id = new_label['uniprots'][idx]
                    # 查看当前uniprot是当前pdb的几号链
                    for fold_candidate in fold_uniprot_label[uniprot_id]:
                        if(pdb_id in fold_candidate):
                            chain_id = fold_candidate[-2].upper()
                    for (fold_item,label) in fold_uniprot_label:
                        item_domain = domain_dict[fold_item].split(':')[-1]
                        if item_domain == "":
                            chain_id = chain_ids[0]
                            from_idx = 0
                            to_idx = len(chain_ids) # 左闭右开区间
                            pass # 整个链
                        else:
                            from_idx,to_idx = item_domain.split('-')
                            from_idx = from_idx -1
                            # 这里的from和to指的都是第几个氨基酸，要转化成是第几个原子
                        # 要和uniprot对应,根据uniprot id查找当前蛋白质的链
                            
                        for idx,ch in enumerate(chain_ids):          
                            if ch == chain_id:
                                base_idx = idx
                                break
                        try:
                            from_idx = base_idx + from_idx
                            to_idx = base_idx + to_idx
                        except:
                            print("{} chain not exist".format(chain_id))
                        for idx in range(from_idx,to_idx):
                            domain_ids[idx].append(fold_item)
            protein_df.insert(protein_df.shape[1],'domain_ids',domain_ids)  
            print("protein_df:")
            print(protein_df)       
            a_complex['atoms_protein'] = protein_df
            a_complex['labels'] = new_label
            tar_complexes.append(a_complex)
        pickle.dump(tar_complexes, open(cache_dir, 'wb'))
    
     
            
                        
                        
            
            
def gen_cache_for_reaction(labels:dict,split:str="train",cache_dir:str="./datasets/ProtFunc/train.cache",remove_hoh:bool=True,remove_hydrogen:bool=False):
    file_path = "./datasets/ProtFunc/{}.txt".format(split)
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
    # with open("./datasets/ProtFunc/uniformed_labels.json", 'r') as f:
    #     labels = json.load(f)  
    # for split in ["train","test","val"]:
    #     gen_cache_for_reaction(labels,split,"./datasets/ProtFunc/{}.cache".format(split))
    # labels = gen_label_for_fold()
    # for split in ["train","test_fold","test","test_superfamily","val"]:
    #     gen_cache_for_fold(labels,split,"./datasets/HomologyTAPE/"+split+".cache")
    gen_cache_for_multi_new_new()