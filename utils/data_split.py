import os
import os.path as osp
from argparse import ArgumentParser
from typing import Dict, List, Tuple, Union
import random
import pandas as pd
import torch
# from atom3d.datasets import deserialize
from Bio.PDB import PDBParser
import re
import pickle
import io
import json
import gzip

from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import logging
import lmdb
from pathlib import Path
import numpy as np
import openbabel
from openbabel import pybel
pdb_fasta_map = {
    "ALA":"A",
    "ARG":"R",
    "ASN":"N",
    "ASP":"D",
    "CYS":"C",
    "GLN":"Q",
    "GLU":"E",
    "GLY":"G",
    "HIS":"H",
    "ILE":"I",
    "LEU":"L",
    "LYS":"K",
    "MET":"M",
    "PHE":"F",
    "PRO":"P",
    "SER":"S",
    "THR":"T",
    "TRP":"W",
    "TYR":"Y",
    "VAL":"V"
}
ec_root = "./datasets/EnzymeCommission/all"
go_root = "./datasets/GeneOntology/all"
lba_root = './datasets/PDBbind/refined-set'
pp_root = './datasets/PDBbind/PP'
reaction_root = './datasets/ProtFunc/all'
def find_structure(item,ec_files,go_files,lba_files,pp_files,reaction_files):
        # ec_files = os.listdir(ec_root)
        # go_files = os.listdir(go_root)
        # lba_files = os.listdir(lba_root)
        # pp_files = os.listdir(pp_root)
        # reaction_files = os.listdir(reaction_root)
        
        if '-' in item:
            if item+'.pdb' in ec_files:
                return os.path.join(ec_root, item+'.pdb'),True,False
            elif item+'.pdb' in go_files:
                return os.path.join(go_root, item+'.pdb'),True,False
            elif item.split('-')[0]+'.pdb' in reaction_files:
                return os.path.join(reaction_root, item.split('-')[0]+'.pdb'), True,True
        else:
            if item + '.ent.pdb' in pp_files:
                return os.path.join(pp_root, item+'.ent.pdb'), False,False
            elif item in lba_files:
                protein_dir = os.path.join(lba_root, item, item + "_protein.pdb")
                # ligand_dir = os.path.join(lba_root, item, item + '_ligand.mol2')
                return protein_dir, False,False
        print(item)
        return -1, False,False
def get_fasta_seq(p,file_path:str="./datasets/PDBbind/refined-set/6i11/6i11_protein.pdb",item:str="6i11",is_chain:bool=False,reaction:bool=False):
    
    
    try:
        structure = p.get_structure(item,file_path)
    except:
        return []
    model = structure[0]
    chains = list(model.get_chains())
    if reaction:
        chain_id = item.split('-')[1]
        for chain in chains:
            if(chain.id==chain_id):
                break
        if(chain.id!=chain_id):
            return []
        chains = [chain]
        
    fasta_data = []
    fasta_id = item
    # print("before in chains")
    # fasta_data.append(">"+fasta_id+'\n')
    # fasta_seq = ""
    for chain in chains:
        # print("chain.id:",len(chain.id))
        if chain.id ==" ":
            continue
        if is_chain :
            fasta_id = item
        else:
            fasta_id = item.upper() + "-" + chain.id
        fasta_data.append(">"+fasta_id+'\n')
        fasta_seq = ""
        for idx,residue in enumerate(chain.get_residues()):
            # print("{} {}".format(idx,residue.get_resname()))
            pdb_res_name = residue.get_resname() # eg:HOH
            
            if pdb_res_name in pdb_fasta_map.keys():
                fasta_seq += pdb_fasta_map[pdb_res_name]
            # else:
            #     if pdb_res_name != "HOH":
            #         print("{} {} is not a res".format(item,pdb_res_name))
            #         continue
        # print("len:",len(fasta_seq)-1)
        for i in range(int((len(fasta_seq)-1)/60)):
            # print("from:{},to:{}".format(i*60,(i+1)*60))
            fasta_data.append(fasta_seq[i*60:(i+1)*60]+'\n')
            # print(fasta_data[-1])
        i = int((len(fasta_seq)-1)/60)
        # print("from:{},to:{}".format(i*60,len(fasta_seq)))
        
        fasta_data.append(fasta_seq[i*60:]+'\n')
        # print(fasta_data[-1])
    # print(fasta_data)
    return fasta_data # ['>1A0A-A', 'MKRESHKHAEQARRNRLAVALHELASLIPAEWKQQNVSAAPSKATTVEAACRYIRHLQQNGST']
def gen_fasta_data_all():
    file_name_list = ['train_all.txt','train.txt','test.txt','val.txt']
    root_dir = "./datasets/MultiTask/"
    output_dir = './output_info/org_DB.fasta'
    item_list = []
    fasta_data = []
    p = PDBParser(QUIET=True)
    for file_name in file_name_list:
        f = open(root_dir + file_name)
        for line in f.readlines():
            item = line.strip('\n')
            item_list.append(item)
    ec_files = os.listdir(ec_root)
    go_files = os.listdir(go_root)
    lba_files = os.listdir(lba_root)
    pp_files = os.listdir(pp_root)
    reaction_files = os.listdir(reaction_root)
    for item in tqdm(item_list):
        pdb_path,is_chain,reaction = find_structure(item,ec_files,go_files,lba_files,pp_files,reaction_files)
        if pdb_path == -1:
            continue
        fasta_data.extend(get_fasta_seq(p,pdb_path,item,is_chain,reaction))
    output_file = open(output_dir,'w')
    output_file.writelines(fasta_data)

def load_chain_clu(clu_tsv_path:str = "../mmseqs/pdbDB_clu.tsv"):
    clusters = list()
    df = pd.read_csv(clu_tsv_path,sep = '\t')
    df.columns = ['clu_name','member']
    groups = df.groupby('clu_name')
    for (clu_name,group) in groups:
        clu = set()
        for idx,row in group.iterrows():
            
            clu.add(row['member'])
        # print(clu)
        clusters.append(clu)
    return clusters
def get_split_with_pdb_fasta(pdb_clus:list,test_ratio=0.35,val_ratio=0.4):
    print("cluster amount:",len(pdb_clus))
    full_label_items = set()
    lack_label_items = set()
    root_dir = './datasets/MultiTask/'
    for file_name in ['train.txt','test.txt','val.txt']:
        file = open(root_dir+file_name)
        for line in file.readlines():
            item = line.strip('\n')
            full_label_items.add(item)
            
    file = open(root_dir+'train_all.txt')
    for line in file.readlines():
        item = line.strip('\n')
        if item not in full_label_items:
            lack_label_items.add(item)
    print("size of lack label:",len(lack_label_items))
    random.shuffle(pdb_clus)
    test_set = set()
    train_val_set = set()
    train_all_set = set()
    for idx,clu in enumerate(pdb_clus):
        test_set = test_set.union(clu.intersection(full_label_items))
        if len(test_set)>= test_ratio * len(full_label_items):
            pdb_clus = pdb_clus[(idx+1):]
            print("break")
            break
    test_set = list(test_set)
    print("size of test set:",len(test_set))
    for idx,clu in enumerate(pdb_clus):
        train_val_set = train_val_set.union(clu.intersection(full_label_items))
        train_all_set = train_all_set.union(clu.intersection(lack_label_items))
        
    train_val_set = list(train_val_set)
    random.shuffle(train_val_set)
    print("shuffled")
    val_set = train_val_set[:int(val_ratio*len(full_label_items))]
    train_set = train_val_set[int(val_ratio*len(full_label_items)):]
    print("test train val train_all:",len(test_set),len(train_set),len(val_set),len(train_all_set))
    # 在method2中保存split结果
    split_list = ['train.txt','test.txt','val.txt','train_all.txt']
    set_list = [train_set,test_set,val_set,train_all_set]
    for idx in range(len(split_list)):
        output_file = open('./output_info/split/method2/'+split_list[idx],'w')
        data = set_list[idx]
        data = [item+'\n' for item in data]
        output_file.writelines(data)
    
def merge_list(L):
    lenth = len(L)
    for i in tqdm(range(1, lenth)):
        for j in range(i):
            if L[i] == {0} or L[j] == {0}:
                continue
            x = L[i].union(L[j])
            y = len(L[i]) + len(L[j])
            if len(x) < y:
                L[i] = x
                L[j] = {0}

    return [i for i in L if i != {0}]
     
def get_split_with_chain_fasta(chain_clus:list,test_ratio=0.35,val_ratio=0.4):
    print("test ratio:",test_ratio)
    # step 1: 把所有的chain用item替代 
    full_label_items = set()
    lack_label_items = set()
    clu_size = torch.zeros(len(chain_clus)) # 记录每个raw clu的item个数
    full_label_clu_size = torch.zeros(len(chain_clus)) # 记录每个raw clu的full label的item个数
    raw_pdb_clu = [ set() for _ in range(len(chain_clus))]
    root_dir = './datasets/MultiTask/'
    for file_name in ['train.txt','test.txt','val.txt']:
        file = open(root_dir+file_name)
        for line in file.readlines():
            item = line.strip('\n')
            full_label_items.add(item)
    
    print("full label item:",len(full_label_items))        
    file = open(root_dir+'train_all.txt')
    for line in file.readlines():
        item = line.strip('\n')
        if item not in full_label_items:
            lack_label_items.add(item)
        
    for idx,chain_clu in enumerate(chain_clus):
        for chain in chain_clu:
            pdb_item = chain.split('-')[0].lower()
            if chain in lack_label_items:
                raw_pdb_clu[idx].add(chain)
                clu_size[idx] += 1
            elif chain in full_label_items:
                raw_pdb_clu[idx].add(chain)
                clu_size[idx] += 1
                full_label_clu_size[idx] += 1
            elif pdb_item in lack_label_items:
                raw_pdb_clu[idx].add(pdb_item)
                clu_size[idx] += 1
            elif pdb_item in full_label_items:
                raw_pdb_clu[idx].add(pdb_item)
                clu_size[idx] += 1
                full_label_clu_size[idx] += 1
            else:
                print("{} not in train_all train test or val".format(chain))
    
    raw_pdb_clu = merge_list(raw_pdb_clu)
    print("cluster amount after merge:",len(raw_pdb_clu))
    # 检查merge后的clu之间是否有交集
    # for i in range(len(raw_pdb_clu)):
    #     for j in range(len(raw_pdb_clu)):
    #         if i == j :
    #             continue
    #         else:
    #             if len(raw_pdb_clu[i].intersection(raw_pdb_clu[j]) ) > 0 :
    #                 print("the merge list has instersection!")
    # print("clu len after merge:",len(raw_pdb_clu))
    while(1):
        test_set = set()
        random.shuffle(raw_pdb_clu)
        for idx,clu in enumerate(raw_pdb_clu):
            test_set = test_set.union(clu.intersection(full_label_items))
        # print("testset:",len(test_set))
            if len(test_set) >= test_ratio*len(full_label_items):
            
                print("break")
                break
        if len(test_set) < (test_ratio+0.1)*len(full_label_items):
            raw_pdb_clu = raw_pdb_clu[(idx+1):]
            break
    # test_set = list(test_set)
    print("test set size:",len(test_set))
    rest_set = set()      
    for clu in raw_pdb_clu:
        if len(clu.intersection(test_set))==0:
            rest_set = rest_set.union(clu)
        else:
            print("this should not happen!")
    # 把全标签的分割成train和val,剩下的作为train_all
    test_set = list(test_set)
    train_all_set = list(rest_set.intersection(lack_label_items))
    train_val_set = list(rest_set.intersection(full_label_items))
    random.shuffle(train_val_set)
    
    val_set = train_val_set[:int(val_ratio*len(full_label_items))]
    train_set = train_val_set[int(val_ratio*len(full_label_items)):]
    print("train set:",len(train_set))
    print("val set:",len(val_set))
    split_list = ['train.txt','test.txt','val.txt','train_all.txt']
    set_list = [train_set,test_set,val_set,train_all_set]
    for idx in range(len(split_list)):
        output_file = open('./output_info/split/method1/'+split_list[idx],'w')
        data = set_list[idx]
        data = [item+'\n' for item in data]
        output_file.writelines(data)
    return train_all_set,train_set,val_set,test_set
    
            
    
    
if __name__ == "__main__":
    # gen_fasta_data_all()
    # get_split_with_chain_fasta(load_chain_clu(clu_tsv_path="../mmseqs/orgDB_clu.tsv"))
    get_split_with_pdb_fasta(load_chain_clu(clu_tsv_path="../mmseqs/orgpdbDB_clu.tsv"))
    
    