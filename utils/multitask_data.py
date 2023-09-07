import os
import os.path as osp
from argparse import ArgumentParser
from typing import Dict, List, Tuple, Union
import gzip
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

from utils import MyData, prot_graph_transform
from utils.hetero_graph import hetero_graph_transform

pybel.ob.obErrorLog.SetOutputLevel(0)
atomic_num_dict = lambda x: {1: 'H', 2: 'HE', 3: 'LI', 4: 'BE', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 11: 'NA',
                   15: 'P', 16: 'S', 17: 'Cl', 20:'Ca', 25: 'MN', 26: 'FE', 30: 'ZN', 35: 'Br', 53: 'I', 80: 'Hg'}.get(x, 'Others')

class CustomMultiTaskDataset(Dataset):
    """
    The Custom MultiTask Dataset with uniform labels
    """
    def __init__(self, root_dir: str = './datasets/MultiTask', label_dir: str = './datasets/MultiTask/uniformed_labels.json',
                remove_hoh = True, remove_hydrogen = True, cutoff = 6, split : str = 'train', task = 'multi', hetero = False, alpha_only=False, info_dict='./output_info/uniprot_dict_test.json'):
        super(CustomMultiTaskDataset, self).__init__(root_dir)
        print("root,dir", root_dir, task)
        print("in Initializing MultiTask Dataset...")
        self.root_dir = root_dir
        self.cache_dir = os.path.join(root_dir, "{}.cache".format(split))
        with open(label_dir, 'r') as f:
            self.labels = json.load(f)
        self.remove_hoh = remove_hoh
        if remove_hydrogen:
            print("Removing Hydrogen...")
        self.remove_hydrogen = remove_hydrogen # 移除氢原子（必要，因为PP的文件里没有给氢原子，需要做到多任务统一）
        self.cutoff = cutoff
        self.hetero = hetero
        self.alpha_only = alpha_only
        file_dir = os.path.join(root_dir, split+'.txt')        
        # self.ec_root = './data/EnzymeCommission/all'
        self.info_dict = info_dict
        # self.ec_root = './data/EC/all'
        self.ec_root = './data/EnzymeCommission/all'
        self.go_root = './data/GeneOntology/all'
        # self.go_root = './data/GO/all'
        self.lba_root = './data/PDBbind/refined-set'
        self.pp_root = './data/PDBbind/PP'
        self.reaction_root = './data/ProtFunc/all'
        self.fold_root = './data/HomologyTAPE/all'
        with open(file_dir, 'r') as f:
            self.files = f.readlines()
            self.files = [i.strip() for i in self.files]
        if split not in ['train', 'val', 'test', 'train_all','train_ec', 'val_ec', 'test_ec']:
            print("Wrong selected split. Have to choose between ['train', 'val', 'test', 'test_all']")
            print("Exiting code")
            exit()
        if task not in ['affinity', 'ec', 'cc', 'mf', 'bp', 'multi', 'go', 'lba', 'ppi', 'reaction', 'fold']:
            print("Wrong selected task. Have to choose between ['affinity', 'ec', 'cc', 'mf', 'bp', 'multi', 'go', 'reaction', 'fold']")
            print("Exiting code")
            exit()
        self.split = split
        self.task = task
        self.process_complexes()
    def find_structure(self, item):
        self.ec_files = os.listdir(self.ec_root)
        self.go_files = os.listdir(self.go_root)
        self.lba_files = os.listdir(self.lba_root)
        self.pp_files = os.listdir(self.pp_root)
        self.reaction_files = os.listdir(self.reaction_root)
        self.fold_files = os.listdir(self.fold_root)
        pdb_files = os.listdir('./data/EC/all')
        if '-' in item:
            if item +'.pdb' in self.ec_files:
                return os.path.join(self.ec_root, item+'.pdb'), -1
            elif item+'.pdb' in self.go_files:
                return os.path.join(self.go_root, item+'.pdb'), -1
            elif item.split('-')[0] +'.pdb.gz' in self.ec_files:
                return os.path.join(self.ec_root, item.split('-')[0] +'.pdb.gz'), -1
            elif item.split('-')[0] +'.pdb' in self.ec_files:
                return os.path.join(self.ec_root, item.split('-')[0] +'.pdb'), -1
            elif item.split('-')[0] +'.pdb.gz' in self.go_files:
                return os.path.join(self.go_root, item.split('-')[0] +'.pdb.gz'), -1
            elif item.split('-')[0] +'.pdb' in self.go_files:
                return os.path.join(self.go_root, item.split('-')[0] +'.pdb'), -1
            elif item.split('-')[0] +'.pdb.gz' in pdb_files:
                return os.path.join('./data/EC/all', item.split('-')[0] +'.pdb.gz'), -1
            elif item.split('-')[0] +'.pdb' in pdb_files:
                return os.path.join('./data/EC/all', item.split('-')[0] +'.pdb'), -1
            elif item.split('-')[0]+'.pdb' in self.reaction_files:
                return os.path.join(self.reaction_root, item.split('-')[0]+'.pdb'), 0
            elif item.split('-')[0] +'.pdb.gz' in self.fold_files:
                return os.path.join(self.fold_root, item.split('-')[0] +'.pdb.gz'), -1
            elif item.split('-')[0] +'.pdb' in self.fold_files:
                return os.path.join(self.fold_root, item.split('-')[0] +'.pdb'), -1
        else:
            if item + '.ent.pdb' in self.pp_files:
                return os.path.join(self.pp_root, item+'.ent.pdb'), -1
            elif item in self.lba_files:
                protein_dir = os.path.join(self.lba_root, item, item + "_protein.pdb")
                ligand_dir = os.path.join(self.lba_root, item, item + '_ligand.mol2')
                return protein_dir, ligand_dir
        print("item", item)
        return -1, -1
    def gen_df(self, coords, elements):
        assert len(coords) == len(elements)
        unified_elements = []
        xs, ys, zs = [coord[0] for coord in coords], [coord[1] for coord in coords], [coord[2] for coord in coords]
        for item in elements:
            if item in ['CL', 'Cl', 'Br', 'BR', 'AT', 'At', 'F', 'I']:
                element = 'Halogen'
            elif item in ['FE', 'ZN', 'MG', 'MN', 'K', 'LI', 'Ca', 'Hg', 'NA']:
                element = 'Metal'
            elif item[0] in ['C','N','O','S','P','H']:
                element = item[0]
            else:
                element = item
            unified_elements.append(element)
        df = pd.DataFrame({'element': unified_elements, 'resname':'LIG', 'x': xs, 'y': ys, 'z': zs})
        return df
    def process_complexes(self):
        p = PDBParser(QUIET=True)
        nmr_files = []
        wrong_number = []
        self.processed_complexes = []
        corrupted = []
        # cache_dir = os.path.join(self.root_dir, self.cache_dir)
        if os.path.exists(self.cache_dir):
            print("Start loading cached Multitask files...")
            self.processed_complexes = pickle.load(open(self.cache_dir, 'rb'))
            print("Complexes Before Checking:", self.len())
            self.check_dataset()
            print("Complexes Before Task Selection:", self.len())
            self.choose_task_items()
            print("Dataset size:", self.len())
            if self.alpha_only:
                print("Only retaining Alpha Carbon atoms for the atom_df")
                self.retain_alpha_carbon()
        else:
            print("Cache not found! Start processing Multitask files...Total Number {}".format(len(self.files)))
            count = 0
            for score_idx, item in enumerate(tqdm(self.files)):
                structure_dir, ligand_dir = self.find_structure(item)
                if ligand_dir != -1 and ligand_dir != 0:
                    ligand = next(pybel.readfile('mol2', ligand_dir))
                    ligand_coords = [atom.coords for atom in ligand]
                    atom_map_lig = [atomic_num_dict(atom.atomicnum) for atom in ligand]
                    ligand_df = self.gen_df(ligand_coords, atom_map_lig)
                else:
                    ligand_df = None
                split_chain_flag = False
                # print(structure_dir)
                file_is_chain = False
                have_chain_id = False
                pdb_id = None
                chain_id = None
                try:
                    if '-' in structure_dir:
                        file_is_chain = True
                    if '.gz' in structure_dir or ligand_dir == 0:
                        if '-' in item:
                            pdb_id, chain_id = item.split('-')
                            file_is_chain = False
                            have_chain_id = True
                        file_handle = gzip.open(structure_dir, 'rt')
                        split_chain_flag = True
                    else:
                        file_handle = structure_dir
                    structure = p.get_structure(item, file_handle)
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
                if file_is_chain:
                    chain = chains[0]
                else:
                    if have_chain_id:
                        for chain in chains:
                            if chain.id == chain_id:
                                break
                if have_chain_id:
                    if chain.id != chain_id:
                        print('cannot find chain:', pdb_id, chain_id)
                chains = [chain]
                pattern = re.compile(r'\d+H.')
                processed_complex = {'complex_id': item, 'num_proteins': protein_numbers, 'labels': self.labels[item],
                                    'atoms_protein': [], 'protein_seq': [], 'atoms_ligand':ligand_df}
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
                        protein_seq.append(residue.get_resname())
                        for atom in residue:
                            # 删除氢原子
                            atom_id = atom.get_id()
                            if self.remove_hydrogen and atom.get_id().startswith('H') or pattern.match(atom.get_id()) != None:
                                continue
                            if atom_id.startswith('H') or pattern.match(atom.get_id()) != None:
                                element = 'H'
                            elif atom_id[0:2] in ['CL', 'Cl', 'Br', 'BR', 'AT', 'At']:
                                element = 'Halogen'
                            elif atom_id[0:2] in ['FE', 'ZN', 'MG', 'MN', 'K', 'LI', 'Ca', 'Hg', 'NA']:
                                element = 'Metal'
                            elif atom_id[0] in ['F', 'I']:
                                element = 'Halogen'
                            elif atom_id[0] in ['C','N','O','S','P']:
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
                        curr_residue += 1
                protein_df = pd.DataFrame({'chain': chain_ids, 'residue': residues, 'resname': resnames, 'element': elements, 'name': names, 'x': xs, 'y': ys, 'z': zs})
                processed_complex['atoms_protein'] = protein_df
                processed_complex['protein_seq'] = protein_seq
                # print(processed_complex)
                self.processed_complexes.append(processed_complex)
                count += 1
                # print("count:", count)
                # print(len(processed_complexes))
                # if count == 128:
                #     break
            print(len(self.processed_complexes))
            print("Structure processed Done, dumping...")
            print("Structures with Wrong numbers:", len(wrong_number), wrong_number)
            print("Structures with NMR methods:", len(nmr_files), nmr_files)
            print("Corrupted:", len(corrupted), corrupted)
            pickle.dump(self.processed_complexes, open(self.cache_dir, 'wb'))
            print("Complexes Before Checking:", self.len())
            self.check_dataset()
            print("Complexes Before Task Selection:", self.len())
            self.choose_task_items()
            print("Dataset size:", self.len())
            if self.alpha_only:
                print("Only retaining Alpha Carbon atoms for the atom_df")
                self.retain_alpha_carbon()
    def correctness_check(self, chain_uniprot_info, complx):
        #Annotation Correctness Check
        correct = True
        chain_ids = list(set(complx['atoms_protein']['chain']))
        if '-' not in complx['complex_id']:
            # if complx['complex_id'] == '4p3y':
            #     print(complx['complex_id'], id, uniprot_id, chain_ids, chain_uniprot_info[complx['complex_id']])
            #     print(labels_uniprot)
            for i, id in enumerate(chain_ids):
                if id in chain_uniprot_info[complx['complex_id']]:
                    uniprot_id = chain_uniprot_info[complx['complex_id']][id]
                    labels_uniprot = complx['labels']['uniprots']
                    if uniprot_id not in labels_uniprot:
                        print("Error, you shouldn't come here!")
                        correct = False
                        print(complx['complex_id'],id, uniprot_id, chain_ids, chain_uniprot_info[complx['complex_id']])
                        print(labels_uniprot)
        return correct
    def cal_length_thres(self, complxes):
        length_list = []
        for complx in complxes:
            length = len(complx['atoms_protein']['element'])
            length_list.append(length)
        sorted_list = sorted(length_list)
        thres = sorted_list[int(0.95*len(sorted_list))]
        print("Cutting Threshold of atom numbers:", thres)
        return thres
    def length_check(self, complx, thres):
        if len(complx['atoms_protein']['element']) > thres:
            return False
        else:
            return True
    def check_dataset(self):
        print("Checking the dataset...")
        info_root = self.info_dict
        print(info_root)
        with open(info_root, 'r') as f:
            chain_uniprot_info = json.load(f)
        thres = self.cal_length_thres(self.processed_complexes)
        if self.split == 'train':
            thres = 6712

        self.processed_complexes = [i for i in tqdm(self.processed_complexes) if self.length_check(i, thres) and self.correctness_check(chain_uniprot_info, i)]
    
    def retain_alpha_carbon(self):
        new_complexes = []
        for item in self.processed_complexes:
            protein_df = item['atoms_protein']
            # print("Original Nodes:", len(protein_df))
            new_protein_df = protein_df[protein_df.name == 'CA'].reset_index(drop=True)
            item['atoms_protein'] = new_protein_df
            # print("Retaining Alpha Carbons:", len(new_protein_df))
            new_complexes.append(item)
        self.processed_complexes = new_complexes
    
    def choose_task_items(self):
        # 根据不同的任务，训练单独的模型
        if self.split in ['train', 'val', 'test']:
            extra_dir = './datasets/MultiTask/{}.txt'.format(self.split)
            with open(extra_dir, 'r') as f:
                extra_info = f.readlines()
                extra_info = [i.strip() for i in extra_info]
        else:
            extra_info = []
        if self.task == 'ec':
            print("Using Enzyme Commission Dataset for training:")
            root_dir = './output_info/enzyme_commission_uniprots.json'
            with open(root_dir, 'r') as f:
                info_dict = json.load(f)
            new_complexes = []
            for item in self.processed_complexes:
                if item['complex_id'] in info_dict or item['complex_id'] in extra_info: #.keys()?
                    labels = item['labels']
                    annot_number = len(labels['uniprots'])
                    for j in range(annot_number):
                        labels['go'][j] = -1
                        labels['reaction'][j] = -1
                        labels['fold'][j] = -1
                    item['labels'] = labels
                    new_complexes.append(item)
            self.processed_complexes = new_complexes
            self.transform_func = GNNTransformEC(task=self.task, hetero=self.hetero, alpha_only=self.alpha_only)
            print("Using EC dataset and transformation")
        elif self.task in ['bp', 'mf', 'cc', 'go']:
            print("Using Gene Ontology {} Dataset for training:".format(self.split))
            # root_dir = './output_info/gene_ontology_uniprots.json'
            root_dir = './output_info/go_uniprots.json'
            print("now")
            with open(root_dir, 'r') as f:
                info_dict = json.load(f)
            new_complexes = []
            print("ok")
            for item in self.processed_complexes:
                if item['complex_id'] in info_dict or item['complex_id'] in extra_info:
                    labels = item['labels']
                    annot_number = len(labels['uniprots'])
                    for j in range(annot_number):
                        labels['ec'][j] = -1
                        labels['reaction'][j] = -1
                        labels['fold'][j] = -1
                    item['labels'] = labels
                    new_complexes.append(item)
            self.processed_complexes = new_complexes
            print([complx['complex_id'] for complx in new_complexes])
            self.transform_func = GNNTransformGO(task=self.task, hetero=self.hetero, alpha_only=self.alpha_only)
        elif self.task in ['affinity', 'lba', 'ppi']:
            print("Using Affinity Dataset for trainisng:")
            root_dir = './output_info/protein_protein_uniprots.json'
            with open(root_dir, 'r') as f:
                info_dict1 = json.load(f)
            root_dir2 = './output_info/protein_ligand_uniprots.json'
            with open(root_dir2, 'r') as f:
                info_dict2 = json.load(f)
            info_dict = {**info_dict1, **info_dict2}
            new_complexes = []
            for item in self.processed_complexes:
                if item['complex_id'] in info_dict or item['complex_id'] in extra_info:
                    labels = item['labels']
                    annot_number = len(labels['uniprots'])
                    for j in range(annot_number):
                        labels['ec'][j] = -1
                        labels['go'][j] = -1
                        labels['reaction'][j] = -1
                        labels['fold'][j] = -1
                    item['labels'] = labels
                    new_complexes.append(item)
            self.processed_complexes = new_complexes
            # print(new_complexes)
            self.transform_func = GNNTransformAffinity(task=self.task, hetero=self.hetero)
        elif self.task == "reaction":
            print("Using Reaction {} Dataset for training:".format(self.split))
            # root_dir = './output_info/gene_ontology_uniprots.json'
            root_dir = './output_info/reaction_uniprots.json'
            print("now")
            with open(root_dir, 'r') as f:
                info_dict = json.load(f)
            new_complexes = []
            print("ok")
            for item in self.processed_complexes:
                if item['complex_id'] in info_dict or item['complex_id'] in extra_info:
                    labels = item['labels']
                    annot_number = len(labels['uniprots'])
                    for j in range(annot_number):
                        labels['ec'][j] = -1
                        labels['go'][j] = -1
                    item['labels'] = labels
                    new_complexes.append(item)
            self.processed_complexes = new_complexes
            self.transform_func = GNNTransformReaction(task=self.task,hetero=self.hetero, alpha_only=self.alpha_only)
        elif self.task == "fold":
            print("Using Fold {} Dataset for training:".format(self.split))
            # root_dir = './output_info/gene_ontology_uniprots.json'
            root_dir = './output_info/fold_uniprots.json'
            print("now")
            with open(root_dir, 'r') as f:
                info_dict = json.load(f)
            new_complexes = []
            print("ok")
            for item in self.processed_complexes:
                if item['complex_id'] in info_dict or item['complex_id'] in extra_info:
                    labels = item['labels']
                    annot_number = len(labels['uniprots'])
                    for j in range(annot_number):
                        labels['ec'][j] = -1
                        labels['go'][j] = -1
                        labels['reaction'][j] = -1
                    item['labels'] = labels
                    new_complexes.append(item)
            self.processed_complexes = new_complexes
            self.transform_func = GNNTransformFold(task=self.task,hetero=self.hetero, alpha_only=self.alpha_only)
        else:
            self.transform_func = GNNTransformMultiTask(hetero=self.hetero, alpha_only=self.alpha_only)
    def len(self):
        return len(self.processed_complexes)
    def get(self, idx):
        # print(type(self.processed_complexes))
        return self.transform_func(self.processed_complexes[idx])

class GNNTransformReaction(object):
    def __init__(
        self,
        cutoff: float = 4.5,
        remove_hydrogens: bool = True,
        max_num_neighbors: int = 32,
        supernode: bool = False,
        offset_strategy: int = 0,
        task='bp', #可能是bp, mf, cc中的一个
        hetero=False,
        alpha_only=False
    ):
        self.cutoff = cutoff
        self.remove_hydrogens = remove_hydrogens
        self.max_num_neighbors = max_num_neighbors
        self.supernode = supernode
        self.offset_strategy = offset_strategy
        self.task = task
        self.hetero = hetero
        self.alpha_only=alpha_only

    def __call__(self, item: Dict) -> MyData:
        # # print("Using Transform Reaction")
        info_root = './output_info/uniprot_dict_all_reaction.json'
        with open(info_root, 'r') as f:
            chain_uniprot_info = json.load(f)
        
        ligand_df = item['atoms_ligand']
        protein_df = item["atoms_protein"]
        atom_df = protein_df
        if isinstance(ligand_df, pd.DataFrame):
            atom_df = pd.concat([protein_df, ligand_df], axis=0)
            if self.remove_hydrogens:
                # remove hydrogens
                atom_df = atom_df[atom_df.element != "H"].reset_index(drop=True)
            lig_flag = torch.zeros(atom_df.shape[0], dtype=torch.long)
            lig_flag[-len(ligand_df):] = 0
        else:
            atom_df = protein_df
            if self.remove_hydrogens:
                # remove hydrogens
                atom_df = atom_df[atom_df.element != "H"].reset_index(drop=True)
            lig_flag = torch.zeros(atom_df.shape[0], dtype=torch.long)
        chain_ids = list(set(protein_df['chain']))
        uniprot_ids = []
        labels = item["labels"]
        pf_ids = []
        #目前是按照肽链来区分不同的蛋白，为了便于Unprot分类
        for i, id in enumerate(chain_ids):
            lig_flag[torch.tensor(list(atom_df['chain'] == id))] = i + 1
            if '-' in item['complex_id']:
                pf_ids.append(0)
                break
            if id in chain_uniprot_info[item['complex_id']]:
                uniprot_id = chain_uniprot_info[item['complex_id']][id]
                uniprot_ids.append(uniprot_id)
                labels_uniprot = labels['uniprots']
                if uniprot_id in labels_uniprot:
                    for idx, u in enumerate(labels_uniprot):
                        if uniprot_id == u:
                            pf_ids.append(idx)
                            break
                else:
                    pf_ids.append(-1)
                    print("Error, you shouldn't come here!")
            else:
                pf_ids.append(-1)
        num_classes = 384
        
        if self.hetero:
            graph = hetero_graph_transform(
                item_name=item['complex_id'],atom_df=atom_df, super_node=self.supernode, flag=lig_flag, protein_seq=item['protein_seq'], alpha_only=self.alpha_only
            )
        else:
            graph = prot_graph_transform(
                atom_df=atom_df, cutoff=self.cutoff, max_num_neighbors=self.max_num_neighbors, flag=lig_flag, super_node=self.supernode, offset_strategy=self.offset_strategy
            )
        reaction = labels['reaction']
        graph.functions = []
        graph.valid_masks = []
        for i, pf_id in enumerate(pf_ids):
            if pf_id == -1:
                valid_mask = torch.zeros(num_classes)
                prop = torch.zeros(num_classes)
                graph.functions.append(prop)
                graph.valid_masks.append(valid_mask)
                continue
            valid_mask = torch.ones(num_classes)
            annotations = []
            reaction_annot = reaction[pf_id]
            if reaction_annot == -1:
                valid_mask[:] = 0
            else:
                annotations = reaction_annot
                
            prop = torch.zeros(num_classes).scatter_(0,torch.tensor(annotations),1)
            graph.functions.append(prop)
            graph.valid_masks.append(valid_mask)
        try:
            graph.functions = torch.vstack(graph.functions)
            graph.valid_masks = torch.vstack(graph.valid_masks)
        except:
            print("PF ids:", pf_ids)
            print(item['complex_id'], chain_ids, labels)
            print(len(graph.functions))
            print(pf_ids)
            print(graph.functions)
            raise RuntimeError    
        graph.chains = lig_flag[lig_flag!=0]
        # print(item['complex_id'])

        graph.lig_flag = lig_flag
        if len(chain_ids) != len(graph.functions):
            print(item['complex_id'])
            print(chain_ids)
            print(len(chain_ids), len(graph.functions))
        graph.prot_id = item["complex_id"]
        graph.type = self.task
        # print("Task Type:", graph.type)
        return graph 

class GNNTransformFold(object):
    def __init__(
        self,
        cutoff: float = 4.5,
        remove_hydrogens: bool = True,
        max_num_neighbors: int = 32,
        supernode: bool = False,
        offset_strategy: int = 0,
        task='reaction', #ec
        hetero=False,
        alpha_only=False
    ):
        self.cutoff = cutoff
        self.remove_hydrogens = remove_hydrogens
        self.max_num_neighbors = max_num_neighbors
        self.supernode = supernode
        self.offset_strategy = offset_strategy
        self.task = task
        self.hetero = hetero
        self.alpha_only=alpha_only

    def __call__(self, item: Dict) -> MyData:
        # # print("Using Transform EC")
        info_root = './output_info/uniprot_dict_all_fold.json'
        with open(info_root, 'r') as f:
            chain_uniprot_info = json.load(f)
        
        ligand_df = item['atoms_ligand']
        protein_df = item["atoms_protein"]
        atom_df = protein_df
        if isinstance(ligand_df, pd.DataFrame):
            atom_df = pd.concat([protein_df, ligand_df], axis=0)
            if self.remove_hydrogens:
                # remove hydrogens
                atom_df = atom_df[atom_df.element != "H"].reset_index(drop=True)
            lig_flag = torch.zeros(atom_df.shape[0], dtype=torch.long)
            lig_flag[-len(ligand_df):] = 0
        else:
            atom_df = protein_df
            if self.remove_hydrogens:
                # remove hydrogens
                atom_df = atom_df[atom_df.element != "H"].reset_index(drop=True)
            lig_flag = torch.zeros(atom_df.shape[0], dtype=torch.long)
        chain_ids = list(set(protein_df['chain']))
        uniprot_ids = []
        labels = item["labels"]
        pf_ids = []
        #目前是按照肽链来区分不同的蛋白，为了便于Unprot分类
        for i, id in enumerate(chain_ids):
            lig_flag[torch.tensor(list(atom_df['chain'] == id))] = i + 1
            if '-' in item['complex_id']:
                pf_ids.append(0)
                break
            if id in chain_uniprot_info[item['complex_id']]:
                uniprot_id = chain_uniprot_info[item['complex_id']][id]
                uniprot_ids.append(uniprot_id)
                labels_uniprot = labels['uniprots']
                if uniprot_id in labels_uniprot:
                    for idx, u in enumerate(labels_uniprot):
                        if uniprot_id == u:
                            pf_ids.append(idx)
                            break
                else:
                    pf_ids.append(-1)
                    print("Error, you shouldn't come here!")
            else:
                pf_ids.append(-1)
        num_classes = 1195
        if self.hetero:
            graph = hetero_graph_transform(
                item_name=item['complex_id'],atom_df=atom_df, super_node=self.supernode, flag=lig_flag, protein_seq=item['protein_seq'], alpha_only=self.alpha_only
            )
        else:
            graph = prot_graph_transform(
                atom_df=atom_df, cutoff=self.cutoff, max_num_neighbors=self.max_num_neighbors, flag=lig_flag, super_node=self.supernode, offset_strategy=self.offset_strategy
            )
        fold = labels['fold']
        graph.functions = []
        graph.valid_masks = []
        for i, pf_id in enumerate(pf_ids):
            if pf_id == -1:
                valid_mask = torch.zeros(num_classes)
                prop = torch.zeros(num_classes)
                graph.functions.append(prop)
                graph.valid_masks.append(valid_mask)
                continue
            valid_mask = torch.ones(num_classes)
            annotations = []
            fold_annot = fold[pf_id]
            if fold_annot == -1:
                valid_mask[:] = 0
            else:
                annotations = fold_annot
                
            prop = torch.zeros(num_classes).scatter_(0,torch.tensor(annotations),1)
            graph.functions.append(prop)
            graph.valid_masks.append(valid_mask)
        try:
            graph.functions = torch.vstack(graph.functions)
            graph.valid_masks = torch.vstack(graph.valid_masks)
        except:
            print("PF ids:", pf_ids)
            print(item['complex_id'], chain_ids, labels)
            print(len(graph.functions))
            print(pf_ids)
            print(graph.functions)
            raise RuntimeError    
        graph.chains = lig_flag[lig_flag!=0]
        # print(item['complex_id'])

        graph.lig_flag = lig_flag
        if len(chain_ids) != len(graph.functions):
            print(item['complex_id'])
            print(chain_ids)
            print(len(chain_ids), len(graph.functions))
        graph.prot_id = item["complex_id"]
        graph.type = self.task
        # print("Task Type:", graph.type)
        return graph 

class GNNTransformGO(object):
    def __init__(
        self,
        cutoff: float = 4.5,
        remove_hydrogens: bool = True,
        max_num_neighbors: int = 32,
        supernode: bool = False,
        offset_strategy: int = 0,
        task='bp', #可能是bp, mf, cc中的一个
        hetero=False,
        alpha_only=False
    ):
        self.cutoff = cutoff
        self.remove_hydrogens = remove_hydrogens
        self.max_num_neighbors = max_num_neighbors
        self.supernode = supernode
        self.offset_strategy = offset_strategy
        self.task = task
        self.hetero = hetero
        self.alpha_only = alpha_only

    def __call__(self, item: Dict) -> MyData:
        # info_root = './output_info/uniprot_dict_all_go.json'
        info_root = './output_info/uniprot_dict_all_reaction.json'
        with open(info_root, 'r') as f:
            chain_uniprot_info = json.load(f)
        # print("Using Transform {}".format(self.task))
        ligand_df = item['atoms_ligand']
        protein_df = item["atoms_protein"]
        atom_df = protein_df
        if isinstance(ligand_df, pd.DataFrame):
            atom_df = pd.concat([protein_df, ligand_df], axis=0)
            if self.remove_hydrogens:
                # remove hydrogens
                atom_df = atom_df[atom_df.element != "H"].reset_index(drop=True)
            lig_flag = torch.zeros(atom_df.shape[0], dtype=torch.long)
            lig_flag[-len(ligand_df):] = 0
        else:
            atom_df = protein_df
            if self.remove_hydrogens:
                # remove hydrogens
                atom_df = atom_df[atom_df.element != "H"].reset_index(drop=True)
            lig_flag = torch.zeros(atom_df.shape[0], dtype=torch.long)
        chain_ids = list(set(protein_df['chain']))
        uniprot_ids = []
        labels = item["labels"]
        pf_ids = []
        #目前是按照肽链来区分不同的蛋白，为了便于Unprot分类
        for i, id in enumerate(chain_ids):
            lig_flag[torch.tensor(list(atom_df['chain'] == id))] = i + 1
            if '-' in item['complex_id']:
                pf_ids.append(0)
                break
            if id in chain_uniprot_info[item['complex_id']]:
                uniprot_id = chain_uniprot_info[item['complex_id']][id]
                uniprot_ids.append(uniprot_id)
                labels_uniprot = labels['uniprots']
                if uniprot_id in labels_uniprot:
                    for idx, u in enumerate(labels_uniprot):
                        if uniprot_id == u:
                            pf_ids.append(idx)
                            break
                else:
                    pf_ids.append(-1)
                    print("Error, you shouldn't come here!")
            else:
                pf_ids.append(-1)
        # print("pf_id",pf_ids)        
        if self.task == 'mf':
            # num_classes = 490
            num_classes = 5348
        elif self.task == 'bp':
            # num_classes = 1944
            num_classes = 10285
        elif self.task == 'cc':
            # num_classes = 321
            num_classes = 1901
        elif self.task == 'go':
            # num_classes = 490 + 1944 + 321
            num_classes = 5348 + 10285 + 1901
        else:
            raise RuntimeError
        # 找个办法把chain和Uniprot对应起来，然后就可以查了
        if self.hetero:
            graph = hetero_graph_transform(
                item_name=item['complex_id'],atom_df=atom_df, super_node=self.supernode, flag=lig_flag, protein_seq=item['protein_seq'], alpha_only=self.alpha_only
            )
        else:
            graph = prot_graph_transform(
                atom_df=atom_df, cutoff=self.cutoff, max_num_neighbors=self.max_num_neighbors, flag=lig_flag, super_node=self.supernode, offset_strategy=self.offset_strategy
            )
        go = labels['go']
        # graph.y = torch.zeros(self.num_classes).scatter_(0,torch.tensor(labels),1)
        graph.functions = []
        graph.valid_masks = []
        if len(go) == 0:
            print(item['complex_id'])
        if item['complex_id'] == '3EWD-A':
            print(go)
        # print("pfids:", pf_ids)
        for i, pf_id in enumerate(pf_ids):
            if pf_id == -1:
                valid_mask = torch.zeros(num_classes)
                prop = torch.zeros(num_classes)
                graph.functions.append(prop)
                graph.valid_masks.append(valid_mask)
                continue
            valid_mask = torch.ones(num_classes)
            annotations = []
            try:
                go_annot = go[pf_id]
            except Exception as e:
                print(go)
                print(item['complex_id'])
            go_annot = go[pf_id]
            # print(go)
            # print("pf_id", pf_id)
            # print(go_annot)
            if item['complex_id'] == '3EWD-A':
                print(go_annot)
            if self.task == 'mf':
                mf_annot = go_annot['molecular_functions'] 
                mf_annot = [j for j in mf_annot]
                if len(mf_annot) == 0:
                    valid_mask[:] = 0
                annotations = mf_annot
            elif self.task == 'bp':
                bp_annot = go_annot['biological_process']
                bp_annot = [j for j in bp_annot]
                if len(bp_annot) == 0:
                    valid_mask[:] = 0
                annotations = bp_annot
            elif self.task == 'cc':
                cc_annot = go_annot['cellular_component']
                cc_annot = [j for j in cc_annot]
                if len(cc_annot) == 0:
                    valid_mask[:] = 0
                annotations = cc_annot
            elif self.task == 'go':
                # print(type(go_annot))
                # print(go_annot)
                # print(go_annot['molecular_functions'] )
                if isinstance(go_annot, int):
                    print(item['complex_id'])
                mf_annot = go_annot['molecular_functions'] 
                mf_annot = [j for j in mf_annot]
                if len(mf_annot) == 0:
                    # valid_mask[: 490] = 0
                    valid_mask[: 5348] = 0
                bp_annot = go_annot['biological_process']
                # bp_annot = [j + 490 for j in bp_annot]
                bp_annot = [j + 5348 for j in bp_annot]
                if len(bp_annot) == 0:
                    # valid_mask[490: 490+1944] = 0
                    valid_mask[5348: 5348+10285] = 0
                cc_annot = go_annot['cellular_component']
                # cc_annot = [j+490+1944 for j in cc_annot]
                cc_annot = [j+5348+10285 for j in cc_annot]
                if len(cc_annot) == 0:
                    # valid_mask[490+1944: ] = 0
                    valid_mask[5348+10285: ] = 0
                annotations = mf_annot + bp_annot + cc_annot
                
            prop = torch.zeros(num_classes).scatter_(0,torch.tensor(annotations),1)
            # print(prop)
            graph.functions.append(prop)
            graph.valid_masks.append(valid_mask)
        try:
            graph.functions = torch.vstack(graph.functions)
            graph.valid_masks = torch.vstack(graph.valid_masks)
        except:
            print("PF ids:", pf_ids)
            print(item['complex_id'], chain_ids, labels)
            print(len(graph.functions))
            print(pf_ids)
            print(graph.functions)
            raise RuntimeError
    
        graph.chains = lig_flag[lig_flag!=0]

        graph.lig_flag = lig_flag
        if len(chain_ids) != len(graph.functions):
            print(item['complex_id'])
            print(chain_ids)
            print(graph.function)
            print(len(chain_ids), len(graph.functions))
        graph.prot_id = item["complex_id"]
        graph.type = self.task
        return graph
    
class GNNTransformAffinity(object):
    def __init__(
        self,
        cutoff: float = 4.5,
        remove_hydrogens: bool = True,
        max_num_neighbors: int = 32,
        supernode: bool = False,
        offset_strategy: int = 0,
        task='affinity', #lba/ppi
        hetero=False,
        alpha_only=False,
    ):
        self.cutoff = cutoff
        self.remove_hydrogens = remove_hydrogens
        self.max_num_neighbors = max_num_neighbors
        self.supernode = supernode
        self.offset_strategy = offset_strategy
        self.task = task
        self.hetero = hetero
        self.alpha_only=alpha_only

    def __call__(self, item: Dict) -> MyData:
        # print("Using Transform Affinity")
        ligand_df = item["atoms_ligand"]
        protein_df = item["atoms_protein"]
        residue_df = protein_df.drop_duplicates(subset=['residue'], keep='first', inplace=False).reset_index(drop=True)
        if isinstance(ligand_df, pd.DataFrame):
            atom_df = pd.concat([protein_df, ligand_df], axis=0)
            res_ligand_df = pd.concat([residue_df, ligand_df], axis=0)
            if self.remove_hydrogens:
                # remove hydrogens
                atom_df = atom_df[atom_df.element != "H"].reset_index(drop=True)
                res_ligand_df = res_ligand_df[res_ligand_df.element != "H"].reset_index(drop=True)
            lig_flag = torch.zeros(res_ligand_df.shape[0], dtype=torch.long)
            lig_flag[-len(ligand_df):] = 0
        else:
            atom_df = protein_df
            if self.remove_hydrogens:
                # remove hydrogens
                atom_df = atom_df[atom_df.element != "H"].reset_index(drop=True)
            lig_flag = torch.zeros(atom_df.shape[0], dtype=torch.long)
        chain_ids = list(set(protein_df['chain']))
        for i, id in enumerate(chain_ids):
            lig_flag[torch.tensor(list(atom_df['chain'] == id))] = i + 1
        labels = item["labels"]
        #目前是按照肽链来区分不同的蛋白，为了便于Unprot分类
        if self.hetero:
            graph = hetero_graph_transform(
                item_name=item['complex_id'],atom_df=atom_df, super_node=self.supernode, flag=lig_flag, protein_seq=item['protein_seq'], alpha_only=self.alpha_only
            )
        else:
            graph = prot_graph_transform(
                atom_df=atom_df, cutoff=self.cutoff, max_num_neighbors=self.max_num_neighbors, flag=lig_flag, super_node=self.supernode, offset_strategy=self.offset_strategy
            )
        
        lba = labels['lba']
        ppi = labels['ppi']
        if lba != -1:
            affinity = lba
            graph.affinity_mask = torch.ones(1)
            graph.y = torch.FloatTensor([affinity])
        elif ppi != -1:
            affinity = ppi
            graph.affinity_mask = torch.ones(1)
            graph.y = torch.FloatTensor([affinity])
        else:
            graph.y = torch.FloatTensor([0])
            graph.affinity_mask = torch.zeros(1)
        graph.chains = lig_flag[lig_flag!=0]

        graph.lig_flag = lig_flag
        graph.prot_id = item["complex_id"]
        graph.type = self.task
        return graph
    

class GNNTransformEC(object):
    def __init__(
        self,
        cutoff: float = 4.5,
        remove_hydrogens: bool = True,
        max_num_neighbors: int = 32,
        supernode: bool = False,
        offset_strategy: int = 0,
        task='ec', #ec
        hetero=False,
        alpha_only=False
    ):
        self.cutoff = cutoff
        self.remove_hydrogens = remove_hydrogens
        self.max_num_neighbors = max_num_neighbors
        self.supernode = supernode
        self.offset_strategy = offset_strategy
        self.task = task
        self.hetero = hetero
        self.alpha_only=alpha_only

    def __call__(self, item: Dict) -> MyData:
        # print("Using Transform EC")
        info_root = './output_info/uniprot_dict_all_reaction.json'
        with open(info_root, 'r') as f:
            chain_uniprot_info = json.load(f)
        protein_df = item["atoms_protein"]
        atom_df = protein_df
        # residue_df = protein_df[protein_df.name == 'CA'].reset_index(drop=True)
        residue_df = protein_df.drop_duplicates(subset=['residue'], keep='first', inplace=False).reset_index(drop=True)
        
        if self.remove_hydrogens:
            # remove hydrogens
            atom_df = atom_df[atom_df.element != "H"].reset_index(drop=True)
            residue_df = residue_df[residue_df.element != "H"].reset_index(drop=True)
        lig_flag = torch.zeros(residue_df.shape[0], dtype=torch.long)
        chain_ids = list(set(protein_df['chain']))
        uniprot_ids = []
        labels = item["labels"]
        pf_ids = []
        #目前是按照肽链来区分不同的蛋白，为了便于Unprot分类
        for i, id in enumerate(chain_ids):
            lig_flag[torch.tensor(list(residue_df['chain'] == id))] = i + 1
            if '-' in item['complex_id']:
                pf_ids.append(0)
                break
            if id in chain_uniprot_info[item['complex_id']]:
                uniprot_id = chain_uniprot_info[item['complex_id']][id]
                uniprot_ids.append(uniprot_id)
                labels_uniprot = labels['uniprots']
                if uniprot_id in labels_uniprot:
                    for idx, u in enumerate(labels_uniprot):
                        if uniprot_id == u:
                            pf_ids.append(idx)
                            break
                else:
                    pf_ids.append(-1)
                    print("Error, you shouldn't come here!")
            else:
                pf_ids.append(-1)
        num_classes =538
        # num_classes = 3615
        # num_classes = 538
        if self.hetero:
            graph = hetero_graph_transform(
                item_name=item['complex_id'],atom_df=atom_df, super_node=self.supernode, protein_seq=item['protein_seq'], alpha_only=self.alpha_only
            )
        else:
            graph = prot_graph_transform(
                atom_df=atom_df, cutoff=self.cutoff, max_num_neighbors=self.max_num_neighbors, flag=lig_flag, super_node=self.supernode, offset_strategy=self.offset_strategy
            )
        ec = labels['ec']
        graph.functions = []
        graph.valid_masks = []
        for i, pf_id in enumerate(pf_ids):
            if pf_id == -1:
                valid_mask = torch.zeros(num_classes)
                prop = torch.zeros(num_classes)
                graph.functions.append(prop)
                graph.valid_masks.append(valid_mask)
                continue
            valid_mask = torch.ones(num_classes)
            annotations = []
            ec_annot = ec[pf_id]
            if ec_annot == -1:
                valid_mask[:] = 0
            else:
                annotations = ec_annot
                
            prop = torch.zeros(num_classes).scatter_(0,torch.tensor(annotations),1)
            graph.functions.append(prop)
            graph.valid_masks.append(valid_mask)
        try:
            graph.functions = torch.vstack(graph.functions)
            graph.valid_masks = torch.vstack(graph.valid_masks)
        except:
            print("PF ids:", pf_ids)
            print(item['complex_id'], chain_ids, labels)
            print(len(graph.functions))
            print(pf_ids)
            print(graph.functions)
            raise RuntimeError    
        graph.chains = lig_flag[lig_flag!=0]
        # print(item['complex_id'])

        graph.lig_flag = lig_flag
        if len(chain_ids) != len(graph.functions):
            print(item['complex_id'])
            print(chain_ids)
            print(len(chain_ids), len(graph.functions))
        graph.prot_id = item["complex_id"]
        graph.type = self.task
        # print("Task Type:", graph.type)
        return graph


class GNNTransformMultiTask(object):
    def __init__(
        self,
        cutoff: float = 4.5,
        remove_hydrogens: bool = True,
        max_num_neighbors: int = 32,
        supernode: bool = False,
        offset_strategy: int = 0,
        hetero = False,
        alpha_only = False
    ):
        self.cutoff = cutoff
        self.remove_hydrogens = remove_hydrogens
        self.max_num_neighbors = max_num_neighbors
        self.supernode = supernode
        self.offset_strategy = offset_strategy
        self.hetero = hetero
        self.alpha_only=alpha_only

    def __call__(self, item: Dict) -> MyData:
        # print("Using Transform LBA")
        info_root = './output_info/uniprot_dict_all_reaction.json'
        with open(info_root, 'r') as f:
            chain_uniprot_info = json.load(f)
        
        ligand_df = item["atoms_ligand"]
        protein_df = item["atoms_protein"]
        residue_df = protein_df.drop_duplicates(subset=['residue'], keep='first', inplace=False).reset_index(drop=True)
        if isinstance(ligand_df, pd.DataFrame):
            atom_df = pd.concat([protein_df, ligand_df], axis=0)
            res_ligand_df = pd.concat([residue_df, ligand_df], axis=0)
        else:
            atom_df = protein_df
            res_ligand_df = residue_df
        if self.remove_hydrogens:
            # remove hydrogens
            atom_df = atom_df[atom_df.element != "H"].reset_index(drop=True)
            res_ligand_df = res_ligand_df[res_ligand_df.element != "H"].reset_index(drop=True)
        # 此时ligand 所对应的atom被自动设置为0
        lig_flag = torch.zeros(res_ligand_df.shape[0], dtype=torch.long)
        chain_ids = list(set(protein_df['chain']))
        uniprot_ids = []
        labels = item["labels"]
        pf_ids = []
        #目前是按照肽链来区分不同的蛋白，为了便于Unprot分类
        for i, id in enumerate(chain_ids):
            lig_flag[torch.tensor(list(res_ligand_df['chain'] == id))] = i + 1
            if '-' in item['complex_id']:
                pf_ids.append(0)
                break
            if id in chain_uniprot_info[item['complex_id']]:
                uniprot_id = chain_uniprot_info[item['complex_id']][id]
                uniprot_ids.append(uniprot_id)
                labels_uniprot = labels['uniprots']
                if uniprot_id in labels_uniprot:
                    for idx, u in enumerate(labels_uniprot):
                        if uniprot_id == u:
                            pf_ids.append(idx)
                            break
                else:
                    pf_ids.append(-1)
                    print("Error, you shouldn't come here!")
            else:
                pf_ids.append(-1)
        # ec, mf, bp, cc
        # num_classes = 538 + 490 + 1944 + 321
        num_classes = [538, 490, 1944, 321, 384, 1195]
        total_classes = 538 + 490 + 1944 + 321 + 384 + 1195
        # num_classes = [3615, 490, 1944, 321]
        # total_classes = 3615 + 490 + 1944 + 321
        # num_classes = [3615, 5348, 10285, 1901]
        # total_classes = 3615 + 5348 + 10285 + 1901
        # 找个办法把chain和Uniprot对应起来，然后就可以查了
        if self.hetero:
            graph = hetero_graph_transform(
                item_name = item['complex_id'], atom_df=atom_df, super_node=self.supernode, protein_seq=item['protein_seq'], alpha_only=self.alpha_only
            )
        else:
            graph = prot_graph_transform(
                atom_df=atom_df, cutoff=self.cutoff, max_num_neighbors=self.max_num_neighbors, flag=lig_flag, super_node=self.supernode, offset_strategy=self.offset_strategy
            )
        lba = labels['lba']
        ppi = labels['ppi']
        ec = labels['ec']
        go = labels['go']
        reaction = labels['reaction']
        fold = labels['fold']
        graph.affinities = torch.FloatTensor([lba, ppi]).unsqueeze(0)
        if lba != -1:
            graph.affinity_mask = torch.tensor([1, 0]).unsqueeze(0)
        elif ppi != -1:
            graph.affinity_mask = torch.tensor([0, 1]).unsqueeze(0)
        else:
            graph.affinity_mask = torch.tensor([0, 0]).unsqueeze(0)

        graph.functions = []
        graph.valid_masks = []
        for i, pf_id in enumerate(pf_ids):
            if pf_id == -1:
                valid_mask = torch.zeros(len(num_classes))
                prop = torch.zeros(total_classes)
                graph.functions.append(prop)
                graph.valid_masks.append(valid_mask)
                continue
            valid_mask = torch.ones(len(num_classes))
            annotations = []
            ec_annot = ec[pf_id]
            go_annot = go[pf_id]
            reaction_annot = reaction[pf_id]
            fold_annot = fold[pf_id]
            if ec_annot == -1:
                valid_mask[0] = 0
            else:
                annotations = annotations + ec_annot
            if go_annot == -1:
                valid_mask[1:] = 0
            else:
                mf_annot = go_annot['molecular_functions'] 
                # mf_annot = [j + 538 for j in mf_annot]
                # mf_annot = [j + 3615 for j in mf_annot]
                mf_annot = [j + 538 for j in mf_annot]
                if len(mf_annot) == 0:
                    valid_mask[1] = 0
                bp_annot = go_annot['biological_process']
                bp_annot = [j + 538 + 490 for j in bp_annot]
                # bp_annot = [j + 3615 + 490 for j in bp_annot]
                # bp_annot = [j + 3615 + 5348 for j in bp_annot]
                if len(bp_annot) == 0:
                    valid_mask[2] = 0
                cc_annot = go_annot['cellular_component']
                cc_annot = [j + 538 + 490 + 1944 for j in cc_annot]
                # cc_annot = [j + 3615 + 490 + 1944 for j in cc_annot]
                # cc_annot = [j + 3615 + 5348 + 10285 for j in cc_annot]
                if len(cc_annot) == 0:
                    valid_mask[3] = 0
                annotations = annotations + mf_annot + bp_annot + cc_annot
            if reaction_annot == -1:
                valid_mask[4] = 0
            else:
                # print("reaction label valid!",reaction_annot)
                reaction_annot = [j+538 + 490 + 1944 + 321 for j in reaction_annot]
                annotations = annotations + reaction_annot  
            if fold_annot == -1:
                valid_mask[5] = 0
            else:
                fold_annot = [j + 538 + 490 + 1944 + 321 + 384 for j in reaction_annot]
                annotations = annotations + fold_annot   
            prop = torch.zeros(total_classes).scatter_(0,torch.tensor(annotations),1)
            graph.functions.append(prop)
            graph.valid_masks.append(valid_mask)
        try:
            graph.functions = torch.vstack(graph.functions)
            graph.valid_masks = torch.vstack(graph.valid_masks)
        except:
            print("PF ids:", pf_ids)
            print(item['complex_id'], chain_ids, labels)
            print(len(graph.functions))
            print(pf_ids)
            print(graph.functions)
            raise RuntimeError
        graph.chains = lig_flag[lig_flag!=0]
        # print(item['complex_id'])

        graph.lig_flag = lig_flag
        if len(chain_ids) != len(graph.functions):
            print(item['complex_id'])
            print(chain_ids)
            print(len(chain_ids), len(graph.functions))
        graph.prot_id = item["complex_id"]        
        graph.type = 'multi'
        return graph