import os
import os.path as osp
from argparse import ArgumentParser
from typing import Dict, List, Tuple, Union

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

from utils import MyData, prot_graph_transform, hetero_graph_transform

pybel.ob.obErrorLog.SetOutputLevel(0)
atomic_num_dict = lambda x: {1: 'H', 2: 'HE', 3: 'LI', 4: 'BE', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 11: 'NA',
                   15: 'P', 16: 'S', 17: 'Cl', 20:'Ca', 25: 'MN', 26: 'FE', 30: 'ZN', 35: 'Br', 53: 'I', 80: 'Hg'}.get(x, 'Others')

class CustomMultiTaskDataset(Dataset):
    """
    The Custom MultiTask Dataset with uniform labels
    """
    def __init__(self, root_dir: str = './datasets/MultiTask', label_dir: str = './datasets/MultiTask/uniformed_labels.json',
                remove_hoh = True, remove_hydrogen = False, cutoff = 6, split : str = 'train', task = 'multi', gearnet = False):
        super(CustomMultiTaskDataset, self).__init__(root_dir)
        print("Initializing MultiTask Dataset...")
        self.root_dir = root_dir
        self.cache_dir = os.path.join(root_dir, "{}.cache".format(split))
        with open(label_dir, 'r') as f:
            self.labels = json.load(f)
        self.remove_hoh = remove_hoh
        self.remove_hydrogen = remove_hydrogen
        self.cutoff = cutoff
        self.gearnet = gearnet
        file_dir = os.path.join(root_dir, split+'.txt')        
        self.ec_root = './datasets/EnzymeCommission/all'
        self.go_root = './datasets/GeneOntology/all'
        self.lba_root = './datasets/PDBbind/refined-set'
        self.pp_root = './datasets/PDBbind/PP'
        self.ec_files = os.listdir(self.ec_root)
        self.go_files = os.listdir(self.go_root)
        self.lba_files = os.listdir(self.lba_root)
        self.pp_files = os.listdir(self.pp_root)
        with open(file_dir, 'r') as f:
            self.files = f.readlines()
            self.files = [i.strip() for i in self.files]
        if split not in ['train', 'val', 'test', 'train_all','train_ec', 'val_ec', 'test_ec']:
            print("Wrong selected split. Have to choose between ['train', 'val', 'test', 'test_all']")
            print("Exiting code")
            exit()
        if task not in ['affinity', 'ec', 'cc', 'mf', 'bp', 'multi', 'go']:
            print("Wrong selected task. Have to choose between ['affinity', 'ec', 'cc', 'mf', 'bp', 'multi', 'go']")
            print("Exiting code")
            exit()
        self.split = split
        self.task = task
        self.process_complexes()
    def find_structure(self, item):
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
    def gen_df(self, coords, elements):
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
            if self.gearnet:
                print("Only retaining Alpha Carbon atoms for the atom_df")
                self.retain_alpha_carbon()
        else:
            print("Cache not found! Start processing Multitask files...Total Number {}".format(len(self.files)))
            # count = 0
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
                                    'atoms_protein': [], 'protein_seq': [], 'atoms_ligand':ligand_df}
                elements = []
                xs = []
                ys = []
                zs = []
                chain_ids = []
                protein_seq = []
                names = []
                resnames = []
                # chain = chains[0]
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
                processed_complex['atoms_protein'] = protein_df
                processed_complex['protein_seq'] = protein_seq
                
                self.processed_complexes.append(processed_complex)
                # count += 1
                # if count == 128:
                #     break
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
            if self.gearnet:
                print("Only retaining Alpha Carbon atoms for the atom_df")
                self.retain_alpha_carbon()
    def correctness_check(self, chain_uniprot_info, complx):
        #Annotation Correctness Check
        correct = True
        chain_ids = list(set(complx['atoms_protein']['chain']))
        if '-' not in complx['complex_id']:
            for i, id in enumerate(chain_ids):
                if id in chain_uniprot_info[complx['complex_id']]:
                    uniprot_id = chain_uniprot_info[complx['complex_id']][id]
                    labels_uniprot = complx['labels']['uniprots']
                    if uniprot_id not in labels_uniprot:
                        print("Error, you shouldn't come here!")
                        correct = False
                        print(complx['complex_id'], chain_ids, chain_uniprot_info[complx['complex_id']])
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
        info_root = './output_info/uniprot_dict_all.json'
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
                if item['complex_id'] in info_dict or item['complex_id'] in extra_info:
                    labels = item['labels']
                    annot_number = len(labels['uniprots'])
                    for j in range(annot_number):
                        labels['go'][j] = -1
                    item['labels'] = labels
                    new_complexes.append(item)
            self.processed_complexes = new_complexes
            self.transform_func = GNNTransformEC(task=self.task, gearnet=self.gearnet)
            print("Using EC dataset and transformation")
        elif self.task in ['bp', 'mf', 'cc', 'go']:
            print("Using Gene Ontology {} Dataset for training:".format(self.split))
            root_dir = './output_info/gene_ontology_uniprots.json'
            with open(root_dir, 'r') as f:
                info_dict = json.load(f)
            new_complexes = []
            for item in self.processed_complexes:
                if item['complex_id'] in info_dict or item['complex_id'] in extra_info:
                    labels = item['labels']
                    annot_number = len(labels['uniprots'])
                    for j in range(annot_number):
                        labels['ec'][j] = -1
                    item['labels'] = labels
                    new_complexes.append(item)
            self.processed_complexes = new_complexes
            self.transform_func = GNNTransformGO(task=self.task, gearnet=self.gearnet)
        elif self.task == 'affinity':
            print("Using Affinity Dataset for training:")
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
                    item['labels'] = labels
                    new_complexes.append(item)
            self.processed_complexes = new_complexes
            self.transform_func = GNNTransformAffinity(task=self.task, gearnet=self.gearnet)
        else:
            self.transform_func = GNNTransformMultiTask(gearnet=self.gearnet)
    def len(self):
        return len(self.processed_complexes)
    def get(self, idx):
        return self.transform_func(self.processed_complexes[idx])

class GNNTransformGO(object):
    def __init__(
        self,
        cutoff: float = 4.5,
        remove_hydrogens: bool = True,
        max_num_neighbors: int = 32,
        supernode: bool = False,
        offset_strategy: int = 0,
        task='bp', #可能是bp, mf, cc中的一个
        gearnet=False
    ):
        self.cutoff = cutoff
        self.remove_hydrogens = remove_hydrogens
        self.max_num_neighbors = max_num_neighbors
        self.supernode = supernode
        self.offset_strategy = offset_strategy
        self.task = task
        self.gearnet = gearnet

    def __call__(self, item: Dict) -> MyData:
        info_root = './output_info/uniprot_dict_all.json'
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
                
        if self.task == 'mf':
            num_classes = 490
        elif self.task == 'bp':
            num_classes = 1944
        elif self.task == 'cc':
            num_classes = 321
        elif self.task == 'go':
            num_classes = 490 + 1944 + 321
        else:
            raise RuntimeError
        # 找个办法把chain和Uniprot对应起来，然后就可以查了
        if self.gearnet:
            graph = hetero_graph_transform(
                atom_df=atom_df, super_node=self.supernode, flag=lig_flag, protein_seq=item['protein_seq']
            )
        else:
            graph = prot_graph_transform(
                atom_df=atom_df, cutoff=self.cutoff, max_num_neighbors=self.max_num_neighbors, flag=lig_flag, super_node=self.supernode, offset_strategy=self.offset_strategy
            )
        go = labels['go']
        # graph.y = torch.zeros(self.num_classes).scatter_(0,torch.tensor(labels),1)
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
            go_annot = go[pf_id]
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
                mf_annot = go_annot['molecular_functions'] 
                mf_annot = [j for j in mf_annot]
                if len(mf_annot) == 0:
                    valid_mask[: 490] = 0
                bp_annot = go_annot['biological_process']
                bp_annot = [j + 490 for j in bp_annot]
                if len(bp_annot) == 0:
                    valid_mask[490: 490+1944] = 0
                cc_annot = go_annot['cellular_component']
                cc_annot = [j+490+1944 for j in cc_annot]
                if len(cc_annot) == 0:
                    valid_mask[490+1944: ] = 0
                annotations = mf_annot + bp_annot + cc_annot
                
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

        graph.lig_flag = lig_flag
        if len(chain_ids) != len(graph.functions):
            print(item['complex_id'])
            print(chain_ids)
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
        gearnet=False
    ):
        self.cutoff = cutoff
        self.remove_hydrogens = remove_hydrogens
        self.max_num_neighbors = max_num_neighbors
        self.supernode = supernode
        self.offset_strategy = offset_strategy
        self.task = task
        self.gearnet = gearnet

    def __call__(self, item: Dict) -> MyData:
        # print("Using Transform Affinity")
        ligand_df = item["atoms_ligand"]
        protein_df = item["atoms_protein"]

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
        for i, id in enumerate(chain_ids):
            lig_flag[torch.tensor(list(atom_df['chain'] == id))] = i + 1
        labels = item["labels"]
        #目前是按照肽链来区分不同的蛋白，为了便于Unprot分类
        if self.gearnet:
            graph = hetero_graph_transform(
                atom_df=atom_df, super_node=self.supernode, flag=lig_flag, protein_seq=item['protein_seq']
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
        gearnet=False
    ):
        self.cutoff = cutoff
        self.remove_hydrogens = remove_hydrogens
        self.max_num_neighbors = max_num_neighbors
        self.supernode = supernode
        self.offset_strategy = offset_strategy
        self.task = task
        self.gearnet = gearnet

    def __call__(self, item: Dict) -> MyData:
        # print("Using Transform EC")
        info_root = './output_info/uniprot_dict_all.json'
        with open(info_root, 'r') as f:
            chain_uniprot_info = json.load(f)
        ligand_df = item["atoms_ligand"]
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
        num_classes =538
        if self.gearnet:
            graph = hetero_graph_transform(
                atom_df=atom_df, super_node=self.supernode, flag=lig_flag, protein_seq=item['protein_seq']
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
        gearnet = False
    ):
        self.cutoff = cutoff
        self.remove_hydrogens = remove_hydrogens
        self.max_num_neighbors = max_num_neighbors
        self.supernode = supernode
        self.offset_strategy = offset_strategy
        self.gearnet = gearnet

    def __call__(self, item: Dict) -> MyData:
        # print("Using Transform LBA")
        info_root = './output_info/uniprot_dict_all.json'
        with open(info_root, 'r') as f:
            chain_uniprot_info = json.load(f)
        
        ligand_df = item["atoms_ligand"]
        protein_df = item["atoms_protein"]

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

        num_classes = 538 + 490 + 1944 + 321
        # 找个办法把chain和Uniprot对应起来，然后就可以查了
        if self.gearnet:
            graph = hetero_graph_transform(
                atom_df=atom_df, super_node=self.supernode, flag=lig_flag, protein_seq=item['protein_seq']
            )
        else:
            graph = prot_graph_transform(
                atom_df=atom_df, cutoff=self.cutoff, max_num_neighbors=self.max_num_neighbors, flag=lig_flag, super_node=self.supernode, offset_strategy=self.offset_strategy
            )
        lba = labels['lba']
        ppi = labels['ppi']
        ec = labels['ec']
        go = labels['go']
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

        # graph.y = torch.zeros(self.num_classes).scatter_(0,torch.tensor(labels),1)
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
            go_annot = go[pf_id]
            if ec_annot == -1:
                valid_mask[:538] = 0
            else:
                annotations = annotations + ec_annot
            if go_annot == -1:
                valid_mask[538:] = 0
            else:
                mf_annot = go_annot['molecular_functions'] 
                mf_annot = [j + 538 for j in mf_annot]
                if len(mf_annot) == 0:
                    valid_mask[538: 538+490] = 0
                bp_annot = go_annot['biological_process']
                bp_annot = [j + 538 + 490 for j in bp_annot]
                if len(bp_annot) == 0:
                    valid_mask[538+490: 538+490+1944] = 0
                cc_annot = go_annot['cellular_component']
                cc_annot = [j + 538 + 490 + 1944 for j in cc_annot]
                if len(cc_annot) == 0:
                    valid_mask[538+490+1944:] = 0
                annotations = annotations + mf_annot + bp_annot + cc_annot
                
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
        graph.type = 'multi'
        return graph