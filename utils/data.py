import os
import os.path as osp
from argparse import ArgumentParser
from typing import Dict, List, Tuple, Union

import pandas as pd
import torch
from atom3d.datasets import deserialize
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


def chunks_n_sized(l, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(l), n):
        yield l[i : i + n]


def chunks_n(l, n):
    """Yield n number of striped chunks from l."""
    for i in range(0, n):
        yield l[i::n]


try:
    PATH = osp.join(osp.dirname(osp.realpath(__file__)), "data")
except NameError:
    PATH = "experiments/lba_aux/data/"


DATA_DIR = osp.join(PATH, "split-by-sequence-identity-30/data")
# DATA_DIR = osp.join(PATH, "split-by-sequence-identity-60/data")

class GNNTransformLBA(object):
    def __init__(
        self,
        cutoff: float = 4.5,
        remove_hydrogens: bool = False,
        pocket_only: bool = True,
        max_num_neighbors: int = 32,
        supernode: bool = False,
        offset_strategy: int = 0,
    ):
        self.pocket_only = pocket_only
        self.cutoff = cutoff
        self.remove_hydrogens = remove_hydrogens
        self.max_num_neighbors = max_num_neighbors
        self.supernode = supernode
        self.offset_strategy = offset_strategy

    def __call__(self, item: Dict) -> MyData:
        # print("Using Transform LBA")
        ligand_df = item["atoms_ligand"]
        if self.pocket_only:
            protein_df = item["atoms_pocket"]
        else:
            protein_df = item["atoms_protein"]

        atom_df = pd.concat([protein_df, ligand_df], axis=0)

        if self.remove_hydrogens:
            # remove hydrogens
            # print("Removing hydrogens")
            atom_df = atom_df[atom_df.element != "H"].reset_index(drop=True)

        labels = item["scores"]
        lig_flag = torch.zeros(atom_df.shape[0], dtype=torch.bool)
        lig_flag[-len(ligand_df):] = 1
        graph = prot_graph_transform(
            atom_df=atom_df, cutoff=self.cutoff, max_num_neighbors=self.max_num_neighbors, flag=lig_flag, super_node=self.supernode, offset_strategy=self.offset_strategy
        )
        graph.y = torch.FloatTensor([labels["neglog_aff"]])


        graph.lig_flag = lig_flag

        graph.prot_id = item["id"]
        # graph.smiles = item["smiles"]
        graph.type = 'lba'
        return graph

class GNNTransformPPI(object):
    def __init__(self, 
        cutoff: float = 4.5,
        remove_hydrogens: bool = False,
        interface_only: bool = False,
        max_num_neighbors: int = 32
    ):
        self.interface_only = interface_only
        self.cutoff = cutoff
        self.remove_hydrogens = remove_hydrogens
        self.max_num_neighbors = max_num_neighbors
    def __call__(self, item: Dict) -> MyData:
        if self.interface_only:
            # print("Using Interface Only!")
            protein_df = item["interface_protein"]
        else:
            protein_df = item["atoms_protein"]
        # if self.remove_hydrogens:
        #     # remove hydrogens
        #     protein_df = protein_df[protein_df.element != "H"].reset_index(drop=True)
        flag_list = []
        for i in range(len(protein_df)):
            flag = i * torch.ones([len(protein_df[i]["element"])])
            flag_list.append(flag)
        # print(flag_list)
        protein_flags = torch.cat(flag_list)
        # print("Protein Flags:", protein_flags.shape)
        labels = item['scores']
        protein_df = pd.concat(protein_df, axis=0)
        graph = prot_graph_transform(
            atom_df=protein_df, cutoff=self.cutoff, max_num_neighbors=self.max_num_neighbors, offset_strategy=1, flag=protein_flags, super_node=True
        )
        graph.y = torch.FloatTensor([labels["neglog_aff"]])
        protein_seqs = item['protein_seq']
        graph.protein_flags = protein_flags
        graph.complex_id = item['complex_id']
        graph.prot_names = item['prot_names']
        graph.protein_seqs = protein_seqs
        graph.type = 'ppi'
        return graph

class GNNTransformProteinFunction(object):
    def __init__(self, 
        cutoff: float = 4.5,
        remove_hydrogens: bool = False,
        max_num_neighbors: int = 32,
        num_classes = 538,
        type = 'ec'
    ):
        self.cutoff = cutoff
        self.remove_hydrogens = remove_hydrogens
        self.max_num_neighbors = max_num_neighbors
        self.num_classes = num_classes
        self.type = 'ec'
    def __call__(self, item: Dict) -> MyData:
        protein_df = item['atoms_protein']
        flag_list = []
        for i in range(len(protein_df)):
            flag = i * torch.ones([len(protein_df[i]["element"])])
            flag_list.append(flag)
        protein_flags = torch.cat(flag_list)
        labels = item['labels']
        protein_df = pd.concat(protein_df, axis=0)
        graph = prot_graph_transform(
            atom_df=protein_df, cutoff=self.cutoff, max_num_neighbors=self.max_num_neighbors
        )
        graph.y = torch.zeros(self.num_classes).scatter_(0,torch.tensor(labels),1)
        graph.protein_flags = protein_flags
        graph.complex_id = item['complex_id']
        graph.protein_seqs = item['protein_seq']
        graph.type = self.type
        return graph


class LMDBDataset(Dataset):
    """
    Creates a dataset from an lmdb file. Adapted from `TAPE <https://github.com/songlab-cal/tape/blob/master/tape/datasets.py>`_.

    :param data_file: path to LMDB file containing dataset
    :type data_file: Union[str, Path]
    :param transform: Transformation function to apply to each item.
    :type transform: Function, optional

    """

    def __init__(self, data_file, transform=None):
        """constructor

        """
        if type(data_file) is list:
            if len(data_file) != 1:
                raise RuntimeError("Need exactly one filepath for lmdb")
            data_file = data_file[0]

        self.data_file = Path(data_file).absolute()
        if not self.data_file.exists():
            raise FileNotFoundError(self.data_file)

        env = lmdb.open(str(self.data_file), max_readers=1, readonly=True,
                        lock=False, readahead=False, meminit=False)

        with env.begin(write=False) as txn:
            self._num_examples = int(txn.get(b'num_examples'))
            self._serialization_format = \
                txn.get(b'serialization_format').decode()
            self._id_to_idx = deserialize(
                txn.get(b'id_to_idx'), self._serialization_format)

        self._env = env
        self._transform = transform
        self.items = []
        self.prepare()

    def __len__(self) -> int:
        return self._num_examples

    def get(self, id: str):
        idx = self.id_to_idx(id)
        return self[idx]

    def id_to_idx(self, id: str):
        if id not in self._id_to_idx:
            raise IndexError(id)
        idx = self._id_to_idx[id]
        return idx

    def ids_to_indices(self, ids):
        return [self.id_to_idx(id) for id in ids]

    def ids(self):
        return list(self._id_to_idx.keys())

    def prepare(self):
        if not os.path.exists(self.data_file / 'data.cache'):
            print("Loading dataset into memory and generating cache...")
            n=64
            count = 0
            for index in tqdm(range(self._num_examples)):
                if not 0 <= index < self._num_examples:
                    raise IndexError(index)

                with self._env.begin(write=False) as txn:

                    compressed = txn.get(str(index).encode())
                    buf = io.BytesIO(compressed)
                    with gzip.GzipFile(fileobj=buf, mode="rb") as f:
                        serialized = f.read()
                    try:
                        item = deserialize(serialized, self._serialization_format)
                    except:
                        return None
                # Recover special data types (currently only pandas dataframes).
                if 'types' in item.keys():
                    for x in item.keys():
                        if item['types'][x] == str(pd.DataFrame):
                            item[x] = pd.DataFrame(**item[x])
                else:
                    logging.warning('Data types in item %i not defined. Will use basic types only.'%index)

                if 'file_path' not in item:
                    item['file_path'] = str(self.data_file)
                if 'id' not in item:
                    item['id'] = str(index)
                if self._transform:
                    item = self._transform(item)
                self.items.append(item)
                count += 1
                if count == n:
                    break
            pickle.dump(self.items, open(self.data_file / 'data.cache', 'wb'))
        else:
            print("Cache found, loading cache...")
            self.items = pickle.load(open(self.data_file / 'data.cache', 'rb'))
            # print(self.items[0])
            # print("Done")

    def __getitem__(self, index: int):
        # 普通的data类型
        item = self.items[index]
        to_return = MyData(
            x=item.x,
            pos=item.pos,
            edge_weights=item.edge_weights,
            edge_index=item.edge_index,
            e=item.e,
            # external_flag=external_edge_flag
            e_pos = item.e_pos,
            e_interaction = item.e_interaction,
            alphas = item.alphas,
            connection_nodes = item.connection_nodes,
            y = item.y,
            lig_flag = item.lig_flag,
            prot_id = item.prot_id,
            smiles = item.smiles,
            type=item.type
            )
        # print(to_return)
        # 变成一个Mydata类型
        return to_return
        # return item


class CustomAuxDataset(Dataset):
    def __init__(self, train_dataset, aux_dataset):
        super(CustomAuxDataset, self).__init__()
        self.train_dataset = train_dataset
        self.aux_dataset = aux_dataset
    def __getitem__(self, idx):
        if idx < len(self.train_dataset):
            return self.train_dataset[idx]
        else:
            return self.aux_dataset[idx - len(self.train_dataset)]
    @property
    def len_main(self):
        return len(self.train_dataset)
    @property
    def len_aux(self):
        return len(self.aux_dataset)
    def __len__(self):
        return len(self.train_dataset) + len(self.aux_dataset)

class CustomProteinFunctionDataset(Dataset):
    def __init__(self, root_dir: str = './data/EnzymeCommission/', split: str = 'train', label_dir: str = './data/EnzymeCommission/nrPDB-EC_annot.tsv',
                 remove_hoh = True, remove_hydrogen = False, cache_dir = 'ec_cache_train.pkl', cutoff = 6, task: str = None):
        super(CustomProteinFunctionDataset, self).__init__(root_dir)
        if 'EnzymeCommission' in root_dir:
            self.type = 'ec'
            print("Initializing Custom EC Dataset...")
        elif 'GeneOntology' in root_dir:
            self.type = 'go'
            print("Initializing Custom GO Dataset...")
        if split not in ['train', 'val', 'test']:
            print("Wrong selected split. Have to choose between ['train', 'val', 'test']")
            print("Exiting code")
            exit()
        self.root_dir = os.path.join(root_dir, split)
        self.remove_hoh = remove_hoh
        self.remove_hydrogen = remove_hydrogen
        self.cache_dir = os.path.join(root_dir, cache_dir)
        self.cutoff = cutoff
        self.task = task
        self.files = os.listdir(self.root_dir)
        self.labels = self.load_annotations(label_dir)
        self.transform_func = GNNTransformProteinFunction(num_classes=self.num_classes, type = self.type)

        self.process_complexes()
    def load_annotations(self, label_dir):
        if self.type == 'ec':
            with open(label_dir, 'r') as f:
                lines = f.readlines()
            ec_classes = lines[1].strip().split('\t')
            label_dict = {}
            pdb_annot_dict = {}
            label_id = 0
            for label in ec_classes:
                if label not in label_dict:
                    label_dict[label] = label_id
                    label_id += 1
            for item in lines[3:]:
                pdb_id, annotations = item.split('\t')
                annotations_list = annotations.strip().split(',')
                pdb_annot_dict[pdb_id] = [label_dict[annot] for annot in annotations_list]
            self.num_classes = label_id
            print("Number of classes in task {} is {}".format('EnzymeCommission', label_id))
        elif self.type == 'go':
            with open(label_dir, 'r') as f:
                lines = f.readlines()
            go_classes_molecular_functions = lines[1].strip().split('\t')
            go_classes_biological_process = lines[5].strip().split('\t')
            go_classes_cellular_component = lines[9].strip().split('\t')
            
            label_dict = {'molecular_functions':{}, 'biological_process':{}, 'cellular_component':{}}
            pdb_annot_dict = {}
            label_id_molecular = 0
            label_id_biological = 0
            label_id_cellular = 0
            for label in go_classes_molecular_functions:
                if label not in label_dict['molecular_functions']:
                    label_dict['molecular_functions'][label] = label_id_molecular
                    label_id_molecular += 1
            for label in go_classes_biological_process:
                if label not in label_dict['biological_process']:
                    label_dict['biological_process'][label] = label_id_biological
                    label_id_biological += 1
            for label in go_classes_cellular_component:
                if label not in label_dict['cellular_component']:
                    label_dict['cellular_component'][label] = label_id_cellular
                    label_id_cellular += 1
            for item in lines[13:]:
                pdb_id, molecular, biological, cellular = item.split('\t')
                molecular_list  = molecular.strip().split(',')
                biological_list = biological.strip().split(',')
                cellular_list = cellular.strip().split(',')
                # print(molecular_list)
                # 列表中会包含一些空的信息
                if self.task == 'go_mf':
                    pdb_annot_dict[pdb_id] = [label_dict['molecular_functions'][annot] for annot in molecular_list if annot != '']
                    self.num_classes = label_id_molecular
                elif self.task == 'go_bp':
                    pdb_annot_dict[pdb_id] = [label_dict['biological_process'][annot] for annot in biological_list if annot != '']
                    self.num_classes = label_id_biological
                elif self.task == 'go_cc':
                    pdb_annot_dict[pdb_id]= [label_dict['cellular_component'][annot] for annot in cellular_list if annot != '']
                    self.num_classes = label_id_cellular
                else:
                    raise NotImplementedError

            print("Number of classes in task {} is {}".format('molecular_functions', label_id_molecular+1))
            print("Number of classes in task {} is {}".format('biological_process', label_id_biological+1))
            print("Number of classes in task {} is {}".format('cellular_component', label_id_cellular+1))
        return pdb_annot_dict
    def process_complexes(self):
        p = PDBParser(QUIET=True)
        self.nmr_files = []
        self.processed_complexes = []
        if os.path.exists(self.cache_dir):
            print("Start loading cached files...")
            self.processed_complexes = pickle.load(open(self.cache_dir, 'rb'))
            print("Dataset size:", self.len())
        else:
            print("Cache not found! Start processing files...")
            # count = 0
            corrupted = []
            for item in tqdm(self.files):
                if self.labels[item.split('_')[0]] == []:
                    # print("Empty!")
                    continue
                try:
                    structure = p.get_structure(item.split('_')[0],os.path.join(self.root_dir, item))
                except:
                    corrupted.append(item)
                    continue
                # print(item)
                model = structure[0]
                chains = list(model.get_chains())
                if len(chains) != 1:
                    print("Found an abnormal structure, containing ", len(chains), " chains")
                    continue
                pattern = re.compile(r'\d+H.')
                processed_complex = {'complex_id': item.split('_')[0], 'labels':[],
                                    'atoms_protein': [], 'protein_seq': []}
                elements = []
                xs = []
                ys = []
                zs = []
                chain_ids = []
                protein_seq = []
                chain = chains[0]
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
                        elements.append(element)
                        chain_ids.append(chain.id)
                        x, y, z = atom.get_vector()
                        xs.append(x)
                        ys.append(y)
                        zs.append(z)
                protein_df = pd.DataFrame({'chain': chain_ids, 'element': elements, 'x': xs, 'y': ys, 'z': zs})
                processed_complex['atoms_protein'].append(protein_df)
                processed_complex['protein_seq'].append(protein_seq)
                processed_complex['labels'] = self.labels[item.split('_')[0]]
                # print(processed_complex)
                self.processed_complexes.append(processed_complex)
                # count += 1
                # if count == 20:
                #     break
            print("Structure processed Done, dumping...")
            print("Corrupted:", len(corrupted), corrupted)
            pickle.dump(self.processed_complexes, open(self.cache_dir, 'wb'))
    def len(self):
        return len(self.processed_complexes)
    def get(self, idx):
        return self.transform_func(self.processed_complexes[idx])
        
class CustomPPIDataset(Dataset):
    def __init__(self, root_dir: str = './data/PDBBind/PP', score_dir: str = './data/protein_protein/pp_affinity.xlsx',
                remove_hoh = True, remove_hydrogen = False, remove_ic50 = True, interface = False, ppi_dir = 'ppi_cache.pkl', cutoff = 6, split : str = 'train'):
        super(CustomPPIDataset, self).__init__(root_dir)
        print("Initializing CustomPPI Dataset...")
        self.root_dir = root_dir
        self.score_dir = score_dir
        self.remove_hoh = remove_hoh
        self.remove_hydrogen = remove_hydrogen
        self.interface = interface
        self.remove_ic50 = remove_ic50
        self.transform_func = GNNTransformPPI(interface_only=interface)
        self.ppi_dir = ppi_dir
        self.cutoff = cutoff
        # self.files = [i for i in os.listdir(root_dir) if i.endswith('.pdb')]
        self.files, self.log_scores = self.load_scores()
        self.process_complexes()
        if split not in ['train', 'val', 'test']:
            print("Wrong selected split. Have to choose between ['train', 'val', 'test']")
            print("Exiting code")
            exit()
        longest = 0
        if split == 'train':
            # self.files = self.files[:int(0.8*len(self.files))]
            # self.log_scores = self.log_scores[:int(0.8*len(self.log_scores))]
            self.processed_complexes = self.processed_complexes[:int(0.8*len(self.processed_complexes))]
            short_complexes = []
            for processed_complex in self.processed_complexes:
                g = self.transform_func(processed_complex)
                if g.pos.shape[0] < 2000:
                    short_complexes.append(processed_complex)
                elif g.pos.shape[0] > longest:
                    longest = g.pos.shape[0]
                # print(g)
            print("PPI Training set size: ", len(self.processed_complexes), len(short_complexes))
            print("Longest:", longest)
            self.processed_complexes = short_complexes
        elif split == 'val':
            # self.files = self.files[int(0.8*len(self.files)):int(0.9*len(self.files))]
            # self.log_scores = self.log_scores[int(0.8*len(self.log_scores)):int(0.9*len(self.log_scores))]
            self.processed_complexes = self.processed_complexes[int(0.8*len(self.processed_complexes)):int(0.9*len(self.processed_complexes))]
            short_complexes = []
            for processed_complex in self.processed_complexes:
                g = self.transform_func(processed_complex)
                if g.pos.shape[0] < 2000:
                    short_complexes.append(processed_complex)
                elif g.pos.shape[0] > longest:
                    longest = g.pos.shape[0]
                # print(g)
            print("PPI Validating set size: ", len(self.processed_complexes), len(short_complexes))
            print("Longest:", longest)
            self.processed_complexes = short_complexes
        else:
            # self.files = self.files[int(0.9*len(self.files)):]
            # self.log_scores = self.log_scores[int(0.9*len(self.log_scores)):]
            self.processed_complexes = self.processed_complexes[int(0.9*len(self.processed_complexes)):]
            short_complexes = []
            for processed_complex in self.processed_complexes:
                g = self.transform_func(processed_complex)
                if g.pos.shape[0] < 2000:
                    short_complexes.append(processed_complex)
                elif g.pos.shape[0] > longest:
                    longest = g.pos.shape[0]
                # print(g)
            print("PPI Testing set size: ", len(self.processed_complexes), len(short_complexes))
            print("Longest:", longest)
            self.processed_complexes = short_complexes

    # 目前这个处理方法删掉了NMR结构和单个/多个PPI相互作用的结构
    def process_complexes(self):
        p = PDBParser(QUIET=True)
        self.nmr_files = []
        self.uneuqal_to_two = []
        self.processed_complexes = []
        cache_dir = os.path.join(self.root_dir, self.ppi_dir)
        if os.path.exists(cache_dir):
            print("Start loading cached PPI files...")
            self.processed_complexes = pickle.load(open(cache_dir, 'rb'))
            print("Dataset size:", self.len())
        else:
            print("Cache not found! Start processing PPI files...")
            for score_idx, item in enumerate(tqdm(self.files)):
                structure = p.get_structure(item[:-8],os.path.join(self.root_dir, item))
                compound_info = structure.header['compound']
                protein_numbers = len(compound_info.items())
                if len(structure) > 1:
                    self.nmr_files.append(item)
                    continue
                if protein_numbers != 2:
                    self.uneuqal_to_two.append(item)
                    continue
                model = structure[0]
                chains_info = []
                names_info = []
                for key_id in compound_info:
                    chains_info.append(compound_info[key_id]['chain'])
                    names_info.append(compound_info[key_id]['molecule'])
                chains = list(model.get_chains())
                pattern = re.compile(r'\d+H.')
                processed_complex = {'complex_id': item[: -8], 'num_proteins': protein_numbers, 'prot_names':names_info, 'scores':{'neglog_aff': self.log_scores[score_idx]},
                                    'atoms_protein': [], 'interface_protein': [], 'protein_seq': []}
                
                # print("Information:", names_info, chains_info)
                for i, name in enumerate(names_info):
                    elements = []
                    xs = []
                    ys = []
                    zs = []
                    chain_ids = []
                    protein_seq = []
                    resnames = []
                    for chain in chains:
                        if chain.id == ' ':
                            continue
                        if chain.id.lower() in chains_info[i]:
                            for residue in chain.get_residues():
                                # 删除HOH原子
                                if self.remove_hoh and residue.get_resname() == 'HOH':
                                    continue
                                protein_seq.append(residue.get_resname())
                                for atom in residue:
                                    # 删除氢原子
                                    atom_id = atom.get_id()
                                    if self.remove_hydrogen and atom_id.startswith('H') or pattern.match(atom_id) != None:
                                        continue
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
                                    elements.append(element)
                                    chain_ids.append(chain.id)
                                    resnames.append(residue.get_resname())
                                    x, y, z = atom.get_vector()
                                    xs.append(x)
                                    ys.append(y)
                                    zs.append(z)
                    protein_df = pd.DataFrame({'chain': chain_ids, 'element': elements, 'x': xs, 'y': ys, 'z': zs, 'resname': resnames})
                    processed_complex['atoms_protein'].append(protein_df)
                    processed_complex['protein_seq'].append(protein_seq)
                #TODO 如果只要蛋白质pocket,那么以6A为分界线，目前只处理2个蛋白的情况
                #TODO 明天DEBUG一下
                if self.interface:
                    vectors = []
                    new_chain_ids_row = []
                    new_element_row = []
                    new_resname_row = []
                    new_chain_ids_column = []
                    new_element_column = []
                    new_resname_column = []
                    for df in processed_complex['atoms_protein']:
                        xs, ys, zs = torch.tensor(df['x']), torch.tensor(df['y']), torch.tensor(df['z'])
                        vectors.append(torch.stack((xs, ys, zs), dim=-1))
                    assert len(vectors) == 2
                    distances = torch.cdist(vectors[0], vectors[1])
                    interaction_matrix = (distances <= 6)
                    row_interaction_idx = (torch.sum(interaction_matrix, dim=1) > 0)
                    column_interaction_idx = (torch.sum(interaction_matrix, dim=0) > 0)
                    row_interface = vectors[0][row_interaction_idx]
                    column_interface = vectors[1][column_interaction_idx]
                    chain_ids_row = processed_complex['atoms_protein'][0]['chain']
                    element_row = processed_complex['atoms_protein'][0]['element']
                    resname_row = processed_complex['atoms_protein'][0]['resname']
                    chain_ids_column = processed_complex['atoms_protein'][1]['chain']
                    element_column = processed_complex['atoms_protein'][1]['element']
                    resname_column = processed_complex['atoms_protein'][1]['element']
                    for i in range(len(chain_ids_row)):
                        if row_interaction_idx[i]:
                            new_chain_ids_row.append(chain_ids_row[i])
                            new_element_row.append(element_row[i])
                            new_resname_row.append(resname_row[i])
                    for i in range(len(chain_ids_column)):
                        if column_interaction_idx[i]:
                            new_chain_ids_column.append(chain_ids_column[i])
                            new_element_column.append(element_column[i])
                            new_resname_column.append(resname_column[i])
                    if len(new_chain_ids_column) > 2000 or len(new_chain_ids_row) > 2000:
                        continue
                    # print(len(new_chain_ids_row), len(new_element_row), row_interface.shape, column_interface.shape)
                    interface_row_df = pd.DataFrame({'chain': new_chain_ids_row, 'element': new_element_row, 'x': row_interface[:,0].tolist(), 'y': row_interface[:,1].tolist(), 'z': row_interface[:,2].tolist(),'resname':new_resname_row})
                    interface_column_df = pd.DataFrame({'chain': new_chain_ids_column, 'element': new_element_column, 'x': column_interface[:,0].tolist(), 'y': column_interface[:,1].tolist(), 'z': column_interface[:,2].tolist(),'resname': new_resname_column})
                    processed_complex['interface_protein'] = [interface_row_df, interface_column_df]

                    # print("Distance Shapes:", vectors[0].shape, vectors[1].shape, distances.shape)

                self.processed_complexes.append(processed_complex)
            pickle.dump(self.processed_complexes, open(cache_dir, 'wb'))

    def load_scores(self):
        pp_info = pd.read_excel(self.score_dir, header=1)
        # print(pp_info.keys())
        orig_scores = pp_info['Affinity Data']
        pdb_names = []
        log_scores = []
        for i, orig in enumerate(orig_scores):
            if self.remove_ic50 and 'IC50' in orig:
                continue
            pdb_names.append(str(pp_info['PDB code'][i])+'.ent.pdb')
            log_scores.append(pp_info['pKd pKi pIC50'][i])
        # print(len(log_scores))
        return pdb_names, log_scores
    def len(self):
        return len(self.processed_complexes)
    def get(self, idx):
        return self.transform_func(self.processed_complexes[idx])


class CustomLBADataset(Dataset):
    """
    Simply subclassing torch_geometric.data.Dataset by using the processed files
    Allows faster loading of batches
    """

    def __init__(self, root: str = PATH, split: str = "train"):
        # no transform, as we have processed the files already
        super(CustomLBADataset, self).__init__(root)
        if not split in ["train", "val", "test"]:
            print("Wrong selected split. Have to choose between ['train', 'val', 'test']")
            print("Exiting code")
            exit()
        root_dir = os.path.join(root, split)
        self.files = [os.path.join(root_dir, f) for f in os.listdir(root_dir)]

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return self.files

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return self.files

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        # print(idx)
        datadict = torch.load(self.files[idx])
        # __getitem__ handled by parent-class.
        # https://github.com/rusty1s/pytorch_geometric/blob/master/torch_geometric/data/dataset.py#L184-L203
        return datadict
    
class PDBBind2016Dataset(Dataset):
    """
    Adjusting PDBBind2016 in SIGN to this dataset
    """
    def __init__(self, root_dir: str = './data/PDBbind_2016/', score_dir: str = './data/PDBBind_2016/refined-set/index/INDEX_general_PL_data.2016',
                remove_hoh = True, remove_hydrogen = False, remove_ic50 = True, pocket = True, cutoff = 6, split : str = 'train', transform=None):
        super(PDBBind2016Dataset, self).__init__(root_dir)
        print("Initializing PDBBind2016 Dataset...")
        self.root_dir = root_dir
        self.score_dir = score_dir
        self.remove_hoh = remove_hoh
        self.remove_hydrogen = remove_hydrogen
        self.remove_ic50 = remove_ic50
        self.transform_func = transform
        self.cutoff = cutoff
        # self.files = [i for i in os.listdir(root_dir) if i.endswith('.pdb')]
        if split not in ['train', 'val', 'test']:
            print("Wrong selected split. Have to choose between ['train', 'val', 'test']")
            print("Exiting code")
            exit()
        self.split = split
        self.process_complexes()
    def process_complexes(self):
        self.processed_complexes = []
        # excluded = ['4bps']
        # excluded = ['4yhq']
        cache_dir = os.path.join(self.root_dir, 'pdbbind2016_{}.pkl'.format(self.split))
        if os.path.exists(cache_dir):
            print("Start loading cached PDBBind2016 files...")
            self.processed_complexes = pickle.load(open(cache_dir, 'rb'))
            # if self.split == 'train':
            #     self.processed_complexes = [i for i in self.processed_complexes if i["id"] in excluded]
                # self.processed_complexes[0]['scores']["neglog_aff"] = torch.tensor(5.01, dtype=torch.float32)
            # self.special = [i for i in self.processed_complexes if i["id"] in excluded]
            # print(self.special)
            print("Dataset size:", self.len())
        else:
            print("Cache not found! Please run preprocess_pdbbind2016 first!")
            exit()
            
    def len(self):
        return len(self.processed_complexes)
    def get(self, idx):
        return self.transform_func(self.processed_complexes[idx])
        



if __name__ == "__main__":

    parser = ArgumentParser(
        description="Preprocessing script to obtain pytorch graph files for the LBA."
    )
    parser.add_argument(
        "--cutoff",
        help="radial cutoff for connecting edges. Defaults to 4.5",
        type=float,
        default=4.5,
    )
    parser.add_argument(
        "--remove_hydrogens",
        help="If hydrogen atoms in the protein/pocket graph should be removed. Defaults to False.",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--pocket_only",
        help="Whether to use the pocket of binding only.",
        default=False,
        action='store_true',
    )

    args = parser.parse_args()

    train_dir = osp.join(PATH, "train")
    val_dir = osp.join(PATH, "val")
    test_dir = osp.join(PATH, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    transform = GNNTransformLBA(
        cutoff=args.cutoff,
        remove_hydrogens=args.remove_hydrogens,
        pocket_only=args.pocket_only,
    )

    train_dataset = LMDBDataset(osp.join(DATA_DIR, "train"), transform=transform)
    ntrain = len(train_dataset)
    val_dataset = LMDBDataset(osp.join(DATA_DIR, "val"), transform=transform)
    nvalid = len(val_dataset)
    test_dataset = LMDBDataset(osp.join(DATA_DIR, "test"), transform=transform)
    ntest = len(test_dataset)

    print("Processing Training set")
    for i in tqdm(range(ntrain), total=ntrain):
        save_path = osp.join(train_dir, f"data_{i}.pth")
        if not osp.exists(save_path):
            data = train_dataset[i]
            torch.save(data, f=save_path)

    print("Processing Validation set")
    for i in tqdm(range(nvalid), total=nvalid):
        save_path = osp.join(val_dir, f"data_{i}.pth")
        if not osp.exists(save_path):
            data = val_dataset[i]
            torch.save(data, f=save_path)

    print("Processing Test set")
    for i in tqdm(range(ntest), total=ntest):
        save_path = osp.join(test_dir, f"data_{i}.pth")
        if not osp.exists(save_path):
            data = test_dataset[i]
            torch.save(data, f=save_path)

    print("Try dataloading with batch-size 16 on train/val/test dataset")
    # we just want to load the processed files
    train_dataset = CustomLBADataset(split="train")
    val_dataset = CustomLBADataset(split="val")
    test_dataset = CustomLBADataset(split="test")

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=16, shuffle=False, num_workers=4
    )
    val_loader = DataLoader(
        dataset=val_dataset, batch_size=16, shuffle=False, num_workers=4
    )
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=16, shuffle=False, num_workers=4
    )

    for data in tqdm(train_loader, total=len(train_loader)):
        pass

    for data in tqdm(val_loader, total=len(val_loader)):
        pass

    for data in tqdm(test_loader, total=len(test_loader)):
        pass
