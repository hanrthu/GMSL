from argparse import ArgumentParser
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
import itertools as it
import json
from operator import length_hint
import os
from pathlib import Path
import pickle

import pandas as pd
import torch
from torch.types import Device
from tqdm import tqdm
from tqdm.auto import tqdm as tqdm_auto
import yaml

from utils import hetero_graph_transform

chain_uniprot_info = json.loads(Path('output_info/uniprot_dict_all.json').read_bytes())
remove_hydrogen: bool = True
hetero: bool
supernode: bool = False
alpha_only: bool
max_seq_len: int = 3000

def process(item: dict, device: Device):
    ligand_df: pd.DataFrame | None = item['atoms_ligand']
    protein_df: pd.DataFrame = item['atoms_protein']
    residue_df = protein_df.drop_duplicates(subset=['residue'], keep='first', inplace=False).reset_index(drop=True)
    if isinstance(ligand_df, pd.DataFrame):
        atom_df = pd.concat([protein_df, ligand_df], axis=0)
        res_ligand_df = pd.concat([residue_df, ligand_df], axis=0)
    else:
        atom_df = protein_df
        res_ligand_df = residue_df
    if remove_hydrogen:
        atom_df = atom_df[atom_df['element'] != 'H'].reset_index(drop=True)
        res_ligand_df = res_ligand_df[res_ligand_df.element != 'H'].reset_index(drop=True)

    # 此时 ligand 所对应的atom被自动设置为0
    lig_flag = torch.zeros(res_ligand_df.shape[0], dtype=torch.long, device=device)
    chain_ids = list(set(protein_df['chain']))
    uniprot_ids = []
    labels = item['labels']
    pf_ids = []
    # 目前是按照肽链来区分不同的蛋白，为了便于Unprot分类
    for i, chain_id in enumerate(chain_ids):
        lig_flag[torch.as_tensor(res_ligand_df['chain'].array == chain_id, device=device)] = i + 1
        if '-' in item['complex_id']:
            pf_ids.append(0)
            break
        if chain_id in chain_uniprot_info[item['complex_id']]:
            uniprot_id = chain_uniprot_info[item['complex_id']][chain_id]
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
    num_classes = [538, 490, 1944, 321]
    total_classes = sum(num_classes)
    # 找个办法把chain和Uniprot对应起来，然后就可以查了
    graph = hetero_graph_transform(
        item_name=item['complex_id'],
        atom_df=atom_df,
        super_node=supernode,
        protein_seq=item['protein_seq'],
        alpha_only=alpha_only,
        device=device,
    )

    if lig_flag.shape[0] != graph.x.shape[0]:
        print(len(lig_flag), len(graph.x))
        print(item['complex_id'])
        raise ValueError

    graph.affinities = torch.tensor([[labels['lba'], labels['ppi']]])
    ec = labels['ec']
    go = labels['go']
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
        if ec_annot == -1:
            valid_mask[0] = 0
        else:
            annotations = annotations + ec_annot
        if go_annot == -1:
            valid_mask[1:] = 0
        else:
            mf_annot = go_annot['molecular_functions']
            mf_annot = [j + 538 for j in mf_annot]
            if len(mf_annot) == 0:
                valid_mask[1] = 0
            bp_annot = go_annot['biological_process']
            bp_annot = [j + 538 + 490 for j in bp_annot]
            if len(bp_annot) == 0:
                valid_mask[2] = 0
            cc_annot = go_annot['cellular_component']
            cc_annot = [j + 538 + 490 + 1944 for j in cc_annot]
            if len(cc_annot) == 0:
                valid_mask[3] = 0
            annotations = annotations + mf_annot + bp_annot + cc_annot

        prop = torch.zeros(total_classes, dtype=torch.get_default_dtype()).scatter_(0, torch.tensor(annotations), 1)
        graph.functions.append(prop)
        graph.valid_masks.append(valid_mask)
    try:
        graph.functions = torch.vstack(graph.functions)
        graph.valid_masks = torch.vstack(graph.valid_masks)
    except:
        print('PF ids:', pf_ids)
        print(item['complex_id'], chain_ids, labels)
        print(len(graph.functions))
        print(pf_ids)
        print(graph.functions)
        raise RuntimeError
    graph.chains = lig_flag[lig_flag != 0]
    # print(item['complex_id'])

    graph.lig_flag = lig_flag
    if len(chain_ids) != len(graph.functions):
        print(item['complex_id'])
        print(chain_ids)
        print(len(chain_ids), len(graph.functions))
    graph.prot_id = item['complex_id']
    graph.type = 'multi'
    return graph

# class GNNTransformMultiTask:
#     def __init__(
#         self,
#         cutoff: float = 4.5,
#         remove_hydrogen: bool = True,
#         max_num_neighbors: int = 32,
#         supernode: bool = False,
#         offset_strategy: int = 0,
#         hetero=False,
#         alpha_only=False
#     ):
#         self.cutoff = cutoff
#         remove_hydrogen = remove_hydrogen
#         self.max_num_neighbors = max_num_neighbors
#         self.supernode = supernode
#         self.offset_strategy = offset_strategy
#         hetero = hetero
#         self.alpha_only = alpha_only
#
#     def __call__(self, item: dict, device: Device) -> MyData:


# almost copy & paste from tqdm
@contextmanager
def ensure_lock(tqdm_class, lock_name=''):
    '''get (create if necessary) and then restore `tqdm_class`'s lock'''
    old_lock = getattr(tqdm_class, '_lock', None)  # don't create a new lock
    lock = old_lock or tqdm_class.get_lock()  # maybe create a new lock
    lock = getattr(lock, lock_name, lock)  # maybe subtype
    tqdm_class.set_lock(lock)
    yield lock
    if old_lock is None:
        del tqdm_class._lock
    else:
        tqdm_class.set_lock(old_lock)

def _executor_map(PoolExecutor: type[ProcessPoolExecutor], fn, *iterables, **tqdm_kwargs):
    kwargs = tqdm_kwargs.copy()
    if 'total' not in kwargs:
        kwargs['total'] = length_hint(iterables[0])
    tqdm_class = kwargs.pop('tqdm_class', tqdm_auto)
    max_workers = kwargs.pop('max_workers', min(32, os.cpu_count() + 4))
    chunksize = kwargs.pop('chunksize', 1)
    lock_name = kwargs.pop('lock_name', '')
    with ensure_lock(tqdm_class, lock_name=lock_name) as lk:
        # share lock in case workers are already using `tqdm`
        with PoolExecutor(max_workers=max_workers, initializer=tqdm_class.set_lock, initargs=(lk,)) as ex:
            return list(tqdm_class(ex.map(fn, *iterables, chunksize=chunksize), **kwargs))

class PostCopyProcessPoolExecutor(ProcessPoolExecutor):
    def map(self, fn: Callable, *iterables: Iterable, timeout: float | None = None, chunksize: int = 1) -> Iterator:
        for x in super().map(fn, *iterables, timeout=timeout, chunksize=chunksize):
            # https://github.com/pytorch/pytorch/issues/973#issuecomment-459398189
            yield deepcopy(x)

def process_map(fn, *iterables, **tqdm_kwargs):
    if 'lock_name' not in tqdm_kwargs:
        tqdm_kwargs = tqdm_kwargs.copy()
        tqdm_kwargs['lock_name'] = 'mp_lock'
    return _executor_map(PostCopyProcessPoolExecutor, fn, *iterables, **tqdm_kwargs)

def get_argparse():
    parser = ArgumentParser(
        description='Main training script for Equivariant GNNs on Multitask Data.'
    )
    parser.add_argument('--config', type=str, default=None)
    args = parser.parse_args()
    if args.config != None:
        with open(args.config, 'r') as f:
            content = f.read()
        config_dict = yaml.load(content, Loader=yaml.FullLoader)
        # print('Config Dict:', config_dict)
    else:
        config_dict = {}
    for k, v in config_dict.items():
        setattr(args, k, v)
    print('Config Dict:', args)
    return args

def length_check(complx, thres):
    if len(complx['atoms_protein']['element']) > thres:
        return False
    else:
        return True

def correctness_check(complx):
    #Annotation Correctness Check
    correct = True
    chain_ids = list(set(complx['atoms_protein']['chain']))
    if '-' not in complx['complex_id']:
        for i, id in enumerate(chain_ids):
            if id in chain_uniprot_info[complx['complex_id']]:
                uniprot_id = chain_uniprot_info[complx['complex_id']][id]
                labels_uniprot = complx['labels']['uniprots']
                if uniprot_id not in labels_uniprot:
                    # print('Error, you shouldn't come here!')
                    correct = False
                    # print(complx['complex_id'], labels_uniprot, chain_uniprot_info[complx['complex_id']])
    return correct

def main():
    global hetero, alpha_only
    torch.set_float32_matmul_precision('high')
    torch.multiprocessing.set_start_method('spawn')
    # torch.multiprocessing.set_sharing_strategy('file_system')
    print('start method:', torch.multiprocessing.get_start_method())
    print('sharing strategy:', torch.multiprocessing.get_sharing_strategy())
    args = get_argparse()
    hetero = args.model_type in ['gearnet', 'hemenet']
    assert hetero
    alpha_only = args.alpha_only
    # seed = args.seed
    # pl.seed_everything(seed, workers=True)
    # cache_dir = os.path.join(self.root_dir, self.cache_dir)
    train_cache_path = Path('datasets/MultiTask/train_all.cache')
    print('Start loading cached Multitask files...')
    with open(train_cache_path, 'rb') as f:
        processed_complexes = pickle.load(f)
    print('Complexes Before Checking:', len(processed_complexes))
    print('Checking the dataset...')
    thres = 3000
    processed_complexes = [
        i for i in tqdm(processed_complexes)
        if length_check(i, thres) and correctness_check(i)
    ]
    print('Dataset size:', len(processed_complexes))
    # transform_func = GNNTransformMultiTask(hetero=hetero, alpha_only=args.alpha_only)
    print('Transforming complexes...')
    # for i in trange(len(processed_complexes), ncols=80):
    #     processed_complexes[i] = transform_func(processed_complexes[i], 0)
    num_gpus = torch.cuda.device_count()
    # import resource
    # rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    # resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))
    processed_complexes = process_map(
        process,
        processed_complexes, it.cycle(range(num_gpus)),
        max_workers=12, ncols=80, total=len(processed_complexes), chunksize=16,
    )

if __name__ == '__main__':
    main()
