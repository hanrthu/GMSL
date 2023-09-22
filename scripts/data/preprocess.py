from argparse import ArgumentParser
from collections.abc import Sequence
import itertools as it
import os
from pathlib import Path
import pickle

import pandas as pd
import torch
from torch.types import Device
from tqdm import tqdm
import yaml

from gmsl.data import AffinityTable, PropertyTable, MAX_CHANNEL, ModelData, MultiChannelData, ResidueData
from utils import KNNEdge, MyData, SequentialEdge

hetero: bool
supernode: bool = False
max_seq_len: int = 3000

def read_item(item_path: Path) -> tuple[ModelData, ResidueData | None]:
    if item_path.resolve().is_dir():
        model = pd.read_pickle(item_path / 'protein.pkl')
        ligand = pd.read_pickle(item_path / 'ligand.pkl')
    else:
        model = pd.read_pickle(item_path)
        ligand = None
    return model, ligand

def hetero_graph_transform(data: MultiChannelData, chain: torch.Tensor, device: Device = None):
    knn = KNNEdge(k=5, min_distance=4.5, max_channel=MAX_CHANNEL)
    # spatial = SpatialEdge(radius=cutoff, min_distance=4.5, max_channel=max_channel)
    sequential = SequentialEdge(max_distance=2)
    # edge_layers = [knn, spatial, sequential]
    edge_layers = [knn, sequential]
    graph = MyData(
        x=data.node_feats,
        residue_elements=data.element,
        num_nodes=data.node_feats.shape[0],
        pos=data.pos,
        channel_weights=data.mask,
        num_residues = chain.shape[0],
        edge_feature=None,
        chain=chain,
    )
    edge_list = []
    num_edges = []
    num_relations = []
    for layer in edge_layers:
        edges, num_relation = layer(graph)
        edge_list.append(edges)
        num_edges.append(len(edges))
        num_relations.append(num_relation)

    edge_list = torch.cat(edge_list)
    num_edges = torch.as_tensor(num_edges, device=device)

    num_relations = torch.as_tensor(num_relations, device=device)
    num_relation = num_relations.sum()
    offsets = (num_relations.cumsum(0) - num_relations).repeat_interleave(num_edges)
    edge_list[:, 2] += offsets

    # 这里把edge_index和edge_relation分开装，是为了能够让处理batch的时候不要把relation也加上offset
    graph.edge_index = edge_list[:, :2].t() # [2, |E|]
    graph.edge_relations = edge_list[:, 2]
    graph.edge_weights = torch.ones(len(edge_list))
    graph.num_relation = num_relation
    return graph

property_table: PropertyTable = PropertyTable.get()
affinity_table: AffinityTable = AffinityTable.get()

def build_property(pdb_id: str, chain_ids: Sequence[str]):
    properties, valid_masks = zip(*(
        property_table.build(pdb_id, chain_id)
        for chain_id in chain_ids
    ))
    return torch.stack(properties), torch.stack(valid_masks)

def build_graph(
    protein: ModelData,
    ligand: ResidueData | None,
    affinities: torch.Tensor,
    pdb_id: str,
    chain_ids: Sequence[str],
    device: Device,
) -> MyData:
    protein_data, chain = protein.to_multi_channel(device)
    chain += 1
    if ligand is None:
        multi_channel_data = protein_data
        lig_flag = chain
    else:
        ligand_data = ligand.to_multi_channel(device)
        multi_channel_data = MultiChannelData.merge(protein_data, ligand_data)
        lig_flag = torch.cat([chain, chain.new_zeros(ligand_data.n)])
    graph = hetero_graph_transform(multi_channel_data, chain, device)
    graph.chains = chain
    graph.lig_flag = lig_flag
    graph.affinities = affinities
    graph.functions, graph.valid_masks = build_property(pdb_id, chain_ids)
    graph.type = 'multi'
    return graph

def process(item_path: Path, device: Device):
    item_id = item_path.stem
    affinities = affinity_table.build(item_id, device)
    protein, ligand = read_item(item_path)
    if '-' in item_id:
        pdb_id, chain_id = item_id.split('-')
        chain_ids = [chain_id]
    else:
        pdb_id = item_id
        chain_ids = protein.chains.items()
    graph = build_graph(protein, ligand, affinities, pdb_id, chain_ids, device)
    torch.save()

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
    hetero = args.model_args['model_type'] in ['gearnet', 'hemenet', 'dymean']
    # assert hetero
    print(hetero)
    alpha_only = args.alpha_only

    # seed = args.seed
    # pl.seed_everything(seed, workers=True)
    # cache_dir = os.path.join(self.root_dir, self.cache_dir)
    splits = [args.train_split, args.val_split, args.test_split]
    train_cache_paths = []
    output_roots = []
    for split in splits:
        train_cache_path = Path(f'datasets/MultiTask_c03_id09/{split}.cache')
        output_root = Path(args.graph_cache_dir + f'_{split}')
        train_cache_paths.append(train_cache_path)
        output_roots.append(output_root)
    threses = [15000, 15000, 15000]
    for i, _ in enumerate(splits):
        train_cache_path = train_cache_paths[i]
        output_root = output_roots[i]
        thres = threses[i]
        print('Start loading cached Multitask files...')
        with open(train_cache_path, 'rb') as f:
            processed_complexes = pickle.load(f)
        print('Complexes Before Checking:', len(processed_complexes))
        print('Checking the dataset...')

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
            processed_complexes, it.cycle(range(num_gpus)), it.cycle([alpha_only]), it.cycle([hetero]),
            max_workers=2, ncols=80, total=len(processed_complexes), chunksize=4,
        )
        os.makedirs(output_root, exist_ok=True)
        for item in tqdm(processed_complexes):
            torch.save(item.cpu(), output_root/(item['prot_id'] + '.pt'))
if __name__ == '__main__':
    main()
