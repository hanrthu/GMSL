from collections.abc import Sequence
import itertools as it
from pathlib import Path

import pandas as pd
import torch
from torch.types import Device
from tqdm.contrib.concurrent import process_map

from gmsl.data import (
    AffinityTable, MAX_CHANNEL, ModelData, MultiChannelData, PropertyTable, ResidueData, SavedSet,
)
from gmsl.data.path import graph_save_dir, parsed_dir
from utils import KNNEdge, MyData, SequentialEdge

def read_item(item_path: Path) -> tuple[ModelData, ResidueData | None]:
    if item_path.resolve().is_dir():
        model = pd.read_pickle(item_path / 'protein.pkl')
        ligand = pd.read_pickle(item_path / 'ligand.pkl')
    else:
        model = pd.read_pickle(item_path)
        ligand = None
    return model, ligand

k_KNN = 7

def hetero_graph_transform(data: MultiChannelData, chain: torch.Tensor, device: Device = None):
    knn = KNNEdge(k=k_KNN, min_distance=5, max_channel=MAX_CHANNEL)
    sequential = SequentialEdge(max_distance=2)
    edge_layers = [knn, sequential]
    graph = MyData(
        x=data.node_feats,
        residue_elements=data.element,
        num_nodes=data.node_feats.shape[0],
        pos=data.pos,
        channel_weights=data.mask,
        num_residues = chain.shape[0],
        edge_feature=None,
        chains=chain,
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
    graph.edge_index = edge_list[:, :2].T # [2, |E|]
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

processed_items = SavedSet(graph_save_dir / '.processed.txt')
max_num_atoms = 15000

def process(item_id: str, item_path: Path, device: Device):
    affinities = affinity_table.build(item_id, device)[None]
    protein, ligand = read_item(item_path)
    if protein.num_atoms > max_num_atoms:
        processed_items.save(item_id)
        return
    if '-' in item_id:
        pdb_id, chain_id = item_id.split('-')
        chain_ids = [chain_id]
    else:
        pdb_id = item_id
        chain_ids = list(protein.chains)
    graph = build_graph(protein, ligand, affinities, pdb_id, chain_ids, device)
    graph.prot_id = item_id
    torch.save(graph, graph_save_dir / f'{item_id}.pt')
    processed_items.save(item_id)

def get_items():
    items = []
    pdb_bind_item_ids = set()
    for item_path in (parsed_dir / 'PDBbind').glob('*/*'):
        item_id = item_path.stem
        assert item_id not in pdb_bind_item_ids
        pdb_bind_item_ids.add(item_id)
        if item_id not in processed_items:
            items.append((item_id, item_path))

    property_items = set()
    for property_dir_name in ['EnzymeCommission', 'GeneOntology']:
        for item_path in (parsed_dir / property_dir_name).iterdir():
            item_id = item_path.stem
            pdb_id, chain_id = item_id.split('-')
            if pdb_id in pdb_bind_item_ids or item_id in property_items:
                continue
            property_items.add(item_id)
            if item_id not in processed_items:
                items.append((item_id, item_path))
    return items

def main():
    torch.set_float32_matmul_precision('high')
    torch.multiprocessing.set_start_method('spawn')
    num_gpus = torch.cuda.device_count()
    items = get_items()
    # from tqdm import tqdm
    # for item_id, item_path in tqdm(items):
    #     process(item_id, item_path, 2)
    process_map(
        process, *zip(*items), it.cycle(range(num_gpus)),
        max_workers=4, ncols=80, total=len(items), chunksize=8,
    )

if __name__ == '__main__':
    main()
