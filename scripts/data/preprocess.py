from collections.abc import Sequence
import itertools as it
from multiprocessing.managers import AcquirerProxy, DictProxy, ListProxy
import os
from pathlib import Path

from torch.multiprocessing import Manager
import torch
from torch.types import Device
from tqdm.contrib.concurrent import process_map

from gmsl.data import (
    AffinityTable, MAX_CHANNEL, ModelData, MultiChannelData, PropertyTable, ResidueData, SavedSet, append_ln,
)
from gmsl.data.path import graph_save_dir, parsed_dir
from utils import KNNEdge, MyData, SequentialEdge

def read_item(item_path: Path, device: torch.device) -> tuple[ModelData, ResidueData | None]:
    if item_path.resolve().is_dir():
        model = torch.load(item_path / 'protein.pt', device)
        ligand = torch.load(item_path / 'ligand.pt', device)
    else:
        model = torch.load(item_path, device)
        ligand = None
    return model, ligand

k_KNN = 7

def hetero_graph_transform(data: MultiChannelData, chain: torch.Tensor):
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
    num_edges = torch.as_tensor(num_edges, device=data.device)

    num_relations = torch.as_tensor(num_relations, device=data.device)
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

def build_property(pdb_id: str, chain_ids: Sequence[str], device: Device):
    properties, valid_masks = zip(*(
        property_table.build(pdb_id, chain_id, device)
        for chain_id in chain_ids
    ))
    return torch.stack(properties), torch.stack(valid_masks)

def build_graph(
    protein: ModelData,
    ligand: ResidueData | None,
    affinities: torch.Tensor,
    pdb_id: str,
    chain_ids: Sequence[str],
) -> MyData:
    protein_data, chain = protein.to_multi_channel(chain_ids)
    chain += 1
    if ligand is None:
        multi_channel_data = protein_data
        lig_flag = chain
    else:
        ligand_data = ligand.to_multi_channel()
        multi_channel_data = MultiChannelData.merge(protein_data, ligand_data)
        lig_flag = torch.cat([chain, chain.new_zeros(ligand_data.n)])
    graph = hetero_graph_transform(multi_channel_data, chain)
    graph.chains = chain
    graph.lig_flag = lig_flag
    graph.affinities = affinities
    graph.functions, graph.valid_masks = build_property(pdb_id, chain_ids, protein.device)
    graph.type = 'multi'
    return graph

processed_items = SavedSet(graph_save_dir / '.processed.txt')
unmatched_chains_save_path = graph_save_dir / '.unmatched-chains.txt'
max_num_atoms = 15000
max_aug = 5

num_devices = torch.cuda.device_count()

def get_device(pid_to_device_id: DictProxy, device_ref_count: ListProxy, lock: AcquirerProxy) -> torch.device:
    pid = os.getpid()
    if (device_id := pid_to_device_id.get(pid)) is None:
        with lock:
            device_id = min(range(num_devices), key=lambda i: device_ref_count[i])
            pid_to_device_id[pid] = device_id
            device_ref_count[device_id] += 1
    return torch.device(device_id)

def process(item_id: str, item_path: Path, mp_args: tuple):
    device = get_device(*mp_args)
    affinities = affinity_table.build(item_id, device)[None]
    protein, ligand = read_item(item_path, device)
    if protein.num_atoms > max_num_atoms:
        processed_items.save(item_id)
        return
    if '-' in item_id:
        pdb_id, chain_id = item_id.split('-')
        all_chain_ids = [chain_id]
    else:
        pdb_id = item_id
        all_chain_ids = list(protein.chains)
    graph = build_graph(protein, ligand, affinities, pdb_id, all_chain_ids)
    graph.prot_id = item_id
    torch.save(graph, graph_save_dir / f'{item_id}.pt')

    if '-' in item_id and (full_protein_path := parsed_dir / 'mmcif' / f'{pdb_id}.pt').exists():
        chain = protein.chains[chain_id]
        full_protein: ModelData = torch.load(full_protein_path, device)
        chain_dis = {}
        if full_protein.chains.get(chain_id) == chain:
            for ref_chain_id, ref_chain in full_protein.chains.items():
                if ref_chain_id == chain_id:
                    continue
                ref_chain_coords = torch.cat([residue.masked_coords for residue in ref_chain.residues], 0)
                dis = torch.empty(chain.num_residues, device=device)
                for i, residue in enumerate(chain.residues):
                    dis[i] = (residue.masked_coords[:, None] - ref_chain_coords[None, :]).pow(2).sum(dim=-1).min()
                chain_dis[ref_chain_id] = dis.min()
            aug_cnt = 0
            sorted_chain_ids = sorted(chain_dis.keys(), key=lambda k: chain_dis[k])

            def search_aug(chain_idx: int, chain_ids: list[str], num_atoms: int):
                nonlocal aug_cnt
                if aug_cnt == max_aug:
                    return
                if chain_idx == len(chain_dis) or num_atoms > max_num_atoms:
                    if len(chain_ids) > 0:
                        aug_cnt += 1
                        aug_graph = build_graph(full_protein, None, affinities, pdb_id, [chain_id] + chain_ids)
                        aug_item_id = f"{pdb_id}-{chain_id}+{','.join(chain_ids)}"
                        graph.prot_id = item_id
                        torch.save(aug_graph, graph_save_dir / f"{aug_item_id}.pt")
                    return
                else:
                    cur_chain_id = sorted_chain_ids[chain_idx]
                    search_aug(chain_idx + 1, chain_ids + [sorted_chain_ids[chain_idx]], num_atoms + full_protein.chains[cur_chain_id].num_atoms)
                    search_aug(chain_idx + 1, chain_ids, num_atoms)

            search_aug(0, [], chain.num_atoms)
        else:
            append_ln(unmatched_chains_save_path, str(item_path))
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
    torch.multiprocessing.set_start_method('spawn')
    torch.set_float32_matmul_precision('high')
    items = get_items()
    with Manager() as manager:
        pid_to_device_id = manager.dict()
        device_ref_count = manager.list([0 for _ in range(num_devices)])
        lock = manager.Lock()
        # from tqdm import tqdm
        # for item_id, item in tqdm(items):
        #     process(item_id, item, (pid_to_device_id, device_ref_count, lock))
        processes_per_device = 3
        process_map(
            process, *zip(*items),
            it.repeat((pid_to_device_id, device_ref_count, lock)),
            max_workers=num_devices * processes_per_device, ncols=80, total=len(items), chunksize=1,
        )

if __name__ == '__main__':
    main()
