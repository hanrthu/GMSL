import itertools as it
from pathlib import Path

import torch
from torch.types import Device
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from gmsl.data import append_ln
from gmsl.data.path import PROCESSED_DIR, PathLike, graph_save_dir
from utils import MyData

log_file = graph_save_dir / '.check.txt'

def check(o_path: PathLike, n_path: PathLike, device: Device):
    device = torch.device(device)
    o: MyData = torch.load(o_path, map_location=device)
    n: MyData = torch.load(n_path, map_location=device)

    def check_tensor_eq(t1: torch.Tensor, t2: torch.Tensor, th: float = 0.1):
        num = t1.numel()
        s = (t1 == t2).sum()
        return num - s > th * num

    if check_tensor_eq(o.x.argmax(dim=1), n.x.argmax(dim=1)):
        append_ln(log_file, f'x {n_path}')
    assert torch.equal(o.affinities, n.affinities)
    # 似乎新老版本之间有的链的顺序不同，导致 functions 的值不同
    if check_tensor_eq(o.functions, n.functions):
        append_ln(log_file, f'functions {n_path}')
    assert o.num_sequential_relations == n.num_sequential_relations
    # 对于非20种氨基酸的残基且原子个数超过 MAX_CHANNEL，由于排序可能导致截取的原子不同，从而导致 KNN 结果不同
    if abs(n.num_knn_relations - o.num_knn_relations) > o.num_knn_relations * 0.1:
        append_ln(log_file, f'knn {n_path}')

def main():
    torch.multiprocessing.set_start_method('spawn')
    log_file.touch()
    ref_dir = PROCESSED_DIR / Path('graph-ref')
    ref_name = 'hetero_fullatom_knn7_ligand_spatial4.5_sequential2_31elements'
    jobs = []
    for n_path in tqdm(list(graph_save_dir.glob('*.pt')), ncols=80, desc='gathering jobs'):
        item_id = n_path.stem
        if '-' in item_id:
            pdb_id, chain_id = item_id.split('-')
            ref_item_id = f'{pdb_id.upper()}-{chain_id}'
        else:
            ref_item_id = item_id
        for split in ['train_all', 'val', 'test']:
            if (o_path := ref_dir / f'{ref_name}_{split}' / f'{ref_item_id}.pt').exists():
                jobs.append((o_path, n_path))
    num_gpus = torch.cuda.device_count()
    process_map(check, *zip(*jobs), it.cycle(range(num_gpus)), ncols=80, max_workers=16, chunksize=16)

if __name__ == '__main__':
    main()
