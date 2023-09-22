from pathlib import Path

import torch
from tqdm import tqdm

from gmsl.data import SavedSet
from gmsl.data.path import PROCESSED_DIR, PathLike, graph_save_dir
from utils import MyData

checked = SavedSet(graph_save_dir / '.checked')

def check(o_path: PathLike, n_path: PathLike):
    o: MyData = torch.load(o_path)
    n: MyData = torch.load(n_path)
    assert not o.is_cuda and n.is_cuda
    o.cuda(n.pos.device)
    n.cuda(n.pos.device)
    def check_tensor_eq(t1: torch.Tensor, t2: torch.Tensor, th: float = 0.1):
        num = t1.numel()
        s = (t1 == t2).sum()
        return num - s > th * num

    if check_tensor_eq(o.functions, n.functions):
        print(n_path, 'functions')
    if check_tensor_eq(o.x.argmax(dim=1), n.x.argmax(dim=1)):
        print(n_path, 'x')
    assert torch.equal(o.affinities, n.affinities)
    assert o.num_sequential_relations == n.num_sequential_relations
    if abs(n.num_knn_relations - o.num_knn_relations) > o.num_knn_relations * 0.1:
        print('knn', n_path)

def main():
    ref_dir = PROCESSED_DIR / Path('graph-ref')
    ref_name = 'hetero_fullatom_knn7_ligand_spatial4.5_sequential2_31elements'
    for n_path in tqdm(list(graph_save_dir.glob('*.pt')), ncols=80):
        item_id = n_path.stem
        if '-' in item_id:
            pdb_id, chain_id = item_id.split('-')
            ref_item_id = f'{pdb_id.upper()}-{chain_id}'
        else:
            ref_item_id = item_id
        for split in ['train_all', 'val', 'test']:
            if (o_path := ref_dir / f'{ref_name}_{split}' / f'{ref_item_id}.pt').exists():
                check(o_path, n_path)
                break

if __name__ == '__main__':
    main()
