import itertools as it
from pathlib import Path

from Bio.PDB.Atom import Atom, DisorderedAtom
from Bio.PDB.Entity import Entity
import fcntl
import numpy as np

from gmsl.data.path import PathLike

__all__ = [
    'get_pdb_ids',
    'normalize_item_id',
    'cmp_entity',
    'append_ln',
    'SavedSet',
]

def get_pdb_ids(save_path: Path | None = None) -> list[str]:
    if save_path is not None and save_path.exists():
        return save_path.read_text().splitlines()
    ids = [
        entry_path_or_dir.name.split('.', 1)[0]
        for subdir in ['PP', 'refined-set']
        for entry_path_or_dir in (Path('datasets') / 'PDBbind' / subdir).iterdir()
        if entry_path_or_dir.name not in ['index', 'readme']
    ] + [
        filepath.stem.split('-', 1)[0]
        for task, split in it.product(
            ['EnzymeCommission', 'GeneOntology'],
            ['train', 'valid', 'test'],
        )
        for filepath in (Path('datasets') / task / split).glob('*.pdb')
    ]
    ret = sorted(set(map(str.lower, ids)))
    if save_path is not None:
        save_path.write_text('\n'.join(ret) + '\n')
    return ret

def normalize_item_id(item_id: str):
    if '-' in item_id:
        pdb_id, chain_id = item_id.split('-')
        return f'{pdb_id.lower()}-{chain_id}'
    return item_id.lower()

def cmp_entity(a: Entity | Atom, b: Entity | Atom):
    if not issubclass(type(a), type(b)) and not issubclass(type(b), type(a)):
        return False

    if isinstance(a, (Atom, DisorderedAtom)):
        return np.allclose(a.coord, b.coord, atol=1e-3) and a.element == b.element

    if len(a) != len(b):
        return False

    for x, y in zip(a, b):
        if not cmp_entity(x, y):
            return False

    return True

def append_ln(filepath: Path, s: str):
    with open(filepath, 'a') as f:
        fcntl.lockf(f, fcntl.LOCK_EX)
        f.write(s + '\n')
        fcntl.lockf(f, fcntl.LOCK_UN)

class SavedSet:
    def __init__(self, save_path: PathLike):
        self.save_path = save_path = Path(save_path)
        if not save_path.exists():
            save_path.parent.mkdir(exist_ok=True, parents=True)
            save_path.touch()
        self.set = set(save_path.read_text().splitlines())

    def __contains__(self, item):
        return item in self.set

    def save(self, item: str):
        if item not in self.set:
            append_ln(self.save_path, item)
