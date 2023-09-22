import itertools as it
from pathlib import Path
from typing import TypeAlias

from Bio.PDB.Atom import Atom, DisorderedAtom
from Bio.PDB.Entity import Entity
import fcntl
import numpy as np

PROCESSED_DIR = Path('processed-data')

PathLike: TypeAlias = str | bytes | Path

__all__ = [
    'PROCESSED_DIR',
    'get_pdb_ids',
    'cmp_entity',
    'append_ln',
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
