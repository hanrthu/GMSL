from functools import cache
import itertools as it
import json
from pathlib import Path
from typing import TypeAlias

import pandas as pd

PROCESSED_DIR = Path('processed-data')

PathLike: TypeAlias = str | bytes | Path

__all__ = [
    'PROCESSED_DIR',
    'get_pdb_ids',
    'get_uniprot_table',
    'get_go_table',
]

def get_pdb_ids(save_path: Path | None = None) -> list[str]:
    if save_path is not None and save_path.exists():
        return save_path.read_text().splitlines()
    ids = [
        entry_dir.name.split('.', 1)[0]
        for subdir in ['PP', 'refined-set']
        for entry_dir in (Path('datasets') / 'PDBbind' / subdir).iterdir()
        if entry_dir.name not in ['index', 'readme']
    ] + [
        chain.split('-', 1)[0]
        for (task, abbr), split in it.product(
            [
                ('EnzymeCommission', 'EC'),
                ('GeneOntology', 'GO'),
            ],
            ['train', 'valid', 'test'],
        )
        for chain in (Path('datasets') / task / f'nrPDB-{abbr}_{split}.txt').read_text().splitlines()
    ]
    ret = sorted(set(map(str.lower, ids)))
    if save_path is not None:
        save_path.write_text('\n'.join(ret) + '\n')
    return ret

class UniProtTable:
    def __init__(self):
        self.table: pd.Series = pd.read_csv(PROCESSED_DIR / 'uniprot.csv', index_col=['pdb_id', 'chain'])['uniprot']

    def __getitem__(self, item) -> str | None:
        if isinstance(item, str):
            pdb_id, chain = item.split('-')
        elif isinstance(item, tuple):
            pdb_id, chain = item
        else:
            raise TypeError
        return self.table.get((pdb_id, chain))

@cache
def get_uniprot_table():
    return UniProtTable()

class GeneOntologyTable:
    def __init__(self):
        self.pdb_to_go = json.loads((PROCESSED_DIR / 'pdb_to_go.json').read_bytes())
        self.uniprot_to_go = json.loads((PROCESSED_DIR / 'uniprot_to_go.json').read_bytes())
        self.uniprot_table = get_uniprot_table()

    def __getitem__(self, pdb_id: str) -> dict[str, list]:
        if pdb_id in self.pdb_to_go:
            return self.pdb_to_go[pdb_id]
        uniprot_id = self.uniprot_table[pdb_id]
        return self.uniprot_to_go[uniprot_id]

@cache
def get_go_table():
    return GeneOntologyTable()
