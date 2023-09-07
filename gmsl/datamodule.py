import itertools as it
from pathlib import Path

import pandas as pd

uniprot_table_path = 'uniprot.csv'

def get_pdb_ids() -> list[str]:
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
    return sorted(set(map(str.upper, ids)))

def get_uniprot_table() -> pd.Series:
    return pd.read_csv(uniprot_table_path, index_col=['pdb_id', 'chain'])['uniprot']
