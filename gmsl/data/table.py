from functools import cache

import pandas as pd
import torch
from torch.types import Device
from torch.nn import functional as nnf

from .utils import PROCESSED_DIR

__all__ = [
    'UniProtTable',
    'PropertyTable',
    'AffinityTable',
]

table_dir = PROCESSED_DIR / 'table'

uniprot_table_path = table_dir / 'uniprot.csv'

def ensure_str(item: str | tuple[str, str]):
    if isinstance(item, str):
        return item
    pdb_id, chain_id = item
    return f'{pdb_id.lower()}-{chain_id}'

class UniProtTable:
    def __init__(self):
        self.table: pd.Series = pd.read_csv(uniprot_table_path, index_col=['pdb_id'])['uniprot']

    def __getitem__(self, item: str | tuple[str, str]) -> str | None:
        return self.table.get(ensure_str(item))

    @classmethod
    @cache
    def get(cls):
        return UniProtTable()

pdb_to_property_path = table_dir / 'pdb_to_property.json'
uniprot_to_property_path = table_dir / 'uniprot_to_property.json'

property_num_classes = {
    'ec': 538,
    'mf': 490,
    'bp': 1944,
    'cc': 321,
}

class PropertyTable:
    def __init__(self):
        self.uniprot_table: UniProtTable = UniProtTable.get()
        self.pdb_map = pd.read_json(pdb_to_property_path, orient='index')
        self.uniprot_map = pd.read_json(uniprot_to_property_path, orient='index')

    def build(self, pdb_id: str, chain_id: str, device: Device = None):
        pdb_id = pdb_id.lower()
        item_id = f'{pdb_id}-{chain_id}'
        task_annotations: dict[str, list[int]]
        if (pdb_property := self.pdb_map.get(item_id)) is None:
            uniprot_id = self.uniprot_table[item_id]
            if uniprot_id is None or (uniprot_property := self.uniprot_map.get(uniprot_id)) is None:
                task_annotations = {}
            else:
                task_annotations = uniprot_property
        else:
            task_annotations = pdb_property
        prop = []
        valid_mask = torch.zeros(len(property_num_classes), dtype=torch.bool, device=device)
        for i, (task, num_classes) in enumerate(property_num_classes.items()):
            if (annotations := task_annotations.get(task)) is None:
                prop.append(torch.zeros(num_classes, dtype=torch.long, device=device))
            else:
                valid_mask[i] = True
                prop.append(nnf.one_hot(torch.as_tensor(annotations, device=device), num_classes))
        return torch.cat(prop), valid_mask

    @classmethod
    @cache
    def get(cls):
        return PropertyTable()

lba_table_path = table_dir / 'lba.csv'
ppi_table_path = table_dir / 'ppi.csv'

class AffinityTable:
    def __init__(self):
        self.tables: dict[str, pd.Series] = {
            task: pd.read_csv(table_dir / f'{task}.csv')[task]
            for task in ['lba', 'ppi']
        }

    def build(self, pdb_id: str, device: Device) -> torch.Tensor:
        pdb_id = pdb_id.lower()
        value = [
            value if (value := table.get(pdb_id)) is not None else -1
            for task, table in self.tables.items()
        ]
        return torch.tensor(value, device=device)

    @classmethod
    @cache
    def get(cls):
        return AffinityTable()
