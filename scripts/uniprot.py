from collections.abc import Callable
import enum
import itertools as it
import json
from pathlib import Path
import textwrap

import cytoolz
import httpx
from tqdm.contrib.concurrent import process_map

def get_entry_ids() -> list[str]:
    ids = [
        entry_dir.name.split('.', 1)[0]
        for subdir in ['PP', 'refined-set']
        for entry_dir in (Path('datasets/PDBbind') / subdir).iterdir()
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

class UniProtFlag(enum.IntFlag):
    NONE = enum.auto()
    MULTI = enum.auto()

    @classmethod
    def whole(cls):
        return (1 << len(cls)) - 1

def parse_entry(entry: dict):
    entry_id: str = entry['rcsb_id']
    entities = entry['polymer_entities']
    entity_dict = {}
    flag = 0
    for entity in entities:
        entity: dict = entity['rcsb_polymer_entity_container_identifiers']
        entity_id = entity.pop('entity_id')
        assert entity_id not in entity_dict
        uniprot_ids: list[str] | None = entity.pop('uniprot_ids')
        if uniprot_ids is None:
            entity['uniprot_id'] = None
            flag |= UniProtFlag.NONE
        elif len(uniprot_ids) == 1:
            entity['uniprot_id'] = uniprot_ids[0]
        else:
            entity['uniprot_id'] = uniprot_ids
            flag |= UniProtFlag.MULTI
        entity_dict[entity_id] = entity
    return flag, (entry_id, entity_dict)

def request(entry_ids: list[str]):
    r =  httpx.post(
        'https://data.rcsb.org/graphql',
        json={
            'operationName': 'structure',
            'query': textwrap.dedent('''\
            query structure($ids: [String!]!) {
                entries(entry_ids: $ids) {
                    rcsb_id
                    polymer_entities {
                        rcsb_polymer_entity_container_identifiers {
                            entity_id
                            asym_ids
                            auth_asym_ids
                            uniprot_ids
                        }
                    }
                }
            }'''),
            'variables': {'ids': entry_ids},
        },
        timeout=None,
    )
    entries: list[dict] = r.json()['data']['entries']
    return list(map(parse_entry, entries))

def main():
    entry_ids = get_entry_ids()
    print(len(entry_ids))
    result = process_map(request, list(cytoolz.partition_all(300, entry_ids)), ncols=80, max_workers=32)
    result = list(cytoolz.concat(result))
    def dump(flag_filter_fn: Callable[[int],bool], filename: str):
        data = dict(entry for flag, entry in result if flag_filter_fn(flag))
        print(filename, len(data))
        Path(filename).write_text(json.dumps(data, indent=4, ensure_ascii=False))

    dump(lambda flag: flag == 0, 'uniprot.json')
    dump(lambda flag: flag & UniProtFlag.NONE, 'uniprot-none.json')
    dump(lambda flag: flag & UniProtFlag.MULTI, 'uniprot-multi.json')

if __name__ == '__main__':
    main()
