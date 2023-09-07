import json
from pathlib import Path
import textwrap

import cytoolz
import httpx
from tqdm.contrib.concurrent import process_map

from gmsl.datamodule import get_pdb_ids

def parse_entry(entry: dict):
    table = {}
    entities = entry['polymer_entities']
    entity_dict = {}
    for entity in entities:
        entity: dict = entity['rcsb_polymer_entity_container_identifiers']
        entity_id = entity.pop('entity_id')
        assert entity_id not in entity_dict
        uniprot_ids: list[str] | None = entity.pop('uniprot_ids')
        if uniprot_ids is None or len(uniprot_ids) != 1:
            uniprot_id = None
        else:
            uniprot_id = uniprot_ids[0]
        for chain_id in entity['auth_asym_ids']:
            table[chain_id] = uniprot_id
    return entry['rcsb_id'], table

def get_entries(entry_ids: list[str]):
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
    entry_ids = get_pdb_ids()
    print(len(entry_ids))
    result = process_map(get_entries, list(cytoolz.partition_all(300, entry_ids)), ncols=80, max_workers=32)
    Path('uniprot.json').write_text(json.dumps(dict(cytoolz.concat(result)), indent=4, ensure_ascii=False))

if __name__ == '__main__':
    main()
