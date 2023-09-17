import textwrap

import cytoolz
import httpx
import pandas as pd
from tqdm.contrib.concurrent import process_map

from gmsl.data import PROCESSED_DIR, get_pdb_ids

def parse_entry(entry: dict) -> list[tuple[str, str, str]]:
    pdb_id = entry['rcsb_id'].lower()
    entities = entry['polymer_entities']
    ret = []
    for entity in entities:
        entity: dict = entity['rcsb_polymer_entity_container_identifiers']
        uniprot_ids = entity.pop('uniprot_ids')
        if uniprot_ids is not None and len(uniprot_ids) == 1:
            uniprot_id = uniprot_ids[0]
            ret.extend([(pdb_id, chain_id, uniprot_id) for chain_id in entity['auth_asym_ids']])
    return ret

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
    return list(cytoolz.concat(map(parse_entry, entries)))

def main():
    chunksize = 300
    entry_ids = get_pdb_ids()
    print(len(entry_ids))
    result = process_map(get_entries, list(cytoolz.partition_all(chunksize, entry_ids)), ncols=80, max_workers=32)
    table = pd.DataFrame(list(cytoolz.concat(result)), columns=['pdb_id', 'chain', 'uniprot'])
    table.to_csv(PROCESSED_DIR / 'uniprot.csv', index=False)

if __name__ == '__main__':
    main()
