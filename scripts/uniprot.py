import json

import httpx

def main():
    r =  httpx.post(
        'https://data.rcsb.org/graphql',
        json={
            "operationName": "structure",
            "query": "query structure($ids: [String!]!) {\n  entries(entry_ids: $ids) {\n    rcsb_id\n    polymer_entities {\n      rcsb_polymer_entity_container_identifiers {\n        entity_id\n        asym_ids\n        auth_asym_ids\n        uniprot_ids\n      }\n    }\n  }\n}",
            "variables": {
                "ids": ["2b10"]
            }
        }
    )
    print(json.dumps(r.json(), indent=2, ensure_ascii=False))

if __name__ == '__main__':
    main()
