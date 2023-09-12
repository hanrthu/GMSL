from pathlib import Path

from gmsl.datamodule import get_uniprot_table

uniprot_table = get_uniprot_table()

def gen_ec_labels():
    lines = Path('datasets/EnzymeCommission/nrPDB-EC_annot.tsv').read_text().splitlines()
    ec_classes = lines[1].strip().split('\t')
    class_to_int = {
        ec_class: i
        for i, ec_class in enumerate(ec_classes)
    }
    uniprot_to_ec = {}
    ref = {}
    inconsistent = {}
    num_consistent = 0
    for item in lines[3:]:
        pdb_chain, annotations = item.split('\t')
        pdb_id, chain_id = pdb_chain.split('-')
        try:
            uniprot_id: str = uniprot_table[pdb_id, chain_id]
        except KeyError:
            continue
        annotations = sorted(set(annotations.strip().split(',')))
        if uniprot_id not in uniprot_to_ec:
            uniprot_to_ec[uniprot_id] = annotations
            ref[uniprot_id] = pdb_chain
        else:
            if annotations != uniprot_to_ec[uniprot_id]:
                inconsistent.setdefault(uniprot_id, []).append(pdb_chain)
            else:
                num_consistent += 1

    print(num_consistent)
    return uniprot_to_ec

def main():
    ec_labels = gen_ec_labels()
    print(233)

if __name__ == '__main__':
    main()
