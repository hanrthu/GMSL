import functools
import json
from pathlib import Path

import cytoolz

from gmsl.data import PROCESSED_DIR, get_uniprot_table

uniprot_table = get_uniprot_table()

class ClassLabel:
    def __init__(self, names: list):
        self.names = sorted(set(names))
        self.index = {name: i for i, name in enumerate(names)}

def gen_ec_labels():
    lines = Path('datasets/EnzymeCommission/nrPDB-EC_annot.tsv').read_text().splitlines()
    ec_label = ClassLabel(lines[1].strip().split('\t'))
    uniprot_to_ec = {}
    ref = {}
    inconsistent = {}
    num_consistent = 0
    for item in lines[3:]:
        pdb_id, annotations = item.split('\t')
        if (uniprot_id := uniprot_table[pdb_id]) is None:
            continue
        annotations = sorted(set(cytoolz.get(annotations.strip().split(','), ec_label.index)))
        if uniprot_id not in uniprot_to_ec:
            uniprot_to_ec[uniprot_id] = annotations
            ref[uniprot_id] = pdb_id
        else:
            if annotations != uniprot_to_ec[uniprot_id]:
                inconsistent.setdefault(uniprot_id, []).append(pdb_id)
            else:
                num_consistent += 1
    (PROCESSED_DIR / 'uniprot_to_ec.json').write_text(json.dumps(uniprot_to_ec, indent=4, ensure_ascii=False))

def gen_go_labels():
    lines = Path('datasets/GeneOntology/nrPDB-GO_annot.tsv').read_text().splitlines()
    class_labels: dict[str, ClassLabel] = {
        label_class: ClassLabel(lines[line_id].split('\t'))
        for label_class, line_id in [
            ('molecular_function', 1),
            ('biological_process', 5),
            ('cellular_component', 9),
        ]
    }
    uniprot_to_go = {}
    pdb_to_go = {}
    for line in lines[13:]:
        pdb_id, *labels = line.split('\t')
        if (uniprot_id := uniprot_table[pdb_id]) is None:
            continue
        labels = {
            label_class: set(cytoolz.get(label.split(',') if label else [], class_label.index))
            for label, (label_class, class_label) in zip(labels, class_labels.items())
        }
        assert pdb_id not in pdb_to_go
        pdb_to_go[pdb_id] = labels

        if (fallback_labels := uniprot_to_go.get(uniprot_id, None)) is None:
            uniprot_to_go[uniprot_id] = labels
        else:
            for label_class, label in labels.items():
                fallback_labels[label_class] &= label
    def dump(obj: dict[str, dict[str, set]], path: Path):
        obj = cytoolz.valmap(functools.partial(cytoolz.valmap, list), obj)
        path.write_text(json.dumps(obj, ensure_ascii=False, indent=4))
    dump(pdb_to_go, PROCESSED_DIR / 'pdb_to_go.json')
    dump(uniprot_to_go, PROCESSED_DIR / 'uniprot_to_go.json')
    return uniprot_to_go, pdb_to_go

def main():
    gen_ec_labels()
    gen_go_labels()

if __name__ == '__main__':
    main()
