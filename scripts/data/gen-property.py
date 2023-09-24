from collections import Counter, defaultdict
from collections.abc import Sequence
import json
from pathlib import Path

import cytoolz

from gmsl.data import UniProtTable, normalize_item_id
from gmsl.data.table import pdb_to_property_path, uniprot_to_property_path

uniprot_table: UniProtTable = UniProtTable.get()

class ClassLabel:
    def __init__(self, names: list):
        self.names = sorted(set(names))
        self.index = {name: i for i, name in enumerate(names)}

    @property
    def num_classes(self):
        return len(self.names)

def nest(data: dict, name: str):
    return {
        k: {name: v}
        for k, v in data.items()
    }

def gen_ec_labels():
    lines = Path('datasets/EnzymeCommission/nrPDB-EC_annot.tsv').read_text().splitlines()
    ec_label = ClassLabel(lines[1].strip().split('\t'))

    pdb_map = {}
    uniprot_map = defaultdict(set)
    for item in lines[3:]:
        item_id, annotations = item.split('\t')
        item_id = normalize_item_id(item_id)
        annotations = set(cytoolz.get(annotations.strip().split(','), ec_label.index))
        if (pdb_annotations := pdb_map.get(item_id)) is None:
            pdb_map[item_id] = list(annotations)
        else:
            assert pdb_annotations == list(annotations)
        if (uniprot_id := uniprot_table[item_id]) is None:
            continue
        uniprot_map[uniprot_id] |= annotations
    for k, v in uniprot_map.items():
        uniprot_map[k] = list(v)
    return nest(pdb_map, 'ec'), nest(uniprot_map, 'ec')

def gen_go_labels():
    lines = Path('datasets/GeneOntology/nrPDB-GO_annot.tsv').read_text().splitlines()
    task_labels: dict[str, ClassLabel] = {
        task: ClassLabel(lines[line_id].split('\t'))
        for task, line_id in [
            ('mf', 1),
            ('bp', 5),
            ('cc', 9),
        ]
    }
    pdb_map = {}
    uniprot_counter = Counter()
    uniprot_map_counter: dict[str, dict[str, Counter]] = defaultdict(lambda: {
        task: Counter()
        for task in task_labels
    })
    for line in lines[13:]:
        item_id, *task_annotations = line.split('\t')
        item_id = normalize_item_id(item_id)
        task_annotations: dict[str, set[int]] = {
            task: set(cytoolz.get(label.split(',') if label else [], class_label.index))
            for label, (task, class_label) in zip(task_annotations, task_labels.items())
        }
        if (pdb_annotations := pdb_map.get(item_id)) is None:
            pdb_map[item_id] = {
                task: list(annotations)
                for task, annotations in task_annotations.items()
            }
        else:
            assert pdb_annotations == task_annotations
        if (uniprot_id := uniprot_table[item_id]) is None:
            continue
        uniprot_counter[uniprot_id] += 1
        for task, counter in uniprot_map_counter[uniprot_id].items():
            annotations = task_annotations[task]
            for i in annotations:
                counter[i] += 1
    uniprot_map = {
        uniprot_id: {}
        for uniprot_id in uniprot_map_counter
    }
    for uniprot_id, task_counters in uniprot_map_counter.items():
        task_maps = uniprot_map[uniprot_id]
        for task, counter in task_counters.items():
            if uniprot_counter[uniprot_id] == 1:
                task_maps[task] = list(counter)
            else:
                task_maps[task] = [k for k, c in counter.items() if c > 1]
    return pdb_map, uniprot_map

def merge(data: Sequence[dict[str, dict]]):
    ret = defaultdict(dict)
    for x in data:
        for k, task_annotations in x.items():
            for task, annotations in task_annotations.items():
                ret[k][task] = annotations
    return ret

def main():
    pdb_map, uniprot_map = zip(gen_ec_labels(), gen_go_labels())
    pdb_to_property_path.write_text(json.dumps(merge(pdb_map), indent=4, ensure_ascii=False))
    uniprot_to_property_path.write_text(json.dumps(merge(uniprot_map), indent=4, ensure_ascii=False))

if __name__ == '__main__':
    main()
