from pathlib import Path

from Bio.PDB import FastMMCIFParser, PDBParser
from tqdm.contrib.concurrent import process_map

from gmsl.data import PROCESSED_DIR, cmp_entity, parse_model

pdb_parser = PDBParser(QUIET=True)
mmcif_parser = FastMMCIFParser(QUIET=True)

def check(pdb_id: str):
    model_pdb = parse_model(f'datasets/pdbx-mmcif/{pdb_id}.cif')
    model_pp = parse_model(f'datasets/PDBbind/PP/{pdb_id}.ent.pdb')
    if model_pdb is None:
        assert model_pp is None
        return None
    assert model_pp is not None
    return None if cmp_entity(model_pdb, model_pp) else pdb_id

def main():
    pdb_ids = [
        filepath.stem.split('.', 1)[0]
        for filepath in Path('datasets/PDBbind/PP').glob('*.ent.pdb')
    ]
    results = process_map(check, pdb_ids, max_workers=32, chunksize=4)
    results = list(filter(lambda x: x is not None, results))
    (PROCESSED_DIR / 'pp-check.txt').write_text('\n'.join(results) + '\n')

if __name__ == '__main__':
    main()
