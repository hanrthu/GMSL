from Bio.PDB.PDBExceptions import PDBConstructionException
import fcntl
import itertools as it
from pathlib import Path
from typing import TextIO

from Bio.PDB.Chain import Chain
import cytoolz
from tqdm.contrib.concurrent import process_map

from gmsl.data import cmp_entity, parse_model
from gmsl.data.path import PROCESSED_DIR

filepaths: dict[str, dict[str, Path]] = {}

output_dir = PROCESSED_DIR / 'chain-check'
checked_path = output_dir / '.checked.txt'

def ex_writeln(f: TextIO, s: str):
    fcntl.lockf(f, fcntl.LOCK_EX)
    f.write(s + '\n')
    fcntl.lockf(f, fcntl.LOCK_UN)

def append_ln(filepath: Path, s: str):
    with open(filepath, 'a') as f:
        fcntl.lockf(f, fcntl.LOCK_EX)
        f.write(s + '\n')
        fcntl.lockf(f, fcntl.LOCK_UN)

def check(pdb_id: str):
    model_pdb, is_nmr = parse_model(f'datasets/pdbx-mmcif/{pdb_id}.cif')
    passed = []
    failed = []
    for chain_id, filepath in filepaths[pdb_id].items():
        pdb_chain = f'{pdb_id.upper()}-{chain_id}'
        if (chain_pdb := model_pdb.child_dict.get(chain_id, None)) is None:
            append_ln(output_dir / 'not-found.txt', pdb_chain)
            continue
        try:
            model_chain, _ = parse_model(filepath)
        except (ValueError, PDBConstructionException):
            append_ln(output_dir / 'corrupted.txt', pdb_chain)
            continue
        assert len(model_chain) == 1
        chain: Chain = next(model_chain.get_chains())
        if cmp_entity(chain_pdb, chain):
            passed.append(pdb_chain)
        else:
            failed.append(pdb_chain)
    if passed:
        append_ln(output_dir / 'passed.txt', '\n'.join(passed))
    if failed:
        append_ln(output_dir / 'failed.txt', '\n'.join(failed))
        if is_nmr:
            append_ln(output_dir / 'failed-nmr.txt', '\n'.join(failed))
    with open(checked_path, 'a') as f:
        ex_writeln(f, pdb_id)

def main():
    output_dir.mkdir(exist_ok=True)
    checked_path.touch()
    checked = set(checked_path.read_text().splitlines())

    for filepath in cytoolz.concat(
        (Path('datasets') / task / split).glob('*.pdb')
        for task, split in it.product(
            ['EnzymeCommission', 'GeneOntology'],
            ['train', 'valid', 'test'],
        )
    ):
        pdb_id, chain_id = filepath.stem.split('_', 1)[0].split('-')
        pdb_id = pdb_id.lower()
        if pdb_id not in checked:
            filepaths.setdefault(pdb_id, {})[chain_id] = filepath
    process_map(check, list(filepaths), max_workers=32, chunksize=4)

if __name__ == '__main__':
    main()
