import itertools as it
from pathlib import Path

from Bio.PDB.PDBExceptions import PDBConstructionException
from openbabel import pybel
import pandas as pd
from tqdm.contrib.concurrent import process_map

from gmsl.data import PROCESSED_DIR, ModelData, append_ln, parse_ligand, parse_structure

output_dir = PROCESSED_DIR / 'parsed'
corrupted_save_path = output_dir / 'corrupted.txt'
nmr_save_path = output_dir / 'nmr.txt'

jobs: list[tuple[Path, Path, ...]] = []
skipped_paths: set[str] = set()

def submit(src: Path, dst: Path, *args):
    if str(src) not in skipped_paths and not dst.exists():
        jobs.append((src, dst, *args))

def process_model(src: Path, dst: Path, ignore_nmr: bool):
    try:
        structure = parse_structure(src)
    except (ValueError, PDBConstructionException):
        append_ln(corrupted_save_path, str(src))
        return
    if ignore_nmr and len(structure) != 1:
        append_ln(nmr_save_path, str(src))
        return
    model = ModelData.from_model(structure[0])
    dst.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(model, dst)

def process_ligand(src: Path, dst: Path):
    ligand = parse_ligand(src)
    pd.to_pickle(ligand, dst)

def main():
    corrupted_save_path.touch()
    skipped_paths.update(set(corrupted_save_path.read_text().splitlines()))
    nmr_save_path.touch()
    skipped_paths.update(set(nmr_save_path.read_text().splitlines()))
    # for src in Path('datasets/pdbx-mmcif').glob('*.cif'):
    #     submit(src, output_dir / 'mmcif' / f'{src.stem}.pkl', False)
    for src in Path('datasets/PDBbind/PP').glob('*.ent.pdb'):
        submit(src, output_dir / 'PDBbind/PP' / (src.stem.split('.')[0] + '.pkl'), True)
    for src in Path('datasets/PDBbind/refined-set').glob('*/*_protein.pdb'):
        submit(src, output_dir / 'PDBbind/refined-set' / src.parent.name / 'protein.pkl', True)
    for task, split in it.product(
        ['EnzymeCommission', 'GeneOntology'],
        ['train', 'valid', 'test'],
    ):
        for src in (Path('datasets') / task / split).glob('*.pdb'):
            pdb_id, chain_id = src.stem.split('_', 1)[0].split('-')
            pdb_id = pdb_id.lower()
            submit(src, output_dir / task / f'{pdb_id}-{chain_id}.pkl', False)
    if jobs:
        process_map(
            process_model, *zip(*jobs),
            max_workers=1, chunksize=4, ncols=80, desc='process model',
        )
        jobs.clear()
    for src in Path('datasets/PDBbind/refined-set').glob('*/*_ligand.mol2'):
        if (case_output_dir := output_dir / 'PDBbind/refined-set' / src.parent.name).exists():
            submit(src, case_output_dir / 'ligand.pkl')
    if jobs:
        pybel.ob.obErrorLog.SetOutputLevel(0)
        process_map(
            process_ligand, *zip(*jobs),
            max_workers=16, ncols=80, chunksize=8, desc='process ligand'
        )

if __name__ == '__main__':
    main()
