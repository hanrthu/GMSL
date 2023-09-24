import itertools as it
from pathlib import Path

from Bio.PDB.PDBExceptions import PDBConstructionException
from openbabel import pybel
import pandas as pd
import torch
from tqdm.contrib.concurrent import process_map

from gmsl.data import ModelData, SavedSet, normalize_item_id, parse_ligand, parse_structure
from gmsl.data.path import parsed_dir

corrupted_paths = SavedSet(parsed_dir / 'corrupted.txt')
nmr_paths = SavedSet(parsed_dir / 'nmr.txt')

jobs: list[tuple[Path, Path, ...]] = []
skipped_paths: set[str] = set()

def submit(src: Path, dst: Path, *args):
    if str(src) not in skipped_paths and not dst.exists():
        jobs.append((src, dst, *args))

def process_model(src: Path, dst: Path, ignore_nmr: bool):
    try:
        structure = parse_structure(src)
    except (ValueError, PDBConstructionException):
        corrupted_paths.save(str(src))
        return
    if ignore_nmr and len(structure) != 1:
        nmr_paths.save(str(src))
        return
    model = ModelData.from_model(structure[0])
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model, dst)

def process_ligand(src: Path, dst: Path):
    ligand = parse_ligand(src)
    torch.save(ligand, dst)

def update_skipped_paths(save_path: Path):
    save_path.touch()
    skipped_paths.update(set(save_path.read_text().splitlines()))

def main():
    global skipped_paths
    skipped_paths = corrupted_paths.set | nmr_paths.set
    for src in Path('datasets/pdbx-mmcif').glob('*.cif'):
        submit(src, parsed_dir / 'mmcif' / f'{src.stem}.pt', False)
    for src in Path('datasets/PDBbind/PP').glob('*.ent.pdb'):
        submit(src, parsed_dir / 'PDBbind/PP' / (src.stem.split('.')[0] + '.pt'), True)
    for src in Path('datasets/PDBbind/refined-set').glob('*/*_protein.pdb'):
        submit(src, parsed_dir / 'PDBbind/refined-set' / src.parent.name / 'protein.pt', True)
    for task, split in it.product(
        ['EnzymeCommission', 'GeneOntology'],
        ['train', 'valid', 'test'],
    ):
        for src in (Path('datasets') / task / split).glob('*.pdb'):
            item_id = src.stem.split('_', 1)[0]
            submit(src, parsed_dir / task / f'{normalize_item_id(item_id)}.pt', False)
    if jobs:
        # from tqdm import tqdm
        # for args in tqdm(jobs):
        #     process_model(*args)
        process_map(
            process_model, *zip(*jobs),
            max_workers=32, chunksize=4, ncols=80, desc='process model',
        )
        jobs.clear()
    for src in Path('datasets/PDBbind/refined-set').glob('*/*_ligand.mol2'):
        if (case_parsed_dir := parsed_dir / 'PDBbind/refined-set' / src.parent.name).exists():
            submit(src, case_parsed_dir / 'ligand.pt')
    if jobs:
        pybel.ob.obErrorLog.SetOutputLevel(0)
        process_map(
            process_ligand, *zip(*jobs),
            max_workers=16, ncols=80, chunksize=8, desc='process ligand'
        )

if __name__ == '__main__':
    main()
