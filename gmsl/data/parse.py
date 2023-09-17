from pathlib import Path

from Bio.PDB import PDBParser, FastMMCIFParser
from Bio.PDB.Model import Model

from gmsl.data.utils import PathLike

pdb_parser = PDBParser(QUIET=True)
mmcif_parser = FastMMCIFParser(QUIET=True)

__all__ = [
    'parse_model',
]

def parse_model(path: PathLike, structure_id: str = '') -> Model | None:
    path = Path(path)
    match path.suffix:
        case '.pdb':
            structure = pdb_parser.get_structure(structure_id, path)
        case '.cif':
            structure = mmcif_parser.get_structure(structure_id, path)
        case _:
            raise ValueError
    if len(structure) != 1:
        return None
    model: Model = structure[0]
    for res in model.get_residues():
        res.child_list.sort()
    return model
