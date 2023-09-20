from dataclasses import dataclass
from pathlib import Path

from Bio.PDB import FastMMCIFParser, PDBParser
from Bio.PDB.Atom import Atom
from Bio.PDB.Chain import Chain
from Bio.PDB.Model import Model
from Bio.PDB.Residue import Residue
from atom3d.util.formats import atomic_number
import numpy as np
from openbabel import pybel
import pandas as pd
import torch
from torch.types import Device

from gmsl.data.utils import PathLike

pdb_parser = PDBParser(QUIET=True)
mmcif_parser = FastMMCIFParser(QUIET=True)

__all__ = [
    'parse_structure',
    'parse_model',
    'ModelData',
    'ResidueData',
    'ChainData',
    'parse_ligand',
]

def parse_structure(path: PathLike, structure_id: str = ''):
    path = Path(path)
    match path.suffix:
        case '.pdb':
            structure = pdb_parser.get_structure(structure_id, path)
        case '.cif':
            structure = mmcif_parser.get_structure(structure_id, path)
        case _:
            raise ValueError
    for res in structure.get_residues():
        res.child_list.sort()
    return structure

def parse_model(path: PathLike, structure_id: str = '') -> tuple[Model, bool]:
    structure = parse_structure(path, structure_id)
    return structure[0], len(structure) != 1

halogen_elements = {'F', 'Cl', 'Br', 'I', 'At'}
metal_elements = {'Fe', 'Zn', 'Mg', 'Mn', 'K', 'Li', 'Ca', 'Hg', 'Na'}
atomic_num_to_name: dict[int, str] = {
    atomic_num: name
    for name, atomic_num in atomic_number.items()
}

reduced_element_index = {
    'H' : 1,
    'C' : 2,
    'N' : 3,
    'O' : 4,
    'S' : 5,
    'P' : 6,
    'Metal': 7,
    'Halogen': 8,
}

def reduce_element(element: str) -> int:
    element = element.capitalize()
    if element in halogen_elements:
        element = 'Halogen'
    elif element in metal_elements:
        element = 'Metal'
    return reduced_element_index.get(element, 9)

MAX_CHANNEL: int = 14
standard_proteinogenic_amino_acids = {
    "GLY": 0,
    "ALA": 1,
    "SER": 2,
    "PRO": 3,
    "VAL": 4,
    "THR": 5,
    "CYS": 6,
    "ILE": 7,
    "LEU": 8,
    "ASN": 9,
    "ASP": 10,
    "GLN": 11,
    "LYS": 12,
    "GLU": 13,
    "MET": 14,
    "HIS": 15,
    "PHE": 16,
    "ARG": 17,
    "TYR": 18,
    "TRP": 19,
}

side_chain_table = {
    'PHE': ['N', 'CA', 'C', 'O', 'CB', 'CD1', 'CD2', 'CE1', 'CE2', 'CG', 'CZ'],
    'GLN': ['N', 'CA', 'C', 'O', 'CB', 'CD', 'CG', 'NE2', 'OE1'],
    'PRO': ['N', 'CA', 'C', 'O', 'CB', 'CD', 'CG'],
    'ASN': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'ND2', 'OD1'],
}

@dataclass
class ResidueData:
    name: str
    coords: np.ndarray
    elements: np.ndarray

    @classmethod
    def from_residue(cls, residue: Residue):
        coords = []
        elements = []
        for atom in residue.get_atoms():
            atom: Atom
            if atom.element == 'H':
                continue
            coords.append(atom.coord)
            elements.append(reduce_element(atom.element))

        name = residue.get_resname()
        if name in standard_proteinogenic_amino_acids:
            assert len(elements) <= MAX_CHANNEL
            atom_names = [atom.name for atom in residue.get_atoms()]
            assert atom_names == side_chain_table[name]

        return ResidueData(name, np.array(coords), np.array(elements))

    # def to_df(self):
    #     return pd.DataFrame({
    #         **{
    #             axis: self.coords[:, axis_idx].tolist()
    #             for axis_idx, axis in enumerate('xyz')
    #         },
    #     })

    def to_multi_channel(self, device: Device = None):
        if self.name == LIGAND_RESIDUE_NAME:
            pass
        else:
            pos = torch.zeros(MAX_CHANNEL, 3, device=device)
            element = self.elements
            # X_protein = protein_pos.new_zeros((len(protein_seq), MAX_CHANNEL, 3))  # [N, n_channel, d]

@dataclass
class ChainData:
    residues: list[ResidueData]

    @classmethod
    def from_chain(cls, chain: Chain):
        residues = []
        for residue_idx, residue in enumerate(chain.get_residues()):
            residue: Residue
            if residue.resname == 'HOH':
                continue
            residues.append(ResidueData.from_residue(residue))
        return ChainData(residues)

    def to_df(self):
        residue_dfs = []
        for residue_idx, residue in enumerate(self.residues):
            residue_df = residue.to_df()
            residue_df['residue'] = residue_idx
            residue_dfs.append(residue_df)
        return pd.concat(residue_dfs)

@dataclass
class ModelData:
    chains: dict[str, ChainData]

    @classmethod
    def from_model(cls, model: Model):
        return ModelData({
            chain_id: ChainData.from_chain(chain)
            for chain_id, chain in model.child_dict.items()
        })

    def to_df(self):
        chain_dfs = []
        for chain_id, chain in self.chains.items():
            chain_df = chain.to_df()
            chain_df['chain'] = chain_id
            chain_dfs.append(chain_df)
        return pd.concat(chain_dfs)

LIGAND_RESIDUE_NAME = 'LIG'

def parse_ligand(path: PathLike):
    ligand = next(pybel.readfile('mol2', str(path)))
    coords = []
    elements = []
    for atom in ligand:
        element = atomic_num_to_name[atom.atomicnum]
        if element == 'H':
            continue
        coords.append(atom.coords)
        elements.append(reduce_element(element))
    return ResidueData(LIGAND_RESIDUE_NAME, np.array(coords), np.array(elements))
