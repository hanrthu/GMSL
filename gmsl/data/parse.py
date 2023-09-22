from dataclasses import dataclass
from pathlib import Path
from typing import Self

from Bio.PDB import FastMMCIFParser, PDBParser
from Bio.PDB.Atom import Atom
from Bio.PDB.Chain import Chain
from Bio.PDB.Model import Model
from Bio.PDB.Residue import Residue
from atom3d.util.formats import atomic_number
import numpy as np
from openbabel import pybel
import torch
from torch.types import Device
from torch.nn import functional as nnf

from gmsl.data.path import PathLike

pdb_parser = PDBParser(QUIET=True)
mmcif_parser = FastMMCIFParser(QUIET=True)

__all__ = [
    'parse_structure',
    'parse_model',
    'ModelData',
    'ResidueData',
    'ChainData',
    'parse_ligand',
    'MultiChannelData',
    'MAX_CHANNEL',
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
    'Super': 0,
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

num_reduced_elements = max(reduced_element_index.values()) + 2

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
def reduce_resname(resname: str):
    return standard_proteinogenic_amino_acids.get(resname.upper(), 20)

num_reduced_resnames = max(standard_proteinogenic_amino_acids.values()) + 2
num_node_feat_classes = num_reduced_resnames + num_reduced_elements

side_chain_table = {
    'PHE': ('N', 'CA', 'C', 'O', 'CB', 'CD1', 'CD2', 'CE1', 'CE2', 'CG', 'CZ'),
    'GLN': ('N', 'CA', 'C', 'O', 'CB', 'CD', 'CG', 'NE2', 'OE1'),
    'PRO': ('N', 'CA', 'C', 'O', 'CB', 'CD', 'CG'),
    'ASN': ('N', 'CA', 'C', 'O', 'CB', 'CG', 'ND2', 'OD1'),
    'GLU': ('N', 'CA', 'C', 'O', 'CB', 'CD', 'CG', 'OE1', 'OE2'),
    'LYS': ('N', 'CA', 'C', 'O', 'CB', 'CD', 'CE', 'CG', 'NZ'),
    'THR': ('N', 'CA', 'C', 'O', 'CB', 'CG2', 'OG1'),
    'GLY': ('N', 'CA', 'C', 'O'),
    'ALA': ('N', 'CA', 'C', 'O', 'CB'),
    'ASP': ('N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'OD2'),
    'CYS': ('N', 'CA', 'C', 'O', 'CB', 'SG'),
    'LEU': ('N', 'CA', 'C', 'O', 'CB', 'CD1', 'CD2', 'CG'),
    'ARG': ('N', 'CA', 'C', 'O', 'CB', 'CD', 'CG', 'CZ', 'NE', 'NH1', 'NH2'),
    'VAL': ('N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2'),
    'SER': ('N', 'CA', 'C', 'O', 'CB', 'OG'),
    'TYR': ('N', 'CA', 'C', 'O', 'CB', 'CD1', 'CD2', 'CE1', 'CE2', 'CG', 'CZ', 'OH'),
    'ILE': ('N', 'CA', 'C', 'O', 'CB', 'CD1', 'CG1', 'CG2'),
    'HIS': ('N', 'CA', 'C', 'O', 'CB', 'CD2', 'CE1', 'CG', 'ND1', 'NE2'),
    'MET': ('N', 'CA', 'C', 'O', 'CB', 'CE', 'CG', 'SD'),
    'TRP': ('N', 'CA', 'C', 'O', 'CB', 'CD1', 'CD2', 'CE2', 'CE3', 'CG', 'CH2', 'CZ2', 'CZ3', 'NE1'),
}

side_chain_to_name = {
    side_chain: name
    for name, side_chain in side_chain_table.items()
}

@dataclass
class MultiChannelData:
    pos: torch.Tensor
    element: torch.Tensor
    mask: torch.Tensor
    node_feats: torch.Tensor

    @property
    def n(self):
        return self.element.shape[0]

    @classmethod
    def merge(cls, a: Self, b: Self):
        # TODO: make this better
        return MultiChannelData(
            torch.cat([a.pos, b.pos]),
            torch.cat([a.element, b.element]),
            torch.cat([a.mask, b.mask]),
            torch.cat([a.node_feats, b.node_feats]),
        )

@dataclass
class ResidueData:
    is_ligand: bool
    coords: np.ndarray
    elements: np.ndarray
    mask: np.ndarray

    @classmethod
    def from_residue(cls, residue: Residue):
        coords = []
        elements = []
        atom_names = ()
        for atom in residue.get_atoms():
            atom: Atom
            if atom.element == 'H' or atom.element == 'D' or atom.name == 'OXT':
                continue
            coords.append(atom.coord)
            elements.append(reduce_element(atom.element))
            atom_names += (atom.name, )

        default_indexes = []
        assert len(coords) > 0
        default_coord = np.mean(coords, axis=0)
        name: str = residue.get_resname()
        if name in standard_proteinogenic_amino_acids:
            assert len(elements) <= MAX_CHANNEL
            inferred_name = side_chain_to_name.get(atom_names)
            if inferred_name is None:
                ref = side_chain_table[name]
                assert set(ref).issuperset(set(atom_names))
                atom_names = list(atom_names)
                for i, ref_name in enumerate(ref):
                    if i == len(atom_names) or atom_names[i] != ref_name:
                        default_indexes.append(i)
                        coords.insert(i, default_coord)
                        elements.insert(i, reduce_element(ref_name[0]))
                        atom_names.insert(i, ref_name)
            elif inferred_name != name:
                name = inferred_name
        mask = np.ones(len(elements), dtype=bool)
        mask[default_indexes] = False
        return name, ResidueData(False, np.array(coords), np.array(elements), mask)

    @property
    def num_atoms(self):
        return self.elements.shape[0]

    def to_multi_channel(self, device: Device = None):
        assert self.is_ligand
        num_atoms = self.num_atoms
        pos = torch.zeros(num_atoms, MAX_CHANNEL, 3, device=device)
        pos[:, 0] = torch.as_tensor(self.coords, device=device)
        element = torch.zeros(num_atoms, MAX_CHANNEL, dtype=torch.long, device=device)
        element[:, 0] = torch.as_tensor(self.elements, device=device)
        mask = torch.zeros(num_atoms, MAX_CHANNEL, dtype=torch.long, device=device)
        mask[:, 0] = 1
        return MultiChannelData(pos, mask, element, nnf.one_hot(element[:, 0] + num_reduced_resnames, num_node_feat_classes))

@dataclass
class ChainData:
    residues: list[ResidueData]
    resname: np.ndarray

    @classmethod
    def from_chain(cls, chain: Chain):
        residues = []
        resnames = []
        for residue_idx, residue in enumerate(chain.get_residues()):
            residue: Residue
            if residue.resname == 'HOH':
                continue
            resname, residue_data = ResidueData.from_residue(residue)
            residues.append(residue_data)
            resnames.append(reduce_resname(resname))
        return ChainData(residues, np.array(resnames))

    @property
    def num_residues(self):
        return len(self.residues)

    @property
    def num_atoms(self):
        return sum(residue.num_atoms for residue in self.residues)

@dataclass
class ModelData:
    chains: dict[str, ChainData]

    @classmethod
    def from_model(cls, model: Model):
        return ModelData({
            chain_id: ChainData.from_chain(chain)
            for chain_id, chain in model.child_dict.items()
            if chain_id != ' '
        })

    @property
    def num_atoms(self):
        return sum(chain.num_atoms for chain in self.chains.values())

    def to_multi_channel(self, device: Device):
        num_residues = sum(len(chain.residues) for chain in self.chains.values())
        pos = torch.zeros(num_residues, MAX_CHANNEL, 3, device=device)
        element = torch.zeros(num_residues, MAX_CHANNEL, dtype=torch.long, device=device)
        mask = torch.zeros(num_residues, MAX_CHANNEL, dtype=torch.long, device=device)
        chain_id = torch.empty(num_residues, dtype=torch.long, device=device)
        resname = torch.empty(num_residues, dtype=torch.long, device=device)

        residue_idx = 0
        for chain_idx, chain in enumerate(self.chains.values()):
            chain_id[residue_idx:residue_idx + chain.num_residues] = chain_idx
            resname[residue_idx:residue_idx + chain.num_residues] = torch.as_tensor(chain.resname, device=device)
            for residue in chain.residues:
                num_atoms = residue.num_atoms
                pos[residue_idx, :num_atoms] = torch.as_tensor(residue.coords[:MAX_CHANNEL], device=device)
                element[residue_idx, :num_atoms] = torch.as_tensor(residue.elements[:MAX_CHANNEL], device=device)
                mask[residue_idx, :num_atoms] = 1
                residue_idx += 1
        return MultiChannelData(pos, element, mask, nnf.one_hot(resname, num_node_feat_classes)), chain_id

def parse_ligand(path: PathLike):
    ligand = next(pybel.readfile('mol2', str(path)))
    coords = []
    elements = []
    for atom in ligand:
        element = atomic_num_to_name[atom.atomicnum]
        if element == 'H' or element == 'D':
            continue
        coords.append(atom.coords)
        elements.append(reduce_element(element))
    return ResidueData(True, np.array(coords), np.array(elements), np.ones(len(elements), dtype=bool))
