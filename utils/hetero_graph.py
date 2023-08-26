import torch
import pandas as pd
from atom3d.util.graph import mol_atoms, one_of_k_encoding_unk
from torch_geometric.data import Data
import torch_cluster
from typing import List
import math
from tqdm import tqdm
from typing import Any
from torch_geometric.typing import OptTensor, SparseTensor
import torch.nn.functional as F
import yaml
import jinja2
import easydict
from .edge_construction import KNNEdge, SpatialEdge, SequentialEdge
from .utils import MyData
NUM_ATOM_TYPES = 10
MAX_CHANNEL = 14 # 4 backbone + sidechain
# The element mapping for atoms
element_mapping = lambda x: {
    'Super': 0,
    'H' : 1,
    'C' : 2,
    'N' : 3,
    'O' : 4,
    'S' : 5,
    'P' : 6,
    'Metal': 7,
    'Halogen':8,
}.get(x, 9)

# The weight mapping for atoms
weight_mapping = lambda x: {
    'H' : 1,    
    'C' : 12,
    'N' : 14,
    'O' : 16,
    'S' : 32,
    'P' : 31,
    'Li': 3,  'LI': 3,
    'Mn': 55, 'MN': 55,
    'Cl': 35.5,
    'K' : 39,
    'Fe': 56, 'FE': 56,
    'Zn': 65, 'ZN': 65,
    'Mg': 24, 'MG': 24,
    'Br': 80, 'BR': 80,
    'I' : 127,
}.get(x, 0)

# The order is the same as TorchDrug.Protein
amino_acids = lambda x: {"GLY": 0, "ALA": 1, "SER": 2, "PRO": 3, "VAL": 4, "THR": 5, "CYS": 6, "ILE": 7, "LEU": 8,
                  "ASN": 9, "ASP": 10, "GLN": 11, "LYS": 12, "GLU": 13, "MET": 14, "HIS": 15, "PHE": 16,
                  "ARG": 17, "TYR": 18, "TRP": 19}.get(x, 20)

def gen_multi_channel_coords(protein_df, ligand_df, protein_seq):
    res_info = protein_df['residue'].values
    protein_pos = torch.as_tensor(protein_df[["x", "y", "z"]].values, dtype=torch.float64)
    protein_element = torch.as_tensor(list(map(element_mapping, protein_df['element'])), dtype=torch.long)
    element_protein = torch.zeros((len(protein_seq), MAX_CHANNEL)) 
    X_protein = torch.zeros((len(protein_seq), MAX_CHANNEL, 3)) # [N, n_channel, d] 
    mask_protein = torch.zeros(X_protein.shape[:-1])
    current_channel = 0
    for i, item in enumerate(res_info):
        X_protein[int(item)-1, current_channel, :] = protein_pos[i]
        element_protein[int(item-1), current_channel] = protein_element[i]
        mask_protein[int(item)-1, current_channel] = 1
        if i < len(res_info-1) and res_info[i] == res_info[i+1]:
            current_channel += 1
        else:
            current_channel = 0
    if len(ligand_df) > 0:
        ligand_coords = torch.as_tensor(ligand_df[["x", "y", "z"]].values, dtype=torch.float64)
        ligand_element = torch.as_tensor(list(map(element_mapping, ligand_df['element'])), dtype=torch.long)
        X_ligand = torch.zeros((len(ligand_df), MAX_CHANNEL, 3))
        element_ligand = torch.zeros((len(ligand_df), MAX_CHANNEL))
        mask_ligand = torch.zeros(X_ligand.shape[:-1])
        for i, item in enumerate(ligand_coords):
            X_ligand[i, 0, :] = item
            element_ligand[i, 0] = ligand_element[i]
            mask_ligand[i, 0] = 1
        X = torch.cat([X_protein, X_ligand], dim=0)
        mask = torch.cat([mask_protein, mask_ligand], dim=0)
        element = torch.cat([element_protein, element_ligand], dim=0)
    else:
        X = X_protein
        mask = mask_protein
        element = element_protein
    return X, mask, element
    

def hetero_graph_transform(
    atom_df: pd.DataFrame,
    protein_seq:pd.DataFrame,
    cutoff: float = 4.5,
    feat_col: str = "resname",
    init_dtype: torch.dtype = torch.float64,
    super_node: bool = False,
    offset_strategy : int = 0,
    flag: torch.Tensor = None, 
    alpha_only = False
    ):
    """
    A function that can generate graph with different kinds of edges
    """
    # TODO: 重构此段代码，需要适配alpha only和全原子的两种情况（目前全原子的情况仅对坐标做multichannel的适配, 氨基酸仍然用一个feature来表示）
    print("Creating Heterogenuous graph!")
    protein_df = atom_df[flag!=0].reset_index(drop=True)
    ligand_df = atom_df[flag==0].reset_index(drop=True)
    if not alpha_only:
        pos, channel_weights, residue_elements = gen_multi_channel_coords(protein_df, ligand_df, protein_seq) # [N, n_channel, d], [N, n_channel], [N, n_channel] 
        # Retains alpha_carbon for protein_node representation
        protein_df = protein_df[protein_df.name == 'CA'].reset_index(drop=True)
        max_channel = MAX_CHANNEL
    else:
        pos = torch.as_tensor(atom_df[["x", "y", "z"]].values, dtype=init_dtype).unsqueeze(1) # [N, 1, d]
        channel_weights = torch.ones((len(pos), 1)) # [N, 1]
        residue_elements = None
        max_channel = 1
    protein_feats = torch.as_tensor(list(map(amino_acids, protein_df[feat_col])), dtype=torch.long)
    if len(ligand_df) != 0:
        ligand_feats = torch.as_tensor(list(map(element_mapping, ligand_df['element'])), dtype=torch.long)
        ligand_feats += torch.max(protein_feats) + 1
        node_feats = torch.cat([protein_feats, ligand_feats], dim=-1)
    else:
        node_feats = protein_feats # Node Features 用于给Node初始化，和下述的atom type不一样
    # print("Node Features:", node_feats.shape)
    # Onehot coding, 21 amino_acids + 10 atoms
    node_feats = torch.zeros((len(node_feats), 31)).scatter_(1, node_feats.unsqueeze(1), 1)
    residues = torch.as_tensor(list(map(amino_acids, protein_seq)), dtype=torch.long)
    # Three types of edges
    knn = KNNEdge(k=10, min_distance=5, max_channel=max_channel)
    spatial = SpatialEdge(radius=10, min_distance=5, max_channel=max_channel)
    sequential = SequentialEdge(max_distance=2, max_channel=max_channel)
    edge_layers = [knn, spatial, sequential]
    graph = MyData(
        x=node_feats,
        residue_elements=residue_elements,
        num_nodes=len(node_feats),
        pos=pos.to(torch.get_default_dtype()),
        channel_weights=channel_weights.long(),
        num_residues = len(residues),
        edge_feature=None,
        chain=atom_df['chain']
    )
    edge_list = []
    num_edges = []
    num_relations = []
    for layer in edge_layers:
        edges, num_relation = layer(graph)
        edge_list.append(edges)
        num_edges.append(len(edges))
        num_relations.append(num_relation)
    # print("Nume Nodes", len(graph.x))
    # print("Num KNN Edges:", len(edge_list[0]))
    # print("Num Spatial Edges:", len(edge_list[1]))
    # print("Num Sequential Edges:", len(edge_list[2]))
    edge_list = torch.cat(edge_list)
    num_edges = torch.tensor(num_edges)
    num_relations = torch.tensor(num_relations)
    num_relation = num_relations.sum()
    offsets = (num_relations.cumsum(0) - num_relations).repeat_interleave(num_edges)
    edge_list[:, 2] += offsets
    # 这里把edge_index和edge_relation分开装，是为了能够让处理batch的时候不要把relation也加上offset
    graph.edge_index = edge_list[:, :2].t()
    graph.edge_relations = edge_list[:, 2]
    graph.edge_weights = torch.ones(len(edge_list))
    graph.num_relation = num_relation
    return graph
    
    
    
        
        
    