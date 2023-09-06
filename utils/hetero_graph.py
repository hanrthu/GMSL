import pandas as pd
import torch
from torch.types import Device

from .edge_construction import KNNEdge, SequentialEdge, SpatialEdge
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

def gen_multi_channel_coords(
    protein_df: pd.DataFrame,
    ligand_df: pd.DataFrame,
    protein_seq: list[str],
    device: Device = None
):
    res_info = torch.as_tensor(protein_df['residue'].array, device=device)
    atom_chain_id = torch.as_tensor(protein_df['chain'].factorize()[0], device=device)
    protein_pos = torch.as_tensor(protein_df[['x', 'y', 'z']].to_numpy(), dtype=torch.float64, device=device)
    protein_element = torch.as_tensor(protein_df['element'].map(element_mapping).array, dtype=torch.long, device=device)
    element_protein = protein_element.new_zeros((len(protein_seq), MAX_CHANNEL))
    X_protein = protein_pos.new_zeros((len(protein_seq), MAX_CHANNEL, 3))  # [N, n_channel, d]
    mask_protein = X_protein.new_zeros(X_protein.shape[:-1])
    start_idx = torch.arange(len(res_info) - 1, device=device)[res_info[:-1] != res_info[1:]] + 1
    chain_id = atom_chain_id.gather(0, torch.cat([start_idx.new_tensor([0]), start_idx]))
    assert len(chain_id) == len(X_protein)
    start_idx = start_idx.cpu()
    for i, (pos, element) in enumerate(zip(
        protein_pos.tensor_split(start_idx),
        protein_element.tensor_split(start_idx),
    )):
        pos = pos[:MAX_CHANNEL]
        element = element[:MAX_CHANNEL]
        num_channels = len(pos)
        X_protein[i, :num_channels] = pos
        element_protein[i, :num_channels] = element
        mask_protein[i, :num_channels] = 1

    if len(ligand_df) > 0:
        ligand_coords = torch.as_tensor(ligand_df[['x', 'y', 'z']].to_numpy(), dtype=torch.float64, device=device)
        ligand_element = torch.as_tensor(ligand_df['element'].map(element_mapping).array, dtype=torch.long, device=device)
        X_ligand = ligand_coords.new_zeros((len(ligand_df), MAX_CHANNEL, 3))
        element_ligand = ligand_element.new_zeros((len(ligand_df), MAX_CHANNEL))
        mask_ligand = X_ligand.new_zeros(X_ligand.shape[:-1])
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
    return X, mask, element, chain_id

# Debug Passed
def hetero_graph_transform(
    item_name: str,
    atom_df: pd.DataFrame,
    protein_seq: list[str],
    cutoff: float = 4.5,
    feat_col: str = "resname",
    init_dtype: torch.dtype = torch.float64,
    super_node: bool = False,
    alpha_only: bool = False,
    device: Device = None,
):
    """
    A function that can generate graph with different kinds of edges
    """
    # 目前全原子的情况仅对坐标做multichannel的适配, 氨基酸仍然用一个feature来表示
    protein_df: pd.DataFrame = atom_df[atom_df.resname != "LIG"].reset_index(drop=True)
    ligand_df: pd.DataFrame = atom_df[atom_df.resname == "LIG"].reset_index(drop=True)
    # start = datetime.now()
    if not alpha_only:
        pos, channel_weights, residue_elements, chain = gen_multi_channel_coords(protein_df, ligand_df, protein_seq, device) # [N, n_channel, d], [N, n_channel], [N, n_channel]
        # Retains alpha_carbon for protein_node representation
        max_channel = MAX_CHANNEL
        protein_feats = torch.as_tensor(list(map(amino_acids, protein_seq)), dtype=torch.long, device=device)
    else:
        pos = torch.as_tensor(atom_df[["x", "y", "z"]].values, dtype=init_dtype).unsqueeze(1) # [N, 1, d]
        channel_weights = torch.ones((len(pos), 1)) # [N, 1]
        residue_elements = None
        max_channel = 1
        protein_feats = torch.as_tensor(list(map(amino_acids, protein_df['element'])), dtype=torch.long)
    # end = datetime.now()
    # print("Time Cost for Generating multichannel feature for ", item_name, " is: ", end - start)
    if len(ligand_df) != 0:
        ligand_feats = torch.as_tensor(list(map(element_mapping, ligand_df['element'])), dtype=torch.long, device=device)
        ligand_feats += torch.max(protein_feats) + 1
        node_feats = torch.cat([protein_feats, ligand_feats], dim=-1)
    else:
        node_feats = protein_feats # Node Features 用于给Node初始化，和下述的atom type不一样
    # Onehot coding, 21 amino_acids + 10 atoms
    node_feats = node_feats.new_zeros((len(node_feats), 31)).scatter_(1, node_feats.unsqueeze(1), 1)
    residues = torch.as_tensor(list(map(amino_acids, protein_seq)), dtype=torch.long, device=device)
    # Three types of edges
    # 为什么 min_distance 是 4.5，不是一个整数？
    knn = KNNEdge(k=5, min_distance=4.5, max_channel=max_channel)
    # spatial = SpatialEdge(radius=cutoff, min_distance=4.5, max_channel=max_channel)
    sequential = SequentialEdge(max_distance=2)
    edge_layers = [knn, sequential]

    if len(node_feats) != len(pos):
        print(item_name)
        print(len(node_feats), len(pos))
        print(atom_df)
        print(ligand_df)
        print(protein_df)
        raise ValueError
    graph = MyData(
        x=node_feats,
        residue_elements=residue_elements,
        num_nodes=len(node_feats),
        pos=pos.to(torch.get_default_dtype()),
        channel_weights=channel_weights.long(),
        num_residues = len(residues),
        edge_feature=None,
        chain=chain,
    )
    edge_list = []
    num_edges = []
    num_relations = []
    # start = datetime.now()
    for layer in edge_layers:
        edges, num_relation = layer(graph)
        edge_list.append(edges)
        num_edges.append(len(edges))
        num_relations.append(num_relation)
    # end = datetime.now()
    # print("Time Cost for generating edge for", item_name, " is: ", end - start)
    edge_list = torch.cat(edge_list)
    num_edges = torch.as_tensor(num_edges, device=device)
    num_relations = torch.as_tensor(num_relations, device=device)
    num_relation = num_relations.sum()
    offsets = (num_relations.cumsum(0) - num_relations).repeat_interleave(num_edges)
    edge_list[:, 2] += offsets
    # 这里把edge_index和edge_relation分开装，是为了能够让处理batch的时候不要把relation也加上offset
    graph.edge_index = edge_list[:, :2].t() # [2, |E|]
    graph.edge_relations = edge_list[:, 2]
    graph.edge_weights = torch.ones(len(edge_list))
    graph.num_relation = num_relation    
    return graph
