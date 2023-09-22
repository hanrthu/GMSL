from typing import Any, List

import easydict
import jinja2
import math
import pandas as pd
import torch
import torch_cluster
from torch_geometric.data import Data
from torch_geometric.typing import OptTensor, SparseTensor
import yaml

# https://github.com/drorlab/gvp-pytorch/blob/main/gvp/atom3d.py
NUM_ATOM_TYPES = 10
# 这个地方需要改一下编号，尽可能把多的加进去
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

# 这里可以按照bond来，其实也可以按照距离来
bond_dict = lambda x: {
    'C-C': 0,
    'C-O': 1, 'O-C': 1,
    'C-N': 2, 'N-C': 2,
    'C-H': 3, 'H-C': 3,
    'C-S': 4, 'S-C': 4,
    'N-H': 5, 'H-N': 5,
    'N-O': 6, 'O-N': 6,
    'N-S': 7, 'S-N': 7,
    'N-N': 8, 
    'O-H': 9, 'H-O': 9,
    'O-S': 10,'S-O': 10,
    'O-O': 11,
    'S-H': 12, 'H-S': 12,
    'H-H': 13,
    # 's1': 14, 
    # 's2': 15,
    # 'ss': 16
}.get(x, 14)

affinity_num_dict = lambda x :{
    'lba': [1],
    'ppi': [1],
    'multi': [1, 1]
}.get(x, [])

class_num_dict = lambda x : {
    'ec': [538],
    'mf': [490],
    'bp': [1944],
    'cc': [321],
    'go': [490, 1944, 321],
    'multi': [538, 490, 1944, 321]
}.get(x, [])

def singleton(cls):
    _instance = {}

    def inner(*args, **kwargs):
        if cls not in _instance:
            _instance[cls] = cls(*args, **kwargs)
        return _instance[cls]
    return inner

def load_config(cfg_file, context=None):
    with open(cfg_file, "r") as fin:
        raw = fin.read()
    template = jinja2.Template(raw)
    instance = template.render(context)
    cfg = yaml.safe_load(instance)
    cfg = easydict.EasyDict(cfg)
    return cfg

# Modified torch_geometric.data.data for better representation ability
class MyData(Data):
    edge_relations: torch.Tensor

    chains: torch.Tensor
    channel_weights: torch.Tensor
    residue_elements: torch.Tensor

    affinities: torch.Tensor
    functions: torch.Tensor

    @property
    def num_knn_relations(self) -> int:
        return (self.edge_relations == 0).sum()

    @property
    def num_sequential_relations(self):
        return self.edge_relations.shape[0] - self.num_knn_relations

    def __init__(
        self, x: OptTensor = None, edge_index: OptTensor = None,
                edge_attr: OptTensor = None, y: OptTensor = None,
                pos: OptTensor = None, **kwargs
    ):
        super(MyData, self).__init__(x, edge_index, edge_attr, y, pos, **kwargs)

    def __inc__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if 'batch' in key:
            return int(value.max()) + 1
        elif 'index' in key or 'face' in key:
            return self.num_nodes
        elif 'interaction' in key:
            return self.num_edges
        elif 'chains' in key:
            return self.num_chains
        else:
            return 0
    def __cat_dim__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if isinstance(value, SparseTensor) and 'adj' in key:
            return (0, 1)
        elif 'index' in key or 'face' in key or 'interaction' in key:
            return -1
        else:
            return 0
    @property
    def num_chains(self):
        return len(torch.unique(self.chains))


def generate_line_graph(pos: torch.Tensor, elements: List[str], edge_index: torch.Tensor, super_idx: List[int]):
    source_dict = {}
    new_pos = []
    new_edge_index = []
    new_node_features = []
    #为新的“边节点”编号
    idx_dict = {}
    #将原有的普通边转换成节点, 并且构造一个source-to-target字典
    current_idx = 0
    for edge in edge_index.transpose(0,1):
        source, target = edge
        source = source.item()
        target = target.item()
        # 这里的转换不包括super node
        if source not in super_idx and target not in super_idx:
            idx = str(source) + '_' + str(target)
            source_element = elements[source]
            target_element = elements[target]
            # print("Take a look of bonds:", source_element + '-' + target_element)
            new_node_features.append(bond_dict(source_element + '-' + target_element))

            idx_dict[idx] = current_idx
            current_idx += 1
            new_postion = (pos[edge[0]] + pos[edge[1]]) / 2
            new_pos.append(new_postion)
        # 这里完整记录所有边
        if source not in source_dict:
            source_dict[source] = [target] 
        else:
            source_dict[source].append(target)
    # 给两个super node添加规定的位置，并添加一个新的超级点
    new_pos = new_pos + [pos[super_idx[0]], pos[super_idx[1]], (pos[super_idx[0]] + pos[super_idx[1]]) / 2]
    # 需要给他们编下号
    idx_dict['s1'] = current_idx
    idx_dict['s2'] = current_idx + 1
    idx_dict['ss'] = current_idx + 2
    new_node_features.extend([bond_dict('s1'), bond_dict('s2'), bond_dict('ss')])
    # print("New node features:", new_node_features)
    # 开始给新的“边节点”连线
    # print("Idx dict:", idx_dict)
    # print("Source dict:", source_dict)
    for edge in edge_index.transpose(0,1):
        source, target = edge
        source = source.item()
        target = target.item()
        if target in source_dict:
            next_targets = source_dict[target]
        else:
            continue
        
        if source in super_idx and target in super_idx:
            if source == super_idx[0]:
                new_edge_index.append([idx_dict['s1'], idx_dict['ss']])
                new_edge_index.append([idx_dict['ss'], idx_dict['s2']])
            else:
                new_edge_index.append([idx_dict['ss'], idx_dict['s1']])
                new_edge_index.append([idx_dict['s2'], idx_dict['ss']])
        elif source in super_idx:
            for next_target in next_targets:
                if target != source:
                    if source == super_idx[0]:
                        source_idx = idx_dict['s1']
                    else:
                        source_idx = idx_dict['s2']
                    if next_target not in super_idx:
                        target_idx = idx_dict[str(target) + '_' + str(next_target)]
                    else:
                        continue
                new_edge_index.append([source_idx, target_idx])
        elif target in super_idx:
            continue
        else: 
            for next_target in next_targets:
                source_idx = idx_dict[str(source) + '_' + str(target)]
                if next_target not in super_idx:
                    target_idx = idx_dict[str(target) + '_' + str(next_target)]
                elif next_target == super_idx[0]:
                    target_idx = idx_dict['s1']
                elif next_target == super_idx[1]:
                    target_idx = idx_dict['s2']
                new_edge_index.append([source_idx, target_idx])
    # 添加完了，打印出来看看
    # new_pos = torch.cat(new_pos, dim=0)
    # new_edge_index = torch.cat(new_edge_index, dim=0)
    # new_pos = torch.tensor(new_pos)
    # new_edge_index = torch.tenspr(new_edge_index)
    node_feats = torch.as_tensor(new_node_features)
    new_pos = torch.stack(new_pos, dim=0)
    new_edge_index = torch.tensor(new_edge_index).transpose(0,1)
    dist = torch.pow(new_pos[new_edge_index[0]] - new_pos[new_edge_index[1]], exponent=2).sum(-1).sqrt()
    graph = MyData(
        x=node_feats,
        pos=new_pos.to(torch.get_default_dtype()),
        edge_weights=dist.to(torch.get_default_dtype()),
        edge_index=new_edge_index,
    )
    # print("Printing...:", new_pos)
    # print("Printing...:", new_edge_index)
    # print(graph)
    return graph

def gen_edge_interactions(
    pos: torch.Tensor, elements: List[str], edge_index: torch.Tensor, super_idx: List[int]
):
    source_dict = {}
    edge_center = []
    new_edge_index = []
    new_edge_features = []
    alphas = []
    idx_dict = {}
    connection_nodes = []
    #记录边的编号
    current_idx = 0
    elements += ['s1','s2']
    for edge in edge_index.transpose(0,1):
        source, target = edge
        source = source.item()
        target = target.item()
        idx = str(source) + '_' + str(target)
        # print(source, target)
        source_element = elements[source]
        target_element = elements[target]
        # print("Take a look of bonds:", source_element + '-' + target_element)
        new_edge_features.append(bond_dict(source_element + '-' + target_element))

        idx_dict[idx] = current_idx
        current_idx += 1
        new_postion = (pos[edge[0]] + pos[edge[1]]) / 2
        edge_center.append(new_postion)
        # 这里完整记录所有边
        if source not in source_dict:
            source_dict[source] = [target] 
        else:
            source_dict[source].append(target)
        
    for edge in edge_index.transpose(0,1):
        source, target = edge
        source = source.item()
        target = target.item()
        if target in source_dict:
            next_targets = source_dict[target]
        else:
            continue
        # 此处复杂度实在太高，尤其是加了supernode以后，节点的度变多了O(mK)
        for next_target in next_targets:
            source_idx = idx_dict[str(source) + '_' + str(target)]
            target_idx = idx_dict[str(target) + '_' + str(next_target)]
            new_edge_index.append([source_idx, target_idx])
            e1 = pos[target] - pos[source]
            e2 = pos[next_target]-pos[target]
            cos_theta = torch.matmul(e1, e2) / (torch.pow(e1, 2).sum().sqrt() * torch.pow(e2, 2).sum().sqrt())
            cos_theta = torch.clamp(cos_theta,-1,1)
            alphas.append(math.acos(cos_theta))
            connection_nodes.append(target)
            
    edge_feats = torch.as_tensor(new_edge_features)
    edge_center = torch.stack(edge_center, dim=0)
    new_edge_index = torch.tensor(new_edge_index).transpose(0,1)
    alphas = torch.as_tensor(alphas)
    connection_nodes = torch.as_tensor(connection_nodes)
    # center_dist = torch.pow(new_pos[new_edge_index[0]] - new_pos[new_edge_index[1]], exponent=2).sum(-1).sqrt()
    return edge_feats, edge_center, new_edge_index, alphas, connection_nodes

def gen_edge_feats( 
    elements: List[str], edge_index: torch.Tensor, flag: torch.Tensor, pos: torch.Tensor
):
    new_edge_features = []
    elements += ['s1','s2']
    flag = torch.cat([flag, 2*torch.ones(2)])
    # print("Flag_idx:", torch.nonzero(flag))
    flag_idx = torch.nonzero(flag)[0,0]
    object_idx_1 = range(0, flag_idx)
    object_idx_2 = range(flag_idx, len(flag))
    external_edge_flag = []
    edge_dist = []
    for edge in edge_index.transpose(0,1):
        source, target = edge
        source = source.item()
        target = target.item()
        rel_pos = pos[source] - pos[target]
        dist = torch.pow(rel_pos, 2).sum(-1).sqrt()
        edge_dist.append(dist)
        # print("Pos_j:", pos[edge_index[0]])
        if (source in object_idx_1 and target in object_idx_2) or (source in object_idx_2 and target in object_idx_1):
            external_edge_flag.append(1)
        else:
            external_edge_flag.append(0)
        source_element = elements[source]
        target_element = elements[target]
        # print("Take a look of bonds:", source_element + '-' + target_element)
        new_edge_features.append(bond_dict(source_element + '-' + target_element))
    edge_feats = torch.as_tensor(new_edge_features)
    external_edge_flag = torch.as_tensor(external_edge_flag)
    edge_dist = torch.as_tensor(edge_dist)
    external_edge_dist = edge_dist[external_edge_flag==1]
    return edge_feats, external_edge_flag, external_edge_dist

def prot_graph_transform(
    atom_df: pd.DataFrame,
    cutoff: float = 4.5,
    feat_col: str = "element",
    max_num_neighbors: int = 32,
    init_dtype: torch.dtype = torch.float64,
    super_node: bool = False,
    offset_strategy : int = 0,
    flag: torch.Tensor = None, 
) -> MyData:
    # 判断此时的任务是单体交互还是多体交互
    pos = torch.as_tensor(atom_df[["x", "y", "z"]].values, dtype=init_dtype)
    # 找质量中心
    # element_weights = torch.as_tensor(list(map(weight_mapping, atom_df[feat_col])), dtype=torch.float32).unsqueeze(-1)
    # 几何中心
    element_weights = torch.ones(pos.shape[0]).unsqueeze(-1)
    # 0:不做区分，1：Binary区分，只区分蛋白的residue和氨基酸的residue, 2:对每一种氨基酸的原子都进行区分
    if offset_strategy == 0:
        res_type = torch.zeros(pos.shape[0], dtype=torch.long)
    elif offset_strategy == 1:
        # res_type = torch.as_tensor(list(map(amino_acids, atom_df['resname'])), dtype=torch.long)
        # res_type[res_type!=0] = 1
        res_type = torch.zeros(pos.shape[0], dtype=torch.long)
        for item in range(len(torch.unique(flag))):
            res_type[flag==item] = item
    elif offset_strategy == 2:
        res_type = torch.as_tensor(list(map(amino_acids, atom_df['resname'])), dtype=torch.long)
    else:
        raise NotImplementedError
    feature_offset = res_type * (element_mapping('nothing'))  # offset 为 vocabulary size

    edge_index = torch_cluster.radius_graph(
        pos, r=cutoff, loop=False, max_num_neighbors=max_num_neighbors
    )
    nodes_added = []
    super_idx = []

    if super_node:
        if len(torch.unique(flag)) == 1:
            # print("Adding One Super Node to the graph!")
            # print("Positions before adding supernode: ", pos.shape)
            # print("Edge Index: ", edge_index.shape)
            # print("Index example", edge_index)
            center = torch.sum(pos*element_weights, dim=0, keepdim=True) / torch.sum(element_weights)
            center_index = pos.shape[0]
            for i in range(pos.shape[0]):
                index_1 = torch.tensor(i, center_index).unsqueeze(dim=1)
                index_2 = torch.tensor(center_index, i).unsqueeze(dim=1)
                edge_index = torch.cat([edge_index, index_1, index_2], dim=1)
            pos = torch.cat([pos, torch.tensor(center)], dim=0)
            nodes_added = [element_mapping('Super')]
            super_idx = center_index
            # print("Positions after adding supernode: ", pos.shape)
            # print("New Edge Index: ", edge_index.shape)
        else:
            # print("Adding Two Super Nodes to the graph!")
            # print("Positions before adding supernode: ", pos.shape)
            # print("Edge Index: ", edge_index.shape)
            # print("Index example", edge_index)
            pos1 = pos[flag==0]
            pos_weight1 = element_weights[flag==0]
            pos2 = pos[flag==1]
            pos_weight2 = element_weights[flag==1]
            center1 = torch.sum(pos1*pos_weight1, dim=0, keepdim=True) / torch.sum(pos_weight1)
            center2 = torch.sum(pos2*pos_weight2, dim=0, keepdim=True) / torch.sum(pos_weight2)
            center1_index = pos.shape[0]
            center2_index = pos.shape[0] + 1
            # print(center1.shape, center2.shape)
            for i in range(pos.shape[0]):
                if flag[i] == 0:
                    index_1 = torch.tensor((i, center1_index)).unsqueeze(dim=1)
                    index_2 = torch.tensor((center1_index, i)).unsqueeze(dim=1)
                else:
                    index_1 = torch.tensor((i, center2_index)).unsqueeze(dim=1)
                    index_2 = torch.tensor((center2_index, i)).unsqueeze(dim=1)
                edge_index = torch.cat([edge_index, index_1, index_2], dim=1)
            pos = torch.cat([pos, center1, center2], dim=0)
            index_1 = torch.tensor((center1_index, center2_index)).unsqueeze(dim=1)
            index_2 = torch.tensor((center2_index, center1_index)).unsqueeze(dim=1)
            edge_index = torch.cat([edge_index, index_1, index_2], dim=1)
            super_idx = [center1_index, center2_index]
            # 这里是为了line graph临时整的
            # if True:

            #     graph = generate_line_graph(pos, list(atom_df[feat_col]), edge_index, super_idx)
            #     return graph
            # print("Positions after adding supernode: ", pos.shape)
            # print("New Edge Index: ", edge_index.shape)
            
            nodes_added = [element_mapping('Super'), element_mapping('Super')]
    dist = torch.pow(pos[edge_index[0]] - pos[edge_index[1]], exponent=2).sum(-1).sqrt()
    # 这个地方有点问题，没有对蛋白和小分子进行区分
    node_feats = torch.as_tensor(list(map(element_mapping, atom_df[feat_col])), dtype=torch.long)
    node_feats += feature_offset
    if nodes_added != []:
        node_feats = torch.cat([node_feats, torch.tensor(nodes_added)])
    # print("Offset shape",node_feats.shape, feature_offset.shape)
    # print("Node_features:", node_feats)
    # 先别这么麻烦了，先整个e
    # edge_feats, new_pos, new_edge_index, alphas, connection_nodes = gen_edge_interactions(pos, list(atom_df[feat_col]), edge_index, super_idx)
    # 这里可以生成edge features， 但为了简化先不用了
    # edge_feats, external_edge_flag, external_edge_dist = gen_edge_feats(list(atom_df[feat_col]), edge_index, flag=flag, pos=pos)
    # new_edge_index = torch.rand([2, 320358])
    # alphas = torch.rand([320358])
    # connection_nodes = torch.ones(320358)
    # new_pos = torch.rand([len(edge_feats), 3])
    graph = MyData(
        x=node_feats,
        pos=pos.to(torch.get_default_dtype()),
        edge_weights=dist.to(torch.get_default_dtype()),
        edge_index=edge_index,
        # e=edge_feats,
        # external_flag=external_edge_flag,
        # external_edge_dist=external_edge_dist.to(torch.get_default_dtype())
        # e_pos = new_pos.to(torch.get_default_dtype()),
        # e_interaction = new_edge_index,
        # alphas = alphas,
        # connection_nodes = connection_nodes
    )

    return graph

# For Gearnet LineGraph
def construct_edge_gearnet(graph, edge_list, num_relation):
    node_in, node_out, r = edge_list.t()
    residue_in, residue_out = graph.atom2residue[node_in], graph.atom2residue[node_out]
    in_residue_type = graph.residue_type[residue_in]
    out_residue_type = graph.residue_type[residue_out]
    sequential_dist = torch.abs(residue_in - residue_out)
    spatial_dist = (graph.pos[node_in] - graph.pos[node_out]).norm(dim=-1)

    return torch.cat([
        torch.zeros((len(in_residue_type), 31)).scatter_(1, in_residue_type.unsqueeze(1), 1),
        torch.zeros((len(out_residue_type), 31)).scatter_(1, in_residue_type.unsqueeze(1), 1),
        torch.zeros((len(r), num_relation)).scatter_(1, r.unsqueeze(1), 1),
        torch.zeros((len(sequential_dist.clamp(max=10)), 11)).scatter_(1, sequential_dist.clamp(max=10).unsqueeze(1), 1),
        spatial_dist.unsqueeze(-1)
    ], dim=-1)


