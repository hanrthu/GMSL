import torch
from torch import nn
from torch_cluster import knn_graph, radius_graph

from torchdrug.layers import functional
class KNNEdge(object):
    """
    Construct edges between each node and its nearest neighbors.

    Parameters:
        k (int, optional): number of neighbors
        min_distance (int, optional): minimum distance between the residues of two nodes
    """

    eps = 1e-10

    def __init__(self, k=10, min_distance=5, max_distance=0):
        super(KNNEdge, self).__init__()
        self.k = k
        self.min_distance = min_distance
        self.max_distance = max_distance

    def __call__(self, graph):
        """
        Return KNN edges constructed from the input graph.

        Parameters:
            graph (Graph): :math:`n` graph(s)

        Returns:
            (Tensor, int): edge list of shape :math:`(|E|, 3)`, number of relations
        """
        edge_list = knn_graph(graph.pos, k=self.k).t()
        relation = torch.zeros(len(edge_list), 1, dtype=torch.long)
        edge_list = torch.cat([edge_list, relation], dim=-1)
        atom2residue = torch.as_tensor(range(len(graph.x)), dtype=torch.long)
        # 这是说在氨基酸序列里不能距离太近
        if self.min_distance > 0:
            node_in, node_out = edge_list.t()[:2]
            mask = (atom2residue[node_in] - atom2residue[node_out]).abs() < self.min_distance
            edge_list = edge_list[~mask]
        # 也不能太远
        if self.max_distance > 0:
            node_in, node_out = edge_list.t()[:2]
            mask = (atom2residue[node_in] - atom2residue[node_out]).abs() > self.max_distance
            edge_list = edge_list[~mask]
            
        node_in, node_out = edge_list.t()[:2]
        mask = (graph.pos[node_in] - graph.pos[node_out]).norm(dim=-1) < self.eps
        edge_list = edge_list[~mask]

        return edge_list, 1

class SpatialEdge(object):
    """
    Construct edges between nodes within a specified radius.

    Parameters:
        radius (float, optional): spatial radius
        min_distance (int, optional): minimum distance between the residues of two nodes
    """

    eps = 1e-10

    def __init__(self, radius=5, min_distance=5, max_num_neighbors=32):
        super(SpatialEdge, self).__init__()
        self.radius = radius
        self.min_distance = min_distance
        self.max_num_neighbors = max_num_neighbors

    def __call__(self, graph):
        """
        Return spatial radius edges constructed based on the input graph.

        Parameters:
            graph (Graph): :math:`n` graph(s)

        Returns:
            (Tensor, int): edge list of shape :math:`(|E|, 3)`, number of relations
        """
        edge_list = radius_graph(graph.pos, r=self.radius, max_num_neighbors=self.max_num_neighbors).t()
        relation = torch.zeros(len(edge_list), 1, dtype=torch.long)
        edge_list = torch.cat([edge_list, relation], dim=-1)
        atom2residue = torch.as_tensor(range(len(graph.x)), dtype=torch.long)
        if self.min_distance > 0:
            node_in, node_out = edge_list.t()[:2]
            # 这是说在氨基酸序列里不能距离太近
            mask = (atom2residue[node_in] - atom2residue[node_out]).abs() < self.min_distance
            edge_list = edge_list[~mask]

        node_in, node_out = edge_list.t()[:2]
        mask = (graph.pos[node_in] - graph.pos[node_out]).norm(dim=-1) < self.eps
        edge_list = edge_list[~mask]

        return edge_list, 1


class SequentialEdge(object):
    """
    Construct edges between atoms within close residues.

    Parameters:
        max_distance (int, optional): maximum distance between two residues in the sequence
    """

    def __init__(self, max_distance=2):
        super(SequentialEdge, self).__init__()
        self.max_distance = max_distance

    def __call__(self, graph):
        """
        Return sequential edges constructed based on the input graph.
        Edge types are defined by the relative distance between two residues in the sequence

        Parameters:
            graph (Graph): :math:`n` graph(s)

        Returns:
            (Tensor, int): edge list of shape :math:`(|E|, 3)`, number of relations
        """
        #注：相比于原版，这里省了很多东西，主要是因为Gearnet只用到了alpha碳，如果要用到其他原子，之后还需要根据原版进一步拓展
        num_node = len(graph.x)
        num_residue = len(graph.x)
        atom2residue = range(len(graph.x))
        edge_list = []
        for i in range(-self.max_distance, self.max_distance + 1):
            node_index = torch.arange(num_node)
            residue_index = torch.arange(num_residue)
            if i > 0:
                node_in = atom2residue[:-i]
                node_out = atom2residue[i:]
            elif i == 0:
                node_in = atom2residue
                node_out = atom2residue
            else:
                node_in = atom2residue[-i:]
                node_out = atom2residue[:i]
            # exclude cross-chain edges
            # print("In:", node_in)
            # print("Out:", node_out)
            # print("Graph Chain:", graph.chain)
            # print("Nodes Chain:", graph.chain[node_in], graph.chain[node_out])
            # print("Len Chains:", len(graph.chain[node_in]), len(graph.chain[node_out]))
            is_same_chain = (graph.chain[node_in].reset_index(drop=True) == graph.chain[node_out].reset_index(drop=True))
            node_in, node_out = torch.as_tensor(node_in, dtype=torch.long), torch.as_tensor(node_out, dtype=torch.long)
            node_in = node_in[is_same_chain]
            node_out = node_out[is_same_chain]
            relation = torch.ones(len(node_in), dtype=torch.long) * (i + self.max_distance)
            edges = torch.stack([node_in, node_out, relation], dim=-1)
            edge_list.append(edges)

        edge_list = torch.cat(edge_list)

        return edge_list, 2 * self.max_distance + 1
