import torch
from torch import nn
from torch_cluster import knn_graph, radius_graph

from torchdrug.layers import functional
def block_knn_graph(pos, channel_weights, k, max_channel=1):
    if max_channel == 1:
        pos = pos.squeeze()
        return knn_graph(pos, k=k)
    else:
        # Implementation of block knn edge construction
        BIGINT = 1e10
        node_idx = torch.arange(0, len(pos))
        row = node_idx.unsqueeze(-1).repeat(1, len(pos)).flatten()
        col = node_idx.repeat(len(pos))
        # print("After Repeat:", row, col)
        X_diff = pos[row].unsqueeze(2) - pos[col].unsqueeze(1) # [|E|, n_channel, n_channel, 3]
        X_dist = torch.norm(X_diff, dim=-1, keepdim=False) # [|E|, n_channel, n_channel]
        source_pad = channel_weights[row] == 0
        target_pad = channel_weights[col] == 0
        pos_pad = torch.logical_or(source_pad.unsqueeze(2) - target_pad.unsqueeze(1)) # [|E|, n_channel, n_channel]
        X_dist = X_dist + pos_pad * BIGINT
        del pos_pad
        X_dist = torch.min(X_dist.reshape(X_dist.shape[0], -1), dim=1)[0] # [|E|]

        dist_mat = torch.ones(len(pos), len(pos), dtype=X_dist.dtype) * BIGINT
        dist_mat[(row, col)] = X_dist
        del X_dist
        dist_neighbors, dst = torch.topk(dist_mat, k, dim=-1, largest=False) # [N, topk]
        src = torch.arange(0, len(pos), device=dst.device).unsqueeze(-1).repeat(1, k)
        src, dst = src.flatten(), dst.flatten()
        dist_neighbors = dist_neighbors.flatten()
        is_valid = dist_neighbors < BIGINT
        src = src.masked_select(is_valid)
        dst = dst.masked_select(is_valid)
        edge_list = torch.stack([src, dst])
        return edge_list

def block_radius_graph(pos, channel_weights, r, max_num_neighbors, max_channel=1):
    if max_channel == 1:
        pos = pos.squeeze()
        return radius_graph(pos, r=r, max_num_neighbors=max_num_neighbors)
    else:
        # Implementation of block knn edge construction
        BIGINT = 1e10
        node_idx = torch.arange(0, len(pos))
        row = node_idx.unsqueeze(-1).repeat(1, len(pos)).flatten()
        col = node_idx.repeat(len(pos))
        # print("After Repeat:", row, col)
        X_diff = pos[row].unsqueeze(2) - pos[col].unsqueeze(1) # [|E|, n_channel, n_channel, 3]
        X_dist = torch.norm(X_diff, dim=-1, keepdim=False) # [|E|, n_channel, n_channel]
        source_pad = channel_weights[row] == 0
        target_pad = channel_weights[col] == 0
        pos_pad = torch.logical_or(source_pad.unsqueeze(2) - target_pad.unsqueeze(1)) # [|E|, n_channel, n_channel]
        X_dist = X_dist + pos_pad * BIGINT
        del pos_pad
        X_dist = torch.min(X_dist.reshape(X_dist.shape[0], -1), dim=1)[0] # [|E|]

        dist_mat = torch.ones(len(pos), len(pos), dtype=X_dist.dtype) * BIGINT
        dist_mat[(row, col)] = X_dist
        del X_dist
        # dist_neighbors, dst = torch.topk(dist_mat, k, dim=-1, largest=False) # [N, topk]
        dist_mat = dist_mat.flatten()
        is_valid = dist_mat <= r
        src = row.masked_select(is_valid)
        dst = col.masked_select(is_valid)
        edge_list = torch.stack([src, dst])
        return edge_list

class KNNEdge(object):
    """
    Construct edges between each node and its nearest neighbors.

    Parameters:
        k (int, optional): number of neighbors
        min_distance (int, optional): minimum distance between the residues of two nodes
    """

    eps = 1e-10

    def __init__(self, k=10, min_distance=5, max_distance=0, max_channel=1):
        super(KNNEdge, self).__init__()
        self.k = k
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.max_channel = max_channel
    def __call__(self, graph):
        """
        Return KNN edges constructed from the input graph.

        Parameters:
            graph (Graph): :math:`n` graph(s)

        Returns:
            (Tensor, int): edge list of shape :math:`(|E|, 3)`, number of relations
        """
        edge_list = block_knn_graph(graph.pos, graph.channel_weights, k=self.k, max_channel=self.max_channel).t()
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

    def __init__(self, radius=5, min_distance=5, max_num_neighbors=32, max_channel=1):
        super(SpatialEdge, self).__init__()
        self.radius = radius
        self.min_distance = min_distance
        self.max_num_neighbors = max_num_neighbors
        self.max_channel = max_channel

    def __call__(self, graph):
        """
        Return spatial radius edges constructed based on the input graph.

        Parameters:
            graph (Graph): :math:`n` graph(s)

        Returns:
            (Tensor, int): edge list of shape :math:`(|E|, 3)`, number of relations
        """
        edge_list = block_radius_graph(graph.pos, graph.channel_weights, r=self.radius, max_num_neighbors=self.max_num_neighbors, max_channel=self.max_channel).t()
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
        #注0815：好像无所谓，只需要在氨基酸层级表示序列就行
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
