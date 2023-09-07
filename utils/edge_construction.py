import einops
import torch
from torch_cluster import knn_graph, radius_graph

from utils.utils import MyData

# @torch.no_grad()
def block_knn_graph(pos: torch.Tensor, channel_weights: torch.Tensor, k: int, eps: float = 1e-10, max_channel: int = 1):
    """
        pos: [N, n_channel, d]
        channel_weights: [N, n_channel]
        k: int
    """
    if max_channel == 1:
        pos = pos.squeeze()
        edge_list = knn_graph(pos, k=k).t()
        node_in, node_out = edge_list.t()[:2]
        mask = (pos[node_in] - pos[node_out]).norm(dim=-1) < eps
        edge_list = edge_list[~mask]
        return edge_list
    else:
        # Implementation of block knn edge construction
        X_dist = (pos[:, None, :, None] - pos[None, :, None, :]).pow(2).sum(dim=-1)
        pos_pad = channel_weights == 0
        pos_pad = pos_pad[:, None, :, None] | pos_pad[None, :, None, :]
        X_dist[pos_pad] = torch.inf
        X_dist = einops.reduce(X_dist, '... n1 n2 -> ...', 'min')
        X_dist.fill_diagonal_(torch.inf)
        k = min(k, len(pos))
        src = einops.repeat(torch.arange(0, len(pos), device=pos.device), 'N -> (N k)', k=k)
        dst = X_dist.topk(k, dim=-1, largest=False)[1].flatten() # [N * k]
        edge_list = torch.stack([src, dst], dim=-1) # [N * k, 2]
        edge_list = edge_list[X_dist[src, dst] >= eps]
        return edge_list # [|Ek|, 2]

@torch.no_grad()
def block_radius_graph(pos: torch.Tensor, channel_weights: torch.Tensor, r: float, max_num_neighbors: int, eps: float = 1e-10, max_channel=1):
    device = pos.device
    if max_channel == 1:
        pos = pos.squeeze()
        edge_list = radius_graph(pos, r=r, max_num_neighbors=max_num_neighbors).t()
        node_in, node_out = edge_list.t()[:2]
        mask = (pos[node_in] - pos[node_out]).norm(dim=-1) < eps
        edge_list = edge_list[~mask]
        return edge_list
    else:
        # Implementation of block knn edge construction
        BIGINT = 1e10
        channel_weights = channel_weights
        node_idx = torch.arange(0, len(pos), device=device)
        row = node_idx.unsqueeze(-1).repeat(1, len(pos)).flatten()
        col = node_idx.repeat(len(pos))
        # print("After Repeat:", row, col)
        X_diff = pos[row].unsqueeze(2) - pos[col].unsqueeze(1) # [|E|, n_channel, n_channel, 3]
        X_dist = torch.norm(X_diff, dim=-1, keepdim=False) # [|E|, n_channel, n_channel]
        source_pad = channel_weights[row] == 0
        target_pad = channel_weights[col] == 0
        pos_pad = torch.logical_or(source_pad.unsqueeze(2), target_pad.unsqueeze(1)) # [|E|, n_channel, n_channel]
        X_dist = X_dist + pos_pad * BIGINT
        del pos_pad
        X_dist = torch.min(X_dist.reshape(X_dist.shape[0], -1), dim=1)[0] # [|E|]

        dist_mat = torch.ones(len(pos), len(pos), dtype=X_dist.dtype, device=pos.device) * BIGINT
        dist_mat[(row, col)] = X_dist
        del X_dist
        # dist_neighbors, dst = torch.topk(dist_mat, k, dim=-1, largest=False) # [N, topk]
        dist_mat_flat = dist_mat.flatten()
        is_valid = dist_mat_flat <= r
        src = row.masked_select(is_valid)
        dst = col.masked_select(is_valid)
        edge_list = torch.stack([src, dst]).t()
        mask = dist_mat[src, dst] < eps
        edge_list = edge_list[~mask]
        return edge_list # [|Er|, 2]

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

    def __call__(self, graph: MyData):
        """
        Return KNN edges constructed from the input graph.

        Parameters:
            graph (Graph): :math:`n` graph(s)

        Returns:
            (Tensor, int): edge list of shape :math:`(|E|, 3)`, number of relations
        """
        edge_list = block_knn_graph(graph.pos, graph.channel_weights, k=self.k, eps=self.eps, max_channel=self.max_channel)
        relation = edge_list.new_zeros((len(edge_list), 1))
        edge_list = torch.cat([edge_list, relation], dim=-1)
        # atom2residue = torch.arange(0, len(graph.pos))
        # 这是说在氨基酸序列里不能距离太近
        if self.min_distance > 0:
            node_in, node_out = edge_list.t()[:2]
            mask = (node_in - node_out).abs() >= self.min_distance
            edge_list = edge_list[mask]
        # 也不能太远
        if self.max_distance > 0:
            node_in, node_out = edge_list.t()[:2]
            mask = (node_in - node_out).abs() <= self.max_distance
            edge_list = edge_list[mask]
        return edge_list, 1

class SpatialEdge(object):
    """
    Construct edges between nodes within a specified radius.

    Parameters:
        radius (float, optional): spatial radius
        min_distance (int, optional): minimum distance between the residues of two nodes
    """

    eps = 1e-10

    def __init__(self, radius: float = 5, min_distance: int = 5, max_num_neighbors: int = 32, max_channel: int = 1):
        super(SpatialEdge, self).__init__()
        self.radius = radius
        self.min_distance = min_distance
        self.max_num_neighbors = max_num_neighbors
        self.max_channel = max_channel

    def __call__(self, graph: MyData):
        """
        Return spatial radius edges constructed based on the input graph.

        Parameters:
            graph (Graph): :math:`n` graph(s)

        Returns:
            (Tensor, int): edge list of shape :math:`(|E|, 3)`, number of relations
        """
        edge_list = block_radius_graph(graph.pos, graph.channel_weights, r=self.radius, max_num_neighbors=self.max_num_neighbors,
                                       eps=self.eps, max_channel=self.max_channel)
        relation = edge_list.new_zeros((len(edge_list), 1))
        edge_list = torch.cat([edge_list, relation], dim=-1)
        # atom2residue = torch.arange(0, len(graph.pos))
        if self.min_distance > 0:
            node_in, node_out = edge_list.t()[:2]
            # 这是说在氨基酸序列里不能距离太近
            mask = (node_in - node_out).abs() < self.min_distance
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

    def __call__(self, graph: MyData):
        """
        Return sequential edges constructed based on the input graph.
        Edge types are defined by the relative distance between two residues in the sequence

        Parameters:
            graph (Graph): :math:`n` graph(s)

        Returns:
            (Tensor, int): edge list of shape :math:`(|E|, 3)`, number of relations
        """
        #注：相比于原版，这里省了很多东西，主要是因为Gearnet只用到了alpha碳，如果要用到其他原子，之后还需要根据原版进一步拓展
        #注 0815：好像无所谓，只需要在氨基酸层级表示序列就行
        edge_list = []
        chain = graph.chain
        for i in range(-self.max_distance, self.max_distance + 1):
            if i > 0:
                slice_in = slice(-i)
                slice_out = slice(i, None)
            elif i == 0:
                slice_in = slice_out = slice(None)
            else:
                slice_in = slice(-i, None)
                slice_out = slice(i)
            # exclude cross-chain edges
            is_same_chain = chain[slice_in] == chain[slice_out]
            node_all = torch.arange(len(chain), device=graph.pos.device)
            node_in = node_all[slice_in][is_same_chain]
            node_out = node_all[slice_out][is_same_chain]
            relation = torch.full_like(node_in, i + self.max_distance)
            edges = torch.stack([node_in, node_out, relation], dim=-1)
            edge_list.append(edges)

        edge_list = torch.cat(edge_list)

        return edge_list, 2 * self.max_distance + 1
