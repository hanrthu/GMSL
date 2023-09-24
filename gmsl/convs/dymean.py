# This script defines an Heterogeneous Multitask Equivariant Network(HeMENet) deal with protein related tasks.
import torch
from torch import nn
from torch_scatter import scatter_add, scatter_mean
from utils import singleton, unsorted_segment_sum, unsorted_segment_mean

@singleton
class RollerPooling(nn.Module):
    '''
    Adaptive average pooling for the adaptive scaler
    '''
    def __init__(self, n_channel, device, dtype) -> None:
        super().__init__()
        self.n_channel = n_channel
        with torch.no_grad():
            pool_matrix = []
            ones = torch.ones((n_channel, n_channel), dtype=dtype, device=device)
            for i in range(n_channel):
                # i start from 0 instead of 1 !!! (less readable but higher implemetation efficiency)
                window_size = n_channel - i
                mat = torch.triu(ones) - torch.triu(ones, diagonal=window_size)
                pool_matrix.append(mat / window_size)
            self.pool_matrix = torch.stack(pool_matrix)
    
    def forward(self, hidden, target_size):
        '''
        :param hidden: [n_edges, n_channel]
        :param target_size: [n_edges]
        '''
        # pool_mat = self.pool_matrix.to(hidden.device).type(hidden.dtype) # [n_channel, n_channel]
        pool_mat = self.pool_matrix[target_size - 1]  # [n_edges, n_channel, n_channel]
        hidden = hidden.unsqueeze(-1)  # [n_edges, n_channel, 1]
        return torch.bmm(pool_mat, hidden)  # [n_edges, n_channel, 1]


# Adaptive multichannel equivariant graph convolutional layer
class AM_EGCL(nn.Module):
    eps = 1e-3

    def __init__(self, input_dim, output_dim, hidden_dim, channel_dim, channel_nf, coords_agg = 'mean',
                edge_input_dim=0, attention=False, activation=nn.SiLU(), dropout = 0.1):
        super(AM_EGCL, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.edge_input_dim = edge_input_dim
        self.attention = attention
        input_edge = input_dim * 2
        self.coords_agg = coords_agg
        self.activation = activation
        self.radial_linear = nn.Linear(channel_nf ** 2, channel_nf)
        self.phi_h = nn.Linear(hidden_dim, output_dim)
        # MLP Phi_m for scalar message 
        self.message_mlp = nn.Sequential(
            nn.Linear(input_edge + channel_nf, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, hidden_dim),
            self.activation
            )
        if self.attention:
            self.att_mlp = nn.Sequential(
                    nn.Linear(hidden_dim, 1),
                    nn.Sigmoid()
                )
        self.dropout = nn.Dropout(dropout)
        # MLP Phi_x for coordinate message
        layer = nn.Linear(hidden_dim, channel_dim, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            self.activation,
            layer
        )
    def generate_radial_feature(self, coords, edge_list, channel_attr, channel_weights: torch.Tensor):
        '''
            Generate radial feature, including scalar and coordinate feature.
            node_s means source node, and node_t means target_node.
            edge_list: [|E|, 2], source, target
            coords: [N, n_channel, d]
            channel_attr: [N, n_channel, channel_nf]
            channel_weights: [N, n_channel]
            edge_attr: [|E|, edge_dim]
            node_attr: [N, node_feature]
        '''
        source, target = edge_list[:, 0], edge_list[:, 1] # j, i
        channel_weights = channel_weights.type_as(coords)
        coord_msg = torch.norm(coords[target][:, :, None, :] - coords[source][:, None, :, :], dim=-1, keepdim=False) # [|E|, n_channel, n_channel]
        coord_msg = coord_msg / (torch.max(coord_msg) + self.eps)
        coord_msg = coord_msg * torch.bmm(channel_weights[target][:, :, None], channel_weights[source][:, None, :]) # [|E|, n_channel, n_channel]
        radial = torch.bmm(channel_attr[target].transpose(-1, -2), coord_msg)
        radial = torch.bmm(radial, channel_attr[source]) # [|E|, channel_nf, channel_nf]
        radial = radial.reshape(radial.shape[0], -1) # [|E|, channel_nf * channel_nf]
        radial_norm = torch.norm(radial, dim=-1, keepdim=True) + self.eps
        # x_tmp = self.radial_linear(radial)
        # norm_tmp = radial_norm
        if torch.isinf(radial).any() or torch.isnan(radial).any():
            print("Here is nan!")
        radial = self.radial_linear(radial / radial_norm) #[|E|, channel_nf]
        
        channel_mask = (channel_weights != 0) # [N, n_channel]
        channel_sum = channel_mask.sum(-1) # [N]
        pooled_col_coord = (coords[source] * channel_mask[source][:, :, None]).sum(1) # [|E|, d]
        pooled_col_coord = pooled_col_coord / (channel_sum[source][:, None] + self.eps)
        coord_diff = coords[target] - pooled_col_coord[:, None, :] # [|E|, n_channel, d]

        return radial, coord_diff
        
    
    def message(self, h, edge_list, coords, channel_attr, channel_weights, edge_attr=None, node_attr=None):
        """
            This is the heterogeneous graph message calculation function
            h: input feature, [N, input_dim]
            edge_list: [|E|, 3], source, target
            coords: [N, n_channel, d]
            channel_attr: [N, n_channel, channel_nf]
            channel_weights: [N, n_channel]
            edge_attr: [|E|, edge_dim]
            node_attr: [N, node_feature]
        """
        radial, coord_diff= self.generate_radial_feature(coords, edge_list, channel_attr, channel_weights)
        # Calculate scalar message mij
        if torch.isnan(radial).any():
            print("Radial Nan")
        if torch.isnan(coord_diff).any():
            print("Coord Nan")
        source, target = edge_list[:, 0], edge_list[:, 1] # j, i
        # max_target = torch.max(target)
        # max_source = torch.max(source)
        # max_nodes = h.shape[0]
        node_s = h[source] 
        node_t = h[target]
        node_message = self.message_mlp(torch.cat([node_t, node_s, radial], dim=-1))
        node_message = self.dropout(node_message) # [|E|, hidden_dim]
        if self.attention:
            node_message = node_message * self.att_mlp(node_message)
        # Calculate coordinate message xij
        n_channel = coords.shape[1]
        edge_feat = self.coord_mlp(node_message) # [|E|, n_channel]
        channel_sum = (channel_weights != 0).sum(-1)
        
        pooled_edge_feat = RollerPooling(n_channel, edge_feat.device, edge_feat.dtype)(edge_feat, channel_sum[target])
        coord_message = coord_diff * pooled_edge_feat # [n_edge, n_channel, d]
        return node_message, coord_message
    
    def aggregate(self, node_message, coord_message, edge_list, edge_weights, num_nodes):
        '''
            This is the heterogeneous graph message aggregation function
            node_message: [|E|, hidden_dim]
            coord_message: [|E|, n_channel, d]
            edge_list: [|E|, 3], source, target
            edge_weights: [|E|]
            num_nodes: int
        '''
        _, target = edge_list[:, 0], edge_list[:, 1] # j, i

        # Calculate scalar aggregation
        node_out = target
        #在这里edgeweight全是1
        # print("Graph Edge Weights:", graph.edge_weights.shape)
        edge_weight = edge_weights.unsqueeze(-1)
        # print("Num Node:", graph.num_nodes, len(graph.x))
        # 此处暂时存疑，可能后续改成scatter_mean
        update = scatter_add(node_message * edge_weight, node_out, dim=0, dim_size=num_nodes)
        node_agg = update.view(num_nodes, self.hidden_dim)
        # Calculate coordinate aggregation
        if self.coords_agg == 'sum':
            coord_agg = unsorted_segment_sum(coord_message, target, num_segments=num_nodes)
        elif self.coords_agg == 'mean':
            coord_agg = unsorted_segment_mean(coord_message, target, num_segments=num_nodes)
        else:
            raise Exception('Please choose the correct aggregation method!')
        
        return node_agg, coord_agg 
    
    def combine(self, h, coords, node_agg, coord_agg):
        '''
            This is the heterogeneous graph message update function
            h: input feature, [N, input_dim]
            coords: [N, n_channel, d]
            node_agg: [num_nodes, hidden_dim]
            coord_agg: [num_nodes, n_channel, d]
        '''
        node_output = self.phi_h(node_agg)
        node_output = h + node_output
        coord_output = coords + coord_agg
        return node_output, coord_output
    
    def forward(self, h, edge_list, coords, channel_attr, channel_weights, 
                edge_weights, edge_attr=None, node_attr=None):
        '''
            h: input feature, [N, input_dim]
            edge_list: [|E|, 3], source, target
            coords: [N, n_channel, d]
            channel_attr: [N, n_channel, channel_nf]
            channel_weights: [N, n_channel]
            edge_weights: [|E|]
            edge_attr: [|E|, edge_dim]
            node_attr: [N, node_feature]
        '''
        num_nodes = h.shape[0]
        node_message, coord_message = self.message(h, edge_list, coords, channel_attr, channel_weights, edge_attr, node_attr)
        if torch.isnan(node_message).any() or torch.isnan(coord_message).any():
            print("Wrong output")
        node_agg, coord_agg = self.aggregate(node_message, coord_message, edge_list, edge_weights, num_nodes)
        if torch.isnan(node_agg).any() or torch.isnan(coord_agg).any():
            print("Wrong output")
        node_output, coord_output = self.combine(h, coords, node_agg, coord_agg)
        if torch.isnan(node_output).any() or torch.isnan(coord_output).any():
            print("Wrong output")
        return node_output, coord_output