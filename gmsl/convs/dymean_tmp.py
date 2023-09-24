# This script defines an Heterogeneous Multitask Equivariant Network(HeMENet) deal with protein related tasks.
import torch
from torch import nn
from torch.nn import functional as F
from torch_scatter import scatter_add, scatter_mean
from collections.abc import Sequence
from utils import singleton, unsorted_segment_sum, unsorted_segment_mean
import math

class MultiLayerPerceptron(nn.Module):
    """
    Multi-layer Perceptron.
    Note there is no batch normalization, activation or dropout in the last layer.

    Parameters:
        input_dim (int): input dimension
        hidden_dim (list of int): hidden dimensions
        short_cut (bool, optional): use short cut or not
        batch_norm (bool, optional): apply batch normalization or not
        activation (str or function, optional): activation function
        dropout (float, optional): dropout rate
    """

    def __init__(self, input_dim, hidden_dims, short_cut=False, batch_norm=False, activation="relu", dropout=0):
        super(MultiLayerPerceptron, self).__init__()

        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        self.dims = [input_dim] + hidden_dims
        self.short_cut = short_cut

        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(nn.Linear(self.dims[i], self.dims[i + 1]))
        if batch_norm:
            self.batch_norms = nn.ModuleList()
            for i in range(len(self.dims) - 2):
                self.batch_norms.append(nn.BatchNorm1d(self.dims[i + 1]))
        else:
            self.batch_norms = None

    def forward(self, input):
        """"""
        layer_input = input

        for i, layer in enumerate(self.layers):
            hidden = layer(layer_input)
            if i < len(self.layers) - 1:
                if self.batch_norms:
                    x = hidden.flatten(0, -2)
                    hidden = self.batch_norms[i](x).view_as(hidden)
                hidden = self.activation(hidden)
                if self.dropout:
                    hidden = self.dropout(hidden)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            layer_input = hidden

        return hidden

# Three types of edge features
class IEConvLayer(nn.Module):
    eps = 1e-6

    def __init__(self, input_dim, hidden_dim, output_dim, edge_input_dim, kernel_hidden_dim=32,
                dropout=0.05, dropout_before_conv=0.2, activation="relu", aggregate_func="sum"):
        super(IEConvLayer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.edge_input_dim = edge_input_dim
        self.kernel_hidden_dim = kernel_hidden_dim
        self.aggregate_func = aggregate_func

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.kernel = MultiLayerPerceptron(edge_input_dim, [kernel_hidden_dim, (hidden_dim + 1) * hidden_dim])
        self.linear2 = nn.Linear(hidden_dim, output_dim)

        self.input_batch_norm = nn.BatchNorm1d(input_dim)
        self.message_batch_norm = nn.BatchNorm1d(hidden_dim)
        self.update_batch_norm = nn.BatchNorm1d(hidden_dim)
        self.output_batch_norm = nn.BatchNorm1d(output_dim)

        self.dropout = nn.Dropout(dropout)
        self.dropout_before_conv = nn.Dropout(dropout_before_conv)

        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

    def message(self, input, edge_input, edge_list):
        node_in = edge_list[:, 0]
        # 先1*1 conv映射一下feature
        message = self.linear1(input[node_in])
        # Batchnorm一下，让整个batch一致
        message = self.message_batch_norm(message)
        # Activation + Dropout
        message = self.dropout_before_conv(self.activation(message))
        # IEConv的kernel，用的是MLP，对edge feature进行处理
        kernel = self.kernel(edge_input).view(-1, self.hidden_dim + 1, self.hidden_dim)
        message = torch.einsum('ijk, ik->ij', kernel[:, 1:, :], message) + kernel[:, 0, :]
        return message
    
    def aggregate(self, message, edge_list, edge_weights, num_node):
        # 这里的Protein类似乎是source to target的形式
        _, node_out = edge_list.t()[:2]
        edge_weight = edge_weights.unsqueeze(-1)
        # node_out是入边
        if self.aggregate_func == "sum":
            update = scatter_add(message * edge_weight, node_out, dim=0, dim_size=num_node) 
        else:
            raise ValueError("Unknown aggregation function `%s`" % self.aggregate_func)
        return update

    def combine(self, input, update):
        output = self.linear2(update)
        return output

    def forward(self, input, edge_input, edge_list, edge_weights):
        input = self.input_batch_norm(input)
        layer_input = self.dropout(self.activation(input))
        num_node = input.shape[0]
        message = self.message(layer_input, edge_input, edge_list)
        update = self.aggregate(message, edge_list, edge_weights, num_node)
        update = self.dropout(self.activation(self.update_batch_norm(update)))
        
        output = self.combine(input, update)
        output = self.output_batch_norm(output)
        return output

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
                edge_input_dim=0, batch_norm=True, attention=False, activation=nn.SiLU(), dropout = 0.1):
        super(AM_EGCL, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.edge_input_dim = edge_input_dim
        self.attention = attention
        input_edge = input_dim * 2
        self.coords_agg = coords_agg
        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(output_dim)
        else:
            self.batch_norm = None
        # if isinstance(activation, str):
        #     self.activation = getattr(F, activation)
        # else:
        self.activation = activation
        self.radial_linear = nn.Linear(channel_nf ** 2, channel_nf)
        self.hetero_linear = nn.Linear(hidden_dim, output_dim)
        # MLP Phi_m for scalar message 
        self.message_mlp = nn.Sequential(
            nn.Linear(input_edge + channel_nf + edge_input_dim, hidden_dim),
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
            edge_list: [|E|, 3], source, target, relation
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
            edge_list: [|E|, 3], source, target, relation
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
            edge_list: [|E|, 3], source, target, relation
            edge_weights: [|E|]
            num_nodes: int
        '''
        # print("Graph Num Relation:", graph.num_relation)
        _, target = edge_list[:, 0], edge_list[:, 1] # j, i

        # assert num_relation[0] == self.num_relation
        # Calculate scalar aggregation
        #乘起来是为了把num_relation种边给分开，然后就可以异质图分别求sumup了
        node_out = target
        #在这里edgeweight全是1
        # print("Graph Edge Weights:", graph.edge_weights.shape)
        edge_weight = edge_weights.unsqueeze(-1)
        # print("Num Node:", graph.num_nodes, len(graph.x))
        # 此处暂时存疑，可能后续改成scatter_mean
        update = scatter_add(node_message * edge_weight, node_out, dim=0, dim_size=num_nodes)
        node_agg = update.view(num_nodes, self.num_relation * self.hidden_dim)
        # Calculate coordinate aggregation
        if self.coords_agg == 'sum':
            coord_agg = unsorted_segment_sum(self.w_r[relation].unsqueeze(-1) * coord_message, target, num_segments=num_nodes)
        elif self.coords_agg == 'mean':
            coord_agg = unsorted_segment_mean(self.w_r[relation].unsqueeze(-1) * coord_message, target, num_segments=num_nodes)
        else:
            raise Exception('Please choose the correct aggregation method!')
        
        return node_agg, coord_agg 
    
    def combine(self, h, coords, node_agg, coord_agg, relation):
        '''
            This is the heterogeneous graph message update function
            h: input feature, [N, input_dim]
            coords: [N, n_channel, d]
            node_agg: [num_nodes, num_relation * hidden_dim]
            coord_agg: [num_nodes, n_channel, d]
        '''
        node_output = self.hetero_linear(node_agg)
        if self.batch_norm:
            node_output = self.batch_norm(node_output)
        if self.activation:
            node_output = self.activation(node_output)
        node_output = h + node_output
        coord_output = coords + coord_agg
        return node_output, coord_output
    
    def forward(self, h, edge_list, coords, channel_attr, channel_weights, 
                edge_weights, edge_attr=None, node_attr=None):
        '''
            h: input feature, [N, input_dim]
            edge_list: [|E|, 3], source, target, relation
            coords: [N, n_channel, d]
            channel_attr: [N, n_channel, channel_nf]
            channel_weights: [N, n_channel]
            edge_weights: [|E|]
            edge_attr: [|E|, edge_dim]
            node_attr: [N, node_feature]
        '''
        num_nodes = h.shape[0]
        relation = edge_list[:, 2]
        node_message, coord_message = self.message(h, edge_list, coords, channel_attr, channel_weights, edge_attr, node_attr)
        if torch.isnan(node_message).any() or torch.isnan(coord_message).any():
            print("Wrong output")
        node_agg, coord_agg = self.aggregate(node_message, coord_message, edge_list, edge_weights, num_nodes)
        if torch.isnan(node_agg).any() or torch.isnan(coord_agg).any():
            print("Wrong output")
        node_output, coord_output = self.combine(h, coords, node_agg, coord_agg, relation)
        if torch.isnan(node_output).any() or torch.isnan(coord_output).any():
            print("Wrong output")
        return node_output, coord_output