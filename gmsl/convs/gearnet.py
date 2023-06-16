import torch
from torch import nn
from torch.nn import functional as F
from torch_scatter import scatter_add
from collections.abc import Sequence
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

#这个是作者简化版的
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

    def message(self, graph, input, edge_input):
        node_in = graph.edge_list[:, 0]
        message = self.linear1(input[node_in])
        message = self.message_batch_norm(message)
        message = self.dropout_before_conv(self.activation(message))
        kernel = self.kernel(edge_input).view(-1, self.hidden_dim + 1, self.hidden_dim)
        message = torch.einsum('ijk, ik->ij', kernel[:, 1:, :], message) + kernel[:, 0, :]

        return message
    
    def aggregate(self, graph, message):
        # 这里的Protein类似乎是source to target的形式
        node_in, node_out = graph.edge_list.t()[:2]
        edge_weight = graph.edge_weight.unsqueeze(-1)
        # 我就假装node_out是入边吧
        if self.aggregate_func == "sum":
            update = scatter_add(message * edge_weight, node_out, dim=0, dim_size=graph.num_node) 
        else:
            raise ValueError("Unknown aggregation function `%s`" % self.aggregate_func)
        return update

    def combine(self, input, update):
        output = self.linear2(update)
        return output

    def forward(self, graph, input, edge_input):
        input = self.input_batch_norm(input)
        layer_input = self.dropout(self.activation(input))
        
        message = self.message(graph, layer_input, edge_input)
        update = self.aggregate(graph, message)
        update = self.dropout(self.activation(self.update_batch_norm(update)))
        
        output = self.combine(input, update)
        output = self.output_batch_norm(output)
        return output
    

class GeometricRelationalGraphConv(nn.Module):
    eps = 1e-6

    def __init__(self, input_dim, output_dim, num_relation, edge_input_dim=None, 
                batch_norm=False, activation="relu"):
        super(GeometricRelationalGraphConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_relation = num_relation
        self.edge_input_dim = edge_input_dim

        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(output_dim)
        else:
            self.batch_norm = None
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

        self.linear = nn.Linear(num_relation * input_dim, output_dim)
        if edge_input_dim:
            self.edge_linear = nn.Linear(edge_input_dim, input_dim)
        else:
            self.edge_linear = None

    def message(self, graph, input, edge_input=None):
        node_in = graph.edge_list[:, 0]
        message = input[node_in]
        if self.edge_linear:
            message += self.edge_linear(graph.edge_feature.float())
        if edge_input is not None:
            assert edge_input.shape == message.shape
            message += edge_input
        return message
    
    def aggregate(self, graph, message):
        #什么是edge_weight? 
        # print("Graph Num Relation:", graph.num_relation)
        assert graph.num_relation[0] == self.num_relation
        #乘起来是为了把num_relation种边给分开，然后就可以异质图分别求sumup了
        node_out = graph.edge_list[:, 1] * self.num_relation + graph.edge_list[:, 2]
        #在这里应该edgeweight全是1
        # print("Graph Edge Weights:", graph.edge_weights.shape)
        edge_weight = graph.edge_weights.unsqueeze(-1)
        # print("Num Node:", graph.num_nodes, len(graph.x))
        update = scatter_add(message * edge_weight, node_out, dim=0, dim_size=graph.num_nodes * self.num_relation)
        update = update.view(graph.num_nodes, self.num_relation * self.input_dim)

        return update
    
    def combine(self, input, update):
        output = self.linear(update)
        if self.batch_norm:
            output = self.batch_norm(output)
        if self.activation:
            output = self.activation(output)
        return output
    
    def forward(self, graph, input, edge_input=None):
        message = self.message(graph, input, edge_input)
        update = self.aggregate(graph, message)
        output = self.combine(input, update)
        return output
    
    
class SpatialLineGraph(nn.Module):
    """
    Spatial line graph construction module from `Protein Representation Learning by Geometric Structure Pretraining`_.

    .. _Protein Representation Learning by Geometric Structure Pretraining:
        https://arxiv.org/pdf/2203.06125.pdf

    Parameters:
        num_angle_bin (int, optional): number of bins to discretize angles between edges
    """

    def __init__(self, num_angle_bin=8):
        super(SpatialLineGraph, self).__init__()
        self.num_angle_bin = num_angle_bin

    def forward(self, graph):
        """
        Generate the spatial line graph of the input graph.
        The edge types are decided by the angles between two adjacent edges in the input graph.

        Parameters:
            graph (PackedGraph): :math:`n` graph(s)

        Returns:
            graph (PackedGraph): the spatial line graph
        """
        line_graph = graph.line_graph()
        node_in, node_out = graph.edge_list[:, :2].t()
        edge_in, edge_out = line_graph.edge_list.t()

        # compute the angle ijk
        node_i = node_out[edge_out]
        node_j = node_in[edge_out]
        node_k = node_in[edge_in]
        vector1 = graph.node_position[node_i] - graph.node_position[node_j]
        vector2 = graph.node_position[node_k] - graph.node_position[node_j]
        x = (vector1 * vector2).sum(dim=-1)
        y = torch.cross(vector1, vector2).norm(dim=-1)
        angle = torch.atan2(y, x)
        relation = (angle / math.pi * self.num_angle_bin).long()
        edge_list = torch.cat([line_graph.edge_list, relation.unsqueeze(-1)], dim=-1)

        return type(line_graph)(edge_list, num_nodes=line_graph.num_nodes, offsets=line_graph._offsets,
                                num_edges=line_graph.num_edges, num_relation=self.num_angle_bin,
                                meta_dict=line_graph.meta_dict, **line_graph.data_dict)
