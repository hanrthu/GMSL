from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from torch_scatter import scatter_add

import gmsl.convs.gearnet as layer
from gmsl.register import Register
register = Register()
@register('gearnet')
class GearNetIEConv(nn.Module):

    def __init__(self, sdim, embedding_dim, hidden_dims, num_relation, edge_input_dim=None,
                 batch_norm=False, activation="relu", concat_hidden=False, short_cut=True, 
                 readout="sum", dropout=0, num_angle_bin=None, layer_norm=False, use_ieconv=False, **kwargs):
        super(GearNetIEConv, self).__init__()

        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        self.input_dim = sdim
        self.embedding_dim = embedding_dim
        self.output_dim = sum(hidden_dims) if concat_hidden else hidden_dims[-1]
        self.dims = [embedding_dim if embedding_dim > 0 else sdim] + list(hidden_dims)
        self.edge_dims = [edge_input_dim] + self.dims[:-1]
        self.num_relation = num_relation
        self.concat_hidden = concat_hidden
        self.short_cut = short_cut
        self.num_angle_bin = num_angle_bin
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden
        self.layer_norm = layer_norm
        self.use_ieconv = use_ieconv  

        # if embedding_dim > 0:
        #     self.linear = nn.Linear(sdim, embedding_dim)
        #     self.embedding_batch_norm = nn.BatchNorm1d(embedding_dim)

        self.layers = nn.ModuleList()
        self.ieconvs = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            # note that these layers are from gearnet.layer instead of torchdrug.layers
            self.layers.append(layer.GeometricRelationalGraphConv(self.dims[i], self.dims[i + 1], num_relation,
                                                                   None, batch_norm, activation))
            if use_ieconv:
                self.ieconvs.append(layer.IEConvLayer(self.dims[i], self.dims[i] // 4, 
                                    self.dims[i+1], edge_input_dim=14, kernel_hidden_dim=32))
        if num_angle_bin:
            # self.spatial_line_graph = layers.SpatialLineGraph(num_angle_bin)
            self.edge_layers = nn.ModuleList()
            for i in range(len(self.edge_dims) - 1):
                self.edge_layers.append(layer.GeometricRelationalGraphConv(
                    self.edge_dims[i], self.edge_dims[i + 1], num_angle_bin, None, batch_norm, activation))

        if layer_norm:
            self.layer_norms = nn.ModuleList()
            for i in range(len(self.dims) - 1):
                self.layer_norms.append(nn.LayerNorm(self.dims[i + 1]))

        self.dropout = nn.Dropout(dropout)

        # if readout == "sum":
        #     self.readout = layers.SumReadout()
        # elif readout == "mean":
        #     self.readout = layers.MeanReadout()
        # else:
        #     raise ValueError("Unknown readout `%s`" % readout)

    def get_ieconv_edge_feature(self, graph):
        pos = graph.pos.squeeze()
        u = torch.ones_like(pos)
        u[1:] = pos[1:] - pos[:-1]
        u = F.normalize(u, dim=-1)
        b = torch.ones_like(pos)
        b[:-1] = u[:-1] - u[1:]
        b = F.normalize(b, dim=-1)
        n = torch.ones_like(pos)
        n[:-1] = torch.cross(u[:-1], u[1:])
        n = F.normalize(n, dim=-1)

        local_frame = torch.stack([b, n, torch.cross(b, n)], dim=-1)

        node_in, node_out = graph.edge_index
        atom2residue = torch.as_tensor(range(len(graph.x)), dtype=torch.long).to(node_in.device)
        t = pos[node_out] - pos[node_in]
        t = torch.einsum('ijk, ij->ik', local_frame[node_in], t)
        r = torch.sum(local_frame[node_in] * local_frame[node_out], dim=1)
        delta = torch.abs(atom2residue[node_in] - atom2residue[node_out]).float() / 6
        delta = delta.unsqueeze(-1)

        return torch.cat([
            t, r, delta, 
            1 - 2 * t.abs(), 1 - 2 * r.abs(), 1 - 2 * delta.abs()
        ], dim=-1)

    def forward(self, graph, input):
        hiddens = []
        layer_input = input
        # print("Layer Input:", layer_input.shape)
        # print("Input and embedding dim:", self.input_dim, self.embedding_dim)
        # 现场给氨基酸进行embedding，所以nodefeature实际上就是氨基酸的序列表示
        # if self.embedding_dim > 0:
        #     layer_input = self.linear(layer_input)
        #     layer_input = self.embedding_batch_norm(layer_input)
        if self.num_angle_bin:
            edge_hidden = None
            # line_graph = self.spatial_line_graph(graph)
            # edge_hidden = line_graph.node_feature.float()
        else:
            edge_hidden = None
        ieconv_edge_feature = self.get_ieconv_edge_feature(graph)

        for i in range(len(self.layers)):
            # edge message passing
            if self.num_angle_bin:
                edge_hidden = None
                # edge_hidden = self.edge_layers[i](line_graph, edge_hidden)
            hidden = self.layers[i](graph, layer_input, edge_hidden)
            # ieconv layer
            if self.use_ieconv:
                hidden = hidden + self.ieconvs[i](graph, layer_input, ieconv_edge_feature)
            hidden = self.dropout(hidden)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            if self.layer_norm:
                hidden = self.layer_norms[i](hidden)
            hiddens.append(hidden)
            layer_input = hidden

        if self.concat_hidden:
            node_feature = torch.cat(hiddens, dim=-1)
        else:
            node_feature = hiddens[-1]
        # graph_feature = self.readout(graph, node_feature)

        return node_feature
