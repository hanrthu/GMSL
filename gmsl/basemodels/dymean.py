from collections.abc import Sequence

import torch
from Bio.Data.IUPACData import atom_weights
from torch import nn
from torch.nn import functional as F

from torch_scatter import scatter_add

import gmsl.convs.dymean as layer
from gmsl.register import Register
register = Register()
@register('dymean')
class DyMEAN(nn.Module):
    def __init__(self, sdim, embedding_dim, hidden_dims, channel_dim, channel_nf, edge_input_dim=None,
                activation=nn.SiLU(), concat_hidden=False, short_cut=True,
                 coords_agg="mean", dropout=0, num_angle_bin=None, layer_norm=False, **kwargs):
        super(DyMEAN, self).__init__()

        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        self.input_dim = sdim
        self.embedding_dim = embedding_dim
        self.output_dim = sum(hidden_dims) if concat_hidden else hidden_dims[-1]
        self.dims = [embedding_dim if embedding_dim > 0 else sdim] + list(hidden_dims)
        self.edge_dims = [edge_input_dim] + self.dims[:-1]
        self.concat_hidden = concat_hidden
        self.short_cut = short_cut
        self.num_angle_bin = num_angle_bin
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden
        self.layer_norm = layer_norm

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(layer.AM_EGCL(self.dims[i], self.dims[i + 1], self.dims[i+1],
                                            channel_dim, channel_nf, coords_agg,
                                            activation=activation, edge_input_dim=edge_input_dim))
        if layer_norm:
            self.layer_norms = nn.ModuleList()
            for i in range(len(self.dims) - 1):
                self.layer_norms.append(nn.LayerNorm(self.dims[i + 1]))

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, edge_list, coords, channel_attr, channel_weights, edge_weights,
                edge_attr=None, node_attr=None):
        '''
            input: input feature, [N, input_dim]
            edge_list: [|E|, 3], source, target, relation
            coords: [N, n_channel, d]
            channel_attr: [N, n_channel, channel_nf]
            channel_weights: [N, n_channel]
            edge_weights: [|E|]
            edge_attr: Optional, [|E|, edge_dim]
            node_attr: Optional, [N, node_feature]
        '''
        node_hiddens = []
        coord_hiddens = []
        layer_input = input
        coord_input = coords

        for i in range(len(self.layers)):
            # message passing
            node_hidden, coord_hidden = self.layers[i](layer_input, edge_list, coord_input, channel_attr, channel_weights,
                edge_weights, edge_attr=edge_attr, node_attr=node_attr)
            node_hidden = self.dropout(node_hidden)
            # if self.short_cut and node_hidden.shape == layer_input.shape:
            #     node_hidden = node_hidden + layer_input
            if self.layer_norm:
                node_hidden = self.layer_norms[i](node_hidden)
            node_hiddens.append(node_hidden)
            coord_hiddens.append(coord_hidden)
            layer_input = node_hidden
            coord_input = coord_hidden

        if self.concat_hidden:
            node_feature = torch.cat(node_hiddens, dim=-1)
        else:
            node_feature = node_hiddens[-1]
        coord_feature = coord_hiddens[-1]
        return node_feature, coord_feature