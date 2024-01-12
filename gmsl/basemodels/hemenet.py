# This script defines an Heterogeneous Multitask Equivariant Network(HeMENet) deal with protein related tasks.
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from torch_scatter import scatter_add

import gmsl.convs.hemenet as layer
from gmsl.register import Register
register = Register()
@register('hemenet')
class HemeNet(nn.Module):
    def __init__(self, sdim, embedding_dim, hidden_dims, num_relation, channel_dim, channel_nf, edge_input_dim=None,
                 batch_norm=False, activation=nn.SiLU(), concat_hidden=False, short_cut=True, 
                 coords_agg="mean", dropout=0, num_angle_bin=None, layer_norm=False, use_ieconv=False, **kwargs):
        super(HemeNet, self).__init__()

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

        self.layers = nn.ModuleList()
        self.ieconvs = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            # note that these layers are from gearnet.layer instead of torchdrug.layers
            self.layers.append(layer.AM_EGCL(self.dims[i], self.dims[i + 1], self.dims[i+1], num_relation,
                                            channel_dim, channel_nf, coords_agg, batch_norm=batch_norm,
                                            activation=activation, edge_input_dim=edge_input_dim))
            if use_ieconv:
                self.ieconvs.append(layer.IEConvLayer(self.dims[i], self.dims[i] // 4, 
                                    self.dims[i+1], edge_input_dim=14, kernel_hidden_dim=32))
        if layer_norm:
            self.layer_norms = nn.ModuleList()
            for i in range(len(self.dims) - 1):
                self.layer_norms.append(nn.LayerNorm(self.dims[i + 1]))

        self.dropout = nn.Dropout(dropout)

    def get_ieconv_edge_feature(self, coords, input, channel_weights, edge_list):
        '''
            coords: [N, n_channel, d]
            input: input feature, [N, input_dim]
            channel_weights: [N, n_channel]
            edge_list: [|E|, 3], source, target, relation
        '''
        # We currently adopts the center of position as the postion for each node
        channel_sum = torch.sum(channel_weights, dim=-1).unsqueeze(1)
        coord_sum = torch.sum(coords, dim=1, keepdim=False)
        center_pos = coord_sum / channel_sum #[N, d]
        u = torch.ones_like(center_pos)
        u[1:] = center_pos[1:] - center_pos[:-1]
        u = F.normalize(u, dim=-1)
        b = torch.ones_like(center_pos)
        b[:-1] = u[:-1] - u[1:]
        b = F.normalize(b, dim=-1)
        n = torch.ones_like(center_pos)
        n[:-1] = torch.cross(u[:-1], u[1:])
        n = F.normalize(n, dim=-1)

        local_frame = torch.stack([b, n, torch.cross(b, n)], dim=-1)

        node_in, node_out = edge_list[:, 0], edge_list[:, 1]
        # atom2residue = torch.arange(0, len(center_pos)).to(node_in.device)
        t = center_pos[node_out] - center_pos[node_in]
        t = torch.einsum('ijk, ij->ik', local_frame[node_in], t)
        r = torch.sum(local_frame[node_in] * local_frame[node_out], dim=1)
        delta = torch.abs(node_in - node_out).float() / 6
        delta = delta.unsqueeze(-1)

        return torch.cat([
            t, r, delta, 
            1 - 2 * t.abs(), 1 - 2 * r.abs(), 1 - 2 * delta.abs()
        ], dim=-1)

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
        # ieconv_edge_feature = self.get_ieconv_edge_feature(coords, input, channel_weights, edge_list)

        for i in range(len(self.layers)):
            # edge message passing
            node_hidden, coord_hidden = self.layers[i](layer_input, edge_list, coord_input, channel_attr, channel_weights, 
                edge_weights, edge_attr=edge_attr, node_attr=node_attr)
            # ieconv layer
            # if self.use_ieconv:
            #     node_hidden = node_hidden + self.ieconvs[i](layer_input, ieconv_edge_feature, edge_list, edge_weights)
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
