import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Batch
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.nn.inits import reset
from typing import Optional
import inspect

from gmsl.modules import DenseLayer, GatedEquivBlock, LayerNorm
from gmsl.basemodels import MultiLayerTAR

# SEGNN
from gmsl.segnn.segnn import SEGNN
from gmsl.segnn.balanced_irreps import BalancedIrreps
from e3nn.o3 import Irreps
from e3nn.o3 import spherical_harmonics
from utils.hetero_graph import MAX_CHANNEL, NUM_ELEMENTS
from utils import class_num_dict, affinity_num_dict
from gmsl.register import Register


class BaseModel(nn.Module):
    def __init__(
        self,
        out_units: int,
        sdim: int = 512,
        vdim: int = 16,
        depth: int = 3,
        r_cutoff: float = 4.5,
        num_radial: int = 32,
        model_type: str = "eqgat",
        dropout: float = 0.1,
        aggr: str = "mean",
        graph_pooling: str = "mean",
        no_feat_attn: bool = False,
        task = 'multi',
        readout = 'vallina',
        batch_norm = True,
        layer_norm = True,
        concat_hidden = True,
        embedding_dim = 512,
        **kwargs
    ):
        if not model_type.lower() in ["painn", "eqgat", "schnet", "egnn", "egnn_edge", "gearnet", "hemenet", "dymean"]:
            print("Wrong select model type")
            print("Exiting code")
            raise ValueError

        super(BaseModel, self).__init__()
        self.sdim = sdim
        self.vdim = vdim
        self.depth = depth
        self.r_cutoff = r_cutoff
        self.num_radial = num_radial
        self.model_type = model_type.lower()
        self.out_units = out_units
        self.task = task

        local_args = locals()
        register = Register()
        self.init_embedding = nn.Embedding(num_embeddings=NUM_ELEMENTS, embedding_dim=embedding_dim)
        # 14 is the n_channel size
        self.channel_attr = nn.Embedding(num_embeddings=MAX_CHANNEL, embedding_dim=sdim) if self.model_type in ['hemenet', 'dymean'] else None
        self.edge_attr = nn.Embedding(num_embeddings=kwargs['num_relation'], embedding_dim=sdim) if self.model_type=='hemenet' else None
        gnn_cls = register[self.model_type]

        argspec = inspect.getfullargspec(gnn_cls.__init__)
        in_args = {}
        for key, value in local_args.items():
            if key in argspec.args and key != 'self':
                in_args[key] = value
        in_args.update(kwargs)
        self.gnn = gnn_cls(**in_args)

        if concat_hidden:
            sdim = 0
            for i in kwargs['hidden_dims']:
                sdim += i
        else:
            sdim = kwargs['hidden_dims'][-1]

        self.use_norm = layer_norm

        if self.model_type in ["schnet", "egnn"]:
            if layer_norm:
                self.post_norm = LayerNorm(dims=(sdim, None))
            else:
                self.post_norm = None
            self.post_lin = DenseLayer(sdim, sdim, bias=False)
        elif self.model_type == 'gearnet':
            self.post_lin = None
            self.post_norm = None
        elif self.model_type in ['hemenet', 'dymean']:
            self.post_norm = None
            self.post_lin = None
            # self.coord_lin = DenseLayer(MAX_CHANNEL*MAX_CHANNEL , sdim, bias=False)
        else:
            if self.use_norm:
                self.post_norm = LayerNorm(dims=(sdim, vdim))
            else:
                self.post_norm = None
            self.post_lin = GatedEquivBlock(in_dims=(sdim, vdim),
                                            out_dims=(sdim, None),
                                            hs_dim=sdim, hv_dim=vdim,
                                            use_mlp=False)
        affinity_num = affinity_num_dict(self.task)
        self.affinity_heads = nn.ModuleList()
        for _ in affinity_num:
            self.affinity_heads.append(
                nn.Sequential(
                DenseLayer(sdim, sdim, activation=nn.SiLU(), bias=True),
                nn.Dropout(dropout),
                DenseLayer(sdim, 1, bias=True)
            ))

        class_nums = class_num_dict(self.task)
        self.property_heads = nn.ModuleList()
        for class_num in class_nums:
            self.property_heads.append(nn.Sequential(
                DenseLayer(sdim, sdim, activation=nn.SiLU(), bias=True),
                nn.Dropout(dropout),
                DenseLayer(sdim, class_num, bias=True),
            ))
        self.readout = readout
        print("Readout Strategy:", readout)
        if readout == 'task_aware_attention':
            self.affinity_prompts = nn.Parameter(torch.ones(1, len(self.affinity_heads), sdim))
            self.property_prompts = nn.Parameter(torch.ones(1, len(self.property_heads), sdim))
            self.global_readout = MultiLayerTAR(in_features=sdim, hidden_size=sdim, out_features=sdim, num_layers=1)
            self.chain_readout = MultiLayerTAR(in_features=sdim, hidden_size=sdim, out_features=sdim, num_layers=1)
            nn.init.kaiming_normal_(self.affinity_prompts)
            nn.init.kaiming_normal_(self.property_prompts)
        elif readout == 'weighted_feature':
            # Different weights for different tasks
            self.affinity_weights = nn.Parameter(torch.ones(len(self.affinity_heads), 1, sdim))
            self.property_weights = nn.Parameter(torch.ones(len(self.property_heads), 1, sdim))
            nn.init.kaiming_normal_(self.affinity_weights)
            nn.init.kaiming_normal_(self.property_weights)
        self.graph_pooling = graph_pooling
        self.apply(reset)

    def forward(self, data: Batch):
        s, pos, batch, channel_weights = data.x, data.pos, data.batch, data.channel_weights
        edge_index, d, lig_flag, chains = data.edge_index, data.edge_weights, data.lig_flag, data.chains
        pos = pos.squeeze()
        # 暂时转一下，之后预处理可以直接存成序号
        if len(s.shape) == 2:
            s = torch.argmax(s, dim=-1)
            s = self.init_embedding(s.int())
        else:
            s = self.init_embedding(s.int())
        row, col = edge_index
        rel_pos = pos[row] - pos[col]
        rel_pos = F.normalize(rel_pos, dim=-1, eps=1e-6)
        edge_attr = d, rel_pos
        coords = None
        if self.model_type in ["painn", "eqgat"]:
            v = torch.zeros(size=[s.size(0), 3, self.vdim], device=s.device)
            s, v = self.gnn((s, v), edge_index=edge_index, edge_attr=edge_attr, batch=batch)
            if self.use_norm:
                s, _ = self.post_norm(x=(s, None), batch=batch)
            s, _ = self.post_lin(x=(s, v))
            if torch.isnan(s).any():
                print("Found Nan!")
        elif self.model_type == "egnn":
            v = pos
            s, coords = self.gnn((s, v), edge_index=edge_index, edge_attr=edge_attr, batch=batch)
            if self.use_norm:
                s, v = self.post_norm(x=(s,v), batch=batch)
            s = self.post_lin(s)
        elif self.model_type == 'gearnet':
            graph = data
            graph.edge_list = torch.cat([graph.edge_index.t(), graph.edge_relations.unsqueeze(-1)], dim=-1)
            input_data = s
            s = self.gnn(graph, input_data)
            # s = self.post_lin(s)
        elif self.model_type == 'hemenet':
            edge_list = torch.cat([data.edge_index.t(), data.edge_relations.unsqueeze(-1)], dim=-1) # [|E|, 3]
            channel_attr = self.channel_attr(data.residue_elements.long())
            edge_attr = self.edge_attr(data.edge_relations.long())
            s, coords = self.gnn(s, edge_list, pos, channel_attr, channel_weights, data.edge_weights, edge_attr)
            # dist_info = torch.norm(coords[:, :, None] - coords[:, None, :], dim=-1, keepdim=False).view(coords.shape[0], -1)
            # s = s + self.coord_lin(dist_info)
        elif self.model_type == 'dymean':
            edge_list = data.edge_index.t()
            channel_attr = self.channel_attr(data.residue_elements.long())
            s, coords = self.gnn(s, edge_list, pos, channel_attr, channel_weights, data.edge_weights)
        else: # schnet
            s = self.gnn(x=s, edge_index=edge_index, edge_attr=d, batch=batch)
            if self.use_norm:
                s, _ = self.post_norm(x=(s, None), batch=batch)
            s = self.post_lin(s)

        if self.readout == 'vallina':
            y_pred = scatter(s, index=batch, dim=0, reduce=self.graph_pooling).unsqueeze(0).repeat_interleave(len(self.affinity_heads), dim=0)
            # 将链从1编号改为从0编号
            chain_pred = scatter(s[(lig_flag!=0).squeeze(), :], index=(chains-1).squeeze(), dim=0, reduce=self.graph_pooling).unsqueeze(0).repeat_interleave(len(self.property_heads), dim=0)
        elif self.readout == 'weighted_feature':
            y_pred = scatter(s * self.affinity_weights, index=batch, dim=1, reduce=self.graph_pooling)
            chain_pred = scatter(s[(lig_flag!=0).squeeze(), :] * self.property_weights, index=(chains-1).squeeze(), dim=1, reduce=self.graph_pooling)
        elif self.readout == 'task_aware_attention':
            global_index = batch
            y_pred = self.global_readout(self.affinity_prompts, s, global_index)
            proteins = s[(lig_flag!=0).squeeze(), :]
            chain_pred = self.chain_readout(self.property_prompts, proteins, (chains-1).squeeze())
        else:
            raise NotImplementedError
        if torch.isnan(y_pred).any() or torch.isnan(chain_pred).any():
            print("Found Nan!")
        if self.task == 'multi':
            affinity_pred = [affinity_head(y_pred[i].squeeze()) for i, affinity_head in enumerate(self.affinity_heads)]
            property_pred = [property_head(chain_pred[i].squeeze()) for i, property_head in
                             enumerate(self.property_heads)]
            if coords is not None:
                return affinity_pred, property_pred, coords
            else:
                return affinity_pred, property_pred, None
        elif self.task in ['ec', 'go', 'mf', 'bp', 'cc']:
            property_pred = [property_head(chain_pred[i].squeeze()) for i, property_head in
                             enumerate(self.property_heads)]
            if coords is not None:
                return property_pred, coords
            else:
                return property_pred, None
        elif self.task in ['lba', 'ppi']:
            affinity_pred = [affinity_head(y_pred[i]) for i, affinity_head in enumerate(self.affinity_heads)]
            if coords is not None:
                return affinity_pred, coords
            else:
                return affinity_pred, None
        else:
            raise RuntimeError

class SEGNNModel(nn.Module):
    def __init__(
        self,
        out_units: int,
        hidden_dim: int,
        lmax: int = 2,
        depth: int = 3,
        graph_level: bool = True,
        layer_norm: bool = True,
    ):
        super(SEGNNModel, self).__init__()
        self.init_embedding = nn.Embedding(num_embeddings=NUM_ELEMENTS, embedding_dim=num_elements)
        self.input_irreps = Irreps(f"{NUM_ELEMENTS}x0e")    # element embedding
        self.edge_attr_irreps = Irreps.spherical_harmonics(lmax)  # Spherical harmonics projection of relative pos.
        self.node_attr_irreps = Irreps.spherical_harmonics(lmax)    # aggregation of spherical harmonics projection
        self.hidden_irreps = BalancedIrreps(lmax, hidden_dim, sh_type=True)   # only considering SO(3)
        self.additional_message_irreps = Irreps("1x0e")  # euclidean distance
        self.output_irreps = Irreps(f"{out_units}x0e")  # SO(3) invariant output quantity
        self.model = SEGNN(self.input_irreps,
                           self.hidden_irreps,
                           self.output_irreps,
                           self.edge_attr_irreps,
                           self.node_attr_irreps,
                           num_layers=depth,
                           norm="instance" if layer_norm else None,
                           pool="mean",
                           task="graph" if graph_level else "node",
                           additional_message_irreps=self.additional_message_irreps)
        self.model.init_pooler(pool="avg")


    def forward(self, data: Batch, subset_idx: Optional[Tensor ] = None) -> Tensor:
        x, pos, batch = data.x, data.pos, data.batch
        edge_index, d = data.edge_index, data.edge_weights
        x = self.init_embedding(x)

        row, col = edge_index
        rel_pos = pos[row] - pos[col]
        rel_pos = F.normalize(rel_pos, dim=-1, eps=1e-6)

        edge_attr = spherical_harmonics(l=self.edge_attr_irreps, x=rel_pos,
                                        normalize=True,
                                        normalization="component"
                                        )
        node_attr = scatter(edge_attr, col, dim=0, reduce="mean", dim_size=x.size(0))

        # to match https://github.com/RobDHess/Steerable-E3-GNN/blob/main/models/segnn/segnn.py#L101-L109
        new_data = data.clone()
        new_data.x = x
        new_data.edge_attr = edge_attr
        new_data.node_attr = node_attr
        new_data.additional_message_features = d.unsqueeze(-1)
        out = self.model(new_data)
        if subset_idx is not None:
            out = out[subset_idx]

        return out


if __name__ == '__main__':

    model0 = BaseModel(
                       out_units=1,
                       sdim=128,
                       vdim=16,
                       depth=3,
                       r_cutoff=5.0,
                       num_radial=32,
                       model_type="eqgat",
                       graph_level=True,
                       layer_norm=True)

    print(sum(m.numel() for m in model0.parameters() if m.requires_grad))
    # 375841
    model1 = SEGNNModel(
                        out_units=1,
                        hidden_dim=128 + 3*16 + 5*8,
                        lmax=2,
                        depth=3,
                        graph_level=True,
                        layer_norm=True)

    print(sum(m.numel() for m in model1.parameters() if m.requires_grad))
    # 350038
