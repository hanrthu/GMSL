import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Batch
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.nn.inits import reset
from typing import Optional

from gmsl.modules import DenseLayer, GatedEquivBlock, LayerNorm, GatedEquivBlockTP
from gmsl.basemodels import EQGATGNN, PaiNNGNN, SchNetGNN, GVPNetwork, \
                            EGNN, EGNN_Edge, GearNetIEConv, MultiLayerTAR, HemeNet

# SEGNN
from gmsl.segnn.segnn import SEGNN
from gmsl.segnn.balanced_irreps import BalancedIrreps
from e3nn.o3 import Irreps
from e3nn.o3 import spherical_harmonics
from utils.hetero_graph import MAX_CHANNEL

class BaseModel(nn.Module):
    def __init__(
        self,
        num_elements: int,
        out_units: int,
        sdim: int = 512,
        vdim: int = 16,
        depth: int = 3,
        r_cutoff: float = 5.0,
        num_radial: int = 32,
        model_type: str = "eqgat",
        graph_level: bool = True,
        dropout: float = 0.1,
        use_norm: bool = True,
        aggr: str = "mean",
        graph_pooling: str = "mean",
        cross_ablate: bool = False,
        no_feat_attn: bool = False,
        enhanced: bool = True,
        task = 'multitask',
        readout = 'vallina',
        batch_norm = True,
        layer_norm = True
    ):
        if not model_type.lower() in ["painn", "eqgat", "schnet", "egnn", "egnn_edge", "gearnet", "hemenet"]:
            print("Wrong select model type")
            print("Exiting code")
            exit()

        super(BaseModel, self).__init__()
        self.sdim = sdim
        self.vdim = vdim
        self.depth = depth
        self.r_cutoff = r_cutoff
        self.num_radial = num_radial
        self.model_type = model_type.lower()
        self.graph_level = graph_level
        self.num_elements = num_elements
        self.out_units = out_units
        self.enhanced = enhanced
        self.task = task
        max_channel = MAX_CHANNEL
        channel_nf = sdim
        
        self.init_embedding = nn.Embedding(num_embeddings=num_elements, embedding_dim=sdim)
        self.init_e_embedding = nn.Embedding(num_embeddings=20, embedding_dim=sdim)
        # 14 is the n_channel size
        self.channel_attr = nn.Embedding(num_embeddings=MAX_CHANNEL, embedding_dim=sdim)
        if self.model_type == "painn":
            self.gnn = PaiNNGNN(
                dims=(sdim, vdim),
                depth=depth,
                num_radial=num_radial,
                cutoff=r_cutoff,
                aggr=aggr,
                use_norm=use_norm,
            )
        elif self.model_type == "eqgat":
            self.gnn = EQGATGNN(
                dims=(sdim, vdim),
                depth=depth,
                cutoff=r_cutoff,
                num_radial=num_radial,
                use_norm=use_norm,
                basis="bessel",
                use_mlp_update=True,
                use_cross_product=not cross_ablate,
                no_feat_attn=no_feat_attn,
                vector_aggr=aggr
            )
        elif self.model_type== "egnn":
            vdim = 3
            self.gnn = EGNN(
                dims=(sdim, vdim),
                depth=depth,
                cutoff=r_cutoff,
                vector_aggr=aggr,
            )
        elif self.model_type == "egnn_edge":
            vdim = 3
            edim = sdim
            self.gnn = EGNN_Edge(
                dims=(sdim, vdim, edim),
                depth=depth,
                cutoff=r_cutoff,
                vector_aggr=aggr,
                enhanced= self.enhanced,
            )
        elif self.model_type == "gearnet":
            concat_hidden = True
            hidden_dims = [512, 512, 512, 512, 512, 512]
            self.gnn = GearNetIEConv(
                input_dim=31,
                embedding_dim=128, # ?怎么没看到config里面设置
                hidden_dims=hidden_dims,
                batch_norm=batch_norm,
                layer_norm=layer_norm,
                concat_hidden=concat_hidden,
                short_cut=True,
                # readout='sum',
                num_relation=7,
                # 这个59维包括了inresidue（20），outresidue（20），relation（7），sequentialdist（11）和spatialdist（1）
                edge_input_dim=59,
                num_angle_bin=8
            )
            if concat_hidden == True:
                sdim = 0
                for i in hidden_dims:
                    sdim += i
            else:
                sdim = hidden_dims[-1]
        elif self.model_type == "hemenet":
            concat_hidden = True
            # TODO: 建立一个Residue的map，可以根据不同的residue来自动学习
            # weights = torch.where(
            #     self.atom_weights_mask,
            #     self.zero_atom_weight,
            #     self.atom_weight
            # )  # [num_aa_classes, max_atom_number(n_channel)]
            # if not self.fix_atom_weights:
            #     weights = F.normalize(weights, dim=-1)
            # weights = weights[residue_types]
            hidden_dims = [512, 512, 512, 512, 512, 512]
            self.gnn = HemeNet(
                input_dim=31,
                embedding_dim=512, # ?怎么没看到config里面设置
                hidden_dims=hidden_dims,
                channel_dim = max_channel,
                channel_nf=channel_nf,
                batch_norm=batch_norm,
                layer_norm=layer_norm,
                concat_hidden=concat_hidden,
                short_cut=True,
                num_relation=7,
                # 这个59维包括了inresidue（20），outresidue（20），relation（7），sequentialdist（11）和spatialdist（1）
                edge_input_dim=59,
                num_angle_bin=8
            )
            if concat_hidden == True:
                sdim = 0
                for i in hidden_dims:
                    sdim += i
            else:
                sdim = hidden_dims[-1]

        elif self.model_type == "schnet":
            self.gnn = SchNetGNN(
                dims=sdim,
                depth=depth,
                cutoff=r_cutoff,
                num_radial=num_radial,
                aggr=aggr,
                use_norm=use_norm,
            )

        self.use_norm = use_norm

        if self.model_type in ["schnet", "egnn", "gearnet"]:
            if use_norm:
                self.post_norm = LayerNorm(dims=(sdim, None))
            else:
                self.post_norm = None
            self.post_lin = DenseLayer(sdim, sdim, bias=False)
        elif self.model_type == "egnn_edge":
            self.post_lin = DenseLayer(sdim, sdim, bias=False)
            self.post_lin_e = DenseLayer(edim, sdim, bias=False)
        else:
            if use_norm:
                self.post_norm = LayerNorm(dims=(sdim, vdim))
            else:
                self.post_norm = None
            self.post_lin = GatedEquivBlock(in_dims=(sdim, vdim),
                                            out_dims=(sdim, None),
                                            hs_dim=sdim, hv_dim=vdim,
                                            use_mlp=False)
        
        if self.task in ['lba', 'ppi', 'multi']:
            self.affinity_heads = nn.ModuleList()
            if self.task == 'lba' or self.task == 'ppi':
                self.affinity_heads.append(nn.Sequential(
                    DenseLayer(sdim, sdim, activation=nn.SiLU(), bias=True),
                    nn.Dropout(dropout),
                    DenseLayer(sdim, 1, bias=True)
                ))
            else:
                lba = nn.Sequential(
                    DenseLayer(sdim, sdim, activation=nn.SiLU(), bias=True),
                    nn.Dropout(dropout),
                    DenseLayer(sdim, 1, bias=True)
                )
                ppi = nn.Sequential(
                    DenseLayer(sdim, sdim, activation=nn.SiLU(), bias=True),
                    nn.Dropout(dropout),
                    DenseLayer(sdim, 1, bias=True)
                )
                self.affinity_heads.append(lba)
                self.affinity_heads.append(ppi)
        if self.task == 'ec':
            class_nums = [538]
        elif self.task == 'mf':
            class_nums = [490]
        elif self.task == 'bp':
            class_nums = [1944]
        elif self.task == 'cc':
            class_nums = [321]
        elif self.task == 'go':
            class_nums = [490, 1944, 321]
        elif self.task == 'multi':
            class_nums = [538, 490, 1944, 321]
        else:
            class_nums = []
        # print("Class Nums:", class_nums)
        if len(class_nums) > 0:
            heads = [nn.Sequential(
                DenseLayer(sdim, sdim, activation=nn.SiLU(), bias=True),
                nn.Dropout(dropout),
                DenseLayer(sdim, class_num, bias=True),
            ) for class_num in class_nums]
            self.property_heads = nn.ModuleList()
            for head in heads:
                self.property_heads.append(head)
        
        if self.model_type == "egnn_edge":
            # 辅助任务，预测距离
            self.aux_pred = nn.Sequential(
                DenseLayer(edim, edim, activation=nn.SiLU(), bias=True),
                nn.Dropout(dropout),
                DenseLayer(sdim, 1, bias=True)
            )
        self.readout = readout
        print("Readout Strategy:", readout)
        if readout == 'task_aware_attention':
            self.affinity_prompts = nn.Parameter(torch.ones(1, len(self.affinity_heads), sdim))
            self.property_prompts = nn.Parameter(torch.ones(1, len(self.property_heads), sdim))
            self.global_readout = MultiLayerTAR(in_features=sdim, hidden_size=sdim, out_features=sdim, num_layers=1)
            self.chain_readout = MultiLayerTAR(in_features=sdim, hidden_size=sdim, out_features=sdim, num_layers=1)
            nn.init.kaiming_normal_(self.affinity_prompts)
            nn.init.kaiming_normal_(self.property_prompts)
            # print(self.affinity_prompts.shape)
            # print(self.property_prompts.shape)
        elif readout == 'weighted_feature':
            # Different weights for different tasks
            self.affinity_weights = nn.Parameter(torch.ones(len(self.affinity_heads), 1, sdim))
            self.property_weights = nn.Parameter(torch.ones(len(self.property_heads), 1, sdim))
            nn.init.kaiming_normal_(self.affinity_weights)
            nn.init.kaiming_normal_(self.property_weights)
        self.graph_pooling = graph_pooling
        self.apply(reset)

    def forward(self, data: Batch, subset_idx: Optional[Tensor] = None) -> Tensor:
        s, pos, batch, channel_weights = data.x, data.pos, data.batch, data.channel_weights
        edge_index, d = data.edge_index, data.edge_weights
        lig_flag = data.lig_flag
        chains = data.chains
        pos = pos.squeeze()
        # print("Unique chains:", len(function_label), len(torch.unique(chains)))
        # # print("Atoms:", s.shape)
        # print("Protein shape:", s[(lig_flag!=0).squeeze()].shape)
        # print("Lig shape, chain shape:", lig_flag.shape, chains.shape)
        # external_flag = data.external_flag
        if self.model_type not in ["gearnet", "hemenet"]:
            s = self.init_embedding(s)
        row, col = edge_index
        rel_pos = pos[row] - pos[col]
        rel_pos = F.normalize(rel_pos, dim=-1, eps=1e-6)
        edge_attr = d, rel_pos
        
        if self.model_type in ["painn", "eqgat"]:
            v = torch.zeros(size=[s.size(0), 3, self.vdim], device=s.device)
            s, v = self.gnn(x=(s, v), edge_index=edge_index, edge_attr=edge_attr, batch=batch)
            if self.use_norm:
                s, v = self.post_norm(x=(s, v), batch=batch)
            s, _ = self.post_lin(x=(s, v))
        elif self.model_type == "egnn":
            v = pos
            s, v = self.gnn(x=(s, v), edge_index=edge_index, edge_attr=edge_attr, batch=batch)
            s = self.post_lin(s)
        elif self.model_type == "egnn_edge":
            v = pos
            e = data.e
            e = self.init_e_embedding(e)
            if self.enhanced:
                s, v, e = self.gnn(x=(s, v, e), edge_index=edge_index, edge_attr=edge_attr, batch=batch)
            else:
                e_pos = data.e_pos
                e_interaction = data.e_interaction,
                alphas = data.alphas,
                connection_nodes = data.connection_nodes
                s, v, e = self.gnn(x=(s, v, e), edge_index=edge_index, edge_attr=edge_attr, batch=batch, e_pos=e_pos, e_interaction=e_interaction[0],
                                   alphas=alphas[0], connection_nodes=connection_nodes)
            s = self.post_lin(s)
            e = self.post_lin_e(e)
        elif self.model_type == 'gearnet':
            graph = data
            graph.edge_list = torch.cat([graph.edge_index.t(), graph.edge_relations.unsqueeze(-1)], dim=-1)
            # print("Graph Edge List:", graph.edge_list.shape)
            input_data = data.x
            s = self.gnn(graph, input_data)
            # print("Gearnet Output Shape:", s.shape)
            s = self.post_lin(s)
        elif self.model_type == 'hemenet':
            edge_list = torch.cat([data.edge_index.t(), data.edge_relations.unsqueeze(-1)], dim=-1) # [|E|, 3]
            input_data = data.x
            channel_attr = self.channel_attr(data.residue_elements.long())
            s, coords = self.gnn(input_data, edge_list, pos, channel_attr, channel_weights, data.edge_weights)
        else:
            s = self.gnn(x=s, edge_index=edge_index, edge_attr=d, batch=batch)
            if self.use_norm:
                s, _ = self.post_norm(x=(s, None), batch=batch)
            s = self.post_lin(s)

        if self.graph_level:
            if self.readout == 'vallina':
                y_pred = scatter(s, index=batch, dim=0, reduce=self.graph_pooling)
                # 将链从1编号改为从0编号
                chain_pred = scatter(s[(lig_flag!=0).squeeze(), :], index=(chains-1).squeeze(), dim=0, reduce=self.graph_pooling)
            elif self.readout == 'weighted_feature':
                # print("Shape:", self.task_weights.shape, s.shape)
                y_pred = scatter(s * self.affinity_weights, index=batch, dim=1, reduce=self.graph_pooling)
                chain_pred = scatter(s[(lig_flag!=0).squeeze(), :] * self.property_weights, index=(chains-1).squeeze(), dim=1, reduce=self.graph_pooling)
                # print("After", y_pred.shape, chain_pred.shape)
                if self.task == 'multi':
                    affinity_pred = [affinity_head(y_pred[i].squeeze()) for i, affinity_head in enumerate(self.affinity_heads)]
                    property_pred = [property_head(chain_pred[i].squeeze()) for i, property_head in enumerate(self.property_heads)]
                    return affinity_pred, property_pred
                elif self.task in ['ec', 'go', 'mf', 'bp', 'cc']:
                    property_pred = [property_head(chain_pred[i].squeeze()) for i, property_head in enumerate(self.property_heads)]
                    return property_pred
                elif self.task in ['lba', 'ppi']:
                    affinity_pred = [affinity_head(y_pred[i]) for i, affinity_head in enumerate(self.affinity_heads)]
                    return affinity_pred
                else:
                    raise RuntimeError
            elif self.readout == 'task_aware_attention':
                global_index = batch
                y_pred = self.global_readout(self.affinity_prompts, s, global_index)
                chain_index = (chains-1).squeeze()
                proteins = s[(lig_flag!=0).squeeze(), :]
                # print("Shapes:", chain_index.shape, proteins.shape)
                chain_pred = self.chain_readout(self.property_prompts, proteins, chain_index)
                if self.task == 'multi':
                    affinity_pred = [affinity_head(y_pred[i].squeeze()) for i, affinity_head in enumerate(self.affinity_heads)]
                    property_pred = [property_head(chain_pred[i].squeeze()) for i, property_head in enumerate(self.property_heads)]
                    return affinity_pred, property_pred
                elif self.task in ['ec', 'go', 'mf', 'bp', 'cc']:
                    property_pred = [property_head(chain_pred[i].squeeze()) for i, property_head in enumerate(self.property_heads)]
                    return property_pred
                elif self.task in ['lba', 'ppi']:
                    affinity_pred = [affinity_head(y_pred[i]) for i, affinity_head in enumerate(self.affinity_heads)]
                    return affinity_pred
                else:
                    raise RuntimeError
            if self.model_type == "egnn_edge":
                edge_batch = batch[edge_index[0]]
                # e_external = e[external_flag==1]
                # edge_external_batch = edge_batch[external_flag==1]
                # e_pool = scatter(e_external, index=edge_external_batch, dim=0, reduce="mean")
                e_pool = scatter(e, index=edge_batch, dim=0, reduce='mean')
                # print("E flags:", external_flag.shape)
                y_pred = torch.cat([y_pred, e_pool], dim=-1)
        else:
            y_pred = s

        if subset_idx is not None:
            y_pred = y_pred[subset_idx]
        # print('Datatype:', data.type)
        if self.task == 'multi':
            affinity_pred = [affinity_head(y_pred) for affinity_head in self.affinity_heads]
            property_pred = [property_head(chain_pred) for property_head in self.property_heads]
            return affinity_pred, property_pred
        elif self.task in ['ec', 'go', 'mf', 'bp', 'cc']:
            property_pred = [property_head(chain_pred) for property_head in self.property_heads]
            return property_pred
        elif self.task in ['lba', 'ppi']:
            affinity_pred = [affinity_head(y_pred) for affinity_head in self.affinity_heads]
            return affinity_pred
        else:
            raise RuntimeError
        
        
        # # 设计了一个简易的辅助任务
        # if self.model_type == "egnn_edge":
        #     e_external = e[external_flag==1]
        #     # print("Data shape:", e_pool.shape, e_external.shape)
        #     y_aux = self.aux_pred(e_external)
        #     return y_pred, y_aux



class SEGNNModel(nn.Module):
    def __init__(
        self,
        num_elements: int,
        out_units: int,
        hidden_dim: int,
        lmax: int = 2,
        depth: int = 3,
        graph_level: bool = True,
        use_norm: bool = True,
    ):
        super(SEGNNModel, self).__init__()
        self.init_embedding = nn.Embedding(num_embeddings=num_elements, embedding_dim=num_elements)
        self.input_irreps = Irreps(f"{num_elements}x0e")    # element embedding
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
                           norm="instance" if use_norm else None,
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

    model0 = BaseModel(num_elements=9,
                       out_units=1,
                       sdim=128,
                       vdim=16,
                       depth=3,
                       r_cutoff=5.0,
                       num_radial=32,
                       model_type="eqgat",
                       graph_level=True,
                       use_norm=True)

    print(sum(m.numel() for m in model0.parameters() if m.requires_grad))
    # 375841
    model1 = SEGNNModel(num_elements=9,
                        out_units=1,
                        hidden_dim=128 + 3*16 + 5*8,
                        lmax=2,
                        depth=3,
                        graph_level=True,
                        use_norm=True)

    print(sum(m.numel() for m in model1.parameters() if m.requires_grad))
    # 350038
