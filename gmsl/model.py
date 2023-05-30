import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Batch
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.nn.inits import reset
from typing import Optional

from gmsl.modules import DenseLayer, GatedEquivBlock, LayerNorm, GatedEquivBlockTP
from gmsl.basemodels import EQGATGNN, PaiNNGNN, SchNetGNN, GVPNetwork, EGNN, EGNN_Edge

# SEGNN
from gmsl.segnn.segnn import SEGNN
from gmsl.segnn.balanced_irreps import BalancedIrreps
from e3nn.o3 import Irreps
from e3nn.o3 import spherical_harmonics


class BaseModel(nn.Module):
    def __init__(
        self,
        num_elements: int,
        out_units: int,
        sdim: int = 128,
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
        task = 'multitask'
    ):
        if not model_type.lower() in ["painn", "eqgat", "schnet", "egnn", "egnn_edge"]:
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
        
        self.init_embedding = nn.Embedding(num_embeddings=num_elements, embedding_dim=sdim)
        self.init_e_embedding = nn.Embedding(num_embeddings=20, embedding_dim=sdim)
        
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

        if self.model_type in ["schnet", "egnn"]:
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
        self.downstream = nn.Sequential(
            DenseLayer(sdim, sdim, activation=nn.SiLU(), bias=True),
            nn.Dropout(dropout),
            DenseLayer(sdim, out_units, bias=True)
        )
        
        if self.task in ['affinity', 'multi']:
            self.affinity = nn.Sequential(
                DenseLayer(sdim, sdim, activation=nn.SiLU(), bias=True),
                nn.Dropout(dropout),
                DenseLayer(sdim, 1, bias=True)
            )
        
        if self.task == 'ec':
            class_num = 538
        elif self.task == 'mf':
            class_num = 490
        elif self.task == 'bp':
            class_num = 1944
        elif self.task == 'cc':
            class_num = 321
        elif self.task == 'go':
            class_num = 490 + 1944 + 321
        elif self.task == 'multi':
            class_num = 538 + 490 + 1944 + 321
        if self.task != "affinity":
            self.property = nn.Sequential(
                DenseLayer(sdim, sdim, activation=nn.SiLU(), bias=True),
                nn.Dropout(dropout),
                DenseLayer(sdim, class_num, bias=True),
                nn.Sigmoid()
            )
        
        if self.model_type == "egnn_edge":
            # 辅助任务，预测距离
            self.aux_pred = nn.Sequential(
                DenseLayer(edim, edim, activation=nn.SiLU(), bias=True),
                nn.Dropout(dropout),
                DenseLayer(sdim, 1, bias=True)
            )
            self.downstream = nn.Sequential(
                DenseLayer(sdim+edim, sdim, activation=nn.SiLU(), bias=True),
                nn.Dropout(dropout),
                DenseLayer(sdim, out_units, bias=True)
            )

        self.graph_pooling = graph_pooling
        self.apply(reset)

    def forward(self, data: Batch, subset_idx: Optional[Tensor] = None) -> Tensor:
        s, pos, batch = data.x, data.pos, data.batch
        edge_index, d = data.edge_index, data.edge_weights
        lig_flag = data.lig_flag
        chains = data.chains
        # print("Unique chains:", len(function_label), len(torch.unique(chains)))
        # # print("Atoms:", s.shape)
        # print("Protein shape:", s[(lig_flag!=0).squeeze()].shape)
        # print("Lig shape, chain shape:", lig_flag.shape, chains.shape)
        # external_flag = data.external_flag
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
        else:
            s = self.gnn(x=s, edge_index=edge_index, edge_attr=d, batch=batch)
            if self.use_norm:
                s, _ = self.post_norm(x=(s, None), batch=batch)
            s = self.post_lin(s)

        if self.graph_level:
            y_pred = scatter(s, index=batch, dim=0, reduce=self.graph_pooling)
            # 将链从1编号改为从0编号
            chain_pred = scatter(s[(lig_flag!=0).squeeze(), :], index=(chains-1).squeeze(), dim=0, reduce=self.graph_pooling)
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
        if 'lba' in data.type:
            y_pred = self.downstream(y_pred)
        # print('Datatype:', data.type)
        if self.task == 'multi':
            affinity_pred = self.affinity(y_pred)
            property_pred = self.property(chain_pred)
            return affinity_pred, property_pred
        elif self.task in ['ec', 'go', 'mf', 'bp', 'cc']:
            property_pred = self.property(chain_pred)
            return property_pred
        elif self.task == 'affinity':
            affinity_pred = self.affinity(y_pred)
            # print("Predictions:", y_pred.shape, affinity_pred.shape)
            return affinity_pred
        else:
            raise RuntimeError
            # print("Property:", affinity_pred.shape, property_pred.shape, function_label.shape)
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
