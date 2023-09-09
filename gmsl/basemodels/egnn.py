from typing import Optional, Tuple
import torch
from torch import Tensor, nn
from torch_geometric.nn.inits import reset
from gmsl.convs import EGCL, EGCL_Edge_Fast, EGCL_Edge_Vallina
from gmsl.register import Register
register = Register()

@register('egnn')
class EGNN(nn.Module):
    def __init__(
        self,
        sdim: int = 128,
        vdim: int = 3,
        hid_dims: int = 64,
        depth: int = 4,
        eps: float = 1e-6,
        cutoff: Optional[float] = 5.0,
        vector_aggr: str = 'mean',
        act_fn = nn.SiLU(),
        **kwargs
    ):
        super(EGNN, self).__init__()
        self.dims = (sdim, vdim)
        self.depth = depth
        self.vector_aggr = vector_aggr
        module = EGCL
        self.convs = nn.ModuleList()
        for _ in range(depth):
            self.convs.append(
                module(
                    in_dims=self.dims,
                    has_v_in=True,
                    out_dims=self.dims,
                    hid_dims=hid_dims,
                    cutoff=cutoff,
                    vector_aggr=vector_aggr,
                    act_fn=act_fn
                )
            )
        self.apply(fn=reset)
    def forward(
        self,
        x: Tuple[Tensor, Tensor],
        edge_index: Tensor,
        edge_attr: Tuple[Tensor, Tensor],
        batch: Tensor,
    ) -> Tuple[Tensor, Tensor]:

        s, v = x
        for i in range(len(self.convs)):
            s, v = self.convs[i](x=(s,v), edge_index=edge_index, edge_attr=edge_attr)
        return s, v

@register('egnn_edge')
class EGNN_Edge(nn.Module):
    def __init__(
        self,
        sdim: int = 128,
        vdim: int = 3,
        edim: int = 128,
        hid_dims: int = 64,
        depth: int = 4,
        eps: float = 1e-6,
        cutoff: Optional[float] = 5.0,
        vector_aggr: str = 'mean',
        act_fn = nn.SiLU(),
        enhanced: bool = True
    ):
        super(EGNN_Edge, self).__init__()
        self.dims = (sdim, vdim, edim)
        self.depth = depth
        self.vector_aggr = vector_aggr
        self.enhanced = enhanced
        if enhanced:
            module = EGCL_Edge_Fast
        else:
            module = EGCL_Edge_Vallina
        self.convs = nn.ModuleList()
        for _ in range(depth):
            self.convs.append(
                module(
                    in_dims=self.dims,
                    has_v_in=True,
                    out_dims=self.dims,
                    hid_dims=hid_dims,
                    cutoff=cutoff,
                    vector_aggr=vector_aggr,
                    act_fn=act_fn
                )
            )
        self.apply(fn=reset)
    def forward(
        self,
        x: Tuple[Tensor, Tensor, Tensor],
        edge_index: Tensor,
        edge_attr: Tuple[Tensor, Tensor],
        batch: Tensor,
        e_pos: Tensor=None, 
        e_interaction: Tensor=None,
        alphas: Tensor=None, 
        connection_nodes: Tensor=None
    ) -> Tuple[Tensor, Tensor]:

        s, v ,e = x
        if self.enhanced:
            for i in range(len(self.convs)):
                s, v, e = self.convs[i](x=(s,v,e), edge_index=edge_index, edge_attr=edge_attr)
        else:
            for i in range(len(self.convs)):
                s, v, e = self.convs[i](x=(s,v,e), edge_index=edge_index, edge_attr=edge_attr, e_pos=e_pos, e_interaction=e_interaction,
                                   alphas=alphas, connection_nodes=connection_nodes)
        return s, v, e