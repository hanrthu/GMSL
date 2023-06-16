import torch
from torch import Tensor
from typing import Tuple, Optional
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import reset
from torch_scatter import scatter
import torch.nn as nn
import torch.nn.functional as F
class EGCL(MessagePassing):
    def __init__(self,
        in_dims: Tuple[int, Optional[int]],
        out_dims: Tuple[int, Optional[int]],
        hid_dims: int,
        cutoff: float,
        eps: float = 1e-6,
        num_radial: int = 1,
        edges_in_df: int = 0,
        has_v_in: bool = True,
        act_fn = nn.SiLU(),
        attention = False,
        vector_aggr: str = "mean"):
        super(EGCL, self).__init__(node_dim=0, aggr=None, flow="source_to_target")
        self.vector_aggr = vector_aggr
        self.in_dims = in_dims
        self.si, self.vi = in_dims
        self.out_dims = out_dims
        self.so, self.vo = out_dims
        self.has_v_in = has_v_in
        self.attention = attention
        self.phi_m = nn.Sequential(
            nn.Linear(2*self.si+num_radial+edges_in_df, hid_dims),
            act_fn,
            nn.Linear(hid_dims, hid_dims)
            )
        layer = nn.Linear(hid_dims, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        self.phi_x = nn.Sequential(
            nn.Linear(hid_dims, hid_dims),
            act_fn,
            layer
        )
        self.phi_h = nn.Sequential(
            nn.Linear(self.si+hid_dims, hid_dims),
            act_fn,
            nn.Linear(hid_dims, self.so)
        )
        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hid_dims, 1),
                nn.Sigmoid())
    def aggregate(self, inputs: Tuple[Tensor, Tensor], index: Tensor, dim_size: Optional[int] = None) -> Tuple[Tensor, Tensor]:
        # print("It's my turn")
        # print(inputs[0].shape, inputs[1].shape, index.shape)
        ms = scatter(inputs[0], index=index, dim=0, dim_size=dim_size, reduce="sum")
        mv = scatter(inputs[1], index=index, dim=0, dim_size=dim_size, reduce="mean")
        # print("Done")
        return ms, mv
    def message(
            self,
            s_i: Tensor, 
            s_j: Tensor, 
            v_i: Tensor, 
            v_j: Tensor,
            d: Tensor,
        ) -> Tensor:
        # print(s_i.shape, s_j.shape, d.shape)
        dist = torch.pow(d, 2).unsqueeze(-1)
        # 其实对于EGNN这里还有一个edge attr，但这里省略了，假设所有的edge attr都是1吧
        a_ij = torch.cat([s_i, s_j, dist], dim=-1)
        ms_j = self.phi_m(a_ij)
        rel_pos = v_i - v_j
        # assert ((v_j - v_i - (d.unsqueeze(-1) * r)) < 1e-3).all()
        mv_j = rel_pos * self.phi_x(ms_j)
        return ms_j, mv_j
    def forward(
        self,   
        x: Tuple[Tensor, Tensor],
        edge_index: Tensor,
        edge_attr: Tuple[Tensor, Tensor]):
        s, v = x
        # r 是相对的方向，单位向量
        d = torch.pow(v[edge_index[0]] - v[edge_index[1]] + 1e-6, exponent=2).sum(-1).sqrt()
        ms, mv = self.propagate(
            edge_index=edge_index,
            dim_size = s.size(0),
            s=s,
            v=v,
            d=d,
        )
        # print(mv.shape, mv.isnan().any())
        s = s + self.phi_h(torch.cat([s, ms], dim=-1))
        v = v + mv
        return s, v 
        
class EGCL_Edge_Vallina(MessagePassing):
    def __init__(self,
        in_dims: Tuple[int, Optional[int], Optional[int]],
        out_dims: Tuple[int, Optional[int], Optional[int]],
        hid_dims: int,
        cutoff: float,
        eps: float = 1e-6,
        num_radial: int = 1,
        angle_nf: int = 1,
        edges_in_df: int = 0,
        has_v_in: bool = True,
        act_fn = nn.SiLU(),
        attention = False,
        vector_aggr: str = "mean"):
        super(EGCL_Edge_Vallina, self).__init__(node_dim=0, aggr=None, flow="source_to_target")
        self.vector_aggr = vector_aggr
        self.in_dims = in_dims
        self.si, self.vi, self.ei = in_dims
        self.out_dims = out_dims
        self.so, self.vo, self.eo = out_dims
        self.has_v_in = has_v_in
        self.attention = attention
        self.phi_m = nn.Sequential(
            nn.Linear(2*self.si+num_radial+edges_in_df+self.ei, hid_dims),
            act_fn,
            nn.Linear(hid_dims, hid_dims)
            )
        layer = nn.Linear(hid_dims, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        self.phi_x = nn.Sequential(
            nn.Linear(hid_dims, hid_dims),
            act_fn,
            layer
        )
        self.phi_h = nn.Sequential(
            nn.Linear(self.si+hid_dims, hid_dims),
            act_fn,
            nn.Linear(hid_dims, self.so)
        )
        self.phi_e = nn.Sequential(
            nn.Linear(angle_nf+num_radial+2*self.ei+self.si, hid_dims),
            act_fn,
            nn.Linear(hid_dims, self.eo)
        )
        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hid_dims, 1),
                nn.Sigmoid())
    def aggregate(self, inputs: Tuple[Tensor, Tensor, Tensor, Tensor], index: Tensor, dim_size: Optional[int] = None) -> Tuple[Tensor, Tensor]:
        ms = scatter(inputs[0], index=index, dim=0, dim_size=dim_size, reduce="sum")
        mv = scatter(inputs[1], index=index, dim=0, dim_size=dim_size, reduce="mean")
        me = scatter(inputs[2], index=inputs[3], dim=0, dim_size=len(torch.unique(inputs[3])), reduce="mean")
        # print("Unique edges:", len(torch.unique(inputs[3])))
        return ms, mv, me
    # def vallina_edge_message(self, edge_index, e_in, s_in, v_in):
    def message(
            self,
            s_i: Tensor, 
            s_j: Tensor, 
            v_i: Tensor, 
            v_j: Tensor,
            d: Tensor,
            e: Tensor,
            e_pos: Tensor,
            e_interaction: Tensor,
            alpha: Tensor,
            connection_nodes: Tensor
        ) -> Tensor:
        # print(s_i.shape, s_j.shape, d.shape)
        # 其实对于EGNN这里还有一个edge attr，但这里省略了，假设所有的edge attr都是1吧
        # 这里的e_interaction特别特别重要，涉及到边的编号问题
        dist = torch.pow(d, 2).unsqueeze(-1)
        a_ij = torch.cat([s_i, s_j, e, dist], dim=-1)
        ms_j = self.phi_m(a_ij)
        rel_pos = v_i - v_j
        # assert ((v_j - v_i - (d.unsqueeze(-1) * r)) < 1e-3).all()
        mv_j = rel_pos * self.phi_x(ms_j)
        c_dist = torch.pow(e_pos[e_interaction[0]] - e_pos[e_interaction[1]], 2).sum(dim=-1).sqrt()
        in_feature_e = e[e_interaction[0]]
        out_feature_e = e[e_interaction[1]]
        in_feature_s = s_j[connection_nodes]
        # print(alpha)
        me_in = self.phi_e(torch.cat([alpha.unsqueeze(1), c_dist.unsqueeze(1), in_feature_e, out_feature_e, in_feature_s], dim=-1))
        update_e_index = e_interaction[1]
        return ms_j, mv_j, me_in, update_e_index
    def forward(
        self,   
        x: Tuple[Tensor, Tensor, Tensor],
        edge_index: Tensor,
        edge_attr: Tuple[Tensor, Tensor],
        e_pos: Tensor,
        e_interaction: Tensor,
        alphas: Tensor,
        connection_nodes: Tensor):
        s, v, e = x
        # r 是相对的方向，单位向量
        d = torch.pow(v[edge_index[0]] - v[edge_index[1]], exponent=2).sum(-1).sqrt()
        rel_pos = v[edge_index[0]] - v[edge_index[1]]
        r = F.normalize(rel_pos, dim=-1, eps=1e-6)
        # print(e_interaction)
        ms, mv, me = self.propagate(
            edge_index=edge_index,
            dim_size = s.size(0),
            s=s,
            v=v,
            e=e,
            d=d,
            e_pos=e_pos,
            e_interaction=e_interaction,
            alpha=alphas,
            connection_nodes=connection_nodes
        )
        
        s = s + self.phi_h(torch.cat([s, ms], dim=-1))
        v = v + mv
        e = e + me
        return s, v, e
    
    
class EGCL_Edge_Fast(MessagePassing):
    def __init__(self,
        in_dims: Tuple[int, Optional[int], Optional[int]],
        out_dims: Tuple[int, Optional[int], Optional[int]],
        hid_dims: int,
        cutoff: float,
        eps: float = 1e-6,
        num_radial: int = 1,
        angle_nf: int = 1,
        edges_in_df: int = 0,
        has_v_in: bool = True,
        act_fn = nn.SiLU(),
        attention = False,
        vector_aggr: str = "mean"):
        super(EGCL_Edge_Fast, self).__init__(node_dim=0, aggr=None, flow="source_to_target")
        self.vector_aggr = vector_aggr
        self.in_dims = in_dims
        self.si, self.vi, self.ei = in_dims
        self.out_dims = out_dims
        self.so, self.vo, self.eo = out_dims
        self.has_v_in = has_v_in
        self.attention = attention
        self.phi_m = nn.Sequential(
            nn.Linear(2*self.si+self.ei+num_radial+edges_in_df, hid_dims),
            act_fn,
            nn.Linear(hid_dims, hid_dims)
            )
        layer = nn.Linear(hid_dims, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        self.phi_x = nn.Sequential(
            nn.Linear(hid_dims, hid_dims),
            act_fn,
            layer
        )
        self.phi_h = nn.Sequential(
            nn.Linear(self.si+hid_dims, hid_dims),
            act_fn,
            nn.Linear(hid_dims, self.so)
        )
        self.phi_me = nn.Sequential(
            nn.Linear(self.ei + self.si, hid_dims),
            act_fn,
            nn.Linear(hid_dims, hid_dims)
        )
        # TODO: Improve this to self-attension version
        self.phi_r = nn.Sequential(
            nn.Linear(self.ei, hid_dims),
            act_fn,
            nn.Linear(hid_dims, 1)
        )
        self.phi_re = nn.Sequential(
            nn.Linear(self.ei + hid_dims, hid_dims),
            act_fn,
            nn.Linear(hid_dims, self.eo)
        )

    def aggregate(self, inputs: Tuple[Tensor, Tensor, Tensor, Tensor], index: Tensor, dim_size: Optional[int] = None) -> Tuple[Tensor, Tensor]:
        ms = scatter(inputs[0], index=index, dim=0, dim_size=dim_size, reduce="sum")
        mv = scatter(inputs[1], index=index, dim=0, dim_size=dim_size, reduce="mean")
        me = scatter(inputs[2], index=index, dim=0, dim_size=dim_size, reduce="mean")
        return ms, mv, me
    def message(
            self,
            s_i: Tensor, 
            s_j: Tensor, 
            v_i: Tensor, 
            v_j: Tensor,
            d: Tensor,
            r: Tensor,
            e: Tensor,
        ) -> Tensor:
        # 其实对于EGNN这里还有一个edge attr，但这里省略了，假设所有的edge attr都是1吧
        # 计算ms
        dist = torch.pow(d, 2).unsqueeze(-1)
        a_ij = torch.cat([s_i, s_j, e, dist], dim=-1)
        ms_j = self.phi_m(a_ij)
        rel_pos = v_i - v_j
        # 计算mx
        mv_j = rel_pos * self.phi_x(ms_j)
        # 计算me
        me_j = self.phi_me(torch.cat([e, s_i], dim=-1))
        return ms_j, mv_j, me_j
    def cal_message_direction(self, e, edge_index, r, dim_size):
        #目前这个版本很简陋，没有考虑边之间的相互作用
        m_ri = scatter(r * self.phi_r(e), index=edge_index[1], dim=0, dim_size=dim_size, reduce='mean')
        return m_ri
    def forward(
        self,   
        x: Tuple[Tensor, Tensor, Tensor],
        edge_index: Tensor,
        edge_attr: Tuple[Tensor, Tensor],
        ):
        s, v, e = x
        # r 是相对的方向，单位向量
        d = torch.pow(v[edge_index[0]] - v[edge_index[1]] + 1e-6, exponent=2).sum(-1).sqrt()
        rel_pos = v[edge_index[1]] - v[edge_index[0]]
        r = F.normalize(rel_pos, dim=-1, eps=1e-6)
        m_ri = self.cal_message_direction(e, edge_index, r, dim_size=s.size(0))
        ms, mv, me = self.propagate(
            edge_index=edge_index,
            dim_size = s.size(0),
            s=s,
            v=v,
            e=e,
            d=d,
            r=r,
        )
        
        s = s + self.phi_h(torch.cat([s, ms], dim=-1))
        v = v + mv
        me_direction = m_ri[edge_index[1]]
        me = me[edge_index[1]]
        # print("E:", e.shape, me.shape, edge_index.shape)
        me_in = me * (me_direction * rel_pos).sum(dim=-1, keepdim=True)
        # print(me_in.shape)
        # ttt = torch.cat([e, me_in], dim=-1)
        # print(ttt.shape)
        e = e + self.phi_re(torch.cat([e, me_in], dim=-1))
        return s, v, e
            

if __name__ == '__main__':
    model = EGCL((16,3), (128,128), 64, 1, 6)
    nodes = torch.rand((100,16))
    pos = torch.rand((100,3))
    edge_index = torch.randint(0, 99, size=(2,50))
    rel_pos = pos[edge_index[0]] - pos[edge_index[1]]
    dist = torch.pow(rel_pos, 2).sum(-1).sqrt()
    # print("Pos_j:", pos[edge_index[0]])
    r = F.normalize(rel_pos, dim=-1, eps=1e-6)
    # print((rel_pos-(dist.unsqueeze(-1)*r) < 1e-3).all())
    edge_attr = (dist, r)
    x = (nodes, pos)
    model(x, edge_index, edge_attr)