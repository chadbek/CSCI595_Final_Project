from __future__ import annotations
import copy
from enum import Enum
 
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data


class GraphMethod(str, Enum):
    STATIC          = 'static'
    FULLY_CONNECTED = 'fully_connected'
    REWEIGHT        = 'reweight'
    AUGMENT         = 'augment'
    PRUNE           = 'prune'


class MLP(nn.Module):
    def __init__(self, 
                 d_in: int, 
                 d_hidden: int, 
                 d_out: int, 
                 use_layer_norm: bool = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(d_hidden, d_out),
        )
        self.norm = nn.LayerNorm(d_out) if use_layer_norm else nn.Identity()

    def forward(self, x: torch.Tensor):
        return self.norm(self.net(x))


def _broadcast_q_to_edges(q: torch.Tensor, 
                          edge_index: torch.Tensor) -> torch.Tensor:
    E = edge_index.size(1)
    if q.dim() == 1:
        return q.unsqueeze(0).expand(E, -1)          # [E, d_q]
    raise ValueError(
        "Pass a 1-D question vector (single graph) or index q[batch_of_edge] "
        "before calling _broadcast_q_to_edges."
    )


class EdgeReweighting(nn.Module):
    
    def __init__(self,
                 d_h: int,
                 d_e: int,
                 d_q: int,
                 d_hidden: int,
                 global_softmax: bool = False):
        super().__init__()
        self.global_softmax = global_softmax

        # MLP_gate: conditioned on node pair + question → scalar gate
        self.mlp_gate = MLP(
            d_in=d_h + d_h + d_q,
            d_hidden=d_hidden,
            d_out=1,
            use_layer_norm=False,
        )

        # MLP_score: conditioned on node pair + edge feature → scalar score
        self.mlp_score = MLP(
            d_in=d_h + d_h + d_e,
            d_hidden=d_hidden,
            d_out=1,
            use_layer_norm=False,
        )

    def forward(self, 
                h: torch.Tensor,
                e: torch.Tensor,
                q: torch.Tensor,
                edge_index: torch.Tensor,edge_batch: torch.Tensor,):
        src, dst = edge_index[0], edge_index[1] # [E]
        h_i = h[src] # [E, d_h]
        h_j = h[dst] # [E, d_h]
        q_e = q[edge_batch]

        gate_input = torch.cat([h_i, h_j, q_e], dim=-1) # [E, d_h+d_h+d_q]
        gate = torch.sigmoid(self.mlp_gate(gate_input)).squeeze(-1) # [E]

        score_input = torch.cat([h_i, h_j, e], dim=-1) # [E, d_h+d_h+d_e]
        score = self.mlp_score(score_input).squeeze(-1) # [E]

        raw = gate * score # [E]

        if self.global_softmax:
            edge_weight = torch.softmax(raw, dim=0)
        else:
            edge_weight = self._dst_softmax(raw, dst, num_nodes=h.size(0))

        return edge_weight # [E]

    @staticmethod
    def _dst_softmax(raw: torch.Tensor,
                     dst: torch.Tensor,
                     num_nodes: int,) -> torch.Tensor:
        max_vals = torch.full((num_nodes,), float('-inf'), device=raw.device)
        max_vals.scatter_reduce_(0, dst, raw, reduce='amax', include_self=True)
        shifted  = raw - max_vals[dst]

        exp_vals = shifted.exp()
        sum_exp  = torch.zeros(num_nodes, device=raw.device)
        sum_exp.scatter_add_(0, dst, exp_vals)
        denom    = sum_exp[dst].clamp(min=1e-9)

        return exp_vals / denom


class EdgeAugmentation(nn.Module):
    
    def __init__(self, 
                 d_h: int, 
                 d_q: int, 
                 d_hidden: int, 
                 top_k: int = 5):
        super().__init__()
        self.top_k = top_k
        self.mlp   = MLP(d_h + d_h + d_q, d_hidden, 1)
 
    def forward(self,
                h: torch.Tensor,
                q: torch.Tensor,
                edge_index: torch.Tensor,
                node_batch: torch.Tensor,):
        device = h.device
        B = q.size(0)
        new_srcs, new_dsts = [], []
 
        for g in range(B):
            node_mask    = node_batch == g
            node_indices = node_mask.nonzero(as_tuple=True)[0]
            N_g = node_indices.size(0)
            if N_g < 2:
                continue
 
            # existing edges in graph g
            edge_g_mask = node_batch[edge_index[0]] == g
            ex_src = edge_index[0][edge_g_mask]
            ex_dst = edge_index[1][edge_g_mask]
            existing = set(zip(ex_src.tolist(), ex_dst.tolist()))
 
            # all ordered non-self pairs in graph g
            ii, jj = torch.meshgrid(node_indices, node_indices, indexing='ij')
            cand_src = ii.reshape(-1)
            cand_dst = jj.reshape(-1)
            keep = cand_src != cand_dst
            cand_src = cand_src[keep]
            cand_dst = cand_dst[keep]
 
            # score
            h_i = h[cand_src]
            h_j = h[cand_dst]
            q_g = q[g].unsqueeze(0).expand(h_i.size(0), -1)
            scores = self.mlp(torch.cat([h_i, h_j, q_g], dim=-1)).squeeze(-1)
 
            # mask existing edges
            exist_mask = torch.tensor(
                [(s.item(), d.item()) in existing
                 for s, d in zip(cand_src, cand_dst)],
                dtype=torch.bool, device=device,
            )
            scores = scores.masked_fill(exist_mask, float('-inf'))
 
            k = min(self.top_k, int((scores != float('-inf')).sum().item()))
            if k > 0:
                _, top_idx = scores.topk(k)
                new_srcs.append(cand_src[top_idx])
                new_dsts.append(cand_dst[top_idx])
 
        if new_srcs:
            new_edges = torch.stack([torch.cat(new_srcs), torch.cat(new_dsts)], dim=0)
            aug_edge_index = torch.cat([edge_index, new_edges], dim=1)
            added_count = new_edges.size(1)
        else:
            aug_edge_index = edge_index
            added_count = 0
 
        return aug_edge_index, added_count


class NeighborPruning(nn.Module):

    def __init__(self, 
                 d_h: int, 
                 d_q: int, 
                 d_hidden: int, 
                 top_k: int = 3,
                 keep_self_loops: bool = True):
        super().__init__()
        self.top_k = top_k
        self.keep_self_loops = keep_self_loops
        self.mlp = MLP(d_h + d_h + d_q, d_hidden, 1)
 
    def forward(self,
                h: torch.Tensor,
                q: torch.Tensor,
                edge_index: torch.Tensor,
                edge_batch: torch.Tensor,):
        src, dst = edge_index[0], edge_index[1]
        E = edge_index.size(1)
        device = h.device
 
        # separate self-loops from regular edges
        if self.keep_self_loops:
            self_mask = src == dst
            reg_mask = ~self_mask
            self_cols = self_mask.nonzero(as_tuple=True)[0]
            regular_cols = reg_mask.nonzero(as_tuple=True)[0]
        else:
            regular_cols = torch.arange(E, device=device)
            self_cols = torch.tensor([], dtype=torch.long, device=device)
 
        if regular_cols.numel() == 0:
            return edge_index, torch.arange(E, device=device)
 
        reg_src = src[regular_cols]
        reg_dst = dst[regular_cols]
        reg_batch = edge_batch[regular_cols]
 
        # score regular edges
        h_i = h[reg_src]
        h_j = h[reg_dst]
        q_e = q[reg_batch]
        scores = self.mlp(torch.cat([h_i, h_j, q_e], dim=-1)).squeeze(-1)

        # top-k per destination node
        keep_mask = torch.zeros(regular_cols.size(0), dtype=torch.bool, device=device)
        unique_dsts = reg_dst.unique()
 
        for node in unique_dsts:
            node_mask = reg_dst == node
            node_scores = scores.clone()
            node_scores[~node_mask] = float('-inf')
            k = min(self.top_k, int(node_mask.sum().item()))
            _, top_idx = node_scores.topk(k)
            keep_mask[top_idx] = True
 
        kept_regular = regular_cols[keep_mask]
 
        all_kept, _ = torch.cat([self_cols, kept_regular]).sort()
        return edge_index[:, all_kept], all_kept


class QuestionConditionedGraphBuilder(nn.Module):
 
    def __init__(self,
                 method: str,
                 d_h: int,
                 d_e: int,
                 d_q: int,
                 d_hidden: int = 256,
                 top_k: int = 5,):

        super().__init__()
        self.method = GraphMethod(method)
 
        if self.method == GraphMethod.REWEIGHT:
            self.builder = EdgeReweighting(d_h, d_e, d_q, d_hidden)
        elif self.method == GraphMethod.AUGMENT:
            self.builder = EdgeAugmentation(d_h, d_q, d_hidden, top_k)
        elif self.method == GraphMethod.PRUNE:
            self.builder = NeighborPruning(d_h, d_q, d_hidden, top_k)
        else:
            raise ValueError(
                f"QuestionConditionedGraphBuilder requires a learned method; "
                f"got '{method}'. Use 'reweight', 'augment', or 'prune'."
            )

 
def apply_question_conditioned_graph_batched(gt_scene_graphs,
                                             h: torch.Tensor,
                                             e: torch.Tensor,
                                             q: torch.Tensor,
                                             builder: QuestionConditionedGraphBuilder):
    edge_index = gt_scene_graphs.edge_index # [2, E]
    node_batch = gt_scene_graphs.batch # [N_total]
    edge_batch = node_batch[edge_index[0]] # [E]
 
    method = builder.method
 
    if method == GraphMethod.REWEIGHT:
        edge_weight = builder.builder(
            h=h, e=e, q=q,
            edge_index=edge_index,
            edge_batch=edge_batch,
        )
        e_out = e * edge_weight.unsqueeze(-1)
        return gt_scene_graphs, h, e_out
 
    elif method == GraphMethod.AUGMENT:
        aug_edge_index, added_count = builder.builder(
            h=h, q=q,
            edge_index=edge_index,
            node_batch=node_batch,
        )
        if added_count > 0:
            pad   = torch.zeros(added_count, e.size(-1), dtype=e.dtype, device=e.device)
            e_out = torch.cat([e, pad], dim=0)
        else:
            e_out = e
        sg_out = _copy_with_new_edge_index(gt_scene_graphs, aug_edge_index)
        return sg_out, h, e_out
 
    elif method == GraphMethod.PRUNE:
        pruned_edge_index, kept_cols = builder.builder(
            h=h, q=q,
            edge_index=edge_index,
            edge_batch=edge_batch,
        )
        e_out  = e[kept_cols]
        sg_out = _copy_with_new_edge_index(gt_scene_graphs, pruned_edge_index)
        return sg_out, h, e_out
 
    else:
        raise ValueError(f"Unknown method in apply: {method}")


def _copy_with_new_edge_index(batch_data, new_edge_index):
    sg_out = copy.copy(batch_data)
    sg_out.edge_index = new_edge_index
    sg_out.added_sym_edge = torch.tensor(
        [], dtype=torch.long, device=new_edge_index.device
    )
    return sg_out

def build_static_scene_graph(x: torch.Tensor,
                             sg_this: dict,
                             objIDs: list,
                             map_objID_to_node_idx: dict,
                             SG_ENCODING_TEXT,graph_method: str,) -> Data:
    if GraphMethod(graph_method) == GraphMethod.FULLY_CONNECTED:
        return _build_fully_connected(x, len(objIDs), SG_ENCODING_TEXT)
 
    # Original symmetric scene-graph edges
    edge_feature_list = []
    edge_topology_list = []
    added_sym_edge_list = []
 
    from_to_set: set[tuple[int, int]] = set()
    for node_idx, objId in enumerate(objIDs):
        for rel in sg_this['objects'][objId]['relations']:
            from_to_set.add((node_idx, map_objID_to_node_idx[rel['object']]))
 
    for node_idx, objId in enumerate(objIDs):
        obj = sg_this['objects'][objId]
 
        edge_topology_list.append([node_idx, node_idx])
        edge_feature_list.append(
            np.array([SG_ENCODING_TEXT.vocab.stoi['<self>']], dtype=np.int_)
        )
 
        for rel in obj['relations']:
            dst_idx = map_objID_to_node_idx[rel['object']]
            edge_tok = np.array(
                [SG_ENCODING_TEXT.vocab.stoi[rel['name']]], dtype=np.int_
            )
            edge_topology_list.append([node_idx, dst_idx])
            edge_feature_list.append(edge_tok)
 
            if (dst_idx, node_idx) not in from_to_set:
                edge_topology_list.append([dst_idx, node_idx])
                edge_feature_list.append(edge_tok)
                added_sym_edge_list.append(len(edge_feature_list) - 1)
 
    edge_index = torch.tensor(edge_topology_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.from_numpy(np.stack(edge_feature_list, axis=0)).long()
    datum = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    datum.added_sym_edge = torch.LongTensor(added_sym_edge_list)
    return datum
 
 
def _build_fully_connected(x, N, SG_ENCODING_TEXT, include_self_loops=True):
    topo, feat = [], []
    self_tok = np.array([SG_ENCODING_TEXT.vocab.stoi['<self>']], dtype=np.int_)
    unk_tok = np.array([SG_ENCODING_TEXT.vocab.stoi.get('<UNK>', 0)], dtype=np.int_)
    for i in range(N):
        for j in range(N):
            if i == j and not include_self_loops:
                continue
            topo.append([i, j])
            feat.append(self_tok if i == j else unk_tok)
    edge_index = torch.tensor(topo, dtype=torch.long).t().contiguous()
    edge_attr = torch.from_numpy(np.stack(feat, axis=0)).long()
    datum = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    datum.added_sym_edge = torch.LongTensor([])
    return datum


if __name__ == '__main__':
    torch.manual_seed(0)

    N, E   = 10, 20
    d_h, d_e, d_q, d_hidden = 64, 32, 128, 64

    h = torch.randn(N, d_h)
    e = torch.randn(E, d_e)
    q = torch.randn(d_q)
    edge_index = torch.randint(0, N, (2, E))

    print("=" * 60)
    print("Method 1 – EdgeReweighting")
    print("=" * 60)
    reweighter  = EdgeReweighting(d_h, d_e, d_q, d_hidden)
    edge_weight = reweighter(h, e, q, edge_index)
    print(f"  Input  edges : {edge_index.shape}")
    print(f"  Output weight: {edge_weight.shape}  (should be [{E}])")
    print(f"  Weight sample: {edge_weight[:5].detach().round(decimals=4)}")
    assert edge_weight.shape == (E,)
    print("  PASSED\n")

    print("=" * 60)
    print("Method 2 – EdgeAugmentation  (top_k=4)")
    print("=" * 60)
    augmenter      = EdgeAugmentation(d_h, d_q, d_hidden, top_k=4)
    aug_edge_index = augmenter(h, q, edge_index, num_nodes=N)
    print(f"  Input  edges : {edge_index.shape}")
    print(f"  Output edges : {aug_edge_index.shape}  (should have ≥ {E} cols)")
    assert aug_edge_index.shape[0] == 2
    assert aug_edge_index.shape[1] >= E
    print("  PASSED\n")

    print("=" * 60)
    print("Method 3 – NeighborPruning  (top_k=2)")
    print("=" * 60)
    pruner             = NeighborPruning(d_h, d_q, d_hidden, top_k=2)
    pruned_edge_index  = pruner(h, q, edge_index)
    print(f"  Input  edges : {edge_index.shape}")
    print(f"  Output edges : {pruned_edge_index.shape}  (should have ≤ {E} cols)")
    assert pruned_edge_index.shape[0] == 2
    assert pruned_edge_index.shape[1] <= E
    print("  PASSED\n")

    print("=" * 60)
    print("QuestionConditionedGraphBuilder wrapper")
    print("=" * 60)
    for m in ('reweight', 'augment', 'prune'):
        kwargs = dict(d_h=d_h, d_q=d_q, d_hidden=d_hidden)
        if m == 'reweight':
            kwargs['d_e'] = d_e
        builder = QuestionConditionedGraphBuilder(m, **kwargs)
        extra   = {'e': e} if m == 'reweight' else {}
        out     = builder(h, q, edge_index, **extra)
        print(f"  method={m}  output shape={out.shape}")
    print("  ALL PASSED")