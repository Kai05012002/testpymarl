# modules/agents/enhanced_gat_agent.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhancedGraphLayer(nn.Module):
    """
    實作「加強版 GAT Layer」： (X_l, E_l) -> (X_{l+1}, E_{l+1})
    參考你的示意圖: W3, W4, W5, W6, Hadamard product, residual, LN...
    """
    def __init__(self, node_in_dim, edge_in_dim, node_out_dim, edge_out_dim):
        super(EnhancedGraphLayer, self).__init__()
        
        # ---- 一、線性轉換 (對應 W3, W4, W5, W6) ----
        # 1) W3 : node_in_dim -> node_out_dim
        self.W3 = nn.Linear(node_in_dim, node_out_dim, bias=False)
        # 2) W4 : edge_in_dim -> edge_out_dim
        self.W4 = nn.Linear(edge_in_dim, edge_out_dim, bias=False)
        # 3) W5 : node_in_dim -> node_out_dim (global proj)
        self.W5 = nn.Linear(node_in_dim, node_out_dim, bias=False)
        # 4) W6 : node_out_dim -> node_out_dim (最後融合 local + global)
        self.W6 = nn.Linear(node_out_dim, node_out_dim, bias=False)

        # ---- 二、注意力打分 ----
        # concat( h_i, h_j, e_ij ) -> scalar
        #  (node_out_dim + node_out_dim + edge_out_dim) -> 1
        self.attn_fc = nn.Linear(node_out_dim * 2 + edge_out_dim, 1, bias=False)

        # ---- 三、LayerNorm & Dropout 等 ----
        self.ln_node = nn.LayerNorm(node_out_dim)
        self.ln_edge = nn.LayerNorm(edge_out_dim)
        self.dropout = nn.Dropout(p=0.1)  # 可自行調整 0.1, 0.0

    def forward(self, X_l, E_l):
        """
        X_l: (bs, n, node_in_dim)
        E_l: (bs, n, n, edge_in_dim)
        return:
          X_{l+1}: (bs, n, node_out_dim)
          E_{l+1}: (bs, n, n, edge_out_dim)
        """
        bs, n, _ = X_l.shape

        # 1) Node投影 => local_proj
        local_proj = self.W3(X_l)        # (bs,n,node_out_dim)

        # 2) Edge投影 => E_proj
        E_proj = self.W4(E_l)           # (bs,n,n,edge_out_dim)

        # 3) Global投影 => global_proj
        global_proj = self.W5(X_l)      # (bs,n,node_out_dim)

        # -- 準備做注意力: e_ij = f( h_i, h_j, E_ij ) --
        h_i = local_proj.unsqueeze(2).expand(bs, n, n, -1)  # (bs,n,n,node_out_dim)
        h_j = local_proj.unsqueeze(1).expand(bs, n, n, -1)  # (bs,n,n,node_out_dim)

        attn_in = torch.cat([h_i, h_j, E_proj], dim=-1)     # (bs,n,n, 2*node_out_dim+edge_out_dim)
        e_ij = self.attn_fc(attn_in).squeeze(-1)            # (bs,n,n)

        # -- softmax --
        d_k = local_proj.size(-1)  # node_out_dim
        e_ij = e_ij / math.sqrt(d_k)     # scale
        # 減去行內最大值 => 避免溢出
        e_ij = e_ij - e_ij.max(dim=-1, keepdim=True)[0]  # (bs,n,n)
        alpha_ij = torch.softmax(e_ij, dim=-1)           # (bs,n,n)
        alpha_ij = self.dropout(alpha_ij).unsqueeze(-1)  # (bs,n,n,1)

        # -- (5) local_info = sum_j [ alpha_ij * (h_j + E_proj ) ]
        local_info = torch.sum(alpha_ij * (h_j + E_proj), dim=2)  # (bs,n,node_out_dim)

        # -- (6) global 先做 mean-pool => hadamard => W6
        global_mean = global_proj.mean(dim=1, keepdim=True)  # (bs,1,node_out_dim)
        global_mean = global_mean.expand(-1, n, -1)          # (bs,n,node_out_dim)
        fused = local_info * global_mean                     # hadamard product

        fused2 = self.W6(fused)  # (bs,n,node_out_dim)

        # -- (7) 殘差 + LayerNorm + ReLU
        X_res = local_proj
        X_next = self.ln_node(fused2 + X_res)
        X_next = torch.relu(X_next)  # (bs,n,node_out_dim)

        # -- (8) 更新 E_{l+1}, 例如: E_new = E_proj + sum_j alpha_ij * X_j_next
        X_j_next = X_next.unsqueeze(1).expand(bs, n, n, -1)   # (bs,n,n,node_out_dim)
        node_interact = alpha_ij * X_j_next                   # (bs,n,n,node_out_dim)
        node_interact_sum = node_interact.sum(dim=2)          # (bs,n,node_out_dim)
        # broadcast 回 (bs,n,n,node_out_dim)
        node_interact_sum = node_interact_sum.unsqueeze(2).expand(bs, n, n, -1)

        E_new = E_proj + node_interact_sum  # simpler residual
        E_next = self.ln_edge(E_new)
        E_next = torch.relu(E_next)

        return X_next, E_next

class EnhancedGATAgent(nn.Module):
    """
    多層Enhanced GraphLayer + Q-head (for DQN / Q-learning).
    """
    def __init__(self, args):
        super(EnhancedGATAgent, self).__init__()
        self.args = args
        self.n_actions = args.n_actions

        # GAT input dims
        self.node_in_dim = args.gat_node_input_dim  # e.g. 9
        self.edge_in_dim = args.gat_edge_input_dim  # e.g. 3
        self.hidden_dim  = args.gat_hidden_dim      # e.g. 32

        # -- 建立多層 GAT Layer (這裡示範 2 層) --
        self.layer1 = EnhancedGraphLayer(
            node_in_dim=self.node_in_dim,
            edge_in_dim=self.edge_in_dim,
            node_out_dim=self.hidden_dim,
            edge_out_dim=self.hidden_dim
        )
        self.layer2 = EnhancedGraphLayer(
            node_in_dim=self.hidden_dim,
            edge_in_dim=self.hidden_dim,
            node_out_dim=self.hidden_dim,
            edge_out_dim=self.hidden_dim
        )

        # -- Q-head --
        # 輸出 dimension = n_actions
        self.q_head = nn.Linear(self.hidden_dim, self.n_actions)

    def forward(self, node_feats, edge_feats):
        """
        node_feats: (bs, n, node_in_dim)
        edge_feats: (bs, n, n, edge_in_dim)
        return => (bs, n, hidden_dim)
        """
        # layer1
        x1, e1 = self.layer1(node_feats, edge_feats)
        # layer2
        x2, e2 = self.layer2(x1, e1)
        # x2 => (bs, n, hidden_dim)
        return x2

    def init_hidden(self):
        """
        為了兼容 BasicMAC 介面 (雖然 GAT 無 RNN)，
        這裡給個 0 Tensor
        """
        device = next(self.parameters()).device
        return torch.zeros(1, self.hidden_dim, device=device)
