#gat_agent.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SimpleGraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, edge_dim, dropout=0.1):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_fc = nn.Linear(out_dim * 2 + edge_dim, 1, bias=False)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(out_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.xavier_uniform_(self.attn_fc.weight)

    def forward(self, x, edge_feats):
        """
        x: (bs, N, in_dim)
        edge_feats: (bs, N, N, edge_dim)
        return: (bs, N, out_dim)
        """
        bs, N, _ = x.shape
        h = self.fc(x)  # (bs, N, out_dim)

        # 檢查 h 是否有非法值
        if torch.isnan(h).any() or torch.isinf(h).any():
            raise ValueError("NaN or Inf detected in h after linear layer")

        # 矩陣運算避免顯式廣播
        h_i = h.unsqueeze(2)  # (bs, N, 1, out_dim)
        h_j = h.unsqueeze(1)  # (bs, 1, N, out_dim)
        e_ij_input = torch.cat([
            h_i.expand(-1, -1, N, -1), 
            h_j.expand(-1, N, -1, -1), 
            edge_feats
        ], dim=-1)  # (bs, N, N, 2*out_dim + edge_dim)

        # 檢查 e_ij_input 是否有非法值
        if torch.isnan(e_ij_input).any() or torch.isinf(e_ij_input).any():
            raise ValueError("NaN or Inf detected in e_ij_input")

        e_ij = self.attn_fc(e_ij_input).squeeze(-1) / math.sqrt(self.fc.out_features)  # (bs, N, N)
        e_ij = e_ij - e_ij.max(dim=-1, keepdim=True)[0]  # Stability improvement

        # 檢查 e_ij 是否有非法值或極端值
        if torch.isnan(e_ij).any() or torch.isinf(e_ij).any():
            raise ValueError("NaN or Inf detected in e_ij after attention FC layer")
        if (e_ij.abs() > 1e6).any():
            raise ValueError("Extreme values detected in e_ij")

        # 計算注意力權重
        alpha_ij = F.softmax(e_ij, dim=-1)  # (bs, N, N)
        if torch.isnan(alpha_ij).any():
            raise ValueError("NaN detected in alpha_ij after softmax")

        alpha_ij = self.dropout(alpha_ij) if self.training else alpha_ij
        alpha_ij = alpha_ij.unsqueeze(-1)  # (bs, N, N, 1)

        # 聚合
        h_prime = torch.sum(alpha_ij * h_j, dim=2)  # (bs, N, out_dim)
        h_prime = self.layer_norm(h_prime)

        # 檢查 h_prime 是否有非法值
        if torch.isnan(h_prime).any() or torch.isinf(h_prime).any():
            raise ValueError("NaN or Inf detected in h_prime after layer normalization")

        return h_prime
class GATAgent(nn.Module):
    """
    GATAgent: 
      - 1 GAT layer => node_embeds
      - Output either Q or pi logits
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.node_in_dim = args.gat_node_input_dim   # 例如: 9
        self.edge_in_dim = args.gat_edge_input_dim   # 例如: 3
        self.hidden_dim  = args.gat_hidden_dim       # 例如: 32
        self.n_actions   = args.n_actions
        self.dropout_prob = args.dropout_prob if hasattr(args, 'dropout_prob') else 0.1  # 默認 Dropout 比例為 0.1

        self.gat_layer = SimpleGraphAttentionLayer(
            in_dim=self.node_in_dim,
            out_dim=self.hidden_dim,
            edge_dim=self.edge_in_dim
        )

        # Q-head 或 Policy-head
        self.q_head  = nn.Linear(self.hidden_dim, self.n_actions)
        self.pi_head = nn.Linear(self.hidden_dim, self.n_actions)
        self.reset_parameters()

        # 定義 Dropout 層
        self.dropout = nn.Dropout(p=self.dropout_prob)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.q_head.weight)
        nn.init.xavier_uniform_(self.pi_head.weight)

    def forward(self, node_feats, edge_feats):
        # 檢查輸入是否有非法值
        assert not torch.isnan(node_feats).any(), "node_feats contains NaN"
        assert not torch.isinf(node_feats).any(), "node_feats contains Inf"
        assert not torch.isnan(edge_feats).any(), "edge_feats contains NaN"
        assert not torch.isinf(edge_feats).any(), "edge_feats contains Inf"

        # 標準化
        node_feats = (node_feats - node_feats.mean(dim=-1, keepdim=True)) / (node_feats.std(dim=-1, keepdim=True) + 1e-8)
        edge_feats = (edge_feats - edge_feats.mean(dim=-1, keepdim=True)) / (edge_feats.std(dim=-1, keepdim=True) + 1e-8)

        # GAT 層處理
        node_embeds = self.gat_layer(node_feats, edge_feats)
        node_embeds = F.relu(node_embeds)          # 使用 ReLU 激活函數
        node_embeds = self.dropout(node_embeds)    # 應用 Dropout

        return node_embeds

    def init_hidden(self):
        """
        For compatibility with BasicMAC's init_hidden();
        GAT 不用 RNN hidden => return a zero
        """
        return torch.zeros(1, self.hidden_dim)