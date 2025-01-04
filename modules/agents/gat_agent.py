# modules/agents/gat_agent.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleGraphAttentionLayer(nn.Module):
    """最簡易的 GAT Layer (單頭)"""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_fc = nn.Linear(out_dim*2, 1, bias=False)

    def forward(self, x, edge_features):
        """
        x: (bs, N, in_dim) node features
        edge_features: (bs, N, N, e_dim) 這裡若 e_dim=1 => distance
        """
        bs, N, _ = x.shape
        h = self.fc(x)  # shape=(bs, N, out_dim)
        
        # 每對 (i,j) => concat [h_i, h_j], 透過 attn_fc -> attention logits
        # shape=(bs, N, N, out_dim), broadcast
        h_i = h.unsqueeze(2).expand(bs, N, N, h.size(-1))
        h_j = h.unsqueeze(1).expand(bs, N, N, h.size(-1))
        alpha_ij_in = torch.cat([h_i, h_j], dim=-1)  # (bs, N, N, 2*out_dim)
        
        e_ij = self.attn_fc(alpha_ij_in).squeeze(-1)  # (bs, N, N)
        # 你可以把 edge_features 也加進 e_ij => e_ij += some_fn(edge_features)
        # or  e_ij = e_ij + w * f(edge_features), etc.

        # attention
        alpha_ij = F.softmax(e_ij, dim=-1)  # (bs, N, N)

        # aggregate
        # shape=(bs, N, out_dim)
        h_prime = torch.einsum("bijn,bin->bjn", alpha_ij.unsqueeze(-1), h_j)

        return h_prime  # shape=(bs, N, out_dim)

class SimpleGAT(nn.Module):
    """示範: 單層 GAT + optional MLP"""
    def __init__(self, node_input_dim, edge_input_dim, hidden_dim, n_heads=1):
        super().__init__()
        self.n_heads = n_heads
        self.heads = nn.ModuleList()
        for _ in range(n_heads):
            self.heads.append(SimpleGraphAttentionLayer(node_input_dim, hidden_dim))

        # 如果要 multi-head concat => out_dim = hidden_dim * n_heads
        # 這裡示範 sum up
        self.merge = nn.Linear(hidden_dim, hidden_dim)  # optional

    def forward(self, node_features, edge_features):
        # node_features: (bs, N, node_input_dim)
        # edge_features: (bs, N, N, edge_input_dim)
        outs = []
        for head in self.heads:
            out = head(node_features, edge_features)  # (bs, N, hidden_dim)
            outs.append(out)

        # merge heads
        # e.g. sum up or concat
        out_sum = torch.stack(outs, dim=0).sum(dim=0)  # shape=(bs,N,hidden_dim)
        emb = self.merge(out_sum)
        return emb
