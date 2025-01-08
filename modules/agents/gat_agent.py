import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleGraphAttentionLayer(nn.Module):
    """示範：單頭 GAT layer"""
    def __init__(self, in_dim, out_dim, edge_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        # attention: concat(hi,hj, e_ij) => alpha
        self.attn_fc = nn.Linear(out_dim*2 + edge_dim, 1, bias=False)

    def forward(self, x, edge_feats):
        """
        x: (bs, N, in_dim)
        edge_feats: (bs, N, N, edge_dim)
        return: (bs, N, out_dim)
        """
        bs, N, _ = x.shape
        h = self.fc(x)  # (bs, N, out_dim)

        # broadcast
        h_i = h.unsqueeze(2).expand(bs, N, N, h.size(-1))  # (bs, N, N, out_dim)
        h_j = h.unsqueeze(1).expand(bs, N, N, h.size(-1))  # (bs, N, N, out_dim)
        e_ij_input = torch.cat([h_i, h_j, edge_feats], dim=-1)  # (bs, N, N, 2*out_dim + edge_dim)
        e_ij = self.attn_fc(e_ij_input).squeeze(-1)             # (bs, N, N)

        alpha_ij = F.softmax(e_ij, dim=-1)                      # (bs, N, N)
        alpha_ij = alpha_ij.unsqueeze(-1)                       # (bs, N, N, 1)

        # aggregate
        h_prime = torch.einsum("bijn,bijn->bin", alpha_ij, h_j)  # (bs, N, out_dim)
        assert not torch.isnan(h_prime).any(), "h_prime contains NaN"
        return h_prime

class GATAgent(nn.Module):
    """
    GATAgent: 
      - 1 GAT layer => node_embeds
      - output either Q or pi logits
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.node_in_dim = args.gat_node_input_dim   # 9
        self.edge_in_dim = args.gat_edge_input_dim   # 3
        self.hidden_dim  = args.gat_hidden_dim       # e.g. 32
        self.n_actions   = args.n_actions

        self.gat_layer = SimpleGraphAttentionLayer(
            in_dim=self.node_in_dim,
            out_dim=self.hidden_dim,
            edge_dim=self.edge_in_dim
        )

        # Q-head or Policy-head
        self.q_head  = nn.Linear(self.hidden_dim, self.n_actions)
        self.pi_head = nn.Linear(self.hidden_dim, self.n_actions)

    def forward(self, node_feats, edge_feats):
        """
        node_feats: (bs, n_nodes, node_in_dim)
        edge_feats: (bs, n_nodes, n_nodes, edge_in_dim)
        return: (bs, n_nodes, hidden_dim)
        """
        return self.gat_layer(node_feats, edge_feats)

    def init_hidden(self):
        """
        For compatibility with BasicMAC's init_hidden();
        GAT 不用 RNN hidden => return a zero
        """
        return torch.zeros(1, self.hidden_dim)