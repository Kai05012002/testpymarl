# modules/agents/gat_agent.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleGraphAttentionLayer(nn.Module):
    """示範：單頭 GAT layer，可自行擴充多頭"""
    def __init__(self, in_dim, out_dim, edge_dim=3):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_fc = nn.Linear(out_dim*2 + edge_dim, 1, bias=False)

    def forward(self, x, edge_feats):
        """
        x: (bs, N, in_dim)
        edge_feats: (bs, N, N, edge_dim)
        回傳: (bs, N, out_dim)
        """
        bs, N, _ = x.shape
        # transform
        h = self.fc(x)  # (bs, N, out_dim)

        # broadcast
        h_i = h.unsqueeze(2).expand(bs, N, N, h.size(-1))  # (bs, N, N, out_dim)
        h_j = h.unsqueeze(1).expand(bs, N, N, h.size(-1))
        e_ij_input = torch.cat([h_i, h_j, edge_feats], dim=-1)  # (bs, N, N, out_dim*2 + edge_dim)
        e_ij = self.attn_fc(e_ij_input).squeeze(-1)  # (bs, N, N)

        alpha_ij = F.softmax(e_ij, dim=-1)  # (bs, N, N)
        # aggregate
        # h_j shape=(bs, N, N, out_dim)
        # alpha_ij shape=(bs, N, N)
        # => out shape=(bs, N, out_dim)
        h_prime = torch.einsum('bijn,bjn->bin', alpha_ij.unsqueeze(-1), h_j)
        return h_prime


class GATAgent(nn.Module):
    """
    只負責一次 GAT forwarding (一層或多層) + Q-head
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        # 你可從 args.gat_node_input_dim, gat_edge_input_dim, gat_hidden_dim 來配置
        self.node_in_dim = getattr(args, "gat_node_input_dim", 9)
        self.edge_in_dim = getattr(args, "gat_edge_input_dim", 3)
        self.hidden_dim = getattr(args, "gat_hidden_dim", 32)

        # GAT layer
        self.gat_layer = SimpleGraphAttentionLayer(self.node_in_dim, self.hidden_dim, edge_dim=self.edge_in_dim)

        # Q-head: (hidden_dim -> n_actions)
        self.q_head = nn.Linear(self.hidden_dim, args.n_actions)

    def forward(self, node_feats, edge_feats):
        """
        node_feats: (bs, n_nodes, node_in_dim)
        edge_feats: (bs, n_nodes, n_nodes, edge_in_dim)
        return: (bs, n_nodes, hidden_dim)
        """
        h_prime = self.gat_layer(node_feats, edge_feats)
        return h_prime

    def init_hidden(self):
        """GAT 不需要 RNN hidden，但為了兼容BasicMAC的介面，可回傳一個零向量"""
        # 這個 hidden 不會實際被用到
        return torch.zeros(1, self.hidden_dim)
