#gat_mac.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .basic_controller import BasicMAC
from modules.agents.gat_agent import GATAgent

class GATMAC(BasicMAC):
    def __init__(self, scheme, groups, args):
        super().__init__(scheme, groups, args)
        # BasicMAC.__init__ 會呼叫 self._build_agents(self.input_shape)
        # 但我們想用 GATAgent, 所以要 override:

    def _build_agents(self, input_shape):
        # 替換成我們自定義的 GATAgent，而非 RNNAgent
        self.agent = GATAgent(self.args)

    def forward(self, ep_batch, t, test_mode=False):
        """
        1) 取出 state => reshape => (bs, 6, 9) = node_feats
        2) 產生 edge_feats => (bs, 6, 6, 3)
        3) GAT => (bs, 6, hidden_dim)
        4) slice 我方 n_agents => shape=(bs, n_agents, hidden_dim)
        5) 通過 pi_head or q_head => shape=(bs, n_agents, n_actions)
        6) 如果是 "pi_logits"，再做 softmax
        """
        bs = ep_batch.batch_size
        state = ep_batch["state"][:, t]   # shape = (bs, 54)
        print("state shape:", state.shape)  # 應為 (bs, 54)

        node_feats = state.view(bs, 6, 9)  # (bs, 6, 9)
        print("node_feats shape:", node_feats.shape)  # 應為 (bs, 6, 9)
        assert not torch.isnan(node_feats).any(), "node_feats contains NaN"

        # Step 2) 產生 edge_feats
        edge_feats = self._build_edge_features(node_feats)  # => (bs, 6, 6, 3)
        assert not torch.isnan(edge_feats).any(), "edge_feats contains NaN"

        # Step 3) forward GAT => (bs, 6, hidden_dim)
        node_embeds = self.agent(node_feats, edge_feats)  # shape=(bs, 6, hidden_dim)
        assert not torch.isnan(node_embeds).any(), "node_embeds contains NaN"

        # Step 4) slice 出前 n_agents (例如 3) => (bs, 3, hidden_dim)
        n_agents = self.args.n_agents
        ally_embeds = node_embeds[:, :n_agents, :]  # (bs, 3, hidden_dim)
        assert not torch.isnan(ally_embeds).any(), "ally_embeds contains NaN"

        # Step 5) 走 pi_head 或 q_head
        if self.args.agent_output_type == "pi_logits":
            logits = self.agent.pi_head(ally_embeds)   # => (bs, 3, n_actions)
            assert not torch.isnan(logits).any(), "logits contains NaN"
            # Step 6) softmax
            agent_outs = F.softmax(logits, dim=-1)     # => (bs, 3, n_actions)
            assert not torch.isnan(agent_outs).any(), "agent_outs contains NaN"
        else:
            q_vals = self.agent.q_head(ally_embeds)    # => (bs, 3, n_actions)
            assert not torch.isnan(q_vals).any(), "q_vals contains NaN"
            agent_outs = q_vals  # => (bs, 3, n_actions)

        return agent_outs

    def _build_edge_features(self, node_feats):
        """
        node_feats: shape=(bs, n_nodes, 9)
          - e.g. node_feats[..., 3:5] = (pos_x, pos_y)
        return: shape=(bs, n_nodes, n_nodes, edge_dim=3)
          => e.g. [dist, visibility, attackable]
        Demo:
        """
        bs, n_nodes, dim = node_feats.shape  # 期待 (bs, 6, 9)

        # 拿出位置
        pos_xy = node_feats[..., 3:5]  # (bs, n_nodes, 2)
        pos_i = pos_xy.unsqueeze(2)    # => (bs, n_nodes, 1, 2)
        pos_j = pos_xy.unsqueeze(1)    # => (bs, 1, n_nodes, 2)
        diff_ij = pos_i - pos_j        # => (bs, n_nodes, n_nodes, 2)
        dist_ij = torch.sqrt(torch.sum(diff_ij**2, dim=-1))  # => (bs, n_nodes, n_nodes)

        # trivial example: threshold
        visibility_ij = (dist_ij < 0.3).float()
        attack_ij = (dist_ij < 0.2).float()

        edge_feats = torch.stack([dist_ij, visibility_ij, attack_ij], dim=-1)  # (bs, n_nodes, n_nodes, 3)
        assert not torch.isnan(edge_feats).any(), "edge_feats contains NaN"

        return edge_feats