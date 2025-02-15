# controllers/enhanced_gat_mac.py

import torch
import torch.nn.functional as F
from .basic_controller import BasicMAC
from modules.agents.enhanced_gat_agent import EnhancedGATAgent

class EnhancedGATMAC(BasicMAC):
    """
    多智能體控制器 (MAC) ，
    每個 time-step 會把 ep_batch["state"][:, t] -> node_feats -> edge_feats,
    送進 GAT agent => slice ally => Q-head => (bs, n_agents, n_actions).
    """
    def __init__(self, scheme, groups, args):
        super().__init__(scheme, groups, args)

    def _build_agents(self, input_shape):
        # 用我們剛定義的 EnhancedGATAgent
        self.agent = EnhancedGATAgent(self.args)

    def forward(self, ep_batch, t, test_mode=False):
        """
        ep_batch["state"][:, t] => shape=(bs, state_dim=54) for 3m
        reshape=> (bs,6,9), build edge => (bs,6,6,3),
        agent => (bs,6, hidden_dim),
        slice ally => Q-head => (bs, n_agents, n_actions).
        """
        bs = ep_batch.batch_size
        n_agents = self.args.n_agents

        # (1) 取 state: (bs, 54) => (bs,6,9)
        state_1d = ep_batch["state"][:, t]  # (bs, 54)
        #print("DEBUG: state_1d shape:", state_1d.shape)
        node_feats = state_1d.view(bs, 6, 9)

        # (2) 邊特徵: (bs,6,6,3)
        edge_feats = self._build_edge_features(node_feats)

        # (3) Forward agent => GAT => (bs,6,hidden_dim)
        node_embeds = self.agent(node_feats, edge_feats)

        # (4) slice ally => (bs,n_agents,hidden_dim)
        #   3m => n_agents=3, 只取 node 0~2
        ally_embeds = node_embeds[:, :n_agents, :]

        # (5) Q-head => (bs,n_agents,n_actions)
        q_values = self.agent.q_head(ally_embeds)

        return q_values

    def _build_edge_features(self, node_feats):
        """
        node_feats => (bs, 6, 9), 其中 pos_x,pos_y 在 col=3,4
        回傳 => (bs, 6, 6, 3): [dist, visibility, attackable]
        """
        bs, n, _ = node_feats.shape

        # 取 (pos_x, pos_y)
        pos_xy = node_feats[..., 3:5]  # (bs,6,2)
        pos_i = pos_xy.unsqueeze(2)    # (bs,6,1,2)
        pos_j = pos_xy.unsqueeze(1)    # (bs,1,6,2)
        diff = pos_i - pos_j           # (bs,6,6,2)
        dist = torch.sqrt((diff ** 2).sum(dim=-1))  # (bs,6,6)

        # 自行定義 threshold
        visibility = (dist < 2.5).float()
        attackable = (dist < 2.0).float()

        edge_feats = torch.stack([dist, visibility, attackable], dim=-1)  # (bs,6,6,3)
        return edge_feats
