import torch
import torch.nn.functional as F
from .basic_controller import BasicMAC
from modules.agents.enhanced_gat_agent import EnhancedGATAgent

class EnhancedGATMAC(BasicMAC):
    """
    專為 1v1 場景硬編碼的 MAC。
    假設 get_state() 回傳 (bs, 18)，對應 2 個 node (ally + enemy)，每個 node 9 維。
    """
    def __init__(self, scheme, groups, args):
        super().__init__(scheme, groups, args)

    def _build_agents(self, input_shape):
        # 用我們自己定義的 EnhancedGATAgent
        self.agent = EnhancedGATAgent(self.args)

    def forward(self, ep_batch, t, test_mode=False):
        """
        假定 state_1d.shape = (bs, 18)，其中 2 個 node，各 9 維。
        reshape => (bs, 2, 9)。
        """
        bs = ep_batch.batch_size

        # (1) 取 state: shape=(bs,18)，reshape => (bs,2,9)
        state_1d = ep_batch["state"][:, t]  # (bs,18)
        node_feats = state_1d.view(bs, 2, 9)  # 針對 1 ally + 1 enemy

        # (2) 邊特徵: (bs, 2, 2, 3)
        edge_feats = self._build_edge_features(node_feats)

        # (3) 送進 GAT => (bs,2, hidden_dim)
        node_embeds = self.agent(node_feats, edge_feats)

        # (4) Ally 只取 node 0 => (bs,1,hidden_dim)
        ally_embeds = node_embeds[:, :1, :]  # 第 0 個 node

        # (5) Q-head => (bs,1,n_actions)
        q_values = self.agent.q_head(ally_embeds)

        return q_values

    def _build_edge_features(self, node_feats):
        """
        node_feats: (bs,2,9)，其中 col=3,4 是 (pos_x,pos_y)。
        計算 (bs,2,2,3) => [dist, visibility, attackable]。
        """
        bs = node_feats.size(0)
        n = 2  # 硬編碼，因為 1 ally + 1 enemy

        # 取 pos_x, pos_y => col=3,4
        pos_xy = node_feats[..., 3:5]  # (bs,2,2)

        pos_i = pos_xy.unsqueeze(2).expand(bs, n, n, 2)  # (bs,2,2,2)
        pos_j = pos_xy.unsqueeze(1).expand(bs, n, n, 2)  # (bs,2,2,2)

        diff = pos_i - pos_j                             # (bs,2,2,2)
        dist = torch.sqrt((diff ** 2).sum(dim=-1))       # (bs,2,2)

        # 硬寫 threshold 產生 visibility / attackable
        visibility = (dist < 2.5).float()
        attackable = (dist < 2.0).float()

        # 堆成 (bs,2,2,3)
        edge_feats = torch.stack([dist, visibility, attackable], dim=-1)
        return edge_feats
