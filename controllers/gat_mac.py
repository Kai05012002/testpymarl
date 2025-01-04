# controllers/gat_mac.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from .basic_controller import BasicMAC
from modules.agents.gat_agent import GATAgent  # 假定我們在 modules/agents/gat_agent.py
import math

class GATMAC(BasicMAC):
    """
    這個MAC一次forward整張圖(6個或N個nodes)，
    透過GATAgent來計算GAT embedding。
    """
    def __init__(self, scheme, groups, args):
        super().__init__(scheme, groups, args)
        # 注意：BasicMAC.__init__() 會呼叫 self._build_agents()，
        #      但我們要改用自己的 agent，因此可改override

    def _build_agents(self, input_shape):
        """
        覆蓋BasicMAC原本的建構行為，
        但這裡直接實例化我們自定義的 GATAgent。
        """
        self.agent = GATAgent(self.args)

    def forward(self, ep_batch, t, test_mode=False):
        """
        主要流程：
          1. 從batch提取 state (或obs) → 建立 node features X_l
          2. 計算 edge features E_l
          3. 一次 GAT forwarding → node embeddings
          4. 取對應 agent node embedding → linear => Q-values
        """
        bs = ep_batch.batch_size

        # 1) 取得 state
        # shape = (bs, state_dim)
        state = ep_batch["state"][:, t]
        # 2) node_features, edge_features
        node_feats = self._build_node_features(state)  # (bs, n_nodes, in_dim)
        edge_feats = self._build_edge_features(node_feats)  # (bs, n_nodes, n_nodes, e_dim)

        # 3) forward GAT
        # GATAgent預設: forward(node_feats, edge_feats) -> (bs, n_nodes, embed)
        node_embeds = self.agent(node_feats, edge_feats)

        # 4) 假設前 N_agents 個node對應我方 agent => each agent's node_emb => MLP => Q
        #    這裡只是示例：3m => 3個我方 + 3個敵方 => n_agents=3 => 我方node在前3
        n_agents = self.args.n_agents
        agent_embeds = node_embeds[:, :n_agents, :]  # shape=(bs, n_agents, hidden_dim)

        # 5) map to Q-values
        #    這裡簡單 linear => (bs, n_agents, n_actions)
        q_values = self.agent.q_head(agent_embeds)

        return q_values

    def _build_node_features(self, state):
        """
        假設6個單位 (3我方+3敵方) → each node feature = local(6dim) + global(3dim)...

        你需要先保證在 starcraft2.py / get_state() 中，把
          [health_i, shield_i, x_i, y_i, type_i, cooldown_i, ..., global_stats...]
        打包到 state。

        這裡的實作取決於你如何把state layout設計好。
        """
        bs = state.shape[0]
        # demo: n_nodes=6, node_input_dim=9
        n_nodes = 6
        node_input_dim = 9
        node_feats = state.view(bs, n_nodes, node_input_dim)
        return node_feats

    def _build_edge_features(self, node_feats):
        """
        e.g. distance, visibility, attack_possible ...
        shape = (bs, n_nodes, n_nodes, e_dim)
        """
        bs, n_nodes, dim = node_feats.shape
        # 例如 node_feats[...,0:2] = (posx, posy), or ...
        pos_xy = node_feats[..., 3:5]  # demo: index=3,4 = x,y
        pos_i = pos_xy.unsqueeze(2).expand(bs, n_nodes, n_nodes, 2)
        pos_j = pos_xy.unsqueeze(1).expand(bs, n_nodes, n_nodes, 2)
        dist_ij = torch.sqrt(torch.sum((pos_i - pos_j)**2, dim=-1))  # (bs, n_nodes, n_nodes)

        # 另外, visibility => dist < 9 => 1 else 0
        # attack => dist < 6 => 1 else 0
        visibility_ij = (dist_ij < 9.0).float()
        attack_ij = (dist_ij < 6.0).float()

        # 只是一個範例
        edge_features = torch.stack([dist_ij, visibility_ij, attack_ij], dim=-1)
        return edge_features
