# controllers/gat_mac.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .basic_controller import BasicMAC
from modules.agents.gat_agent import SimpleGAT  # 假設你在 gat_agent.py 寫一個簡易 GAT
# 也可以不透過 gat_agent.py，直接在這檔案裡寫 GAT class

class GATMAC(BasicMAC):
    def __init__(self, scheme, groups, args):
        super().__init__(scheme, groups, args)
        self.args = args
        # BasicMAC 裡面已經有 self.n_agents, self.action_selector, etc.
        # 但 "敵軍數" 要從 env_info 或 config 取得，你可在 runner/setup 時賦值或做個固定
        # 3m 地圖可以先寫死 enemy_num=3，簡化demo
        self.n_enemies = 3  
        self.n_nodes = self.n_agents + self.n_enemies  # 3 + 3 = 6

        # 建立我們的 GAT 網路
        # 例如 simpleGAT 裡面需要 (node_input_dim, num_heads, hidden_dim...) 這些參數
        self.gat_net = SimpleGAT(
            node_input_dim=args.gat_node_input_dim,
            edge_input_dim=args.gat_edge_input_dim,
            hidden_dim=args.gat_hidden_dim,
            n_heads=args.gat_num_heads
        )

        # Q-head: 把對應我方Marine的 node embedding -> Q-values
        # 假設我們先用一個簡單 linear
        self.q_head = nn.Linear(args.gat_hidden_dim, self.args.n_actions)

    def forward(self, ep_batch, t, test_mode=False):
        """ep_batch: EpisodeBatch
           t: time index
        """
        bs = ep_batch.batch_size  # batch_size
        # 取得我方/敵方的狀態資訊，組成 Xl
        # 這裡示範從 state 中萃取，或用 obs + 你自定義的收集方式

        # step 1: 取出 state
        # shape = (bs, state_dim)，裡面包含(3我方 + 3敵方)單位座標/血量/etc. 
        # 你要確定 starcraft2env 的 state 是否足以重建整個圖
        state = ep_batch["state"][:, t]  # (bs, state_dim)

        # step 2: 重建 6 個 node feature (X_l)
        # 這裡只是 pseudo code, 你要自己把 state slice -> node0 features, node1 features, ...
        # e.g. node_features shape = (bs, n_nodes, node_input_dim)
        node_features = self._build_node_features_3m(state)

        # step 3: 構建 edge features (El)
        # e.g. shape = (bs, n_nodes, n_nodes, edge_input_dim)
        edge_features = self._build_edge_features_3m(node_features)

        # step 4: 一次 forward GAT
        # 你可以 batch 化 GAT，或用 for loop 每個 batch sample 跑
        # 這裡假設 SimpleGAT forward: (bs, N, node_input_dim), (bs, N, N, edge_input_dim) -> (bs, N, hidden_dim)
        node_embeddings = self.gat_net(node_features, edge_features)  # shape = (bs, n_nodes, hidden_dim)

        # step 5: 取前 3 個 node embedding (對應我方Marine)
        # shape = (bs, n_agents, hidden_dim)
        agent_node_emb = node_embeddings[:, :self.n_agents, :]

        # step 6: 每個 agent node emb -> Q-head
        # shape = (bs, n_agents, n_actions)
        q_values = self.q_head(agent_node_emb)  # Broadcasting: (bs, n_agents, hidden_dim) -> (bs, n_agents, n_actions)

        # step 7: 若要 mask unavailable actions, 在這裡做
        # e.g. q_values[avail_actions==0] = -9999999

        return q_values

    def _build_node_features_3m(self, state):
        """示範: 從 state 中 slice 出 6 個Marine的 [hp, x, y, ...] 特徵"""
        # 具體要看 StarCraft2Env 的 state layout
        # Demo: node_features shape = (bs, 6, node_input_dim)
        bs = state.shape[0]
        # 假裝 state_dim=6*5=30, 每個Marine 5個特徵
        node_input_dim = 5
        node_features = state.view(bs, self.n_nodes, node_input_dim)
        return node_features

    def _build_edge_features_3m(self, node_features):
        """計算距離/可見性/攻擊可能等 (bs, n_nodes, n_nodes, edge_input_dim)"""
        bs = node_features.shape[0]
        n_nodes = node_features.shape[1]

        # Demo: 只計算距離 => edge_feat_dim=1
        # positions = node_features[..., 1:3] -> x,y 假設index=1,2
        positions = node_features[..., 1:3]  # shape=(bs, n_nodes, 2)
        # broadcast
        pos_i = positions.unsqueeze(2).expand(bs, n_nodes, n_nodes, 2)
        pos_j = positions.unsqueeze(1).expand(bs, n_nodes, n_nodes, 2)
        dist_ij = torch.sqrt(torch.sum((pos_i - pos_j)**2, dim=-1)) # (bs, n_nodes, n_nodes)

        # shape=(bs, n_nodes, n_nodes, 1)
        edge_features = dist_ij.unsqueeze(-1)
        return edge_features
