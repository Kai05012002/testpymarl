import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class QMeta(nn.Module):
    def __init__(self, args):
        super(QMeta, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))
        self.embed_dim = args.mixing_embed_dim

        self.d = self.embed_dim**0.5

        # 定義全連接層生成權重矩陣
        self.hyper_w_k = nn.Linear(self.state_dim, self.embed_dim * self.n_agents)
        self.hyper_w_q = nn.Linear(self.state_dim, self.embed_dim * self.n_agents)
        
        self.hyper_w_agent_q = nn.Linear(self.n_agents, self.embed_dim * self.n_agents)


        # 權重初始化（使用 Xavier 初始化）
        nn.init.xavier_uniform_(self.hyper_w_k.weight)
        nn.init.xavier_uniform_(self.hyper_w_q.weight)
        nn.init.xavier_uniform_(self.hyper_w_agent_q.weight)
       
        
   
    def forward(self, agent_qs, states):#, hidden_state
        bs = agent_qs.size(0)  # 批次大小
        states = states.reshape(-1, self.state_dim)  # 展平 state 維度
        agent_qs2 = agent_qs.reshape(-1, self.n_agents)
        agent_qs = agent_qs.view(-1, 1, self.n_agents)  # 調整 agent_qs 的形狀

        # 計算權重矩陣 Wv, Wk, Wq，並調整形狀
        
        wk = self.hyper_w_k(states).view(-1, self.n_agents, self.embed_dim)
        wq = self.hyper_w_q(states).view(-1, self.n_agents, self.embed_dim)
        w_agentqs = self.hyper_w_agent_q(agent_qs2).view(-1, self.embed_dim, self.n_agents)

        # 計算 Key 和 Query
        
        k = th.bmm(wk, w_agentqs)  # (B, n_agents, n_agents)
        q = th.bmm(wq, w_agentqs)  # (B, n_agents, n_agents)
        
      
        a = th.matmul(k, q) / self.d  
        # 加入數值穩定性
        a = a - a.max(dim=-1, keepdim=True).values  
        alpha = F.softmax(a, dim=-1)   # (B, n_agents, n_agents)
        # print("before al : ", alpha.shape)
        alpha = alpha.sum(dim=(-2), keepdim=True)
        # print("after al : ", alpha.shape)
        # print("agents : ", agent_qs.shape)
        y= agent_qs*alpha
        # print("y : ", y.shape)
        # 計算總輸出 Q_tot
        y = y.sum(dim=(-1), keepdim=True)
        q_tot = y.view(bs, -1, 1)  # 調整形狀為 (batch_size, seq_len, 1)
        # print("q_tot : ", q_tot.shape)

        return q_tot








