import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GPQMixer(nn.Module):
    def __init__(self, args):
        super(GPQMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))

        self.embed_dim = args.mixing_embed_dim # 32
        # RNN for State processing
        self.state_rnn = nn.GRU(self.state_dim, self.state_dim, batch_first=True)

        # Transformation layers for Keys, Queries, and Values
        self.key_layer = nn.Linear(self.state_dim, self.embed_dim*self.n_agents)
        self.query_layer = nn.Linear(self.state_dim, self.embed_dim*self.n_agents)


        self.hyper_w_agent_q = nn.Linear(self.n_agents, self.embed_dim*self.n_agents)

        # 權重初始化（使用 Xavier 初始化）
        nn.init.xavier_uniform_(self.key_layer.weight)
        nn.init.xavier_uniform_(self.query_layer.weight)
        nn.init.xavier_uniform_(self.hyper_w_agent_q.weight)

    def forward(self, agent_qs, states):
        """
        Args:
            agent_qs: Tensor of shape (batch_size, seq_len, n_agents)
            states: Tensor of shape (batch_size, seq_len, state_dim)

        Returns:
            q_tot: Tensor of shape (batch_size, seq_len, 1)
        """
        batch_size, seq_len, _ = states.size()
        states = states.reshape(-1, 1, self.state_dim)  # 展平 state 維度
        agent_qs2 = agent_qs.reshape(-1, self.n_agents)
        agent_qs = agent_qs.view(-1, 1, self.n_agents)  # 調整 agent_qs 的形狀
        # print(states.shape)
        # print(agent_qs2.shape)
        # print(agent_qs.shape)
        # Process State through RNN
        states, _ = self.state_rnn(states)  # (batch_size, seq_len, state_dim) (32,seq,48)
        # print("===================================",states.shape)
        # Compute k_t, q_t
        k_t = self.key_layer(states).view(-1, self.n_agents, self.embed_dim)
        q_t = self.query_layer(states).view(-1, self.n_agents, self.embed_dim)
        # print("----k_t----",k_t.shape)
        # print("----q_t----",q_t.shape)
        # Process agent_qs through hyper_w_agent_q
        w_agentqs = self.hyper_w_agent_q(agent_qs2).view(-1, self.embed_dim, self.n_agents)
        # print("----w_agentqs-----",w_agentqs.shape)
        # 計算 Key 和 Query
        
        k = torch.bmm(k_t, w_agentqs)  # (B, n_agents, n_agents)
        q = torch.bmm(q_t, w_agentqs)  # (B, n_agents, n_agents)
        # print(k.shape,q.shape)

        a = torch.matmul(k, q) / self.embed_dim**0.5
        # 加入數值穩定性
        a = a - a.max(dim=-1, keepdim=True).values  
        alpha = F.softmax(a, dim=-1)   
        # print(alpha.shape,alpha)
        # print(agent_qs.shape,agent_qs)
        # print("++++++++++++++++++++++++++++++++++++++++++++++")
        # Weighted sum
        y = alpha * agent_qs  # (batch_size, seq_len, 1, embed_dim)
        # print(y.shape,y)
        y_reduce = torch.sum(y, dim=1)  # (batch_size, 1, embed_dim)
        # print(y_reduce.shape,y_reduce)
        # Compute final Q_total_t
        Q_total_t = torch.sum(y_reduce, dim=-1, keepdim=True)  # (batch_size, 1)
        Q_total_t = Q_total_t.view(batch_size, -1, 1)
        # print(Q_total_t.shape,Q_total_t)
        return Q_total_t

        # k_t = self.key_layer(sliding_states)#.view(batch_size, seq_len, self.embed_dim, 1)  # (batch_size, seq_len, embed_dim, 1)
        # print("----k_t----",k_t.shape,k_t)
        # q_t = self.query_layer(sliding_states)#.view(batch_size, seq_len, 1, self.embed_dim)  # (batch_size, seq_len, 1, embed_dim)
        # print("----q_t----",q_t.shape,q_t)


        # print("+++++++++++++++++++++++++++++++++++++++++++++++++")
        # v_t = self.value_layer(agent_qs).view(batch_size, seq_len, 1, self.embed_dim)  # (batch_size, seq_len, 1, embed_dim)

        # # Transform agent_qs into embeddings
        # agent_qs_embed = agent_qs.view(-1, n_agents)  # Flatten for linear layers

        # # Compute Keys, Queries, and Values
        # keys = self.key_layer(agent_qs_embed).view(batch_size, seq_len, n_agents, -1)
        # queries = self.query_layer(agent_qs_embed).view(batch_size, seq_len, n_agents, -1)
        # values = self.value_layer(agent_qs_embed).view(batch_size, seq_len, n_agents, -1)

        # # Compute attention scores (scaled dot-product attention)
        # attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / np.sqrt(self.embed_dim)
        # attention_scores = F.softmax(attention_scores, dim=-1)  # [batch, seq_len, n_agents, n_agents]

        # # Compute weighted values
        # weighted_values = torch.matmul(attention_scores, values)  # [batch, seq_len, n_agents, embed_dim]

        # # Aggregate agent Q-values using attention weights
        # q_att = weighted_values.sum(dim=2)  # Sum over agents, [batch, seq_len, embed_dim]

        # # Compute state-dependent baseline
        # state_baseline = self.state_baseline_layer(states).squeeze(-1)  # [batch, seq_len]

        # # Compute total Q-value
        # q_tot = q_att.sum(dim=-1) + torch.abs(state_baseline)  # [batch, seq_len]

        # return q_tot.unsqueeze(-1)  # [batch, seq_len, 1]
    
#batch_size, seq_len, n_agents = agent_qs.size()
        # #print("----agent_qs----", agent_qs.shape, agent_qs[0]) # (batch_size, seq_len , n_agents) (1, seq_len, n_agents)
        # #print("----states----",states.shape, states[0]) # (batch_size, seq_len, state_dim) (1, seq_len, state_dim)

        # 1/3 不進行Data preprocessing
        # # input 進行 sliding window
        # agent_qs = agent_qs.squeeze(0)
        # agent_qs = agent_qs.detach().cpu().numpy()
        # states = states.squeeze(0)
        # states = states.detach().cpu().numpy()
        # sliding_window_size = 5
        # slided_seq_size =  states.shape[0]-sliding_window_size+1
        # # (seq, n_agents) to (seq-4,1,sliding_window_size, n_agents)
        # # reshape to (seq-4,sliding_window_size,n_agents)
        # agent_sliding_qs = np.lib.stride_tricks.sliding_window_view(agent_qs, (sliding_window_size, agent_qs.shape[1])).reshape(slided_seq_size, sliding_window_size, agent_qs.shape[1])
        # # 從 NumPy 陣列轉為 PyTorch 張量
        # agent_sliding_qs = torch.from_numpy(agent_sliding_qs).to("cuda")  
        # agent_sliding_qs.requires_grad = True
        # sliding_states = np.lib.stride_tricks.sliding_window_view(states, (sliding_window_size, states.shape[1])).reshape(slided_seq_size, sliding_window_size, states.shape[1])
        # sliding_states = torch.from_numpy(sliding_states).to("cuda")
        # sliding_states.requires_grad = True
        # #input 預處理完成