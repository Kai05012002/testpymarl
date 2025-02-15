import copy
import torch
import torch.nn.functional as F
from torch.optim import RMSprop  # 或 Adam, 隨需求
from modules.critics.basic_value import BasicValueNet  # 你需要額外寫一個簡單的 Value 網路
from components.episode_buffer import EpisodeBatch

class A2CLearner:
    """
    單一 Agent 的 A2C。
    假設你只需要 policy 與 value (critic) 來計算 Advantage，並進行策略梯度更新。

    參考 PyMARL 的 coma_learner, q_learner 等寫法。
    """

    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac            # 多半是一個 MAC (這裡只有 1 agent 也可以)
        self.logger = logger

        # Policy network 的參數
        self.agent_params = list(mac.parameters())

        # 建立一個簡易的 Value Network (Critic)，
        # 假設輸入 shape = state_shape, 輸出 shape = 1
        self.critic = BasicValueNet(args.state_shape, args.critic_hidden_dim)
        self.critic_params = list(self.critic.parameters())

        # 把 actor + critic 的參數合在一起
        self.params = self.agent_params + self.critic_params

        # Optimizer
        self.optimiser = RMSprop(self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # logger
        self.log_stats_t = -self.args.learner_log_interval - 1

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        """
        A2C 主要步驟:
          1) 取出 state, action, reward, (terminated), mask
          2) 計算 V(s) & V(s')
          3) Advantage = [r + gamma * V(s') - V(s)]
          4) Policy loss = - log pi(a|s) * Advantage
          5) Critic loss = Advantage^2  (或 0.5*(TD error^2))
        """
        bs = batch.batch_size
        max_t = batch.max_seq_length

        # (1) 取出資料
        rewards = batch["reward"][:, :-1]           # shape=(bs,t,1)
        actions = batch["actions"][:, :-1]          # shape=(bs,t,1,1)
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])  # valid transitions

        # Actor 需要 action logits (policy)，Critic 需要 value(s)
        # 先 rollout policy (actor)
        mac_out = []
        self.mac.init_hidden(batch_size=bs)
        for t in range(max_t):
            actor_outs = self.mac.forward(batch, t=t)  # shape=(bs, 1, n_actions)，單一 agent
            mac_out.append(actor_outs)
        mac_out = torch.stack(mac_out, dim=1)  # shape=(bs, T, 1, n_actions)

        # 取出對應 actions 的 log pi(a|s)
        pi = torch.softmax(mac_out[:, :-1], dim=-1)                # (bs,t,1,n_actions)
        pi_taken = pi.gather(dim=3, index=actions).squeeze(3)      # (bs,t,1) -> (bs,t)
        log_pi_taken = torch.log(pi_taken + 1e-10)                 # 避免 log(0)

        # Critic: 計算 V(s)
        vs = []
        for t in range(max_t):
            state_t = batch["state"][:, t]  # shape=(bs, state_shape)
            v_t = self.critic(state_t)
            vs.append(v_t)
        vs = torch.stack(vs, dim=1)   # shape=(bs, T, 1)
        vs_next = vs[:, 1:]          # shape=(bs, T-1, 1)
        vs      = vs[:, :-1]         # shape=(bs, T-1, 1)

        # (2) TD(0) Advantage: r + gamma * V(s') - V(s)
        gamma = self.args.gamma
        targets = rewards + gamma * vs_next * (1 - terminated)
        advantages = (targets - vs).detach()   # (bs, T-1, 1)

        # (3) Policy loss = -( advantage * log_prob ) * mask
        log_pi_taken = log_pi_taken.squeeze(2)  # (bs,t)
        advantages   = advantages.squeeze(2)    # (bs,t)
        mask = mask.squeeze(2)                 # (bs,t)
        policy_loss = - (advantages * log_pi_taken * mask).sum() / (mask.sum() + 1e-10)

        # (4) Critic loss = MSE( vs, targets )
        td_error = (vs.squeeze(2) - targets.squeeze(2)) * mask
        critic_loss = (td_error ** 2).sum() / (mask.sum() + 1e-10)

        # 總 loss
        loss = policy_loss + self.args.value_loss_coef * critic_loss

        # (5) 更新
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        # (6) logging
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("policy_loss", policy_loss.item(), t_env)
            self.logger.log_stat("critic_loss", critic_loss.item(), t_env)
            self.logger.log_stat("advantage_mean", (advantages * mask).sum().item() / mask.sum().item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            self.log_stats_t = t_env

    def cuda(self):
        self.mac.cuda()
        self.critic.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        torch.save(self.critic.state_dict(), "{}/critic_a2c.torch".format(path))
        torch.save(self.optimiser.state_dict(), "{}/opt_a2c.torch".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.critic.load_state_dict(torch.load("{}/critic_a2c.torch".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(torch.load("{}/opt_a2c.torch".format(path), map_location=lambda storage, loc: storage))
