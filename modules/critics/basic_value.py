import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicValueNet(nn.Module):
    """
    最簡單的 Value function：輸入 state -> 輸出 V(s)
    """
    def __init__(self, state_dim, hidden_dim=64):
        super(BasicValueNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.v_out = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        # state: (bs, state_dim)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        v = self.v_out(x)  # shape=(bs,1)
        return v
