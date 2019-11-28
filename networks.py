import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):

    def __init__(self, s_dim, a_dim):
        super(DQN, self).__init__()
        self.a_dim = a_dim
        # Build the network
        self.fc1 = nn.Linear(s_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, a_dim)
        # Initialize weights
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x.view(x.size(0), -1)
