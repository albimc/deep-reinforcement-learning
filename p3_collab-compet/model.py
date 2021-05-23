import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class ActorNetwork(nn.Module):
    def __init__(self, state_size, hidden_in_dim, hidden_out_dim, action_size, seed):

        super(ActorNetwork, self).__init__()

        self.seed = torch.manual_seed(seed)
        self.bn = nn.BatchNorm1d(state_size)
        self.fc1 = nn.Linear(state_size, hidden_in_dim)
        self.fc2 = nn.Linear(hidden_in_dim, hidden_out_dim)
        self.fc3 = nn.Linear(hidden_out_dim, action_size)
        self.nonlin = F.leaky_relu   # relu  # leaky_relu
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-1e-3, 1e-3)

    def forward(self, states):

        if states.dim() == 1:
            x = self.bn(states.unsqueeze(0))
        else:
            x = self.bn(states)

        h1 = self.nonlin(self.fc1(x))
        h2 = self.nonlin(self.fc2(h1))
        h3 = (self.fc3(h2))
        return F.tanh(h3)


class CriticNetwork(nn.Module):
    def __init__(self, state_size, action_size, num_agents, hidden_in_dim, hidden_out_dim, seed):

        super(CriticNetwork, self).__init__()

        self.seed = torch.manual_seed(seed)

        self.bn = nn.BatchNorm1d(state_size*num_agents)
        self.fc1 = nn.Linear(state_size*num_agents, hidden_in_dim)
        # self.fc2 = nn.Linear(hidden_in_dim + action_size * num_agents, hidden_out_dim)
        self.fc2 = nn.Linear(hidden_in_dim + action_size*num_agents, hidden_out_dim)
        self.fc3 = nn.Linear(hidden_out_dim, 1)
        self.nonlin = F.leaky_relu   # relu  # leaky_relu
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-1e-3, 1e-3)

    def forward(self, states, actions):

        if states.dim() == 1:
            x = self.bn(states.unsqueeze(0))
        else:
            x = self.bn(states)

        hs = self.nonlin(self.fc1(x))
        h1 = torch.cat((hs, actions), dim=1)
        h2 = self.nonlin(self.fc2(h1))
        return self.fc3(h2)
