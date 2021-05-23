# individual network settings for each actor + critic pair
# see networkforall for details

from model import ActorNetwork, CriticNetwork
from utilities import hard_update, gumbel_softmax, onehot_from_logits
from torch.optim import Adam
import torch
import numpy as np
import pdb

# add OU noise for exploration
from OUNoise import OUNoise

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'


class DDPGAgent:
    def __init__(self, state_size, action_size, num_agents,
                 hidden_in_actor, hidden_out_actor,
                 hidden_in_critic, hidden_out_critic,
                 seed=1, device='cpu', lr_actor=1e-4, lr_critic=3e-4, weight_decay_critic=0):
        super(DDPGAgent, self).__init__()

        self.device = device

        # Actor
        self.actor = ActorNetwork(state_size, hidden_in_actor, hidden_out_actor, action_size, seed).to(device)
        self.target_actor = ActorNetwork(state_size, hidden_in_actor, hidden_out_actor, action_size, seed).to(device)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr_actor)

        # Target
        self.critic = CriticNetwork(state_size, action_size, num_agents, hidden_in_critic, hidden_out_critic, seed).to(device)
        self.target_critic = CriticNetwork(state_size, action_size, num_agents, hidden_in_critic, hidden_out_critic, seed).to(device)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr_critic, weight_decay=weight_decay_critic)

        # Noise
        self.noise = OUNoise(action_size, seed, scale=1.0)

        # initialize targets same as original networks
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)

    def reset(self):
        self.noise.reset()

    def act(self, obs, noise_factor=0.0):

        if torch.is_tensor(obs):
            states = obs
        else:
            states = torch.from_numpy(obs).float().to(self.device)

        self.actor.eval()
        with torch.no_grad():
            actions = self.actor(states).cpu().data.numpy()
        self.actor.train()
        actions += noise_factor*self.noise.sample()
        return np.clip(actions, -1, 1)

    def target_act(self, obs):

        if torch.is_tensor(obs):
            states = obs
        else:
            states = torch.from_numpy(obs).float().to(self.device)

        self.target_actor.eval()
        with torch.no_grad():
            actions = self.target_actor(states).cpu().data.numpy()
        self.target_actor.train()
        return np.clip(actions, -1, 1)
