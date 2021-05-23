import numpy as np
from collections import namedtuple, deque
import random
import torch

# from utilities import transpose_list


class ReplayBuffer:
    def __init__(self, size, seed, device):
        self.size = size
        self.memory = deque(maxlen=self.size)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.device = device

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self, batchsize):
        """sample from the buffer"""
        experiences = random.sample(self.memory, k=batchsize)
        return experiences

    def __len__(self):
        return len(self.memory)
