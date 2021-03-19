import numpy as np
from collections import defaultdict

class Agent:
    
    def __init__(self, nA=6):
        """ Initialize agent.
        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.ones(self.nA)*0)
        self.alpha = 0.1
        self.gamma = 1.0
        self.eps_start = 1.0
        self.eps_decay = 0.9999
        self.eps_min = 0.00001
        self.epsilon = self.eps_start

    def select_action(self, state):
        """ Given the state, select an action.
        Params
        ======
        - state: the current state of the environment
        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        self.epsilon = max(self.epsilon*self.eps_decay, self.eps_min)
        policy_s = np.ones(self.nA) * self.epsilon / self.nA
        policy_s[np.argmax(self.Q[state])] = 1 - self.epsilon + (self.epsilon / self.nA)
        return np.random.choice(np.arange(self.nA), p=policy_s)

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.
        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        Qsa_now = self.Q[state][action]
        policy_s = np.ones(self.nA) * self.epsilon / self.nA
        policy_s[np.argmax(self.Q[next_state])] = 1 - self.epsilon + (self.epsilon / self.nA)
        Qsa_next = np.dot(self.Q[next_state], policy_s)
        self.Q[state][action] += self.alpha * (reward + (self.gamma * Qsa_next) - Qsa_now)
