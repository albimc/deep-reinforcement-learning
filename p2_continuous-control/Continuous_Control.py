import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from unityagents import UnityEnvironment
import torch

# Import DDPG Libraries
from ddpg_agent import Agent
from ddpg_model import Actor, Critic
from ddpg_interact import Interact

import time

# ###################
# Unity Environment #
# ###################
# env = UnityEnvironment(file_name="./Reacher_Linux_NoVis/Reacher.x86_64")
env = UnityEnvironment(file_name="./Reacher_Linux_NoVis_20/Reacher.x86_64")

# Check Environment Specs
print(str(env))
env.__dict__
dir(env)

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
print(brain_name)

# reset the environment set to Train Mode
env_info = env.reset(train_mode=True)[brain_name]
print(dir(env_info))

# number of agents in the environment
print('Agents:', env_info.agents)
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# number of actions
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)

# examine the state space
states = env_info.vector_observations
state_size = states.shape[1]
print('States have length:', state_size)
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])

# examine the reward
rewards = env_info.rewards
print('Reward:', rewards)

# ################
# Random Actions #
# ################
# env_info = env.reset(train_mode=False)[brain_name]  # reset the environment
# states = env_info.vector_observations               # get the current state (for each agent)
# scores = np.zeros(num_agents)                       # initialize the score (for each agent)

# while False:
#     actions = np.random.randn(num_agents, action_size)  # select an action (for each agent)
#     actions = np.clip(actions, -1, 1)                   # all actions between -1 and 1
#     env_info = env.step(actions)[brain_name]            # send all actions to tne environment
#     next_states = env_info.vector_observations          # get next state (for each agent)
#     rewards = env_info.rewards                          # get reward (for each agent)
#     dones = env_info.local_done                         # see if episode finished
#     scores += env_info.rewards                          # update the score (for each agent)
#     states = next_states                                # roll over states to next time step
#     if np.any(dones):                                   # exit loop if episode finished
#         break
#
# print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))


##############################################################################################################
#
##############################################################################################################

# #######
# AGENT #
# #######
agent = Agent(state_size=state_size, action_size=action_size, num_agents=num_agents, seed=1, timestamp=1)

# load the weights from file
# agent.actor_local.load_state_dict(torch.load('checkpoint_actor.pth'))
# agent.critic_local.load_state_dict(torch.load('checkpoint_critic.pth'))

for name, param in agent.actor_local.named_parameters():
    if param.requires_grad:
        print(name, len(param.data), param.data)

# #############
# TRAIN AGENT #
# #############
scores = Interact(env, agent, brain_name, num_agents, train_mode=True, add_noise=True, n_episodes=1000,
                  max_t=1000, eps_start=1, eps_end=0.2, eps_decay=0.999, n_window=10, tgt_score=45)

rolling_mean = pd.Series(scores).rolling(10).mean()


# #############
# PLOT SCORES #
# #############
# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.plot(rolling_mean, lw=3)
plt.axhline(y=30, color='r', linestyle='dashed')
# plt.title('Navigation Project: Collecting Bananas')
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()
fig.savefig('Average_Score.pdf')
fig.savefig('Average_Score.jpg')


# ##################
# PLAY SMART AGENT #
# ##################

# load the weights from file
agent.qnetwork_local.load_state_dict(torch.load('checkpoint_actor.pth'))
agent.critic_local.load_state_dict(torch.load('checkpoint_critic.pth'))

for i in range(3):
    env_info = env.reset(train_mode=False)[brain_name]
    states = env_info.vector_observations
    scores = np.zeros(num_agents)
    for j in range(200):
        actions = agent.act(states)
        env_info = env.step(actions)[brain_name]
        next_states = env_info.vector_observations    # get the next state
        rewards = env_info.rewards                    # get the reward
        dones = env_info.local_done                   # see if episode has finished
        scores += rewards                             # update the score
        states = next_states                          # roll over the state to next time step
        if np.any(dones):
            break

print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))

env.close()


# ##############################################################################################################
# ##############################################################################################################
