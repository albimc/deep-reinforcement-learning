import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from unityagents import UnityEnvironment
import torch

# Import DQN Libraries
from dqn_agent import Agent
from dqn_model import QNetwork
from dqn_monitor import Interact

# ###################
# Unity Environment #
# ###################

# With Visual
# env = UnityEnvironment(file_name="./Banana_Linux/Banana.x86_64")

# Without Visual
env = UnityEnvironment(file_name="./Banana_Linux_NoVis/Banana.x86_64")

# Check Environment Specs
print(str(env))
env.__dict__
dir(env)

# get the default brain
brain_name = env.brain_names[0]
print(brain_name)
brain = env.brains[brain_name]

# number of actions
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]
print(dir(env_info))

# number of agents in the environment
print('Agents:', env_info.agents)
print('Number of agents:', len(env_info.agents))

# examine the state space
state = env_info.vector_observations[0]
print('States look like:\n', state)
state_size = len(state)
print('States have length:', state_size)

# examine the reward
reward = env_info.rewards[0]
print('Reward:', reward)

# ################
# Random Actions #
# ################
env_info = env.reset(train_mode=False)[brain_name]  # reset the environment
state = env_info.vector_observations[0]             # get the current state
score = 0                                           # initialize the score
while True:
    action = np.random.randint(action_size)         # select an action
    env_info = env.step(action)[brain_name]         # send the action to the environment
    next_state = env_info.vector_observations[0]    # get the next state
    reward = env_info.rewards[0]                    # get the reward
    done = env_info.local_done[0]                   # see if episode has finished
    score += reward                                 # update the score
    state = next_state                              # roll over the state to next time step
    if done:                                        # exit loop if episode finished
        break

print("Score: {}".format(score))


# ##############################################################################################################
# ##############################################################################################################

# #######
# AGENT #
# #######
agent = Agent(state_size=state_size, action_size=action_size, seed=0)

# #############
# TRAIN AGENT #
# #############
scores = Interact(env, agent, brain_name)
rolling_mean = pd.Series(scores).rolling(100).mean()

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.plot(rolling_mean, lw=3)
plt.axhline(y=13, color='r', linestyle='dashed')
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
agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))

for i in range(3):
    env_info = env.reset(train_mode=False)[brain_name]
    state = env_info.vector_observations[0]
    for j in range(200):
        action = agent.act(state)
        env_info = env.step(action)[brain_name]
        next_state = env_info.vector_observations[0]    # get the next state
        reward = env_info.rewards[0]                    # get the reward
        done = env_info.local_done[0]                   # see if episode has finished
        score += reward                                 # update the score
        state = next_state                              # roll over the state to next time step
        if done:
            break

print("Score: {}".format(score))

env.close()


# ##############################################################################################################
# ##############################################################################################################
