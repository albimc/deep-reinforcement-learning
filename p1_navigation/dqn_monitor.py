from collections import deque
# import sys
# import math
import numpy as np
import torch


def Interact(env, agent, brain_name, train_mode=True, n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995, n_window=100, tgt_score=15):
    """Deep Q-Learning.
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    print('\rRunning for {} episodes with target score of {}'.format(n_episodes, tgt_score))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('\rUsing {}'.format(device))
    # Initialise
    scores = []                                             # list containing scores from each episode
    scores_window = deque(maxlen=n_window)                  # last 100 scores
    eps = eps_start                                         # initialize epsilon
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=train_mode)[brain_name]    # reset the environment
        state = env_info.vector_observations[0]                    # get the current state
        score = 0
        for t in range(max_t):
            # Agent Selects Action #
            action = agent.act(state, eps)                         # Agent action
            # Agent Performs Action #
            env_info = env.step(action)[brain_name]                # Environment reacts
            next_state = env_info.vector_observations[0]           # get the next state
            reward = env_info.rewards[0]                           # get the reward
            done = env_info.local_done[0]                          # see if episode has finished
            # Agent Observes State #
            agent.step(state, action, reward, next_state, done)    # Agent observes new state
            state = next_state                                     # roll over the state to next time step
            score += reward                                        # update the score
            if done:
                break
        scores_window.append(score)        # save most recent score
        scores.append(score)               # save most recent score
        eps = max(eps_end, eps_decay*eps)  # decrease epsilon
        # Periodic Check
        if i_episode % n_window == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window) >= tgt_score:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-n_window, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            break
    return scores
