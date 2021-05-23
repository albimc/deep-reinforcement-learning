from collections import deque
import numpy as np
import torch
import time


def Interact(env, agent, brain_name, num_agents, train_mode=True, add_noise=True,
             n_episodes=1000, max_t=1000, eps_start=1.0, eps_end=0.1, eps_decay=0.999, n_window=40, tgt_score=30):
    """
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
    start = time.time()

    # Initialise
    scores = []                                             # list containing scores from each episode
    scores_window = deque(maxlen=n_window)                  # last 100 scores
    eps = eps_start                                         # initialize epsilon

    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=train_mode)[brain_name]    # reset the environment
        states = env_info.vector_observations                    # get the current state
        score = np.zeros(num_agents)
        for t in range(max_t):
            # Agent Selects Action #
            actions = agent.act(states, add_noise=add_noise, eps=eps)     # Agent action
            # Agent Performs Action #
            env_info = env.step(actions)[brain_name]                # Environment reacts
            next_states = env_info.vector_observations           # get the next state
            rewards = env_info.rewards                           # get the reward
            dones = env_info.local_done                          # see if episode has finished
            # Agent Observes State #
            agent.step(states, actions, rewards, next_states, dones)    # Agent observes new state
            states = next_states                                     # roll over the state to next time step
            score += rewards                                        # update the score
            if any(dones):
                break
        scores_window.append(np.mean(score))        # save most recent score
        scores.append(np.mean(score))               # save most recent score
        eps = max(eps_end, eps_decay*eps)  # decrease epsilon
        # Periodic Check
        if i_episode % n_window == 0:
            end = time.time()
            print('\nElapsed time {:.1f}'.format((end - start)/60), 'Steps {}'.format(i_episode*max_t), 'Agent Steps {}'.format(agent.timestamp), end="")
            print('\nEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
            print('\nEpisode {}\tLast Action:'.format(i_episode), actions[0], end="")
            print('\nEpisode {}\tLast Epsilon: {:.2f}'.format(i_episode, eps), end="")
            torch.save(agent.actor_local.state_dict(), './Data/checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), './Data/checkpoint_critic.pth')
        if np.mean(scores_window) >= tgt_score:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-n_window, np.mean(scores_window)))
            break

    return scores
