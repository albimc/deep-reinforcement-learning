# TD Exercise #

import sys
import gym
import numpy as np
from collections import defaultdict, deque
import matplotlib.pyplot as plt


import check_test
from plot_utils import plot_values

# #############
# Environment #
# #############
env = gym.make('CliffWalking-v0')
# ###
print(env.action_space)
print(env.observation_space)

# ##############################
# Optimal state-value function #
# ##############################

V_opt = np.zeros((4, 12))
print(V_opt)
V_opt[0][0:13] = -np.arange(3, 15)[::-1]
V_opt[1][0:13] = -np.arange(3, 15)[::-1] + 1
V_opt[2][0:13] = -np.arange(3, 15)[::-1] + 2
V_opt[3][0] = -13
print(V_opt)

plot_values(V_opt)
plt.show()

# ###########################
# Part 1: TD Control: Sarsa #
# ###########################


def update_Q_sarsa(Qsa, Qsa_next, reward, alpha, gamma):
    """ updates the action-value function estimate using the most recent time step """
    return Qsa + (alpha * (reward + (gamma * Qsa_next) - Qsa))


def epsilon_greedy_probs(env, Q_s, epsilon):
    """ obtains the action probabilities corresponding to epsilon-greedy policy """
    policy_s = np.ones(env.nA) * epsilon / env.nA
    policy_s[np.argmax(Q_s)] = 1 - epsilon + (epsilon / env.nA)
    return policy_s


def sarsa(env, num_episodes, alpha, gamma=1.0, eps_start=1.0, eps_decay=.99999, eps_min=0.05, plot_every=100):
    Q = defaultdict(lambda: np.zeros(env.nA))  # initialize action-value function (empty dictionary of arrays)
    epsilon = eps_start                        # initialize epsilon
    # initialize performance monitor
    tmp_scores = deque(maxlen=plot_every)
    scores = deque(maxlen=num_episodes)
    # loop over episodes
    for i_episode in range(1, num_episodes+1):
        # monitor progress
        if i_episode % 100 == 0:
            print("\rEpisode {}/{}".format(i_episode, num_episodes), end="")
            sys.stdout.flush()
        # initialize score
        score = 0
        # begin an episode, observe S
        state = env.reset()
        # set value of epsilon
        epsilon = max(epsilon*eps_decay, eps_min)
        # get epsilon-greedy action probabilities
        policy_s = epsilon_greedy_probs(env, Q[state], epsilon)
        # pick action A
        action = np.random.choice(np.arange(env.nA), p=policy_s)
        # limit number of time steps per episode
        # for t_step in np.arange(300):
        while True:
            # take action A, observe R, S'
            next_state, reward, done, info = env.step(action)
            # add reward to score
            score += reward
            if not done:
                # get epsilon-greedy action probabilities
                policy_s = epsilon_greedy_probs(env, Q[next_state], epsilon)
                # pick next action A'
                next_action = np.random.choice(np.arange(env.nA), p=policy_s)
                # update TD estimate of Q
                Q[state][action] = update_Q_sarsa(Q[state][action], Q[next_state][next_action], reward, alpha, gamma)
                # S <- S'
                state = next_state
                # A <- A'
                action = next_action
            if done:
                # update TD estimate of Q
                Q[state][action] = update_Q_sarsa(Q[state][action], 0, reward, alpha, gamma)
                # append score
                tmp_scores.append(score)
                break
        if (i_episode % plot_every == 0):
            scores.append(np.mean(tmp_scores))
    # plot performance
    plt.plot(np.linspace(0, num_episodes, len(scores), endpoint=False), np.asarray(scores))
    plt.xlabel('Episode Number')
    plt.ylabel('Average Reward (Over Next %d Episodes)' % plot_every)
    plt.show()
    # print best 100-episode performance
    print(('Best Average Reward over %d Episodes: ' % plot_every), np.max(scores))
    return Q


# obtain the estimated optimal policy and corresponding action-value function
Q_sarsa = sarsa(env, num_episodes=5000, alpha=0.01, gamma=1.0, eps_start=1.0, eps_decay=0.5, eps_min=1/5000, plot_every=100)

# print the estimated optimal policy
policy_sarsa = np.array([np.argmax(Q_sarsa[key]) if key in Q_sarsa else -1 for key in np.arange(48)]).reshape(4, 12)
check_test.run_check('td_control_check', policy_sarsa)
print("\nEstimated Optimal Policy (UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3, N/A = -1):")
print(policy_sarsa)

# plot the estimated optimal state-value function
V_sarsa = ([np.max(Q_sarsa[key]) if key in Q_sarsa else 0 for key in np.arange(48)])
plot_values(V_sarsa)


# #########################################
# Part 2: TD Control: Q-learning Sarsamax #
# #########################################

def sarsamax(env, num_episodes, alpha, gamma=1.0, eps_start=1.0, eps_decay=.99999, eps_min=0.05, plot_every=100):
    Q = defaultdict(lambda: np.zeros(env.nA))  # initialize action-value function (empty dictionary of arrays)
    epsilon = eps_start                        # initialize epsilon
    # initialize performance monitor
    tmp_scores = deque(maxlen=plot_every)
    scores = deque(maxlen=num_episodes)
    # loop over episodes
    for i_episode in range(1, num_episodes+1):
        # monitor progress
        if i_episode % 100 == 0:
            print("\rEpisode {}/{}".format(i_episode, num_episodes), end="")
            sys.stdout.flush()
        # initialize score
        score = 0
        # begin an episode, observe S
        state = env.reset()
        # set value of epsilon
        epsilon = max(epsilon*eps_decay, eps_min)
        while True:
            # get epsilon-greedy action probabilities
            policy_s = epsilon_greedy_probs(env, Q[state], epsilon)
            # pick action A
            action = np.random.choice(np.arange(env.nA), p=policy_s)
            # take action A, observe R, S'
            next_state, reward, done, info = env.step(action)
            # add reward to score
            score += reward
            # pick next best action A'
            next_best_action = np.argmax(Q[next_state])
            # update TD estimate of Q
            Q[state][action] = update_Q_sarsa(Q[state][action], Q[next_state][next_best_action], reward, alpha, gamma)
            # S <- S'
            state = next_state
            # until S is terminal
            if done:
                # update TD estimate of Q
                Q[state][action] = update_Q_sarsa(Q[state][action], 0, reward, alpha, gamma)
                # append score
                tmp_scores.append(score)
                break
        if (i_episode % plot_every == 0):
            scores.append(np.mean(tmp_scores))
    # plot performance
    plt.plot(np.linspace(0, num_episodes, len(scores), endpoint=False), np.asarray(scores))
    plt.xlabel('Episode Number')
    plt.ylabel('Average Reward (Over Next %d Episodes)' % plot_every)
    plt.show()
    # print best 100-episode performance
    print(('Best Average Reward over %d Episodes: ' % plot_every), np.max(scores))
    return Q


# obtain the estimated optimal policy and corresponding action-value function
Q_sarsamax = sarsamax(env, num_episodes=5000, alpha=0.01, gamma=1.0, eps_start=1.0, eps_decay=0.1, eps_min=1/5000, plot_every=100)

# print the estimated optimal policy
policy_sarsamax = np.array([np.argmax(Q_sarsamax[key]) if key in Q_sarsamax else -1 for key in np.arange(48)]).reshape((4, 12))
check_test.run_check('td_control_check', policy_sarsamax)
print("\nEstimated Optimal Policy (UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3, N/A = -1):")
print(policy_sarsamax)

# plot the estimated optimal state-value function
plot_values([np.max(Q_sarsamax[key]) if key in Q_sarsamax else 0 for key in np.arange(48)])


# ####################################
# Part 3: TD Control: Expected Sarsa #
# ####################################


def expsarsa(env, num_episodes, alpha, gamma=1.0, eps_start=1.0, eps_decay=.99999, eps_min=0.05, plot_every=100):
    Q = defaultdict(lambda: np.zeros(env.nA))  # initialize action-value function (empty dictionary of arrays)
    epsilon = eps_start                        # initialize epsilon
    # initialize performance monitor
    tmp_scores = deque(maxlen=plot_every)
    scores = deque(maxlen=num_episodes)
    # loop over episodes
    for i_episode in range(1, num_episodes+1):
        # monitor progress
        if i_episode % 100 == 0:
            print("\rEpisode {}/{}".format(i_episode, num_episodes), end="")
            sys.stdout.flush()
        # initialize score
        score = 0
        # begin an episode, observe S
        state = env.reset()
        # set value of epsilon
        epsilon = max(epsilon*eps_decay, eps_min)
        while True:
            # get epsilon-greedy action probabilities
            policy_s = epsilon_greedy_probs(env, Q[state], epsilon)
            # pick action A
            action = np.random.choice(np.arange(env.nA), p=policy_s)
            # take action A, observe R, S'
            next_state, reward, done, info = env.step(action)
            # add reward to score
            score += reward
            # pick next best action A'
            policy_next_s = epsilon_greedy_probs(env, Q[next_state], epsilon)
            exp_next_Q = np.dot(Q[next_state], policy_next_s)
            # update TD estimate of Q
            Q[state][action] = update_Q_sarsa(Q[state][action], exp_next_Q, reward, alpha, gamma)
            # S <- S'
            state = next_state
            # until S is terminal
            if done:
                # update TD estimate of Q
                Q[state][action] = update_Q_sarsa(Q[state][action], 0, reward, alpha, gamma)
                # append score
                tmp_scores.append(score)
                break
        if (i_episode % plot_every == 0):
            scores.append(np.mean(tmp_scores))
    # plot performance
    plt.plot(np.linspace(0, num_episodes, len(scores), endpoint=False), np.asarray(scores))
    plt.xlabel('Episode Number')
    plt.ylabel('Average Reward (Over Next %d Episodes)' % plot_every)
    plt.show()
    # print best 100-episode performance
    print(('Best Average Reward over %d Episodes: ' % plot_every), np.max(scores))
    return Q


# obtain the estimated optimal policy and corresponding action-value function
Q_expsarsa = expsarsa(env, num_episodes=5000, alpha=0.1, gamma=1.0, eps_start=1.0, eps_decay=0.5, eps_min=1/5000, plot_every=100)

# print the estimated optimal policy
policy_expsarsa = np.array([np.argmax(Q_expsarsa[key]) if key in Q_expsarsa else -1 for key in np.arange(48)]).reshape(4, 12)
check_test.run_check('td_control_check', policy_expsarsa)
print("\nEstimated Optimal Policy (UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3, N/A = -1):")
print(policy_expsarsa)

# plot the estimated optimal state-value function
plot_values([np.max(Q_expsarsa[key]) if key in Q_expsarsa else 0 for key in np.arange(48)])
