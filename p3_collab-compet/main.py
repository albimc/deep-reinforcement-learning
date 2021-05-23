# main function that sets up environments
# perform training loop
from unityagents import UnityEnvironment
# import envs
from replaybuffer import ReplayBuffer
from maddpg import MADDPG
import torch
import numpy as np
from tensorboardX import SummaryWriter
import progressbar as pb
import time

from collections import deque
import matplotlib.pyplot as plt

import os
import pdb
# pdb.set_trace()
# keep training awake
# from workspace_utils import keep_awake


def seeding(seed=1):
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'
    # seed
    seed = 1
    seeding(seed)
    # number of training episodes.
    number_of_episodes = 50000
    episode_length = 1000
    buffer_size = int(1e5)
    batchsize = 512
    # how many episodes before update
    episode_per_update = 2
    # how many episodes to save policy and gif
    save_interval = 100
    #
    discount_factor = 1
    tau = 1e-2
    # amplitude of OU noise
    # this slowly decreases to 0
    noise_factor = 2
    noise_reduction = 0.9999
    noise_floor = 0.0

    # logger
    log_path = os.getcwd()+"/log"
    model_dir = os.getcwd()+"/model_dir"
    logger = SummaryWriter(log_dir=log_path)
    os.makedirs(model_dir, exist_ok=True)

    ####################
    # Load Environment #
    ####################

    # torch.set_num_threads(parallel_envs)
    # env = envs.make_parallel_env(parallel_envs)
    env = UnityEnvironment(file_name="./Tennis_Linux_NoVis/Tennis.x86_64")

    # Get brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    print('Brain Name:', brain_name)

    # Reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # Number of Agents
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)

    # size of each action
    action_size = brain.vector_action_space_size
    print('Size of each action:', action_size)

    # examine the state space
    states = env_info.vector_observations
    state_size = states.shape[1]
    print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))

    # Replay Buffer
    rebuffer = ReplayBuffer(buffer_size, seed, device)

    # initialize Multi Agent policy and critic
    maddpg = MADDPG(state_size, action_size, num_agents, discount_factor, tau, seed, device)

    load_model = True
    if load_model:
        load_dict_list = torch.load(os.path.join(model_dir, 'episode-saved.pt'))
        for i in range(num_agents):
            maddpg.maddpg_agent[i].actor.load_state_dict(load_dict_list[i]['actor_params'])
            maddpg.maddpg_agent[i].actor_optimizer.load_state_dict(load_dict_list[i]['actor_optim_params'])
            maddpg.maddpg_agent[i].critic.load_state_dict(load_dict_list[i]['critic_params'])
            maddpg.maddpg_agent[i].critic_optimizer.load_state_dict(load_dict_list[i]['critic_optim_params'])

    # initialize scores
    scores_history = []
    scores_window = deque(maxlen=save_interval)

    ####################
    # Show Progressbar #
    ####################
    widget = ['episode: ', pb.Counter(), '/', str(number_of_episodes), ' ',
              pb.Percentage(), ' ', pb.ETA(), ' ', pb.Bar(marker=pb.RotatingMarker()), ' ']

    timer = pb.ProgressBar(widgets=widget, maxval=number_of_episodes).start()
    start = time.time()

    ################
    # TRINING LOOP #
    ################
    i_episode = 0

    for i_episode in range(number_of_episodes):

        timer.update(i_episode)
        # reset and reduce noise_factor
        maddpg.reset()
        noise_factor = max(noise_floor, noise_factor*noise_reduction)

        env_info = env.reset(train_mode=True)[brain_name]  # Reset Environmet
        states = env_info.vector_observations
        scores = np.zeros(num_agents)

        # save info or not
        save_info = ((i_episode) % save_interval == 0 or i_episode == number_of_episodes)

        # episode_t = 0
        for episode_t in range(episode_length):

            # explore with decaying factor
            actions = maddpg.act(states, noise_factor=noise_factor)
            env_info = env.step(actions)[brain_name]             # Environment reacts
            next_states = env_info.vector_observations           # get the next state
            rewards = env_info.rewards                           # get the reward
            dones = env_info.local_done                          # see if episode has finished

            # for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            #     rebuffer.add(state, action, reward, next_state, done)

            rebuffer.add(states, actions, rewards, next_states, dones)

            scores += rewards
            states = next_states

            if any(dones):
                break

            # update once after every episode_per_update
            # if len(rebuffer) > batchsize and episode_t % episode_per_update == 0:
            #     for a_i in range(num_agents):
            #         samples = rebuffer.sample(batchsize)
            #         maddpg.update(samples, a_i, logger)
            #     maddpg.update_targets()  # soft update the target network towards the actual networks

        scores_window.append(np.max(scores))        # save most recent score
        scores_history.append(np.max(scores))       # save most recent score

        # update once after every episode_per_update
        if len(rebuffer) > batchsize and i_episode % episode_per_update == 0:
            for a_i in range(num_agents):
                samples = rebuffer.sample(batchsize)
                maddpg.update(samples, a_i, logger)
            maddpg.update_targets()  # soft update the target network towards the actual networks

        if i_episode % save_interval == 0 or i_episode == number_of_episodes-1:
            end = time.time()
            avg_rewards = np.mean(scores_window)
            logger.add_scalars('rewards', {'Avg Reward': avg_rewards, 'Noise Factor': noise_factor}, i_episode)
            print('\nElapsed time {:.1f} \t Update Count {}'.format((end - start)/60, maddpg.update_count),
                  '\nEpisode {} \tEpisode t {}\tAverage Score: {:.2f} Noise Factor {:2f}'.format(i_episode, episode_t, np.mean(scores_window), noise_factor), end="\n")
        # if np.mean(scores_window) > 0.5:
        #    break

        # saving model
        save_dict_list = []
        if save_info:
            for i in range(num_agents):

                save_dict = {'actor_params': maddpg.maddpg_agent[i].actor.state_dict(),
                             'actor_optim_params': maddpg.maddpg_agent[i].actor_optimizer.state_dict(),
                             'critic_params': maddpg.maddpg_agent[i].critic.state_dict(),
                             'critic_optim_params': maddpg.maddpg_agent[i].critic_optimizer.state_dict()}
                save_dict_list.append(save_dict)

                torch.save(save_dict_list, os.path.join(model_dir, 'episode-{}.pt'.format(i_episode)))

    env.close()
    logger.close()
    timer.finish()

    return scores_window, scores_history


if __name__ == '__main__':
    scores_window, scores_history = main()
