# main code that contains the neural network setup
# policy + critic updates
# see ddpg.py for other details in the network

from ddpg import DDPGAgent
import torch
import torch.nn.functional as F
from utilities import soft_update, transpose_to_tensor, transpose_list, tensor_flatten
import numpy as np

import pdb
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'


class MADDPG:
    def __init__(self, state_size, action_size, num_agents,
                 discount_factor=1, tau=0.02, seed=1, device='cpu'):
        super(MADDPG, self).__init__()

        # critic input = obs_full + actions = 24+2+2=28
#        self.maddpg_agent = [DDPGAgent(state_size, action_size, num_agents,
#                                       hidden_in_actor=256, hidden_out_actor=128,
#                                       hidden_in_critic=256, hidden_out_critic=128,
#                                       seed=seed, device=device),
#                             DDPGAgent(state_size, action_size, num_agents,
#                                       hidden_in_actor=256, hidden_out_actor=128,
#                                       hidden_in_critic=256, hidden_out_critic=128,
#                                       seed=seed, device=device)]

        self.maddpg_agent = [DDPGAgent(state_size, action_size, num_agents,
                                       hidden_in_actor=512, hidden_out_actor=256,
                                       hidden_in_critic=512, hidden_out_critic=256,
                                       seed=seed, device=device,
                                       lr_actor=1e-4, lr_critic=3e-4, weight_decay_critic=0) for _ in range(num_agents)]

        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.discount_factor = discount_factor
        self.tau = tau
        self.seed = seed
        self.device = device
        self.update_count = 0

    def reset(self):
        for ddpg_agent in self.maddpg_agent:
            ddpg_agent.reset()

    def actor(self, obs_all_agents):
        actions = [ddpg_agent.actor(obs) for ddpg_agent, obs in zip(self.maddpg_agent, obs_all_agents)]
        return actions

    def target_actor(self, obs_all_agents):
        target_actions = [ddpg_agent.target_actor(obs) for ddpg_agent, obs in zip(self.maddpg_agent, obs_all_agents)]
        return target_actions

    def act(self, obs_all_agents, noise_factor=0.0):
        """get actions from all agents in the MADDPG object"""
        actions = [ddpg_agent.act(obs, noise_factor) for ddpg_agent, obs in zip(self.maddpg_agent, obs_all_agents)]
        return np.squeeze(actions)

    def target_act(self, obs_all_agents):
        """get target network actions from all the agents in the MADDPG object """
        target_actions = [ddpg_agent.target_act(obs) for ddpg_agent, obs in zip(self.maddpg_agent, obs_all_agents)]
        return np.squeeze(target_actions)

    def update(self, samples, agent_number, logger):
        """update the critics and actors of all the agents """

        # --- Experiences ---
        states = torch.from_numpy(np.stack(transpose_list([e.state for e in samples if e is not None]))).float().to(self.device)
        actions = torch.from_numpy(np.stack(transpose_list([e.action for e in samples if e is not None]))).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([max(e.reward) for e in samples if e is not None])).float().to(self.device).t()[0]
        next_states = torch.from_numpy(np.stack(transpose_list([e.next_state for e in samples if e is not None]))).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in samples if e is not None]).astype(np.uint8)).float().to(self.device)

        # --- Agent --------------------#
        agent = self.maddpg_agent[agent_number]
        # pdb.set_trace()
        # ------- Update Critic ------------------------------------------#
        agent.critic_optimizer.zero_grad()

        states_flat = tensor_flatten(states, self.state_size*self.num_agents)[0]
        actions_flat = tensor_flatten(actions, self.action_size*self.num_agents)[0]

        next_states_flat = tensor_flatten(next_states, self.state_size*self.num_agents)[0]

        # target_next_actions = torch.from_numpy(self.target_actor(next_states)).float().to(self.device)
        target_next_actions_flat = torch.cat(self.target_actor(next_states), dim=1)
        # target_next_actions_flat = tensor_flatten(target_next_actions, self.action_size*self.num_agents)[0]

        with torch.no_grad():
            target_next_q = agent.target_critic(next_states_flat, target_next_actions_flat).t()[0]

        # target_q = rewards[:, agent_number] + self.discount_factor * target_next_q * (1 - dones[:, agent_number])
        target_q = rewards + self.discount_factor * target_next_q * (1 - dones[:, agent_number])
        local_q = agent.critic(states_flat, actions_flat).t()[0]

        critic_loss = F.mse_loss(local_q, target_q)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 1)  #
        agent.critic_optimizer.step()

        # ------- Update Actor ------------------------------------------#
        agent.actor_optimizer.zero_grad()

        local_actions_flat = torch.cat(self.actor(states), dim=1)
        # local_actions = torch.from_numpy(self.actor(states)).float().to(self.device)
        # local_actions_flat = tensor_flatten(local_actions, self.action_size*self.num_agents)[0]

        actor_loss = -agent.critic(states_flat, local_actions_flat).mean()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 1)  #
        agent.actor_optimizer.step()

        # ---- Save Loss -------------------------------#
        aloss = actor_loss.cpu().detach().item()
        closs = critic_loss.cpu().detach().item()
        logger.add_scalars('agent%i/losses' % agent_number, {'critic loss': closs, 'actor_loss': aloss}, self.update_count)

    def update_targets(self):
        """soft update targets"""
        self.update_count += 1
        for ddpg_agent in self.maddpg_agent:
            soft_update(ddpg_agent.target_actor, ddpg_agent.actor, self.tau)
            soft_update(ddpg_agent.target_critic, ddpg_agent.critic, self.tau)
