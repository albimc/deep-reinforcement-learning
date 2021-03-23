import utils
import matplotlib.pyplot as plt
import numpy as np
import syntheticChrissAlmgren as sca
from ddpg_agent import Agent
from collections import deque


# Set the default figure size
plt.rcParams['figure.figsize'] = [17.0, 7.0]

# Set the number of days to follow the stock price
n_days = 100


# PRICE DYNAMICS #

# Plot the stock price as a function of time
utils.plot_price_model(seed=0, num_days=n_days)


# MARKET IMPACT #
# Set the number of days to sell all shares (i.e. the liquidation time)
l_time = 60

# Set the number of trades
n_trades = 60

# Set the trader's risk aversion
t_risk = 1e-9

# Plot the trading list and trading trajectory. If show_trl = True, the data frame containing the values of the
# trading list and trading trajectory is printed
foo = utils.plot_trade_list(lq_time=l_time, nm_trades=n_trades, tr_risk=t_risk, show_trl=True)
print(str(foo))


# Get the default financial and AC Model parameters
financial_params, ac_params = utils.get_env_param()

print(str(financial_params))
print(str(ac_params))

# #
# Set the random seed
sd = 0

# Set the number of days to sell all shares (i.e. the liquidation time)
l_time = 60

# Set the number of trades
n_trades = 60

# Set the trader's risk aversion
t_risk = 1e-6

# Implement the trading list for the given parameters
utils.implement_trade_list(seed=sd, lq_time=l_time, nm_trades=n_trades, tr_risk=t_risk)


# Set the liquidation time
l_time = 60

# Set the number of trades
n_trades = 60

# Set trader's risk aversion
t_risk = 1e-6

# Set the number of episodes to run the simulation
episodes = 100

utils.get_av_std(lq_time=l_time, nm_trades=n_trades, tr_risk=t_risk, trs=episodes)
plt.show()

# Get the AC Optimal strategy for the given parameters
ac_strategy = utils.get_optimal_vals(lq_time=l_time, nm_trades=n_trades, tr_risk=t_risk)
print(str(ac_strategy))

# Get the minimum impact and minimum variance strategies
minimum_impact, minimum_variance = utils.get_min_param()
print(str(minimum_impact))
print(str(minimum_variance))

# Plot the efficient frontier for the default values. The plot points out the expected shortfall and variance of the
# optimal strategy for the given the trader's risk aversion. Valid range for the trader's risk aversion (1e-7, 1e-4).
utils.plot_efficient_frontier(tr_risk=1e-6)
plt.show()

# ###################
# OPTIMAL EXECUTION #
# ###################

# Get the default financial and AC Model parameters
financial_params, ac_params = utils.get_env_param()
print(str(financial_params))
print(str(ac_params))

# Create simulation environment
env = sca.MarketEnvironment()

# Initialize Feed-forward DNNs for Actor and Critic models.
agent = Agent(state_size=env.observation_space_dimension(), action_size=env.action_space_dimension(), random_seed=0)

# Set the liquidation time
lqt = 60
# Set the number of trades
n_trades = 60
# Set trader's risk aversion
tr = 1e-6
# Set the number of episodes to run the simulation
episodes = 10000

shortfall_hist = np.array([])
shortfall_deque = deque(maxlen=100)

for episode in range(episodes):
    # Reset the enviroment
    cur_state = env.reset(seed=episode, liquid_time=lqt, num_trades=n_trades, lamb=tr)

    # set the environment to make transactions
    env.start_transactions()

    for i in range(n_trades + 1):

        # Predict the best action for the current state.
        action = agent.act(cur_state, add_noise=True)

        # Action is performed and new state, reward, info are received.
        new_state, reward, done, info = env.step(action)

        # current state, action, reward, new state are stored in the experience replay
        agent.step(cur_state, action, reward, new_state, done)

        # roll over new state
        cur_state = new_state

        if info.done:
            shortfall_hist = np.append(shortfall_hist, info.implementation_shortfall)
            shortfall_deque.append(info.implementation_shortfall)
            break

    if (episode + 1) % 100 == 0:  # print average shortfall over last 100 episodes
        print('\rEpisode [{}/{}]\tAverage Shortfall: ${:,.2f}'.format(episode + 1, episodes, np.mean(shortfall_deque)))

print('\nAverage Implementation Shortfall: ${:,.2f} \n'.format(np.mean(shortfall_hist)))
