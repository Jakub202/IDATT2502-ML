import gym
import numpy as np
import time
import matplotlib.pyplot as plt

from CartPoleFunctions import CartPoleFunctions

env=gym.make('CartPole-v1')
(state,_)=env.reset()


# here define the parameters for state discretization
upper_bounds = env.observation_space.high
lower_bounds = env.observation_space.low
cart_velocity_min = -3
cart_velocity_max = 3
pole_angle_velocity_min = -10
pole_angle_velocity_max = 10
upper_bounds[1] = cart_velocity_max
upper_bounds[3] = pole_angle_velocity_max
lower_bounds[1] = cart_velocity_min
lower_bounds[3] = pole_angle_velocity_min

number_of_bins_position = 30
number_of_bins_velocity = 30
number_of_bins_angle = 30
number_of_bins_angle_velocity = 30
number_of_bins = [number_of_bins_position, number_of_bins_velocity, number_of_bins_angle, number_of_bins_angle_velocity]

# define the parameters
alpha=0.1
gamma=1
epsilon=0.2
numberEpisodes=15000

Q1 = CartPoleFunctions(env, alpha, gamma, epsilon, numberEpisodes, number_of_bins, lower_bounds, upper_bounds)

Q1.simulateEpisodes()
# simulate the learned strategy
(obtainedRewardsOptimal,env1)=Q1.simulateLearnedStrategy()

plt.figure(figsize=(12, 5))
# plot the figure and adjust the plot parameters
plt.plot(Q1.sumRewardsEpisode,color='blue',linewidth=1)
plt.xlabel('Episode')
plt.ylabel('Sum of Rewards in Episode')
plt.yscale('log')
plt.savefig('convergence.png')
plt.show()

# close the environment
env1.close()
# get the sum of rewards
np.sum(obtainedRewardsOptimal)

# now simulate a random strategy
(obtainedRewardsRandom,env2)=Q1.simulateRandomStrategy()
plt.hist(obtainedRewardsRandom)
plt.xlabel('Sum of rewards')
plt.ylabel('Percentage')
plt.savefig('histogram.png')
plt.show()

# run this several times and compare with a random learning strategy
(obtainedRewardsOptimal,env1)=Q1.simulateLearnedStrategy()
