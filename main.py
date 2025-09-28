import gymnasium as gym
import numpy as np

import Train
from Agent import Agent

if __name__ == '__main__':
    n_episodes = 5000000
    learning_rate = 0.01
    learning_rate = 1
    discount = 0.95
    start_epsilon = 1
    final_epsilon = 0.01
    epsilon_decay_factor = 0.0001
    lr_decay_factor = 0.0001


    gym.envs.registration.register(
        id = 'GridWorld-v0',
        entry_point = 'Environment:Environment',
        max_episode_steps = 300
    )

    env = gym.make('GridWorld-v0', grid_size=(10, 10), target_positions=np.array([[1, 3], [6, 7], [9,2]]), render_mode = 'rgb_array')

    agent = Agent(env, learning_rate, start_epsilon, final_epsilon, discount)

    Train.train_record(env, agent, n_episodes, period= 5000, show_results=True)

    Train.test_record(env, agent, 5)
