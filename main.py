import gymnasium as gym
import numpy as np

import PrioritizedBuffer
import Train
from Agent import Agent

if __name__ == '__main__':
    n_episodes = 2500000
    learning_rate = 1
    final_lr = 0.0
    discount = 0.95
    start_epsilon = 1
    final_epsilon = 0.0
    epsilon_decay_factor = 0.001
    lr_decay_factor = 0.001

    load_and_test = False

    buffer_size = 10000
    buffer = PrioritizedBuffer.PrioritizedBuffer(buffer_size, n_episodes)

    gym.envs.registration.register(
        id = 'GridWorld-v0',
        entry_point = 'Environment:Environment',
        max_episode_steps = 300
    )

    env = gym.make('GridWorld-v0', grid_size=(10, 10), target_positions=np.array([[1, 3], [6, 7], [9,2], [1, 8], [5, 1]]), render_mode = 'rgb_array')
    # env = gym.make('GridWorld-v0', grid_size=(5, 5), target_positions=np.array([[1, 3], [4, 2], [4, 4]]), render_mode = 'rgb_array')

    agent = Agent(env, learning_rate, final_lr, start_epsilon, final_epsilon, buffer, discount, epsilon_decay_factor, lr_decay_factor)

    if not load_and_test:
        Train.train_record(env, agent, n_episodes, period= 5000, show_results=True)
        agent.save_table_on_file()
    else:
        agent.load_table_from_file()

    Train.test_record(env, agent, 20)
