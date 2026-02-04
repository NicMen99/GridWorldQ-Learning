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
    lr_decay_factor = 0.0001

    load_and_test = True

    buffer_size = 10000
    buffer = PrioritizedBuffer.PrioritizedBuffer(buffer_size, n_episodes)

    gym.envs.registration.register(
        id = 'GridWorld-v0',
        entry_point = 'Environment:Environment',
        max_episode_steps = 300
    )

    env = gym.make('GridWorld-v0', grid_size=(10, 10), target_positions=np.array([[7, 4], [0, 2], [4, 8], [9, 0], [1, 6]]), render_mode = 'rgb_array')

    agent = Agent(env, learning_rate, final_lr, start_epsilon, final_epsilon, buffer, discount, epsilon_decay_factor, lr_decay_factor)

    if not load_and_test:
        Train.train_record(env, agent, n_episodes, period= 5000, show_results=True)
        agent.save_table_on_file(filename = "final_table.json")
    else:
        agent.load_table_from_file(filename="best_table.json")

    Train.test_record(env, agent, 20)
