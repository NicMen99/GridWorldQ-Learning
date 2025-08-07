import gymnasium as gym
import numpy as np

import Train
from Agent import Agent

if __name__ == '__main__':
    n_episodes = 200000
    learning_rate = 0.01
    discount = 0.9
    start_epsilon = 1
    epsilon_decay = start_epsilon /(n_episodes/2)
    final_epsilon = 0.1


    gym.envs.registration.register(
        id = 'GridWorld-v0',
        entry_point = 'Environment:Environment',
        max_episode_steps = 300
    )

    env = gym.make('GridWorld-v0', grid_size=(5, 5), target_positions=np.array([[2, 3], [1, 4], [4, 4]]))
    print(env.observation_space)

    agent = Agent(env, learning_rate, start_epsilon, epsilon_decay, final_epsilon, discount)

    print(dict(agent.QTable))

    Train.train(env, agent, n_episodes, show_results=True)

    print("\n\n")
    print(dict(agent.QTable))
