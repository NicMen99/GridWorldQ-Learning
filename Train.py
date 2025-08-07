import gymnasium as gym
import matplotlib.pyplot as plt
import Agent
from tqdm import tqdm

def train(env: gym.Env, agent:Agent, n_episodes=1000, show_results = False
          ):
    episode_rewards = []
    for episode in tqdm(range(n_episodes)):
        obs, info = env.reset()
        done = False

        total_reward = 0

        while not done:
            action = agent.epsilon_greedy_policy(obs)

            next_obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward

            agent.update_table(obs, action, reward, terminated, next_obs)

            done = terminated or truncated
            obs = next_obs

        agent.decay_epsilon()

        episode_rewards.append(total_reward)

    if show_results:
        plt.plot(episode_rewards)
        plt.show()