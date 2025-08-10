import gymnasium as gym
import matplotlib.pyplot as plt
from gymnasium.wrappers import RecordEpisodeStatistics

import Agent
from tqdm import tqdm

def train(env: gym.Env, agent:Agent, n_episodes=1000, show_results = False):
    episode_rewards = []
    for _ in tqdm(range(n_episodes)):
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

def test(env: gym.Env, agent:Agent, n_episodes=1000):
    episode_rewards = []

    old_epsilon = agent.epsilon
    agent.epsilon = 0.0

    for _ in tqdm(range(n_episodes)):
        obs, info = env.reset()
        done = False

        total_reward = 0

        while not done:
            action = agent.epsilon_greedy_policy(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        episode_rewards.append(total_reward)

    agent.epsilon = old_epsilon

    plt.plot(episode_rewards)
    plt.show()

    # Aggiungi metriche che possono essere utili
    # Ricompensa media
    # Success rate
    # lungezza dell'episodio (passi e tempo)
    # Varianza nelle ricompense
    # MFA
    # Stabilità della Q-Table ?

def train_record(env: gym.Env, agent:Agent, n_episodes=1000, period = 500):
    reg_env = gym.wrappers.RecordVideo(env, video_folder="robot_training", name_prefix="training", episode_trigger=lambda x: x % period == 0)

    reg_env = RecordEpisodeStatistics(reg_env)

    for _ in tqdm(range(n_episodes)):
        obs, info = reg_env.reset()
        done = False

        while not done:
            action = agent.epsilon_greedy_policy(obs)

            next_obs, reward, terminated, truncated, info = reg_env.step(action)

            agent.update_table(obs, action, reward, terminated, next_obs)

            done = terminated or truncated
            obs = next_obs

        agent.decay_epsilon()

    reg_env.close()