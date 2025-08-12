import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium.wrappers import RecordEpisodeStatistics

import Agent
from tqdm import tqdm

def test_record(env: gym.Env, agent:Agent, n_episodes=10):
    reg_env = gym.wrappers.RecordVideo(env, video_folder="robot_training/eval", name_prefix="eval", episode_trigger=lambda x: True)

    reg_env = RecordEpisodeStatistics(reg_env)

    episode_metrics = {
        "rewards": [],
        "steps": [],
        "success": []
    }

    old_epsilon = agent.epsilon
    agent.epsilon = 0.0

    for _ in tqdm(range(n_episodes)):
        obs, info = reg_env.reset()
        done = False

        while not done:
            action = agent.epsilon_greedy_policy(obs)
            obs, reward, terminated, truncated, info = reg_env.step(action)
            done = terminated or truncated

        episode_metrics["rewards"].append(info["episode"]["r"])
        episode_metrics["steps"].append(info["episode"]["l"])
        episode_metrics["success"].append(1 if info["targets reached"]==3 else 0)

    agent.epsilon = old_epsilon

    reg_env.close()

    avg_reward = sum(episode_metrics["rewards"]) / len(episode_metrics["rewards"])
    std_reward = np.std(episode_metrics["rewards"])
    avg_steps = sum(episode_metrics["steps"]) / len(episode_metrics["steps"])
    std_steps = np.std(episode_metrics["steps"])

    success_rate = np.sum(episode_metrics["success"]) / len(episode_metrics["success"])

    print('\nEval Summary')
    print(f'Episode rewards: {avg_reward:.2f} +- {std_reward:.2f}')
    print(f'Episode steps: {avg_steps} +- {std_steps}')
    print(f'Success rate: {success_rate:.2f}%')
    # Aggiungi metriche che possono essere utili
    # MFA
    # Stabilità della Q-Table ?


def train_record(env: gym.Env, agent:Agent, n_episodes=1000, period = 500, show_results = False):
    reg_env = gym.wrappers.RecordVideo(env, video_folder="robot_training/training", name_prefix="training", episode_trigger=lambda x: x % period == 0)

    reg_env = RecordEpisodeStatistics(reg_env)

    episode_metrics = {
        "rewards" : [],
        "steps": [],
        "targets" : [],
        "success": []
    }

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

        episode_metrics["rewards"].append(info["episode"]["r"])
        episode_metrics["steps"].append(info["episode"]["l"])
        episode_metrics["targets"].append(info["targets reached"])
        episode_metrics["success"].append(1 if info["targets reached"]==3 else 0)

    if show_results:
        fig, axs = plt.subplots(2, 1)

        data1 = episode_metrics["rewards"]
        data2 = episode_metrics["steps"]
        data3 = episode_metrics["targets"]
        data4 = episode_metrics["success"]

        window = int(n_episodes / 50)
        data1 = np.convolve(data1, np.ones(window) / window, "same")
        data2 = np.convolve(data2, np.ones(window) / window, "same")
        data3 = np.convolve(data3, np.ones(window) / window, "same")
        data4 = np.convolve(data4, np.ones(window) / window, "same")

        color = "tab:blue"
        axs[0].set_ylabel("Reward", color=color)
        axs[0].plot(data1, color=color)

        axs01 = axs[0].twinx()
        color = "tab:red"
        axs01.set_ylabel("Number of Steps", color=color)
        axs01.plot(data2, color=color)

        color = "tab:blue"
        axs[1].set_xlabel("Episodes")
        axs[1].set_ylabel("Success Rate", color=color)
        axs[1].plot(data4, color=color)

        axs11 = axs[1].twinx()
        color = "tab:red"
        axs11.set_ylabel("Objectives reached", color=color)
        axs11.plot(data3, color=color)

        fig.suptitle("Episode Metrics")
        fig.tight_layout()
        plt.show()

    reg_env.close()

