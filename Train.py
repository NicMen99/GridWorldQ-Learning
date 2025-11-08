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
        "targets": [],
        "success": []
    }

    old_epsilon = agent.current_epsilon
    agent.current_epsilon = 0.0

    for _ in tqdm(range(n_episodes)):
        obs, info = reg_env.reset()
        done = False

        while not done:
            action = agent.epsilon_greedy_policy(obs)
            obs, reward, terminated, truncated, info = reg_env.step(action)
            done = terminated or truncated

        episode_metrics["rewards"].append(info["episode"]["r"])
        episode_metrics["steps"].append(info["episode"]["l"])
        episode_metrics["targets"].append(info["targets reached"])
        episode_metrics["success"].append(1 if info["targets reached"]==len(env.unwrapped.world_targets) and np.array_equal(obs["agent position"], info["starting position"]) else 0)

    agent.current_epsilon = old_epsilon

    reg_env.close()

    avg_reward = sum(episode_metrics["rewards"]) / len(episode_metrics["rewards"])
    std_reward = np.std(episode_metrics["rewards"])
    avg_steps = sum(episode_metrics["steps"]) / len(episode_metrics["steps"])
    std_steps = np.std(episode_metrics["steps"])
    avg_targets_per_episode = np.sum(episode_metrics["targets"]) / len(episode_metrics["targets"])
    std_targets = np.std(episode_metrics["targets"])

    success_rate = np.sum(episode_metrics["success"]) / len(episode_metrics["success"]) * 100

    print('\nEval Summary')
    print(f'Episode rewards: {avg_reward:.2f} +- {std_reward:.2f}')
    print(f'Episode steps: {avg_steps} +- {std_steps}')
    print(f'Targets_reached: {avg_targets_per_episode:.2f}/{len(env.unwrapped.world_targets)} +- {std_targets:.2f}')
    print(f'Success rate: {success_rate:.2f}%')

def train_record(env: gym.Env, agent:Agent, n_episodes=1000, period = 500, show_results = False):
    reg_env = gym.wrappers.RecordVideo(env, video_folder="robot_training/training", name_prefix="training", episode_trigger=lambda x: x % period == 0)

    reg_env = RecordEpisodeStatistics(reg_env)

    episode_metrics = {
        "rewards" : [],
        "steps": [],
        "targets" : [],
        "success": []
    }

    checkpoints = []

    for i in tqdm(range(n_episodes)):
        obs, info = reg_env.reset()
        done = False

        while not done:
            action = agent.epsilon_greedy_policy(obs)

            next_obs, reward, terminated, truncated, info = reg_env.step(action)

            agent.update_table(obs, action, reward, terminated, next_obs)

            done = terminated or truncated
            obs = next_obs

        agent.decay()

        episode_metrics["rewards"].append(info["episode"]["r"])
        episode_metrics["steps"].append(info["episode"]["l"])
        episode_metrics["targets"].append(info["targets reached"])
        episode_metrics["success"].append(1 if info["targets reached"]==len(env.unwrapped.world_targets) and np.array_equal(obs["agent position"], info["starting position"]) else 0)

        if i % period == 0:
            old_epsilon = agent.epsilon
            agent.epsilon = 0.0

            obs, info = reg_env.reset()
            done = False

            while not done:
                action = agent.epsilon_greedy_policy(obs)
                obs, reward, terminated, truncated, info = reg_env.step(action)
                done = terminated or truncated

            checkpoints.append(info["episode"]["r"])

            agent.epsilon = old_epsilon

    if show_results:
        fig, axs = plt.subplots(3, 1)

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
        axs[1].set_ylabel("Reward", color=color)
        axs[1].plot(checkpoints, color=color)

        color = "tab:blue"
        axs[2].set_xlabel("Episodes")
        axs[2].set_ylabel("Success Rate", color=color)
        axs[2].plot(data4, color=color)

        axs21 = axs[2].twinx()
        color = "tab:red"
        axs21.set_ylabel("Objectives reached", color=color)
        axs21.plot(data3, color=color)

        fig.suptitle("Episode Metrics")
        fig.tight_layout()
        plt.show()

    reg_env.close()

