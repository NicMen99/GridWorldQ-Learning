import gymnasium as gym

if __name__ == '__main__':
    gym.envs.registration.register(
        id = 'GridWorld-v0',
        entry_point = 'Environment:Environment',
        max_episode_steps = 300
    )

    env = gym.make('GridWorld-v0')
    pass
