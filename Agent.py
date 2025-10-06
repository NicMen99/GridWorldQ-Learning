from collections import defaultdict
import numpy as np
import gymnasium as gym
import random

class Agent:
    def __init__(self,
                 env: gym.Env,
                 learning_rate: float,
                 final_lr: float,
                 initial_epsilon: float,
                 final_epsilon: float,
                 discount_factor: float = 0.95,
                 epsilon_decay_factor: float = 0.01,
                 lr_decay_factor: float = 0.01
                 ):
        self.env = env

        self.QTable = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = learning_rate
        self.final_lr = final_lr
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.epsilon_decay_factor = epsilon_decay_factor
        self.lr_decay_factor = lr_decay_factor

        self.episode_count = 0
        self.current_lr = self.lr
        self.current_epsilon = self.epsilon

    def epsilon_greedy_policy(self, state: dict):
        state = (tuple(state['agent position']), state['agent charge'], tuple(state['visited positions']))

        r = random.random()
        if r < self.current_epsilon:
            return self.env.action_space.sample()
        else:
            return int(np.argmax(self.QTable[state]))

    def update_table(self, state: dict,
                     action: int,
                     reward: float,
                     terminated: bool,
                     next_state: dict):

        state = (tuple(state['agent position']), state['agent charge'], tuple(state['visited positions']))
        next_state = (tuple(next_state['agent position']), next_state['agent charge'], tuple(next_state['visited positions']))

        future_q = (not terminated) * np.max(self.QTable[next_state])
        target = reward + self.discount_factor * future_q
        temporal_difference = target - self.QTable[state][action]
        
        self.QTable[state][action] = self.QTable[state][action] + self.current_lr * temporal_difference

    def decay(self):
        self.current_epsilon = max(self.final_epsilon, self.epsilon / (self.episode_count * self.epsilon_decay_factor + 1))
        self.current_lr = max(self.final_lr, self.lr / (self.episode_count * self.lr_decay_factor + 1))
        self.episode_count += 1