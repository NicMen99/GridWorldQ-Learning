from collections import defaultdict
import numpy as np
import gymnasium as gym
import random

class Agent:
    def __init__(self,
                 env: gym.Env,
                 learning_rate: float,
                 initial_epsilon: float,
                 epsilon_decay: float,
                 final_epsilon: float,
                 c1: float,
                 c2: float,
                 discount_factor: float = 0.95
                 ):
        self.env = env

        self.QTable = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

    def epsilon_greedy_policy(self, obs: tuple[int, int, np.ndarray[np.bool_]]):
        r = random.random()
        if r < self.epsilon:
            return self.env.action_space.sample()
        else:
            return int(np.argmax(self.QTable[obs]))

    def optimal_policy(self, state):
        return np.argmax(self.QTable[state])

    def update_table(self, state: tuple[int, int, np.ndarray[np.bool_]],
                     action: int,
                     reward: float,
                     terminated: bool,
                     next_state: tuple[int, int, np.ndarray[np.bool_]]):

        future_q = (not terminated) * np.max(self.QTable[next_state])
        target = reward + self.discount_factor * future_q
        temporal_difference = target - self.QTable[state][action]
        
        self.QTable[state][action] = self.QTable[state][action] + self.lr * temporal_difference

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)