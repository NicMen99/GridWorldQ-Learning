from enum import Enum
import numpy as np
import random

class Actions(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

class Agent:
    def __init__(self, discount_factor, c1, c2):
        self.reward = None
        self.learning_rate = None
        self.epsilon = 0.8
        self.discount = discount_factor
        self.c1 = c1
        self.c2 = c2

    def set_reward_function(self, function):
        self.reward = function

    def learning_rate(self, t = 0):
        return self.c1 / (self.c2 + t)

    def epsilon_greedy_policy(self, state, table, epsilon):
        r = random.random()
        if r < epsilon and table[state]:
            action = np.argmax(table[state])
        else:
            action = int(random.random() * 4)
        return action

    def optimal_policy(self, state, table):
        return np.argmax(table[state])