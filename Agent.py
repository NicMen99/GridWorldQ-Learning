import ast
import PrioritizedBuffer
from collections import defaultdict
import numpy as np
import gymnasium as gym
import random
import json

class Agent:
    def __init__(self,
                 env: gym.Env,
                 learning_rate: float,
                 final_lr: float,
                 initial_epsilon: float,
                 final_epsilon: float,
                 buffer: PrioritizedBuffer,
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
        self.experience_buffer = buffer

    def epsilon_greedy_policy(self, state: dict):
        state = (tuple(int(x) for x in state['agent position']),
                 int(state['agent charge']),
                 tuple(float(x) for x in state['visited positions'].tolist())
        )

        r = random.random()
        if r < self.current_epsilon:
            return self.env.action_space.sample()
        else:
            return int(np.argmax(self.QTable[state]))

    def calc_td(self, state: tuple,
                     action: int,
                     reward: float,
                     terminated: bool,
                     next_state: tuple):


        future_q = (not terminated) * np.max(self.QTable[next_state])
        target = reward + self.discount_factor * future_q
        temporal_difference = target - self.QTable[state][action]
        return temporal_difference

    def update_table(self, state: dict,
                     action: int,
                     reward: float,
                     terminated: bool,
                     next_state: dict):

        state = (tuple(int(x) for x in state['agent position']),
                 int(state['agent charge']),
                 tuple(float(x) for x in state['visited positions'].tolist())
                 )
        next_state = (tuple(int(x) for x in next_state['agent position']),
                      int(next_state['agent charge']),
                      tuple(float(x) for x in next_state['visited positions'].tolist())
                      )

        temporal_difference = self.calc_td(state, action, reward, terminated, next_state)
        
        self.QTable[state][action] = self.QTable[state][action] + self.current_lr * temporal_difference

    def update_table_replay(self, state: dict,
                            action: int,
                            reward: float,
                            terminated: bool,
                            next_state: dict,
                            weight):

        state = (tuple(int(x) for x in state['agent position']),
                 int(state['agent charge']),
                 tuple(float(x) for x in state['visited positions'].tolist())
                 )
        next_state = (tuple(int(x) for x in next_state['agent position']),
                      int(next_state['agent charge']),
                      tuple(float(x) for x in next_state['visited positions'].tolist())
                      )

        temporal_difference = self.calc_td(state, action, reward, terminated, next_state)

        self.QTable[state][action] = self.QTable[state][action] + self.current_lr * weight * temporal_difference

        return temporal_difference

    def decay(self):
        self.episode_count += 1
        self.current_epsilon = max(self.final_epsilon, self.epsilon / (self.episode_count * self.epsilon_decay_factor + 1))
        self.current_lr = max(self.final_lr, self.lr / (self.episode_count * self.lr_decay_factor + 1))

    def save_table_on_file(self, directory: str = "."):
        qTable = dict(self.QTable)
        saveable = {
            str(k): v.tolist() for k, v in qTable.items()
        }
        with open(directory + "/QTable.json", "w") as f:
            json.dump(saveable, f, indent=4)

    def load_table_from_file(self, directory: str = "."):
        with open(directory + "/QTable.json", "r") as f:
            saveable = json.load(f)
        loaded_table = {}

        for k_str, v_list in saveable.items():
            key_tuple = ast.literal_eval(k_str)
            value_np = np.array(v_list)
            loaded_table[key_tuple] = value_np

        self.QTable = defaultdict(lambda : np.zeros(self.env.action_space.n), loaded_table)