import gymnasium as gym
import numpy as np

if __name__ == '__main__':
    pos = np.array([1, 0, 1])
    print(np.where(pos == 0)[0])