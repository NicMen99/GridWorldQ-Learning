from enum import Enum
import gymnasium as gym
import numpy as np

class Actions(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

class Environment(gym.Env):
    def __init__(self, grid_size:tuple[int, int] = (5, 5), target_positions:np.ndarray[np.ndarray[np.int32]] = np.array([[4, 4]])):
        self.world_size:tuple = grid_size
        self.world_targets = target_positions
        self.world_recharge_station = np.array([0, 0], dtype=np.int32)

        self.agent_location = self.world_recharge_station.copy()
        self.max_charge = 10
        self.agent_charge = self.max_charge
        self.visited_positions = np.zeros(len(self.world_targets))

        self.observation_space = gym.spaces.Dict(
            {
                "agent position": gym.spaces.Box(low=0, high=self.world_size[0] - 1, shape=(2,), dtype=np.int32),
                "agent charge": gym.spaces.Discrete(11),
                "visited positions": gym.spaces.MultiBinary(len(self.world_targets))
            }
        )

        self.action_space = gym.spaces.Discrete(4)

        self.action_map = {
            Actions.RIGHT.value: np.array([1, 0], dtype=np.int32),
            Actions.UP.value: np.array([0, 1], dtype=np.int32),
            Actions.LEFT.value: np.array([-1, 0], dtype=np.int32),
            Actions.DOWN.value: np.array([0, -1], dtype=np.int32)
        }

    def reset(self, seed: int = None, options: dict = None):
        super().reset(seed=seed)

        self.agent_location = self.world_recharge_station.copy()
        self.agent_charge = self.max_charge

        return self.get_observations(), self.get_info()

    def step(self, action: int):
        terminated = False
        truncated = False

        direction = self.action_map[action]
        prev_agent_location = self.agent_location.copy()

        # Move agent
        self.agent_location = np.clip(self.agent_location + direction, 0, self.world_size[0]-1)
        self.agent_charge -= 1

        # Check if position matches one of targets, then update state
        match = np.where(np.all(self.agent_location == self.world_targets, axis = 1))[0]
        if match:
            self.visited_positions[match[0]] = 1

        # Check if charge station, then recharge the robot
        charging = np.array_equal(self.agent_location, self.world_recharge_station)
        if charging:
            self.agent_charge = self.max_charge

        # Check for closest non visited target and calculate position
        remaining = np.where(self.visited_positions == 0)[0]
        closest_target_distance = float('inf')
        closest_target = None

        for i in remaining:
            dist = np.sum(np.abs(self.agent_location - self.world_targets[i]))
            if dist < closest_target_distance:
                closest_target_distance = dist
                closest_target = self.world_targets[i]

        prev_dist = np.sum(np.abs(prev_agent_location - closest_target))

        # Calculus of the reward function
        reward = 0.0

        if np.all(self.visited_positions):
            reward += 1000
            terminated = True
            return self.get_observations(), reward, terminated, truncated, self.get_info()

        if self.agent_charge <= 0:
            reward -= 1000
            terminated = True
            return self.get_observations(), reward, terminated, truncated, self.get_info()

        return_cost = np.sum(np.abs(self.agent_location - self.world_recharge_station))
        if self.agent_charge <= return_cost:
            reward -= 50

        if closest_target:
            distance_change = prev_dist - closest_target_distance
            reward += distance_change * 5.0

        if len(match):
            reward += 100

        if np.array_equal(prev_agent_location, self.agent_location):
            reward -= 10
        else:
            reward -= 1

        return self.get_observations(), reward, terminated, truncated, self.get_info()

    def get_observations(self):
        return {"agent": self.agent_location, "agent charge": self.agent_charge, "visited positions": self.visited_positions}

    def get_info(self):
        return {}