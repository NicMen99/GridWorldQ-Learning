import gymnasium as gym
import numpy as np


class Environment(gym.Env):
    def __init__(self, grid_size:tuple[int, int] = (5, 5), target_positions:np.ndarray[np.ndarray[np.int32]] = np.array([[0, 0]])):
        self.world_size:tuple = grid_size
        self.world_targets =target_positions
        self.world_recharge_station = np.array([0, 0], dtype=np.int32)

        self.agent_location = self.world_recharge_station
        self.max_charge = 10
        self.agent_charge = self.max_charge
        self.visited_positions = np.zeros(len(self.world_targets))

        # A cosa serve??
        self.observation_space = gym.spaces.Dict(
            {
                "agent position": gym.spaces.Box(low=0, high=self.world_size[0] - 1, shape=(2,), dtype=np.int32),
                "agent charge": gym.spaces.Discrete(11),
                "visited positions": gym.spaces.MultiBinary(len(self.world_targets))
            }
        )

        self.action_space = gym.spaces.Discrete(4)

        self.action_map = {
            0: np.array([1, 0], dtype=np.int32),
            1: np.array([0, 1], dtype=np.int32),
            2: np.array([-1, 0], dtype=np.int32),
            3: np.array([0, -1], dtype=np.int32)
        }

    def reset(self, seed: int = None, options: dict = None):
        super().reset(seed=seed)

        # Forse inutili
        self.world_recharge_station = self.world_recharge_station
        self.agent_location = self.world_recharge_station
        self.world_targets = self.world_targets
        self.agent_charge = self.max_charge

        return self.get_observations(), self.get_info()

    def step(self, action: int):
        truncated = False

        direction = self.action_map[action]

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

        # Check if all position have been visited
        complete = np.all(self.visited_positions)

        # Check for closest non visited target


        # Calculus of the reward function
        reward = 0.0

        if self.agent_charge <= 0:
            reward -= 1000
            complete = True
            return self.get_observations(), reward, complete, truncated, self.get_info()

        return_cost = np.sum(np.abs(self.agent_location - self.world_recharge_station))
        if self.agent_charge <= return_cost:
            reward -= 50

        return self.get_observations(), reward, complete, truncated, self.get_info()

    def get_observations(self):
        return {"agent": self.agent_location, "agent cahrge": self.agent_charge, "visited positions": self.visited_positions}

    def get_info(self):
        return {}