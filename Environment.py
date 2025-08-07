from enum import Enum
import gymnasium as gym
import numpy as np
import pygame

class Actions(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

class Environment(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(self, grid_size:tuple[int, int] = (5, 5), target_positions:np.ndarray[np.ndarray[np.int32]] = np.array([[4, 4]]), render_mode = None):
        self.world_size:tuple = grid_size
        self.window_size = 720
        self.world_targets = target_positions
        self.world_recharge_station = np.array([0, 0], dtype=np.int32)

        self.agent_location = self.world_recharge_station.copy()
        self.max_charge = (2 * np.sum(np.abs(self.agent_location - (np.array(self.world_size) - 1)))) + 1
        self.agent_charge = self.max_charge
        self.visited_positions = np.zeros(len(self.world_targets))

        self.observation_space = gym.spaces.Dict(
            {
                "agent position": gym.spaces.Box(low=0, high=self.world_size[0] - 1, shape=(2,), dtype=np.int32),
                "agent charge": gym.spaces.Discrete(int(self.max_charge + 1)),
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

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

    def reset(self, seed: int = None, options: dict = None):
        super().reset(seed=seed)

        self.agent_location = self.world_recharge_station.copy()
        self.agent_charge = self.max_charge
        self.visited_positions = np.zeros(len(self.world_targets))

        return self.get_observations(), self.get_info()

    def step(self, action: int):
        terminated = False
        truncated = False

        direction = self.action_map[action]
        prev_agent_location = self.agent_location.copy()

        # Move agent
        self.agent_location = np.clip(self.agent_location + direction, 0, self.world_size[0]-1)
        self.agent_charge -= 1

        # Calculus of the reward function
        reward = 0.0

        if np.array_equal(prev_agent_location, self.agent_location):
            reward -= 10
        else:
            reward -= 1

        # Check if position matches one of targets, then update state and reward
        match = np.where(np.all(self.agent_location == self.world_targets, axis=1))[0]
        if match.size > 0:
            if self.visited_positions[match[0]] == 0:
                self.visited_positions[match[0]] = 1
                reward += 100

        # Check if charge station, then recharge the agent
        charging = np.array_equal(self.agent_location, self.world_recharge_station)
        if charging:
            self.agent_charge = self.max_charge
            reward += 20

        return_cost = np.sum(np.abs(self.agent_location - self.world_recharge_station))
        if self.agent_charge <= return_cost:
            reward -= 50

        # Check for closest non visited target and calculate position
        remaining = np.where(self.visited_positions == 0)[0]
        if remaining.size > 0:
            closest_target_distance = float('inf')
            closest_target = None

            for i in remaining:
                dist = np.sum(np.abs(self.agent_location - self.world_targets[i]))
                if dist < closest_target_distance:
                    closest_target_distance = dist
                    closest_target = self.world_targets[i]

            prev_dist = np.sum(np.abs(prev_agent_location - closest_target))
            distance_change = prev_dist - closest_target_distance
            reward += distance_change * 5.0

        if np.all(self.visited_positions):
            reward += 1000
            terminated = True
        elif self.agent_charge <= 0:
            reward -= 1000
            terminated = True

        if self.render_mode == "human":
            self._render_frame()

        return self.get_observations(), reward, terminated, truncated, self.get_info()

    def get_observations(self):
        return {"agent position": self.agent_location, "agent charge": self.agent_charge, "visited positions": self.visited_positions}

    def get_info(self):
        return {}

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        pix_square_size = (self.window_size/self.world_size[0])

        for i in self.world_targets:
            pygame.draw.rect(
                canvas,
                (255, 0, 0),
                pygame.Rect(
                    pix_square_size * i,
                    (pix_square_size, pix_square_size)
                )
            )

        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self.agent_location + 0.5) * pix_square_size,
            pix_square_size/3
        )

        for i in range(self.world_size[0] + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * i),
                (self.window_size, pix_square_size * i),
                width=3
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * i, 0),
                (pix_square_size * i, self.window_size),
                width=3
            )

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            self.clock.tick(self.metadata['render_fps'])

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()