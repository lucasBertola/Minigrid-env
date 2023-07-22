import gymnasium
from gymnasium import spaces
import pygame
import math
import numpy as np


class MiniGridEnv(gymnasium.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 12}

    def __init__(self, render_mode=None, size=5, output_is_picture=False, max_step=None, pixel_max_value=255):
        """
        Initialize the MiniGrid environment.

        :param render_mode: The mode to render the environment in.
        :param size: The size of the square grid.
        :param output_is_picture: Whether the output should be a picture.
        """
        self.size = size
        self.window_size = 512
        self.output_is_picture = output_is_picture
        self.max_step = max_step
        self.pixel_max_value = pixel_max_value

        if self.output_is_picture and pixel_max_value == 255:
            self.observation_space = spaces.Box(low=0, high=255, shape=(self.size, self.size, 1), dtype=np.uint8)
        elif self.output_is_picture:
            print('pixel_max_value',pixel_max_value)
            self.observation_space = spaces.Box(low=0, high=pixel_max_value, shape=(1,self.size, self.size), dtype=np.uint8)
        else:
            self.observation_space = spaces.Box(0, size - 1, shape=(4,), dtype=int)

        self.action_space = spaces.Discrete(4)
        
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        assert pixel_max_value >=2 and pixel_max_value <= 255
        self.render_mode = render_mode

        self.window = None
        self.clock = None

    def _get_obs(self):
        """
        Get the current observation of the environment.

        :return: The current observation.
        """
        if self.output_is_picture and self.pixel_max_value == 255:
            grid = np.zeros((self.size, self.size, 1), dtype=np.uint8)
            grid[self._PositionX, self._PositionY, 0] = 125  # Agent position with gray color
            grid[self._TargetX, self._TargetY, 0] = 255  # Target position with white color
            return grid
        elif self.output_is_picture:
            grid = np.zeros((1,self.size, self.size), dtype=np.uint8)
            grid[0,self._PositionX, self._PositionY] = self.pixel_max_value//2  # Agent position with gray color
            grid[0,self._TargetX, self._TargetY] = self.pixel_max_value  # Target position with white color
            return grid
        else:
            return np.array([self._PositionX, self._PositionY, self._TargetX, self._TargetY]).reshape(-1)

    def _get_distance(self):
        """
        Calculate the Euclidean distance between the current position and the target.

        :return: The distance between the current position and the target.
        """
        return math.sqrt((self._TargetX - self._PositionX) ** 2 + (self._TargetY - self._PositionY) ** 2)

    def _get_info(self):
        """
        Get the current information about the environment.

        :return: A dictionary containing the current distance to the target.
        """
        return {
            "distance": self._get_distance()
        }

    def _set_random_and_different_position_and_target(self):
        """
        Set a random position and target, ensuring they are different.
        """
        self._PositionX = self.np_random.integers(0, self.size, size=1, dtype=int)[0]
        self._PositionY = self.np_random.integers(0, self.size, size=1, dtype=int)[0]
        self._TargetX = self.np_random.integers(0, self.size, size=1, dtype=int)[0]
        self._TargetY = self.np_random.integers(0, self.size, size=1, dtype=int)[0]

        while self._is_done():
            self._set_random_and_different_position_and_target()

    def _is_done(self):
        """
        Check if the current position is the same as the target.

        :return: True if the current position is the same as the target, False otherwise.
        """
        return (self._PositionX == self._TargetX and self._PositionY == self._TargetY)

    def reset(self, seed=None, options=None):
        """
        Reset the environment.

        :param seed: The random seed to use.
        :param options: Additional options for resetting the environment.
        :return: The initial observation and information.
        """
        super().reset(seed=seed)

        self._set_random_and_different_position_and_target()

        observation = self._get_obs()
        info = self._get_info()
        self.total_steps = 0

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        """
        Execute one time step within the environment.

        :param action: The action to take.
        :return: The new observation, reward, termination status, and information.
        """
        action = action.item() if isinstance(action, np.ndarray) else action

        if action == 0:
            self._PositionX = min(self._PositionX + 1, self.size - 1)
        elif action == 1:
            self._PositionY = min(self._PositionY + 1, self.size - 1)
        elif action == 2:
            self._PositionX = max(self._PositionX - 1, 0)
        elif action == 3:
            self._PositionY = max(self._PositionY - 1, 0)
        else:
            raise Exception('Invalid action')

        terminated = self._is_done()

        reward = (-1 * self._get_distance()) if not terminated else 500

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        truncated = False

        self.total_steps += 1
        if(self.max_step != None and self.total_steps >= self.max_step):
            truncated = True
    
        return observation, reward, terminated, truncated, info

    def render(self):
        """
        Render the environment to the screen.

        :return: The rendered frame if in rgb_array mode, None otherwise.
        """
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        """
        Render a frame of the environment.

        :return: The rendered frame if in rgb_array mode, None otherwise.
        """
        self._target_location = np.array([self._TargetX, self._TargetY])
        self._agent_location = np.array([self._PositionX, self._PositionY])

        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((128, 128, 128))

        padding = self.window_size * 0.05
        grid_size = self.window_size - 2 * padding
        pix_square_size = grid_size / self.size

        pygame.draw.rect(
            canvas,
            (0, 0, 0),
            pygame.Rect(
                padding, padding, grid_size, grid_size
            ),
        )

        pygame.draw.rect(
            canvas,
            (0, 255, 0),
            pygame.Rect(
                padding + pix_square_size * self._target_location[0],
                padding + pix_square_size * self._target_location[1],
                pix_square_size, pix_square_size,
            ),
        )

        pygame.draw.circle(
            canvas,
            (255, 0, 0),
            (padding + (self._agent_location + 0.5) * pix_square_size),
            pix_square_size / 3,
        )

        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                (20, 20, 20),
                (padding, padding + pix_square_size * x),
                (self.window_size - padding, padding + pix_square_size * x),
                width=1,
            )
            pygame.draw.line(
                canvas,
                (20, 20, 20),
                (padding + pix_square_size * x, padding),
                (padding + pix_square_size * x, self.window_size - padding),
                width=1,
            )

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        """
        Close the environment.
        """
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()