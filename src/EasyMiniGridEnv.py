import gymnasium
from gymnasium import spaces
import pygame
import math
import numpy as np

class EasyMiniGridEnv(gymnasium.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 12}

    def __init__(self, render_mode=None, size=5, output_is_picture=False):
        self.size = size  # Define the size of the square grid
        self.window_size = 512  # Define the size of the PyGame window
        self.output_is_picture = output_is_picture
   
        if self.output_is_picture:
            self.observation_space = spaces.Box(low=0, high=2, shape=(1,self.size, self.size), dtype=np.int32)
        else:
            # The observation space includes PositionX, PositionY, TargetX, TargetY
            self.observation_space = spaces.Box(0, size - 1, shape=(4,), dtype=int)
        
        # Define the action space. We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = spaces.Discrete(4)

        # Check if the render mode is valid
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        if self.output_is_picture:
            grid = np.zeros((1, self.size, self.size))
            grid[0, self._PositionX, self._PositionY] = 1
            grid[0, self._TargetX, self._TargetY] = 2 
            return grid
        else:
            return np.array([self._PositionX, self._PositionY, self._TargetX, self._TargetY]).reshape(-1)

    def _get_distance(self):
        # Calculate the Euclidean distance between the current position and the target
        return math.sqrt((self._TargetX - self._PositionX)**2 + (self._TargetY - self._PositionY)**2)

    def _get_info(self):
        # Return the current distance to the target
        return {
            "distance": self._get_distance()
        }

    def _setRandomAndDifferentPositionAndTarget(self):
        # Set a random position and target, ensuring they are different
        self._PositionX = self.np_random.integers(0, self.size, size=1, dtype=int)[0]
        self._PositionY = self.np_random.integers(0, self.size, size=1, dtype=int)[0]
        self._TargetX = self.np_random.integers(0, self.size, size=1, dtype=int)[0]
        self._TargetY = self.np_random.integers(0, self.size, size=1, dtype=int)[0]

        while self._isDone():
            self._setRandomAndDifferentPositionAndTarget()

    def _isDone(self):
        # Check if the current position is the same as the target
        return (self._PositionX == self._TargetX and self._PositionY == self._TargetY)

    def reset(self, seed=None, options=None):
        # Reset the environment
        super().reset(seed=seed)

        self._setRandomAndDifferentPositionAndTarget()

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # Execute one time step within the environment
        action = action.item() if isinstance(action, np.ndarray) else action

        if (action == 0):
            self._PositionX = min(self._PositionX + 1, self.size - 1)
        elif (action == 1):
            self._PositionY = min(self._PositionY + 1, self.size - 1)
        elif (action == 2):
            self._PositionX = max(self._PositionX - 1, 0)
        elif (action == 3):
            self._PositionY = max(self._PositionY - 1, 0)
        else:
            raise Exception('Invalid action')

        terminated = self._isDone()

        # Define the reward. If the agent is on the target, the reward is 1000. If the agent is not on the target, the reward is negative and decreases with distance
        reward = (-1 * self._get_distance()) if not terminated else 500

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        # Render the environment to the screen
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        # Render a frame of the environment
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
        canvas.fill((128, 128, 128))  # Set background to grey

        # Define the size of the padding
        padding = self.window_size * 0.05

        # Define the size of the grid (excluding padding)
        grid_size = self.window_size - 2 * padding

        pix_square_size = (
            grid_size / self.size
        )  # The size of a single grid square in pixels

        # Draw the grid background
        pygame.draw.rect(
            canvas,
            (0, 0, 0),  # Set grid background to black
            pygame.Rect(
                padding, padding, grid_size, grid_size
            ),
        )

        # Draw the target
        pygame.draw.rect(
            canvas,
            (0, 255, 0),  # Set target to green
            pygame.Rect(
                padding + pix_square_size * self._target_location[0],
                padding + pix_square_size * self._target_location[1],
                pix_square_size, pix_square_size,
            ),
        )
        # Draw the agent
        pygame.draw.circle(
            canvas,
            (255, 0, 0),
            (padding + (self._agent_location + 0.5) * pix_square_size),
            pix_square_size / 3,
        )

        # Add gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                (20, 20, 20),  # Set gridlines to black
                (padding, padding + pix_square_size * x),
                (self.window_size - padding, padding + pix_square_size * x),
                width=1,  # Set gridlines to be very thin
            )
            pygame.draw.line(
                canvas,
                (20, 20, 20),  # Set gridlines to black
                (padding + pix_square_size * x, padding),
                (padding + pix_square_size * x, self.window_size - padding),
                width=1,  # Set gridlines to be very thin
            )

        if self.render_mode == "human":
            # Copy our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # Ensure that human-rendering occurs at the predefined framerate
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        # Close the environment
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()