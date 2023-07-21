import sys
sys.path.append('../')
import torch as th
import torch.nn as nn
from gymnasium import spaces
from gymnasium_minigrid.MiniGridEnv import MiniGridEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import PPO


# Constants
SIZE = 6
NUMBER_OUT_CHANNEL = 28
KERNEL_SIZE = 5
FEATURE_DIM = 296


class CustomCNN(BaseFeaturesExtractor):
    """
    Custom CNN for feature extraction.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]

        # Define the CNN architecture
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, NUMBER_OUT_CHANNEL,
                      kernel_size=KERNEL_SIZE, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(NUMBER_OUT_CHANNEL, NUMBER_OUT_CHANNEL * 2,
                      kernel_size=KERNEL_SIZE // 2, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        # Define the linear layers
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


# Define policy kwargs for the PPO model
policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=FEATURE_DIM),
)

# Create the environment
env = MiniGridEnv(size=SIZE, output_is_picture=True)

# Initialize the PPO model
model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1)

# Train the model
model.learn(total_timesteps=20000)

print('Training finished')

# Test the trained model
env = MiniGridEnv(render_mode="human", size=SIZE, output_is_picture=True)

obs, _ = env.reset()
for i in range(2000):
    action, _states = model.predict(obs)
    obs, rewards, dones, truncated, info = env.step(action)
    env.render()
    if truncated or dones:
        obs, _ = env.reset()