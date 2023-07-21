import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# Constants
NUMBER_OUT_CHANNEL = 28
KERNEL_SIZE = 3
FEATURE_DIM = 296


class CustomCNN(BaseFeaturesExtractor):
    """
    Custom Convolutional Neural Network (CNN) feature extractor for Stable-Baselines3.
    This class extends the BaseFeaturesExtractor class and overrides the forward method.

    :param observation_space: (gym.Space) The observation space of the environment.
    :param features_dim: (int) Number of features extracted. This corresponds to the number of units for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-processing or wrapper
        n_input_channels = observation_space.shape[0]

        # Define the CNN architecture
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, NUMBER_OUT_CHANNEL, kernel_size=KERNEL_SIZE, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(NUMBER_OUT_CHANNEL, NUMBER_OUT_CHANNEL * 2, kernel_size=KERNEL_SIZE // 2, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]

        # Define the linear layers
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        """
        Forward pass to extract features from the input observations.

        :param observations: (th.Tensor) Input observations.
        :return: (th.Tensor) Extracted features.
        """
        return self.linear(self.cnn(observations))


# Define the policy_kwargs dictionary for use with Stable-Baselines3 algorithms
policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=FEATURE_DIM),
)