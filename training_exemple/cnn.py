from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import PPO
import sys
sys.path.append('../')
from src.MiniGridEnv import MiniGridEnv
import torch as th
import torch.nn as nn
from gymnasium import spaces
import gymnasium as gym


SIZE = 17
NUMBER_OUT_CHANNEL = 28
KERNEL_SIZE = 5
FEATURE_DIM = 296


class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        print(KERNEL_SIZE//2)
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, NUMBER_OUT_CHANNEL,
                      kernel_size=KERNEL_SIZE, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(NUMBER_OUT_CHANNEL, NUMBER_OUT_CHANNEL*2,
                      kernel_size=KERNEL_SIZE//2, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=FEATURE_DIM),
)
env = MiniGridEnv(size=SIZE,output_is_picture=True)
# env = gym.make('BreakoutNoFrameskip-v4')
model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
model.learn(total_timesteps=20000)

print('training finish')
env = MiniGridEnv(render_mode="human", size=SIZE,output_is_picture=True)
# env = gym.make('BreakoutNoFrameskip-v4', render_mode="human")
# # env = gym.make("LunarLander-v2", render_mode="human")
# # Testez le modèle
obs, _ = env.reset()
for i in range(2000):
    action, _states = model.predict(obs)
    obs, rewards, dones, truncated, info = env.step(action)
    env.render()
    if (truncated or dones):
        obs, _ = env.reset()
