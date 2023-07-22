import sys
sys.path.append('../')
from gymnasium_minigrid.MiniGridEnv import MiniGridEnv
from stable_baselines3 import PPO


# Constants
SIZE = 36 #minimal size of default CnnPolicy policy

# Create the environment
env = MiniGridEnv(size=SIZE, output_is_picture=True,pixel_max_value=255)

# Initialize the PPO model
model = PPO("CnnPolicy", env, verbose=1)

# Train the model
model.learn(total_timesteps=120000)

print('Training finished')

# Test the trained model
env = MiniGridEnv(render_mode="human", size=SIZE, output_is_picture=True,pixel_max_value=255)

obs, _ = env.reset()
for i in range(2000):
    action, _states = model.predict(obs)
    obs, rewards, dones, truncated, info = env.step(action)
    env.render()
    if truncated or dones:
        obs, _ = env.reset()