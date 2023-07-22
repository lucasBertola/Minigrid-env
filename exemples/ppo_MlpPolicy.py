import sys
sys.path.append('../')
import time
from gymnasium_minigrid.MiniGridEnv import MiniGridEnv
from stable_baselines3 import PPO

# Set the grid size
SIZE = 36

# Create the MiniGrid environment
def create_environment(size, render_mode=None):
    return MiniGridEnv(size=size, render_mode=render_mode)

# Create and train the PPO model
def train_model(env, timesteps=100000):
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=timesteps)
    return model

# Test the trained model
def test_model(model, env, steps=1000):
    obs, _ = env.reset()
    for i in range(steps):
        action, _states = model.predict(obs,deterministic=True)
        obs, rewards, dones, truncated, info = env.step(action)
        env.render()
        if truncated or dones:
            obs, _ = env.reset()

# Main function
def main():
    # Create and train the model
    env = create_environment(SIZE)
    model = train_model(env)

    # Test the model with human-readable rendering
    test_env = create_environment(SIZE, render_mode="human")
    test_model(model, test_env)

if __name__ == "__main__":
    main()