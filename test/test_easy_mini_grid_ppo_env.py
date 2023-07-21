import sys
import numpy as np
import pytest
from gymnasium_minigrid.MiniGridEnv import MiniGridEnv
from stable_baselines3 import PPO

sys.path.append('../')


def test_is_determinist_with_ppo():
    # Initialize the environment and the PPO model
    env = MiniGridEnv(size=3, output_is_picture=False)
    model = PPO("MlpPolicy", env, verbose=0, seed=0)

    # Train the model
    model.learn(total_timesteps=1000)

    # Test the model's determinism
    obs, _ = env.reset(seed=0)
    steps = []

    for i in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, truncated, info = env.step(action)

        if truncated or dones:
            steps.append(i)
            obs, _ = env.reset()

    # Check if the model is deterministic
    assert np.sum(steps) == 41997, "The model is not determinist"


def test_is_working_with_ppo():
    # Initialize the environment and the PPO model
    env = MiniGridEnv(size=5, output_is_picture=False)
    model = PPO("MlpPolicy", env, verbose=0, seed=0)

    # Train the model
    model.learn(total_timesteps=20000)

    # Test the model's performance
    obs, _ = env.reset(seed=0)
    step_to_reach_target = []
    step_number = 0

    for i in range(1000):
        step_number += 1
        action, _states = model.predict(obs)
        obs, rewards, dones, truncated, info = env.step(action)

        if truncated or dones:
            step_to_reach_target.append(step_number)
            step_number = 0
            obs, _ = env.reset()

    # Check if the model finishes the game in less than 8 steps on average
    assert np.mean(step_to_reach_target) <= 8, f"The model has not learned to play the game. Score {np.mean(step_to_reach_target)}"


if __name__ == "__main__":
    pytest.main()