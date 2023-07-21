import sys
import numpy as np
import pytest
from stable_baselines3 import PPO

sys.path.append('../')
from gymnasium_minigrid.MiniGridEnv import MiniGridEnv
from cnn_network import policy_kwargs


def test_is_determinist_with_ppo():
    """
    Test if the model is deterministic with PPO.
    """
    # Initialize the environment and the model
    env = MiniGridEnv(size=5, output_is_picture=True)
    model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=0, seed=0)

    # Train the model
    model.learn(total_timesteps=2000)

    # Test the model
    obs, _ = env.reset(seed=0)
    steps = []
    for i in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, truncated, info = env.step(action)
        if truncated or dones:
            steps.append(i)
            obs, _ = env.reset()

    # Check if the model finishes the game in less than 5 steps on average
    assert np.sum(steps) == 2717, f"The model is not deterministic, sum: {np.sum(steps)}"


def test_is_working_with_ppo():
    """
    Test if the model is working with PPO.
    """
    # Initialize the environment and the model
    env = MiniGridEnv(size=5, output_is_picture=True)
    model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=0, seed=2)

    # Train the model
    model.learn(total_timesteps=20000)

    # Test the model
    obs, _ = env.reset(seed=2)
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

    # Check if the model finishes the game in less than 20 steps on average
    assert np.mean(step_to_reach_target) <= 20, f"The model has not learned to play the game. Score: {np.mean(step_to_reach_target)}"


if __name__ == "__main__":
    pytest.main()