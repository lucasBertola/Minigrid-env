import sys 
sys.path.append('../')
from src.EasyMiniGridEnv import EasyMiniGridEnv
from stable_baselines3 import PPO
import time
import gymnasium as gym
import numpy as np
import pytest

def test_is_determinist_with_ppo():
    env = EasyMiniGridEnv(size=3) 
    model = PPO("MlpPolicy", env, verbose=0, seed=0)
    model.learn(total_timesteps=1000)
    obs , _=  env.reset(seed=0)
    steps = []
    for i in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, truncated,info = env.step(action)
        if(truncated or dones):
            steps.append(i)
            obs , _=  env.reset()
    # Vérifiez que le modèle finit le jeu en moins de 5 étapes en moyenne
    assert np.sum(steps) ==28062 , "The model is not detrminist"

def test_is_working_with_ppo():
    env = EasyMiniGridEnv(size=5) 
    model = PPO("MlpPolicy", env, verbose=0, seed=0)
    model.learn(total_timesteps=20000)
    obs , _=  env.reset(seed=0)
    stepToReachTarget = []
    stepNumber = 0
    for i in range(1000):
        stepNumber +=1
        action, _states = model.predict(obs)
        obs, rewards, dones, truncated,info = env.step(action)
        if(truncated or dones):
            stepToReachTarget.append(stepNumber)
            stepNumber = 0
            obs , _=  env.reset()
    # Vérifiez que le modèle finit le jeu en moins de 5 étapes en moyenne
    assert np.mean(stepToReachTarget) <=8 , "The model have not learn to play the game. Score "+str(np.mean(stepToReachTarget))

if __name__ == "__main__":
    pytest.main()