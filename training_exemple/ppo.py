import sys 
sys.path.append('../')
from src.EasyMiniGridEnv import EasyMiniGridEnv
from stable_baselines3 import PPO
import time
import gymnasium as gym

SIZE = 17

# Créez l'environnement
env = EasyMiniGridEnv(size=SIZE,output_is_picture=False)
# env = gym.make("LunarLander-v2")
# Créez le modèle PPO
model = PPO("MlpPolicy", env, verbose=1)

# Entraînez le modèle
model.learn(total_timesteps=10000)

env = EasyMiniGridEnv(render_mode="human",size=SIZE,output_is_picture=False) 

# env = gym.make("LunarLander-v2", render_mode="human")
# Testez le modèle
obs , _=  env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, truncated,info = env.step(action)
    env.render()
    if(truncated or dones):
        obs , _=  env.reset()

