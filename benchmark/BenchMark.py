import sys
import time
import matplotlib.pyplot as plt
sys.path.append('../')
from gymnasium_minigrid.MiniGridEnv import MiniGridEnv
from stable_baselines3 import A2C, DQN, PPO

# Set the grid size
SIZE = 10

# Create the MiniGrid environment
def create_environment(size, render_mode=None):
    return MiniGridEnv(size=size, render_mode=render_mode)

# Create and train the model
def train_model(algorithm, env, timesteps=100):
    model = algorithm("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=timesteps)
    return model

# Evaluate the model
def evaluate_model(model, env):
    
    obs, _ = env.reset()
    rewards_list = []
    NUMBER_OF_EPISODES = 100
    for i in range(NUMBER_OF_EPISODES):
        total_rewards = 0
        for i in range(SIZE*10):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, dones, truncated, info = env.step(action)
            total_rewards += reward
            if truncated or dones:
                obs, _ = env.reset()
                break
        rewards_list.append(total_rewards)

    return sum(rewards_list) / len(rewards_list)

# Main function
def main():
    algorithms = [A2C, DQN, PPO]
    algorithm_names = ["A2C", "DQN", "PPO"]
    training_times = []
    evaluations = []

    env = create_environment(SIZE)

    for algorithm, name in zip(algorithms, algorithm_names):
        print(f"Training {name}...")
        model = train_model(algorithm, env)
        evaluation = [evaluate_model(model, env)]  # Add initial evaluation at 0s
        
        for i in range(30):
            start_time = time.time()
            elapsed_time = 0
            while elapsed_time < 1:
                model.learn(total_timesteps=100)
                elapsed_time = time.time() - start_time
            evaluation.append(evaluate_model(model, env))
            print(f"Algorithm: {name}, Evaluation: {evaluation[-1]}")
            start_time = time.time()

        training_times.append(list(range(0, 31)))  # Add 0s to the training times
        evaluations.append(evaluation)

    # Plot the results
    plt.figure(figsize=(10, 6))
    for i, name in enumerate(algorithm_names):
        plt.plot(training_times[i], evaluations[i], label=name)
    plt.xlabel("Training Time (s)")
    plt.ylabel("Evaluation GRID "+str(SIZE))
    plt.legend()
    plt.savefig(f"algorithm_comparison_grid_{SIZE}.png")
    plt.show()

if __name__ == "__main__":
    main()