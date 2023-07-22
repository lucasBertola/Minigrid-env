<p align="center">
    <img src="https://github.com/lucasBertola/Minigrid/blob/main/showProject.gif" width="500px"/>
</p>

# MiniGrid Environment

Welcome to the MiniGrid Environment repository! This is a simple yet powerful environment designed for reinforcement learning agents. The goal of the game is to navigate a grid and reach a target location. The environment is highly customizable and can be used with both Fully Connected Networks (FCN) and Convolutional Neural Networks (CNN).

## Features

- **Customizable Grid Size**: You can easily adjust the size of the grid to challenge your agent with different levels of complexity.
- **FCN and CNN Compatible**: The environment is designed to work seamlessly with both FCN and CNN policies. This makes it a great playground to experiment with different types of neural networks.
- **Human Render Mode**: Watch your agent learn in real-time with the human render mode. It's not just about the numbers, it's also about the journey!
- **OpenAI Gym / Gymnasium Compatible**: MiniGrid follows the OpenAI Gym / Gymnasium interface, making it compatible with a wide range of reinforcement learning libraries and algorithms.

## Installation

To use the MiniGrid environment, you can install it directly into your project using pip:

```bash
pip install gymnasium-minigrid
```

## Usage

Import the `MiniGridEnv` class:

```python
from gymnasium_minigrid import MiniGridEnv
```

Here is a simple example of how to use the environment with a FCN policy:

```python
from gymnasium_minigrid import MiniGridEnv
from stable_baselines3 import PPO

env = MiniGridEnv(size=5)
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)
```

For detailed usage instructions and examples, please refer to the [examples](https://github.com/lucasBertola/Minigrid/tree/main/exemples) directory or check out our [Colab Notebook](https://colab.research.google.com/github/lucasBertola/Minigrid/blob/main/exemples/PPO_MlpPolicy.ipynb).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lucasBertola/Minigrid/blob/main/exemples/PPO_MlpPolicy.ipynb)

## Environment Details

### Action Space

The action space is discrete with 4 possible actions, corresponding to "right", "up", "left", "down".

### Observation Space

The observation space includes PositionX, PositionY, TargetX, TargetY. 
If output_is_picture is set to true, the observation space is a 2D grid with 0 : empty, 1 self position, 2 target position

### Rewards

The agent receives a reward of 500 for reaching the target and a penalty proportional to the distance to the target for each step taken.

### Episode Termination

The episode terminates when the agent reaches the target.

## Testing

We believe in the importance of testing. That's why we have included a suite of tests in the `test` directory. To run the tests, simply use the command `pytest`.

## Reinforcement Learning Algorithm Benchmark

We have conducted a simple benchmark to compare the performance of different reinforcement learning algorithms on the MiniGrid environment. The following algorithms were tested:

- A2C
- DQN
- PPO

The benchmark was run with the default hyperparameters for each algorithm and was executed only once. The performance of each algorithm was evaluated every second for a total of 30 seconds. The graph below shows the evaluation score (y-axis) as a function of the total training time (x-axis).

![algorithm_comparison](https://github.com/lucasBertola/Minigrid/blob/main/benchmark/algorithm_comparison_grid_10.png)

Please note that this benchmark is not exhaustive and should be taken as an indicative comparison of the algorithms' performance. For a more rigorous evaluation, multiple runs with different random seeds and hyperparameter tuning should be performed. However, this graph provides a good representation of our experience with these algorithms on the MiniGrid environment.


## Contribute & Support

We warmly welcome contributions from the community. If you have an idea for an improvement or found a bug, don't hesitate to open an issue or submit a pull request. Your input is greatly appreciated, and our project is made better by your participation!

If you find this repository useful, please give it a star!

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.