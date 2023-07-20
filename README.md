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

## Action Space

The action space is discrete with 4 possible actions, corresponding to "right", "up", "left", "down".

## Observation Space

The observation space includes PositionX, PositionY, TargetX, TargetY. 
If output_is_picture is set to true, the observation space is a 2D grid with 0 : empty, 1 self position, 2 target position

## Import

```python
from gymnasium_minigrid import MiniGridEnv
```

## Description

The agent starts at a random position in the grid and must navigate to the target position. The agent receives a reward for reaching the target and a penalty for each step taken. The episode ends when the agent reaches the target.

## Rewards

The agent receives a reward of 500 for reaching the target and a penalty proportional to the distance to the target for each step taken.

## Starting State

The agent and target positions are randomly initialized at the start of each episode.

## Episode Termination

The episode terminates when the agent reaches the target.

## Getting Started

To use the MiniGrid environment, you can install it directly into your project using pip:

```bash
pip install gymnasium-minigrid
```

Then, you can import the `MiniGridEnv` class:

```python
from gymnasium_minigrid import MiniGridEnv
```

For detailed usage instructions and examples, please refer to the [examples](https://github.com/lucasBertola/Minigrid/tree/main/exemples) directory or check out our [Colab Notebook](https://colab.research.google.com/github/lucasBertola/Minigrid/blob/main/exemples/PPO_MlpPolicy.ipynb).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lucasBertola/Minigrid/blob/main/exemples/PPO_MlpPolicy.ipynb)

## Examples

We have provided several examples in the [examples](https://github.com/lucasBertola/Minigrid/tree/main/exemples) directory to help you get started. These examples demonstrate how to use the environment with different types of policies and neural networks.

## Usage Example

Here is a simple example of how to use the environment with a FCN policy:

```python
from gymnasium_minigrid import MiniGridEnv
from stable_baselines3 import PPO

env = MiniGridEnv(size=5)
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)
```

## Testing

We believe in the importance of testing. That's why we have included a suite of tests in the `test` directory. To run the tests, simply use the command `pytest`.

## Contribute & Support

We warmly welcome contributions from the community. If you have an idea for an improvement or found a bug, don't hesitate to open an issue or submit a pull request. Your input is greatly appreciated, and our project is made better by your participation!

If you find this repository useful, please give it a star!

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.