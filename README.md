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
