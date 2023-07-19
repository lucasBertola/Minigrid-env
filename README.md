<p align="center">
    <img src="https://github.com/lucasBertola/Minigrid/blob/main/showProject.gif" width="500px"/>
</p>

# EasyMiniGrid Environment

Welcome to the EasyMiniGrid Environment repository! This is a simple yet powerful environment designed for reinforcement learning agents. The goal of the game is to navigate a grid and reach a target location. The environment is highly customizable and can be used with both Fully Connected Networks (FCN) and Convolutional Neural Networks (CNN).

## Features

- **Customizable Grid Size**: You can easily adjust the size of the grid to challenge your agent with different levels of complexity.
- **FCN and CNN Compatible**: The environment is designed to work seamlessly with both FCN and CNN policies. This makes it a great playground to experiment with different types of neural networks.
- **Human Render Mode**: Watch your agent learn in real-time with the human render mode. It's not just about the numbers, it's also about the journey!
- **OpenAI Gym Compatible**: EasyMiniGrid follows the OpenAI Gym interface, making it compatible with a wide range of reinforcement learning libraries and algorithms.

## Getting Started

To use the EasyMiniGrid environment, you need to import the `EasyMiniGridEnv` class from the `src` directory. This class provides all the functionality you need to interact with the environment. For detailed usage instructions, please refer to the documentation in the `src` directory.

## Examples

We have provided several examples in the `examples` directory to help you get started. These examples demonstrate how to use the environment with different types of policies and neural networks.

## Testing

We believe in the importance of testing. That's why we have included a suite of tests in the `test` directory. To run the tests, simply use the command `pytest`.

## Usage Example

Here is a simple example of how to use the environment with a FCN policy:

```python
from src.EasyMiniGridEnv import EasyMiniGridEnv
from stable_baselines3 import PPO

env = EasyMiniGridEnv(size=5)
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)
```

## Contribute & Support

Feel free to fork this repository and contribute. We appreciate any feedback and support. If you find this repository useful, please give it a star!

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.