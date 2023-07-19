<p align="center">
    <img src="https://github.com/lucasBertola/Minigrid/blob/main/showProject.gif" width="500px"/>
</p>
# MiniGrid: A Simple Gridworld Environment for Reinforcement Learning

Welcome to MiniGrid, a simple and intuitive gridworld environment designed for reinforcement learning experiments. This environment is built using the Gymnasium framework and is perfect for beginners who are just starting out in reinforcement learning or for experts who want to test their algorithms in a controlled setting.

## Features

- **Customizable Grid Size**: You can easily adjust the size of the square grid according to your needs.
- **Four Action Space**: The agent can move in four directions - right, up, left, and down.
- **Distance-based Reward System**: The agent is rewarded based on its distance from the target. The closer it is to the target, the higher the reward.
- **Pygame Rendering**: The environment uses Pygame for rendering, providing a visual representation of the agent's movements and interactions with the environment.
- **Randomized Initial Positions**: The initial positions of the agent and the target are randomized in each episode, ensuring a diverse range of scenarios for the agent to learn from.

## Installation

To install MiniGrid, you need to have Python 3.6 or later. You can install it using pip:

```bash
pip install gymnasium
pip install pygame
```

Then, clone this repository and navigate into it:

```bash
git clone https://github.com/yourusername/MiniGrid.git
cd MiniGrid
```

## Usage

Here is a basic example of how to use the environment:

```python
import gym

# Create the environment
env = gym.make('MiniGrid-v0')

# Reset the environment
obs = env.reset()

# Take a step
action = env.action_space.sample()
obs, reward, done, info = env.step(action)

# Render the environment
env.render()
```

## Contributing

We welcome contributions to MiniGrid! If you have a feature request, bug report, or want to improve the code, please open an issue or a pull request.

## License

MiniGrid is released under the MIT license. For more details, see the [LICENSE](LICENSE) file.

## Contact

If you have any questions or feedback, feel free to reach out to us!

Happy Reinforcement Learning!
