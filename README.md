# Multi-Agent-Reinforecment-Learning

This repository contains the code and implementation of a reinforcement learning system applied to the classic **Ms. Pac-Man** Arcade Game.

## Game Setup with Pygame

The Pac-Man game environment is created using **Pygame**, which handles the graphical rendering of the maze, agents (Pac-Man and ghosts), and the game logic. The environment simulates the classic Pac-Man game, where Pac-Man navigates through a maze, collecting pellets and avoiding ghosts. This setup can be extended to support multiple agents, where both Pac-Man and the ghosts are treated as individual reinforcement learning agents.

For more information on Pygame, refer to the official documentation:  
[Pygame Documentation](https://www.pygame.org/docs/)

## Single-Agent Setup with Gymnasium

For the **single-agent** setup, **Gymnasium** is used to define our custom environment and provide reinforcement learning algorithms like **SARSA** and **DQN (Deep Q-Network)** to train Pac-Man. In this setup:
- Ms. Pac-Man is trained independently, learning to navigate the maze while avoiding ghosts that follow a predefined policy.
- The state space for Pac-Man includes information about the maze layout, Pac-Man's position, and the positions of the ghosts.
- **SARSA** and **DQN** algorithms are implemented to help Pac-Man learn the best actions to maximize its score and survive in the maze.

For more information on Gymnasium, refer to the official documentation:  
[Gymnasium Documentation](https://gymnasium.farama.org)

## Multi-Agent Setup with PettingZoo

After implementing and training the single-agent setup, the project transitions to a **multi-agent** setup using **PettingZoo**. In this setup:
- Both Pac-Man and the ghosts are treated as independent reinforcement learning agents.
- **MADDPG (Multi-Agent Deep Deterministic Policy Gradient)** is used to train multiple agents, allowing Pac-Man and the ghosts to learn their respective policies while interacting with each other in a shared environment.
- The **PettingZoo** library facilitates multi-agent interactions by providing a framework for training and testing multiple agents with different actions and policies.

For more information on PettingZoo, refer to the official documentation:  
[PettingZoo Documentation](https://pettingzoo.farama.org/)

The multi-agent environment allows for collaboration or competition between agents, where Pac-Man learns to navigate and collect pellets while avoiding ghost agents that are also learning to chase and capture Pac-Man.
