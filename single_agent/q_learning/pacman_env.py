import pygame
from pygame.locals import *
import gymnasium as gym
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env
import numpy as np
from gymnasium import spaces
from run import GameController
from constants import *
from DQN_model import *
from stable_baselines3 import DQN , PPO
from modified_tensorboard import TensorboardCallback
from stable_baselines3.dqn import MultiInputPolicy
from torch.optim import RMSprop, Adam
import os
import copy

GHOST_MODES = {SCATTER: 0, CHASE: 0, FREIGHT: 1, SPAWN: 2}


if "pacman-v0" not in gym.envs.registry:
    register(id="pacman-v0", entry_point="pacman_env:PacmanEnv", max_episode_steps=1000)


class PacmanEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None , mode = SAFE_MODE , move_mode = DISCRETE_STEPS_MODE, clock_tick = 0 , pacman_lives = 1 , maze_mode = MAZE3 , pac_pos_mode = RANDOM_PAC_POS):

        self.game = GameController(rlTraining = True , mode = mode , move_mode = move_mode , clock_tick = clock_tick , pacman_lives = pacman_lives , maze_mode=maze_mode , pac_pos_mode = pac_pos_mode)
        self.num_pellets_last = 0
        self.game_score = 0
        self.useless_steps = 0
        self.episode_steps = 0

        self.num_frames_obs = 4
        
        self.observation_space = spaces.Box(
                    low = 0, high = 13 , shape = (self.num_frames_obs , GAME_ROWS , GAME_COLS) , dtype=np.int_
                )
        
        self.action_space = spaces.Discrete(5, start=0)

        self._maze_map = np.zeros(shape=(GAME_ROWS , GAME_COLS), dtype=np.int_)
        self._last_obs = np.zeros(shape=(GAME_ROWS , GAME_COLS), dtype=np.int_)
        self.observation_buffer = np.zeros(shape=(self.num_frames_obs , GAME_ROWS , GAME_COLS), dtype=np.int_)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        if self.render_mode == "human":
            self.window = self.game.screen
            self.clock = self.game.clock

    def _getobs(self):
        self._maze_map = self.game.observation
        #self._maze_map = np.expand_dims(self._maze_map , axis=0)
        return self._maze_map

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.restartGame()
        self.game.done = False
        self.game_score = 0

        observation = self._getobs()
        for i in range (self.num_frames_obs):
            self.observation_buffer[i] = observation
        #obs_buf = np.expand_dims(self.observation_buffer , axis=0) 
        info = {}
        return self.observation_buffer, info

    def step(self, action):
        if self.game.move_mode == CONT_STEPS_MODE:
            action -= 2
            step_reward = TIME_PENALITY
            while True:
                if self.render_mode == "human":
                    self.game.update(
                        agent_direction=action,
                        render=True
                        #clocktick=self.metadata["render_fps"],
                    )
                else:
                    self.game.update(
                        agent_direction=action,
                        render=False
                        #clocktick=self.metadata["render_fps"],
                    )
                
                terminated = self.game.done
                truncated = False
                reward = self.game.RLreward
                observation = self._getobs()
                info = {}

                if reward != TIME_PENALITY:
                    step_reward = reward

                if not np.array_equal(observation , self._last_obs): 
                    self.num_pellets_last = len(self.game.pellets.pelletList)
                    np.copyto(self._last_obs , observation)
                    self.game_score += step_reward

                    if self.game.mode == SAFE_MODE:
                        if reward == TIME_PENALITY or reward == HIT_WALL_PENALITY:
                            self.useless_steps +=1
                            if self.useless_steps >= MAX_USELESS_STEPS:
                                self.game.done = True
                                terminated = self.game.done
                                self.useless_steps = 0
                        # else:
                        #     self.useless_steps = 0
                    self.episode_steps +=1
                    if terminated:
                        self.episode_steps = 0
                    return observation, step_reward, terminated, truncated, info 


        elif self.game.move_mode == DISCRETE_STEPS_MODE:
            action -= 2
            #step_reward = TIME_PENALITY
            if self.render_mode == "human":
                self.game.update(
                    agent_direction=action,
                    render=True
                    #clocktick=self.metadata["render_fps"],
                )
            else:
                self.game.update(
                    agent_direction=action,
                    render=False
                    #clocktick=self.metadata["render_fps"],
                )
            self.num_pellets_last = len(self.game.pellets.pelletList)
            terminated = self.game.done
            truncated = False
            reward = self.game.RLreward
            observation = self._getobs()
            info = {}

            #if not np.array_equal(observation , self._last_obs): 
            #np.copyto(self._last_obs , observation)
            self.game_score += reward

            if self.game.mode == SAFE_MODE:
                if reward == TIME_PENALITY or reward == HIT_WALL_PENALITY:
                    self.useless_steps += 1
                    if self.useless_steps >= MAX_USELESS_STEPS:
                        self.game.done = True
                        terminated = self.game.done
                        self.useless_steps = 0
                else:
                    self.useless_steps = 0
            # if reward > 0:
            #     print(reward)

            self.observation_buffer[:-1] = self.observation_buffer[1:]
            self.observation_buffer[-1] = observation
            #obs_buf = np.expand_dims(self.observation_buffer , axis=0)
            # print("***********")
            # print(reward)
            # print(terminated)
            # print("episode steps: " , self.episode_steps)
            self.episode_steps +=1
            if terminated:
                self.episode_steps = 0
            return self.observation_buffer, reward, terminated, truncated, info


    def render(self):
        if self.render_mode == "human":
            self.game.render()

    def close(self):
        if self.window is not None:
            pygame.event.post(pygame.event.Event(QUIT))


