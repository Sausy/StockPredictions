#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dez  28 10:20:41 2020

@author: sausy

FrozenLake-v0
The agent controls the movement of a character in a grid world. Some tiles of
the grid are walkable, and others lead to the agent falling into the water.
Additionally, the movement direction of the agent is uncertain and only
partially depends on the chosen direction. The agent is rewarded for finding a
walkable path to a goal tile.

Das spielfeld
SFFF       (S: starting point, safe)
FHFH       (F: frozen surface, safe)
FFFH       (H: hole, fall to your doom)
HFFG       (G: goal, where the frisbee is located)
"""

import numpy as np
import pandas as pd

import copy

import gym


def main():
    env = gym.make("CartPole-v1")
    observation = env.reset()
    for _ in range(1000):
      env.render()
      action = env.action_space.sample() # your agent here (this takes random actions)
      observation, reward, done, info = env.step(action)

      if done:
        observation = env.reset()
    env.close()




if __name__ == "__main__":
    main()
