#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dez  28 10:20:41 2020

@author: sausy
https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0
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
