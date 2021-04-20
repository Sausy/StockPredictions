#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dez  8 10:53:41 2020

@author: sausy
"""
import keras
print("\n===============\nKeras Version: ")
print(keras.__version__)


import tensorflow as tf
print("\n===============\nTF-Version: ")
print(tf.__version__)
print()

print("\n===============")
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print("GPUs Available: ", tf.config.experimental.list_physical_devices('GPU'))
