#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dez  8 10:53:41 2020

@author: sausy
"""

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print("GPUs Available: ", tf.config.experimental.list_physical_devices('GPU'))
