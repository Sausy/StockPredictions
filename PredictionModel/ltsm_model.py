#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dez  9 12:53:41 2020

@author: sausy
includes two models that are fusied together
*)SC-CNN
*)LTSM
"""

import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn import metrics as skMet

import tensorflow as tf

class modelLtsm(tf.keras.Model):
    def __init__(self,inputShape, outputShape):
        print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
        print("GPUs Available: ", tf.config.experimental.list_physical_devices('GPU'))


        tf.keras.backend.clear_session()
        tf.random.set_seed(4)

        self.N = inputShape[0]
        self.AmountParallelSeries = inputShape[1]
        self.AmountFeatures = inputShape[2] #outputShape #inputShape[2]
        self.outputSize = outputShape

        print("\ninit Model 1")
        print(inputShape)
        print(outputShape)


        self.UnitCountHiddenLayer1 = int(self.AmountParallelSeries*1.8) #int(self.AmountFeatures*1.6)

        print("UnitCountHiddenLayer1: {}\n==============".format(self.UnitCountHiddenLayer1))

        super(modelLtsm, self).__init__()


        #inputs: A 3D tensor with shape [batch, timesteps, feature]  .... for LSTM
        self.lin = tf.keras.layers.InputLayer(input_shape=inputShape)
        self.l1 = tf.keras.layers.LSTM(self.AmountParallelSeries, activation='tanh') #input_shape=(self.AmountParallelSeries, self.AmountFeatures)
        self.dropout = tf.keras.layers.Dropout(0.2)
        #Conv2D(filters, kernelsize, ...)
        #self.conv = tf.keras.layers.Conv2D(10, self.AmountFeatures, activation='tanh', input_shape=input_shape[1:])
        self.bn = tf.keras.layers.BatchNormalization()

        #self.l2 = tf.keras.layers.LSTM(self.UnitCountHiddenLayer1, activation='tanh')
        self.l21 = tf.keras.layers.Dense(int(self.UnitCountHiddenLayer1*1.6), activation='tanh')
        self.dropout21 = tf.keras.layers.Dropout(0.2)

        self.l22 = tf.keras.layers.Dense(int(self.UnitCountHiddenLayer1*1.6), activation='tanh')
        self.dropout22 = tf.keras.layers.Dropout(0.2)

        self.l23 = tf.keras.layers.Dense(int(self.UnitCountHiddenLayer1*1.6), activation='tanh')
        self.dropout23 = tf.keras.layers.Dropout(0.2)



        #self.l3 = tf.keras.layers.Dense(self.AmountFeatures+5, activation='tanh')
        self.l3 = tf.keras.layers.Dense(self.UnitCountHiddenLayer1, activation='tanh')
        self.dropout3 = tf.keras.layers.Dropout(0.2)

        #self.l3 = tf.keras.layers.Dense(100, activation='sigmoid')

    #    model.add(tf.keras.layers.Dense(5, activation='sigmoid'))
     #   model.add(tf.keras.layers.Dense(1, activation='linear'))
        self.l4 = tf.keras.layers.Dense(20, activation='relu')

        self.lout = tf.keras.layers.Dense(self.outputSize, activation='linear')


    def call(self,inputs, training=False):
        #!!Call is a redefinition ... because we are makiing a subclass hence it is
        #not directly called by your code

        #x = self.lin(inputs)
        x = self.l1(inputs)
        if training:
            x = self.dropout(x, training=training)
            #x = self.bn(x, training=training)

        x = self.l21(x)
        if training:
            x = self.dropout21(x, training=training)

        x = self.l22(x)
        if training:
            x = self.dropout22(x, training=training)

        x = self.l23(x)
        if training:
            x = self.dropout23(x, training=training)

        #x = self.bn(x)
        x = self.l3(x)

        if training:
            x = self.dropout3(x, training=training)
        #x = self.l4(x)
        #x = self.l4(x)

        return self.lout(x)

    def showMetric(self,evalDat,predictDat):
        print("\n========[Show Evalution metrics] =====")
        print("explained_variance_score: (best:1)\t{:.4f}".format(skMet.explained_variance_score(evalDat, predictDat)))
        print("max_error: \t\t{:.4f}".format(skMet.max_error(evalDat, predictDat)))
        print("mean_absolute_error: \t{:.4f}".format(skMet.mean_absolute_error(evalDat, predictDat)))
        print("mean_squared_error: \t{:.4f}".format(skMet.mean_squared_error(evalDat, predictDat)))

        #print("f1_score: {:.4f}".format(f1_score(evalDat, predictDat)))

def main():
    pass

if __name__ == "__main__":
    main()
