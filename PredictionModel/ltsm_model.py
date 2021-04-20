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


        self.N = inputShape[0]
        self.AmountParallelSeries = inputShape[1]
        self.AmountFeatures = inputShape[2] #outputShape #inputShape[2]
        self.outputSize = outputShape

        print("\ninit Model 1")
        print(inputShape)
        print(outputShape)


        self.UnitCountHiddenLayer1 = 80#int(self.AmountParallelSeries*1.8) #int(self.AmountParallelSeries*1.8)

        print("UnitCountHiddenLayer1: {}\n==============".format(self.UnitCountHiddenLayer1))

        super(modelLtsm, self).__init__()


        #inputs: A 3D tensor with shape [batch, timesteps, feature]  .... for LSTM
        #self.lin = tf.keras.layers.InputLayer(input_shape=inputShape)

        #self.lstm1 = tf.keras.layers.LSTM(self.AmountParallelSeries, activation='tanh') #input_shape=(self.AmountParallelSeries, self.AmountFeatures)
        self.lstm1 = tf.keras.layers.LSTM(self.UnitCountHiddenLayer1, activation='tanh') #input_shape=(self.AmountParallelSeries, self.AmountFeatures)
        #self.lstm1 = tf.keras.layers.LSTM(self.AmountParallelSeries, activation='tanh', return_sequences=True) #input_shape=(self.AmountParallelSeries, self.AmountFeatures)

        self.lstmDropout = tf.keras.layers.Dropout(0.2)

        #self.lstm2 = tf.keras.layers.LSTM(self.AmountFeatures, activation='tanh') #input_shape=(self.AmountParallelSeries, self.AmountFeatures)
        #self.dropout = tf.keras.layers.Dropout(0.2)

        #Conv2D(filters, kernelsize, ...)
        #self.conv = tf.keras.layers.Conv2D(10, self.AmountFeatures, activation='tanh', input_shape=input_shape[1:])
        #self.bn = tf.keras.layers.BatchNormalization()

        #self.l2 = tf.keras.layers.LSTM(self.UnitCountHiddenLayer1, activation='relu')
        self.l21 = tf.keras.layers.Dense(int(self.UnitCountHiddenLayer1*1.6), activation='tanh')
        self.dropout21 = tf.keras.layers.Dropout(0.2)

        self.l22 = tf.keras.layers.Dense(int(self.UnitCountHiddenLayer1*1.6), activation='tanh')
        self.dropout22 = tf.keras.layers.Dropout(0.2)

        #self.l23 = tf.keras.layers.Dense(int(self.UnitCountHiddenLayer1*1.6), activation='tanh')
        #self.dropout23 = tf.keras.layers.Dropout(0.2)



        #self.l3 = tf.keras.layers.Dense(self.AmountFeatures+5, activation='relu')
        self.l3 = tf.keras.layers.Dense(self.UnitCountHiddenLayer1, activation='tanh')
        self.dropout3 = tf.keras.layers.Dropout(0.2)

        #self.l3 = tf.keras.layers.Dense(100, activation='sigmoid')

    #    model.add(tf.keras.layers.Dense(5, activation='sigmoid'))
     #   model.add(tf.keras.layers.Dense(1, activation='linear'))
        #self.l4 = tf.keras.layers.Dense(20, activation='relu')

        self.lout = tf.keras.layers.Dense(self.outputSize, activation='linear')


    def call(self,inputs, training=False):
        #!!Call is a redefinition ... because we are makiing a subclass hence it is
        #not directly called by your code

        #x = self.lin(inputs)
        x = self.lstm1(inputs)
        if training:
            x = self.lstmDropout(x, training=training)
        #x = self.lstm2(x)
        #if training:
        #    x = self.dropout(x, training=training)
            #x = self.bn(x, training=training)

        x = self.l21(x)
        if training:
            x = self.dropout21(x, training=training)

        x = self.l22(x)
        if training:
            x = self.dropout22(x, training=training)

        #x = self.l23(x)
        #if training:
        #    x = self.dropout23(x, training=training)

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


class modelCnn(tf.keras.Model):
    def __init__(self,inputShape, outputShape):
        print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
        print("GPUs Available: ", tf.config.experimental.list_physical_devices('GPU'))


        #tf.keras.backend.clear_session()
        #tf.random.set_seed(4)

        self.N = inputShape[0]
        self.AmountParallelSeries = inputShape[1]
        #self.AmountFeatures = inputShape[2] #outputShape #inputShape[2]
        self.outputSize = outputShape

        print("\ninit Model 1")
        print(inputShape)
        print(outputShape)


        self.UnitCountHiddenLayer1 = 80#int(self.AmountParallelSeries*1.8) #int(self.AmountParallelSeries*1.8)

        print("UnitCountHiddenLayer1: {}\n==============".format(self.UnitCountHiddenLayer1))

        super(modelCnn, self).__init__()

        #input layer
        §self.lstm1 = tf.keras.layers.LSTM(self.UnitCountHiddenLayer1, activation='tanh') #input_shape=(self.AmountParallelSeries, self.AmountFeatures)
        self.conv1 = tf.keras.layers.Conv3D(56, 7, strides=2, input_shape=inputShape)(x)

        #===LTSM====
        #input dropout layer lstm
        self.lstmDropout = tf.keras.layers.Dropout(0.2)


        self.l21 = tf.keras.layers.Dense(int(self.UnitCountHiddenLayer1*1.6), activation='tanh')
        self.dropout21 = tf.keras.layers.Dropout(0.2)


        #===CNN====
        #x = BatchNormalization()
        #x = Activation(activations.relu)
        self.maxpooling = tf.keras.layers.MaxPooling3D((3, 3, 3), strides=(2, 2))

        #downscaled ResNet 50
        self.avgPooling = AveragePooling3D((2, 2, 2), padding='same')



        #===concatenate Layer =====
        self.concatted = tf.keras.layers.Concatenate()


        #=== fused Modell ===
        self.l22 = tf.keras.layers.Dense(int(self.UnitCountHiddenLayer1*1.6), activation='tanh')
        self.dropout22 = tf.keras.layers.Dropout(0.2)

        self.l3 = tf.keras.layers.Dense(self.UnitCountHiddenLayer1, activation='tanh')
        self.dropout3 = tf.keras.layers.Dropout(0.2)

        self.lout = tf.keras.layers.Dense(self.outputSize, activation='linear')

    def res_identity(x, filters):
        #renet block where dimension doesnot change.
        #The skip connection is just simple identity conncection
        #we will have 3 blocks and then input will be added

        x_skip = x # this will be used for addition with the residual block
        f1, f2 = filters

        #first block
        x = Conv2D(f1, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=l2(0.001))(x)
        x = BatchNormalization()(x)
        x = Activation(activations.relu)(x)

        #second block # bottleneck (but size kept same with padding)
        x = Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(0.001))(x)
        x = BatchNormalization()(x)
        x = Activation(activations.relu)(x)

        # third block activation used after adding the input
        x = Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=l2(0.001))(x)
        x = BatchNormalization()(x)
        # x = Activation(activations.relu)(x)

        # add the input
        x = Add()([x, x_skip])
        x = Activation(activations.relu)(x)

        return x

    def res_conv(x, s, filters):
        '''
        here the input size changes'''
        x_skip = x
        f1, f2 = filters

        # first block
        x = Conv2D(f1, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_regularizer=l2(0.001))(x)
        # when s = 2 then it is like downsizing the feature map
        x = BatchNormalization()(x)
        x = Activation(activations.relu)(x)

        # second block
        x = Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(0.001))(x)
        x = BatchNormalization()(x)
        x = Activation(activations.relu)(x)

        #third block
        x = Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=l2(0.001))(x)
        x = BatchNormalization()(x)

        # shortcut
        x_skip = Conv2D(f2, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_regularizer=l2(0.001))(x_skip)
        x_skip = BatchNormalization()(x_skip)

        # add
        x = Add()([x, x_skip])
        x = Activation(activations.relu)(x)

        return x



    def call(self,inputs, training=False):

        ltsmX = self.lstm1(inputs)
        conX = self.conv1(inputs)

        #===LTSM====
        if training:
            ltsmX = self.lstmDropout(ltsmX, training=training)

        ltsmX = self.l21(ltsmX)
        if training:
            ltsmX = self.dropout21(ltsmX, training=training)


        #===CNN====
        conX = self.maxpooling(conX)

        #ResNet50
        #2nd stage
        # frm here on only conv block and identity block, no pooling

        x = res_conv(x, s=1, filters=(64, 256))
        x = res_identity(x, filters=(64, 256))
        x = res_identity(x, filters=(64, 256))

        # 3rd stage

        x = res_conv(x, s=2, filters=(128, 512))
        x = res_identity(x, filters=(128, 512))
        x = res_identity(x, filters=(128, 512))
        x = res_identity(x, filters=(128, 512))

        # 4th stage

        x = res_conv(x, s=2, filters=(256, 1024))
        x = res_identity(x, filters=(256, 1024))
        x = res_identity(x, filters=(256, 1024))
        x = res_identity(x, filters=(256, 1024))
        x = res_identity(x, filters=(256, 1024))
        x = res_identity(x, filters=(256, 1024))

        # 5th stage

        x = res_conv(x, s=2, filters=(512, 2048))
        x = res_identity(x, filters=(512, 2048))
        x = res_identity(x, filters=(512, 2048))


        #===concatenate Layer =====
        x = self.concatted([ltsmX,conX])



        #=== fused Model  ===
        x = self.l22(x)
        if training:
            x = self.dropout22(x, training=training)

        x = self.l3(x)

        if training:
            x = self.dropout3(x, training=training)

        return self.lout(x)


    def showMetric(self,evalDat,predictDat):
        print("\n========[Show Evalution metrics] =====")
        print("explained_variance_score: (best:1)\t{:.4f}".format(skMet.explained_variance_score(evalDat, predictDat)))
        print("max_error: \t\t{:.4f}".format(skMet.max_error(evalDat, predictDat)))
        print("mean_absolute_error: \t{:.4f}".format(skMet.mean_absolute_error(evalDat, predictDat)))
        print("mean_squared_error: \t{:.4f}".format(skMet.mean_squared_error(evalDat, predictDat)))


def main():
    pass

if __name__ == "__main__":
    main()
