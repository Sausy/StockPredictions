#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 15:14:05 2020

@author: sausy
"""

from getData import cryptocurrency


#dataGrab = getData.cryptocurrency()
dataGrab = cryptocurrency()
from dataTransform import csv_to_dataset

#from tensorflow.keras import backend

import matplotlib.pyplot as plt 
import numpy as np
print('set random.seed(4)')
np.random.seed(4)

import tensorflow as tf

import keras
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, Activation
from keras import optimizers
print('==[Data]==')

#from tensorflow import set_random_seed
#set_random_seed(4)
#from util import csv_to_dataset, history_points


#np.random.seed(4)
#import tensorflow
#from tensorflow import set_random_seed
#set_random_seed(4)
tf.random.set_seed(4)

def runNN():
    '''
    process data 
    '''
    print('==[Data]==')
    dataGrab.setUp()
    
    '''
    Training
    '''
    print('==[Training]==')
    foo_.append(rowdata for rowdata in dataGrab.HistorData[0])
    ohlcv_histories, _, next_day_open_values, unscaled_y, y_normaliser = csv_to_dataset(foo_)
    print('==[after csv]==')
    test_split = 0.9
    n = int(ohlcv_histories.shape[0] * test_split)
    
    ohlcv_train = ohlcv_histories[:n]
    y_train = next_day_open_values[:n]
    
    ohlcv_test = ohlcv_histories[n:]
    y_test = next_day_open_values[n:]
    
    unscaled_y_test = unscaled_y[n:]
    
    print(ohlcv_train.shape)
    print(ohlcv_test.shape)
    
    
    '''
    setUp Model
    '''
    print('==[setup model]==')
    history_points = 50
    lstm_input = Input(shape=(history_points, 5), name='lstm_input')
    #lstm_input = dataGrab.
    
    x = LSTM(50, name='lstm_0')(lstm_input)
    x = Dropout(0.2, name='lstm_dropout_0')(x)
    x = Dense(64, name='dense_0')(x)
    x = Activation('sigmoid', name='sigmoid_0')(x)
    x = Dense(1, name='dense_1')(x)
    output = Activation('linear', name='linear_output')(x)
    
    '''
    model
    '''
    print('==[model]==')
    model = Model(inputs=lstm_input, outputs=output)
    adam = optimizers.Adam(lr=0.0005)
    
    model.compile(optimizer=adam, loss='mse')  
    print('==[model]==')
    model.fit(x=ohlcv_train, y=y_train, batch_size=32, epochs=50, shuffle=True, validation_split=0.1)
    
    
    #####
    '''
    dataTransform.csv_to_dataset
    data_normaliser = preprocessing.MinMaxScaler()
    data_normalised = data_normaliser.fit_transform(dataGrab.HistorData[0][:])
    
    
    next_day_open_values = np.array([data[:, 0][i + history_points].copy() for i in range(len(data) - history_points)])
    next_day_open_values = np.expand_dims(next_day_open_values, -1)
        
    y_normaliser = preprocessing.MinMaxScaler()
    y_normaliser.fit(next_day_open_values)
    '''
    ######
    
    
    
    unscaled_y_test = unscaled_y[n:]
    #y_scaler = 
    
    
    model.fit(x=ohlcv_train, y=y_train, batch_size=32, epochs=50, shuffle=True, validation_split=0.1)
    evaluation = model.evaluate(ohlcv_test, y_test)
    print(evaluation)
    
    '''
    Evaluation
    '''
    print('==[Evaluation]==')
    y_test_predicted = model.predict(ohlcv_test)
    y_test_predicted = y_normaliser.inverse_transform(y_test_predicted)
    y_predicted = model.predict(ohlcv_histories)
    y_predicted = y_normaliser.inverse_transform(y_predicted)
    
    assert unscaled_y_test.shape == y_test_predicted.shape
    real_mse = np.mean(np.square(unscaled_y_test - y_test_predicted))
    scaled_mse = real_mse / (np.max(unscaled_y_test) - np.min(unscaled_y_test)) * 100
    print(scaled_mse)
    
    '''
    Plot
    '''
    plt.gcf().set_size_inches(22, 15, forward=True)
    
    start = 0
    end = -1
    
    real = plt.plot(unscaled_y_test[start:end], label='real')
    pred = plt.plot(y_test_predicted[start:end], label='predicted')
    
    plt.legend(['Real', 'Predicted'])
    
    plt.show()
    
    '''
    Clear it
    '''
    
try:
    runNN()
except:
    print("there was an error")

tf.keras.backend.clear_session()
keras.backend.clear_session()