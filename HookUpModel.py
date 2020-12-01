#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 20:10:51 2020

@author: sausy
"""
import numpy as np 
np.random.seed(4)

from preprocessing import preprocessing


#import keras
#from keras.models import Model
#from keras.layers import Dense, Dropout, LSTM, Input, Activation
#from keras import optimizers

import tensorflow as tf 


def fit_model1(Xtrain,Xeval,Ytrain,Yeval, batch_sizes):
    #lstm_input = Input(shape=(Xtrain.shape[1], Xtrain.shape[2]), name='lstm_input')
    #x = LSTM(Xtrain.shape[1], name='lstm_0')(lstm_input)
    #x = Dropout(0.2, name='lstm_dropout_0')(x)
    lstm_input = tf.keras.Input(shape=(Xtrain.shape[1], Xtrain.shape[2]))
    # set the first layer as LSTM 
    x = tf.keras.layers.LSTM(Xtrain.shape[1], dropout=0.2)(lstm_input)
    
    #x = tf.keras.layers.LSTM(Xtrain.shape[1], name='lstm_0', dropout=0.5)(x)
    #x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dense(64, activation='sigmoid')(x)
    
    #output = tf.keras.layers.Dense(1, activation='linear')(x)
    output = tf.keras.layers.Dense(1, activation='linear')(x)
    
    #x = Dense(64, name='dense_0')(x)
    #x = Activation('sigmoid', name='sigmoid_0')(x)
    #x = Dense(1, name='dense_1')(x)
    
    #output = Activation('linear', name='linear_output')(x)
    
    model = tf.keras.Model(inputs=lstm_input, outputs=output)
    adam = tf.keras.optimizers.Adam(lr=0.0005)
    model.compile(optimizer=adam, loss='mse')
    model.fit(x=Xtrain, y=Ytrain, batch_size=batch_sizes, epochs=50, shuffle=True, validation_split=0.1)
    
    print("Fit Model -- Done")
    
    return model



def fit_model2(Xtrain,Xeval,Ytrain,Yeval, batch_sizes, epochVal):
    #sizeX1 = Xtrain.shape[1]
    tf.keras.backend.clear_session()
    tf.random.set_seed(4)
    
    N = Xtrain.shape[0]
    AmountParallelSeries = Xtrain.shape[1]
    AmountFeatures = Xtrain.shape[2]
    UnitCountHiddenLayer1 = AmountFeatures*2
    
    
    model = tf.keras.models.Sequential()
    #model.add(tf.keras.layers.LSTM(UnitCountHiddenLayer1, dropout=0.2, return_sequences=True, input_shape=(AmountParallelSeries, AmountFeatures)))
    #model.add(tf.keras.layers.LSTM(UnitCountHiddenLayer1, dropout=0.5))
    
    #inputShape = (N, AmountParallelSeries, AmountFeatures)
    '''
    model.add(tf.keras.layers.Conv1D(60, 5, activation='relu', input_shape=(AmountParallelSeries, AmountFeatures)))
    model.add(tf.keras.layers.MaxPool1D(2))
    model.add(tf.keras.layers.LSTM(AmountFeatures, dropout=0.2))
    '''
    
    #model.add(tf.keras.layers.LSTM(AmountFeatures, dropout=0.2, input_shape=(AmountParallelSeries, AmountFeatures)))
    model.add(tf.keras.layers.LSTM(UnitCountHiddenLayer1, activation='relu', return_sequences=True, input_shape=(AmountParallelSeries, AmountFeatures)))
    model.add(tf.keras.layers.LSTM(UnitCountHiddenLayer1, activation='relu'))
    
    model.add(tf.keras.layers.Dense(AmountFeatures))


    
    ##model.add(tf.keras.layers.Dense(100, activation='tanh'))
    #model.add(tf.keras.layers.Dense(AmountFeatures*6, activation='relu'))
    ##model.add(tf.keras.layers.Dense(AmountFeatures*2, activation='tanh'))
    #model.add(tf.keras.layers.Dense(5*5, activation='sigmoid'))

    #model.add(tf.keras.layers.Dense(AmountFeatures, activation='linear'))
    
    #define optimization
    #adam = tf.keras.optimizers.Adam(lr=0.0006)
    #model.compile(optimizer=adam, loss='mse')
    model.compile(optimizer='adam', loss='mse')
    
    #fit it
    model.fit(x=Xtrain, y=Ytrain, batch_size=batch_sizes, epochs=epochVal, shuffle=True, validation_split=0.1)
    
    '''
    #The best result till now
    
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(AmountFeatures, dropout=0.2, input_shape=(AmountParallelSeries, AmountFeatures)))
    model.add(tf.keras.layers.Dense(AmountFeatures*3, activation='relu'))
    
    model.add(tf.keras.layers.Dense(AmountFeatures*3, activation='relu'))
    model.add(tf.keras.layers.Dense(5, activation='sigmoid'))
    model.add(tf.keras.layers.Dense(1, activation='linear'))
    
    #define optimization
    adam = tf.keras.optimizers.Adam(lr=0.0006)
    model.compile(optimizer=adam, loss='mse')
    
    #fit it
    model.fit(x=Xtrain, y=Ytrain, batch_size=batch_sizes, epochs=epochVal, shuffle=True, validation_split=0.1)
   
    
    '''

    
    
    '''
    lstm_input = tf.keras.Input(shape=(AmountParallelSeries, AmountFeatures))
    # set the first layer as LSTM 
    print("size of lstm_input {}".format(lstm_input.shape))
    
    x = tf.keras.layers.LSTM(UnitCountHiddenLayer1, dropout=0.2)(lstm_input)
    #x = tf.keras.layers.LSTM(sizeX1, dropout=0.5)(x)
    print("size of x {}".format(x.shape))
    
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    
    #output = tf.keras.layers.Dense(1, activation='linear')(x)
    output = tf.keras.layers.Dense(1, activation='linear')(x)
    
    model = tf.keras.Model(inputs=lstm_input, outputs=output)
    adam = tf.keras.optimizers.Adam(lr=0.0005)
    model.compile(optimizer=adam, loss='mse')
    model.fit(x=Xtrain, y=Ytrain, batch_size=batch_sizes, epochs=50, shuffle=True, validation_split=0.1)
    
    print("Fit Model -- Done")
    
    '''
    
    return model

def fit_model3(Xtrain,Xtech,Ytrain, batch_sizes, epochVal):
    print("additional modell")
    
    tf.keras.backend.clear_session()
    tf.random.set_seed(4)
    
    N = Xtrain.shape[0]
    AmountParallelSeries = Xtrain.shape[1]
    AmountFeatures = Xtrain.shape[2]
    UnitCountHiddenLayer1 = AmountFeatures*2
    
    
    
    y = tf.keras.layers.LSTM(UnitCountHiddenLayer1, activation='relu', return_sequences=True, input_shape=(AmountParallelSeries, AmountFeatures))(Xtrain)
    y = tf.keras.layers.LSTM(UnitCountHiddenLayer1, activation='relu')(y)
    y = tf.keras.layers.Dense(AmountFeatures)(y)
    lstm_branch = tf.keras.models.Model(inputs=Xtrain, outputs=y)
    
    dense_input = tf.keras.layers.Input(shape=(Xtech.shape[1],), name='tech_input')
    y2 = tf.keras.layers.Dense(20, name='tech_dense_0')(dense_input)
    y2 = tf.keras.layers.Activation("relu", name='tech_relu_0')(y2)
    y2 = tf.keras.layers.Dropout(0.2, name='tech_dropout_0')(y2)
    technical_indicators_branch = tf.keras.models.Model(inputs=Xtech, outputs=y2)
    
    #model.add(tf.keras.layers.Dense(AmountFeatures))
    
    combined = tf.keras.layers.concatenate([lstm_branch.output, technical_indicators_branch.output], name='concatenate')
    
    
    z = tf.keras.layers.Dense(64, activation="sigmoid", name='dense_pooling')(combined)
    z = tf.keras.layers.Dense(1, activation="linear", name='dense_out')(z)
    
    model = tf.keras.models.Model(inputs=[lstm_branch.input, technical_indicators_branch.input], outputs=z)
    #adam = optimizers.Adam(lr=0.0005)
    #model.compile(optimizer=adam, loss='mse')
    #model.fit(x=[ohlcv_train, tech_ind_train], y=y_train, batch_size=32, epochs=50, shuffle=True, validation_split=0.1)
    
    model.compile(optimizer='adam', loss='mse')
    model.fit(x=[Xtrain,Xtech], y=Ytrain, batch_size=batch_sizes, epochs=epochVal, shuffle=True, validation_split=0.1)
    
    return model
    
    
    # the second branch opreates on the second input
    #dense_input = Input(shape=(technical_indicators.shape[1],), name='tech_input')
    

def main():     
    import matplotlib.pyplot as plt 
    
    #tf.random.set_seed(4)

    
    #==== Definitions ==== 
    splitVal = 0.97
    PredictDayIntoFuture = 1
    HistoryExpan = 50 #how fare shall we go back in time
    
    #batch_sizes = [4, 32, 64] 
    #batch_sizes = [128, 254] 
    #batch_sizes = [4, 254] 
    #batch_sizes = [16, 16, 16, 16]
    #batch_sizes = [128] 
    #epochs = [2] 
    #batch_sizes = [16, 32, 128] 
    #epochs = [70, 80, 250] 
    #batch_sizes = [32, 128] 
    #epochs = [80, 250] 
    
    #batch_sizes = [32, 128] 
    #epochs = [60, 200]
    batch_sizes = [512,128] 
    epochs = [50,1]
    
    
    #to figure out how our modell behaves on different batch sizes... 
    
    #==== Parameters of Modell 
    
    
    #=== Preprocess it ==== 
    pp = preprocessing()
    [Xtrain,Xeval,Ytrain,Yeval] = pp.setUP(splitVal,HistoryExpan, PredictDayIntoFuture)
    
    
    #====== PLOT DATA =====     
    XPlot1 = np.linspace(0, splitVal, len(Xtrain))
    XPlot2 = np.linspace(splitVal, 1, len(Xeval))
    
    dumpData = np.linspace(XPlot1[0], XPlot1[0], len(Xeval))
    XPlot1 = np.append(dumpData,XPlot1)
    #plotTrainData = np.zeros((len(Xeval),Xeval.shape[1]))
    
    dumpData = np.linspace(XPlot2[0], XPlot2[0], len(Xtrain))
    XPlot2 = np.append(dumpData,XPlot2)
    
    plotTrainData= np.linspace(Xtrain[0], Xtrain[0], len(Xeval))
    plotEvalData = np.linspace(Xeval[0], Xeval[0], len(Xtrain))
    
    
    #if plotTrainData.shape[1] < 5 :
    #    fig, axs = plt.subplots(plotTrainData.shape[1])
    #else:
    #    fig, axs = plt.subplots(5)
    
    fig, axs = plt.subplots(plotTrainData.shape[2],2, figsize=(6,10))
        
    fig.suptitle('different Features')
    for i in range(plotTrainData.shape[2]):
        #axs[i].plot(XPlot1 , np.append(plotTrainData[:,i],Xtrain[:,i],axis=0)) #'r+'
        #axs[i].plot(XPlot2 , np.append(plotEvalData[:,i],Xeval[:,i],axis=0))
        axs[i,0].plot(np.linspace(0, 10, Yeval.shape[0]),Yeval[:,0], 'k')
        axs[i,1].plot(np.linspace(0, 10, Ytrain.shape[0]),Ytrain[:,0], 'k')
        
    plt.show()
    
    
    
    print("Fit Model")
    print("Xtrain {}".format(Xtrain.shape))
    print("Xeval {}".format(Xeval.shape))
    print("Ytrain {}".format(Ytrain.shape))
    print("Yeval {}".format(Yeval.shape))
    
    
    fig, axs = plt.subplots(len(batch_sizes),5, figsize=(20,10))
    #fig2, axs = plt.subplots(len(batch_sizes), figsize=(8,10))
    plotX = np.linspace(0, 10, Yeval.shape[0])
    plot2 = np.linspace(0, 10, Ytrain.shape[0])
    
    #histVar = np.array([])
    
    #====== MOdel 1 ===== 
    Variance = np.array([[]])
    foo = 0.0
    mean_vec = Yeval.shape[0]   
    
    YMean = (Yeval[:,1]+Yeval[:,2])/2
    YRESHAPED = Yeval[:,:,0]
    
    #print("shape YRESHAPED{}".format(YRESHAPED.shape))
    #print("shape Yeval {}".format(Yeval.shape))
    #exit
        
    fig.suptitle('different batch sizes')
    for n in range(len(batch_sizes)):
        model = fit_model2(Xtrain,Xeval,Ytrain,Yeval,batch_sizes[n], epochs[n])
        Ypredict = model.predict(Xeval)
        Ypredict2 = model.predict(Xtrain)
        
        #calculate variance
        #print("shape Yeval{}".format(YRESHAPED.shape))
        #print("shape Ypredict {}".format(Ypredict.shape))
        
        yDiv = YRESHAPED[:,0] - Ypredict[:,0]
        #print("Yeval {}".format(Yeval[:,0][0]))
        #print("Ypredict {}".format(Ypredict[:,0][0]))
        #print("div {}".format(yDiv))
        #print("div2 {}".format(yDiv[0]))
        
        #print("shape {}".format(yDiv.shape))
        
        
        foo = np.sum(np.abs(yDiv))/mean_vec
        print("sum {}".format(np.sum(np.abs(yDiv))))
        foo2 = np.sum(np.abs(Ytrain[:,0] - Ypredict2[:,0]))/Ytrain.shape[0]
        Variance = np.append(Variance, np.array([foo,foo2]))
        plt.title("Var {}".format(foo))
        
        #print("shape {}".format(Yeval))
        
        axs[n,0].plot(plotX,Yeval[:,0], 'k')
        axs[n,0].plot(plotX,Ypredict[:,0], 'g')
        
        axs[n,1].plot(plotX,Yeval[:,3], 'k')
        axs[n,1].plot(plotX,Ypredict[:,3], 'r')
        
        #axs[n,0].plot(plotX,yDiv, 'r')
        
        #axs[n,1].plot(plotX,YMean, 'k')
        axs[n,2].plot(plotX,Yeval[:,1], 'k')
        axs[n,2].plot(plotX,Ypredict[:,1], 'b')
        axs[n,2].plot(plotX,Ypredict[:,2], 'y')
        
        
        axs[n,3].plot(plotX,Yeval[:,2], 'k')
        axs[n,3].plot(plotX,Ypredict[:,1], 'b')
        axs[n,3].plot(plotX,Ypredict[:,2], 'y')
        
        axs[n,3].plot(plotX,Ytrain[:,0], 'k')
        axs[n,3].plot(plotX,Ypredict2[:,0], 'b')
        axs[n,3].plot(plotX,Ypredict2[:,1], 'y')
        
        #axs[n,3].plot(plot2,Ytrain[:,0], 'k')
        #axs[n,3].plot(plot2,Ypredict2[:,0], 'g')
        
        
        #print(np.var(Yeval,))
        
        
    plt.show()
    
    #histVar = np.append(histVar,Variance,axis=1)
    histVar = Variance
    print("Variance list {}".format(histVar))
    
    '''
    #====== MOdel 2 ===== 
    
    print("Variance list {}".format(Variance))
    fig, axs = plt.subplots(len(batch_sizes), figsize=(8,10))
    
    Variance = np.append(Variance, np.array([foo]))
    
    Variance = np.array([])
    foo = 0.0
        
    fig.suptitle('different batch sizes')
    for n in range(len(batch_sizes)):
        model = fit_model2(Xtrain,Xeval,Ytrain,Yeval,batch_sizes[n])
        Ypredict = model.predict(Xeval)
        
        #calculate variance
        foo = np.sum(np.abs(Yeval - Ypredict))/len(Yeval)        
        Variance = np.append(Variance, np.array([foo]))
        plt.title("Var {}".format(foo))
        
        axs[n].plot(plotX,Yeval, 'k')
        axs[n].plot(plotX,Ypredict, 'g')
        
        #print(np.var(Yeval,))
        
        
    plt.show()
    
    histVar = np.array([Variance, histVar])

    print("Variance list {}".format(Variance))
    print("Variance list {}".format(histVar))
    '''

    print("Model prediction")
    #====== Evaluate =====    
    #Ypredict = model.predict(Xeval)
    #Yeval
    
    print("Size of Ypredict {}|{}".format(Ypredict.shape,Ypredict[:5]))
    print("Size of Yeval {}|{}".format(Yeval.shape,Yeval[:5]))
    
    
    
    
    
    
    #y_test_predicted = model.predict(Xtrain)
    #y_test_predicted = y_normaliser.inverse_transform(y_test_predicted)
    #y_predicted = model.predict(ohlcv_histories)
    #y_predicted = y_normaliser.inverse_transform(y_predicted)
    
    
    #====== Rescaling for furhter procssing ===== 
    
    '''
    assert unscaled_y_test.shape == y_test_predicted.shape
    real_mse = np.mean(np.square(unscaled_y_test - y_test_predicted))
    scaled_mse = real_mse / (np.max(unscaled_y_test) - np.min(unscaled_y_test)) * 100
    print(scaled_mse)
    
    import matplotlib.pyplot as plt
    
    plt.gcf().set_size_inches(22, 15, forward=True)
    
    start = 0
    end = -1
    
    real = plt.plot(unscaled_y_test[start:end], label='real')
    pred = plt.plot(y_test_predicted[start:end], label='predicted')
    
    # real = plt.plot(unscaled_y[start:end], label='real')
    # pred = plt.plot(y_predicted[start:end], label='predicted')
    
    plt.legend(['Real', 'Predicted'])
    
    plt.show()
    '''
    
    '''
    lstm_input = Input(shape=(history_points, 5), name='lstm_input')
    dense_input = Input(shape=(technical_indicators.shape[1],), name='tech_input')
    
    # the first branch operates on the first input
    x = LSTM(50, name='lstm_0')(lstm_input)
    x = Dropout(0.2, name='lstm_dropout_0')(x)
    lstm_branch = Model(inputs=lstm_input, outputs=x)
    
    # the second branch opreates on the second input
    y = Dense(20, name='tech_dense_0')(dense_input)
    y = Activation("relu", name='tech_relu_0')(y)
    y = Dropout(0.2, name='tech_dropout_0')(y)
    technical_indicators_branch = Model(inputs=dense_input, outputs=y)
    
    # combine the output of the two branches
    combined = concatenate([lstm_branch.output, technical_indicators_branch.output], name='concatenate')
    
    z = Dense(64, activation="sigmoid", name='dense_pooling')(combined)
    z = Dense(1, activation="linear", name='dense_out')(z)
    
    # our model will accept the inputs of the two branches and
    # then output a single value
    model = Model(inputs=[lstm_branch.input, technical_indicators_branch.input], outputs=z)
    adam = optimizers.Adam(lr=0.0005)
    model.compile(optimizer=adam, loss='mse')
    model.fit(x=[ohlcv_train, tech_ind_train], y=y_train, batch_size=32, epochs=50, shuffle=True, validation_split=0.1)
    '''
    

if __name__ == "__main__":
    main()