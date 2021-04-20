#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dez  8 10:53:41 2020

@author: sausy
"""

import numpy as np
import pandas as pd

import plotly.graph_objects as go

import copy

#from getData import cryptocurrency
#from StockIndicators import indicators
#from sklearn import preprocessing
#import tensorflow as tf

from preprocessing import preProcessRawData as prepro
from preprocessing import StockIndicators as preStock
from plot2D import stockPlot as plt_modul

import matplotlib.pyplot as plt


from sklearn import preprocessing
from sklearn import metrics as skMet

import tensorflow as tf

def res_identity(x, filters):
    #renet block where dimension doesnot change.
    #The skip connection is just simple identity conncection
    #we will have 3 blocks and then input will be added

    x_skip = x # this will be used for addition with the residual block
    f1, f2 = filters

    #first block
    x = tf.keras.layers.Conv2D(f1, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(tf.keras.activations.relu)(x)

    #second block # bottleneck (but size kept same with padding)
    x = tf.keras.layers.Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(tf.keras.activations.relu)(x)

    # third block activation used after adding the input
    x = tf.keras.layers.Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.Activation(activations.relu)(x)

    # add the input
    x = tf.keras.layers.Add()([x, x_skip])
    x = tf.keras.layers.Activation(tf.keras.activations.relu)(x)

    return x


def res_conv(x, s, filters):
    '''
    here the input size changes'''
    x_skip = x
    f1, f2 = filters

    # first block
    x = tf.keras.layers.Conv2D(f1, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    # when s = 2 then it is like downsizing the feature map
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(tf.keras.activations.relu)(x)

    # second block
    x = tf.keras.layers.Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(tf.keras.activations.relu)(x)

    #third block
    x = tf.keras.layers.Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # shortcut
    x_skip = tf.keras.layers.Conv2D(f2, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x_skip)
    x_skip = tf.keras.layers.BatchNormalization()(x_skip)

    # add
    x = tf.keras.layers.Add()([x, x_skip])
    x = tf.keras.layers.Activation(tf.keras.activations.relu)(x)

    return x

def resnet50(data,ySize):

    #Because RESNET50 or better speaking CNN needs a different format for the input
    # (batch, time, features) -> (batch, time, features, 1)
    #print("x type: {}".format(type(x1_eval)))
    dimX = len(data.shape)

    #print("x Cnn training Data shape: {}".format(xCnn_train.shape))
    #print("x Cnn eval Data shape: {}".format(xCnn_eval.shape))
    #print("x Cnn type: {}".format(type(xCnn_eval)))

    resFilterSize = [64,256]
    #The stockmarket doesn't have rgb xD
    #hence
    # (ImageLabel,imgHigh,imgWidth,rgb)
    ##input_im = tf.keras.layers.Input(shape=(data.shape[0], data.shape[1], data.shape[2])) # cifar 10 images size
    # becomes
    # (batch, time, features) -> (batch, time, features, 1)
    input_cnn = tf.keras.layers.Input(shape=(data.shape[1], data.shape[2], 1))
    input_ltsm = tf.keras.layers.Input(shape=(data.shape[1], data.shape[2]))

    #xCnn = tf.expand_dims(input_im, axis=dimX) # np.expand_dims(input_im, axis=dimX)
    #x = tf.keras.layers.ZeroPadding2D(padding=(3, 3))(input_cnn)
    #x = tf.keras.layers.Conv2D(56, kernel_size=(7, 7), strides=(2, 2))(input_cnn)

    #xltsm = tf.keras.layers.LSTM(100, activation='tanh')(input_ltsm)
    #x = tf.keras.layers.Dropout(0.2)(x)
    xltsm = tf.keras.layers.LSTM(100, activation='tanh',dropout=0.2, return_sequences=True)(input_ltsm)
    xltsm = tf.keras.layers.LSTM(100, activation='tanh',dropout=0.2)(xltsm)


    # 1st stage
    # here we perform maxpooling, see the figure above

    x = tf.keras.layers.Conv2D(56, kernel_size=(7, 7), strides=(2, 2))(input_cnn)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(tf.keras.activations.relu)(x)
    x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

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
    #it was proposed to only use 4 stages
    #x = res_conv(x, s=2, filters=(512, 2048))
    #x = res_identity(x, filters=(512, 2048))
    #x = res_identity(x, filters=(512, 2048))

    # ends with average pooling and dense connection

    x = tf.keras.layers.AveragePooling2D((7, 7), padding='same')(x)

    x = tf.keras.layers.Flatten()(x)


    #==== now Concatenate ltsm and resnet 50 ====
    x = tf.keras.layers.concatenate([x,xltsm])
    x = tf.keras.layers.Dense(500, activation='tanh')(x) #multi-class
    x = tf.keras.layers.Dropout(0.5)(x)


    x = tf.keras.layers.Dense(100, activation='tanh')(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    x = tf.keras.layers.Dense(25, activation='tanh')(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    x = tf.keras.layers.Dense(ySize, activation='linear')(x)
    #x = tf.keras.layers.Dense(ySize, activation='tanh')(x)


    #x = tf.keras.layers.Dense(self.UnitCountHiddenLayer1, activation='tanh')(x)
    #x = tf.keras.layers.Dropout(0.2)(x)
    # define the model

    model = tf.keras.models.Model(inputs=[input_cnn,input_ltsm], outputs=x, name='Resnet50')

    return model

def simplLstm(data,ySize):

    input_ltsm = tf.keras.layers.Input(shape=(data.shape[1], data.shape[2]))

    x = tf.keras.layers.LSTM(100, activation='tanh',dropout=0.2, return_sequences=True)(input_ltsm)
    x = tf.keras.layers.LSTM(100, activation='tanh',dropout=0.2)(x)
    #x = tf.keras.layers.LSTM(80, activation='tanh',dropout=0.2)(input_ltsm)

    x = tf.keras.layers.Dense(int(80*1.8), activation='tanh')(x) #multi-class
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(int(80*1.8), activation='tanh')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    #x = tf.keras.layers.Dense(int(80*2.2), activation='tanh')(x)
    #x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(int(80), activation='tanh')(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    out = tf.keras.layers.Dense(ySize, activation='linear')(x)

    model = tf.keras.models.Model(inputs=input_ltsm, outputs=out, name='simpleltsm')

    return model

def divRawData(df):
    colList = list(df.columns)
    colList.remove('date')
    colList.remove('daysin')
    colList.remove('daycos')
    colList.remove('weeksin')
    colList.remove('weekcos')
    colList.remove('yearsin')
    colList.remove('yearcos')
    colList.remove('rsi')



    ret = pd.DataFrame()

    #create data shifted by 1 iteration
    shiftData = df.shift(1,fill_value=1)
    #divide current value with last one
    df_r = df
    df_r[colList] = df[colList].div(shiftData[colList])

    #because the first row was just div with 1
    ret = df_r.drop(df_r.index[0])

    #now get the log of it
    ret[colList] = np.log10(ret[colList])

    return ret

def log10Data(df):
    colList = list(df.columns)
    colList.remove('date')
    colList.remove('daysin')
    colList.remove('daycos')
    colList.remove('weeksin')
    colList.remove('weekcos')
    colList.remove('yearsin')
    colList.remove('yearcos')
    colList.remove('rsi')

    #now get the log of it
    ret = df
    ret[colList] = np.log10(df[colList])

    return ret



def hotfixLinearShift(x1_train,y1_train,y1_predict):
    #lastX = x1_train[x1_train.shape[0]-1:]
    #lastX = x1_train[x1_train.shape[0]-1:]
    ret = y1_predict

    return ret

def main():

    import sys

    tf.keras.backend.clear_session()
    tf.random.set_seed(4)

    if len(sys.argv) < 3:
        print("usage: ./programm <inputFile> <outputPath>")
        print("will proceed with ./CSV/rdyCSV/rdy.csv")
        inputFile = "./CSV/rdyCSV/rdy.csv"
        outputPath= "./CSV/rdyCSV/"
    else:
        inputFile = sys.argv[1]
        outputPath= sys.argv[2]

    outputFile= outputPath + "predicted.csv"
    outputFileEval= outputPath + "eval.csv"

    print("Sys INPUTFILE \t<{}>\nOUTPUT FILE: \t<{}>".format(inputFile,outputFile))



    #===========================================================
    # PreProcessData
    # Basic Parameters
    #===========================================================
    TicksIntoFuture = 10
    TicksIntoPast = 40#192=1std * 24 * 8 ..... #8days => 8[day]*24[std/day] = 192[std]
    ##the present is not included into this value hence TicksIntoPast can be 0
    #and the batch size is TicksIntoPast+1
    pp = prepro.preprocessing(ticksIntoPast=TicksIntoPast,ticksIntoFuture=TicksIntoFuture, debug=True)
    data = pd.read_csv(inputFile)
    print(data)
    #plt = plt_modul.stockPlot(data)

    '''
    lfoo = data.shape[0]
    fooOut = data[['date','open']][lfoo-246832:]#,columns=LabelList
    print(fooOut)
    fooOut.to_csv(outputPath + "dbg_start.csv",index=False)
    return
    '''

    '''
    Divide each row by the previouse one   Pt/P(t-1)
    '''
    #data = divRawData(data)


    '''
    make data logarithmic
    '''
    data = log10Data(data)


    print("\n===========================================================")
    print("Scale Data")
    #due to preprocessing, certain data needs scaling
    CsvFeatureList_needsScaling =  ['open',
                                    'high',
                                    'low',
                                    'close',
                                    'daysin',
                                    'daycos',
                                    'weeksin',
                                    'weekcos',
                                    'yearsin',
                                    'yearcos',
                                    'avg',
                                    'sma10',
                                    'sma100',
                                    'ema10',
                                    'ema21',
                                    'ema100',
                                    "rsi"]
    data = pp.scaleData(data, CsvFeatureList_needsScaling, method='normalize')
    #data = pp.scaleData(data, CsvFeatureList_needsScaling, method='standardize')

    print("\n++++++++[Down Scaled X]+++++")
    print(data)


    #dataUpscaled = pp.upscaleData(data)
    #plt.addPlot(dataUpscaled,'high',"Shifted")
    #plt.addCandle(dataUpscaled,"Shifted")

    #lfoo = data.shape[0]
    #fooOut = data[['date','open']][lfoo-246832:]#,columns=LabelList
    #fooOut.to_csv(outputPath + "dbg_DownScaledX.csv",index=False)


    print("\n===========================================================")
    print("Create Y\n")
    print("The output equals X(t+n) so x as seen t+n times in the future")
    #because oure model needs knowlege into the past
    #it needs to be defined what features shall be "time shifted"
    #TODO : y needs a date label not for training but to match it
    #when presenting the data again
    LabelList =    ['open',
                    'high',
                    'low',
                    'close',
                    'avg',
                    'sma10',
                    'sma100',
                    'ema10',
                    'ema21',
                    'ema100',
                    "rsi"]
    #LabelList =    ['rsi']
    LabelList =    ['ema21']


    print("\n++++++++[y simple forcast ]+++++")
    [data,y] = pp.genForcastY(data, LabelList=LabelList) #, includeAllFuturDays=False
    print(y)



    print("\n===========================================================")
    print("remove non Trainable Columns in data\n")
    try:
        xNonTrain = copy.deepcopy(data)
        data = data.drop(columns=['date'])
        data = data.values
    except:
        print("[ERROR] X no label name date found")


    try:
        yNonTrain = copy.deepcopy(y)
        y = y.drop(columns=['date'])
        y = y.values
    except:
        print("[ERROR] Y no label name date found")

    #x = x[featureList][:y.shape[0]].values
    print("\n++++++++[xNonTrain]+++++")
    print(xNonTrain)
    print("\n++++++++[yNonTrain]+++++")
    print(yNonTrain)


    print("\n===========================================================")
    print("Create Time Shifted data\n")
    print("Data will be shaped acording to tensor flows [batch, time, features] ... windows")
    [data,y] = pp.genTimeSeries(data,y)

    print("\nY Data is \n")
    print(y)
    print("\nTime Shifted data is  \n")
    print(data)


    print("\nY Shape \t{}".format(y.shape))
    print("\ndata Shape\t{}".format(data.shape))



    #===========================================================
    # Datashaping
    #===========================================================
    print("\n===========================================================")
    print("Data will be shaped acording to tensor flows [batch, time, features] ... windows")


    #===========================================================
    # model parameters
    #===========================================================
    splitVal = 0.98
    #Max batch size= available GPU memory bytes / 4 / (size of tensors + trainable parameters)
    #was training on a gtx970
    batch_sizes = 128#5632 #4096#2048#1024
    epochs = 64#64

    dataSize = int(data.shape[0] * splitVal)
    x1_train = data[:dataSize]
    x1_eval = data[dataSize:]
    y1_train = y[:dataSize]
    y1_eval = y[dataSize:]


    #==========================================================================
    #===========================[TENSORFLOW]===================================
    #==========================================================================

    dimX = len(data.shape)
    xCnn_train = np.expand_dims(x1_train, axis=dimX)
    xCnn_eval = np.expand_dims(x1_eval, axis=dimX)

    #m = resnet50(xCnn_train,y1_train.shape[1])
    m = resnet50(x1_train,y1_train.shape[1])
    m2 = simplLstm(x1_train,y1_train.shape[1])


    #opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    opt = tf.keras.optimizers.SGD(learning_rate=0.05, momentum=0.9, nesterov=True, name='SGD') #not sure if its a good idea to use nesterov momentum
    #well i my imagination I don't have much outliers
    #this is because we are trying to aproximate the sma ore emi value
    #which is a "filtered" stockprice value
    m.compile(opt, loss='mse', metrics=['accuracy'])
    #m2.compile(opt, loss='mae', metrics=['accuracy'])
    m2.compile(opt, loss='mse', metrics=['accuracy'])
    #m2.compile(opt, loss='mse')



    #set callback to tensorboard
    #logFileName = time.strftime("%d_%H_%M_%S", time.gmtime())
    tb_callback = tf.keras.callbacks.TensorBoard('logs\\log', update_freq='epoch') #update tensorboard every 300 batches ... 'epoch'

    # Train
    #m.fit(x=xCnn_train, y=y1_train, batch_size=batch_sizes, epochs=epochs, shuffle=True, validation_split=0.1)#, callbacks=[cp_callback] #shuffle=True, validation_split=0.1

    #m.fit(x=[xCnn_train,x1_train], y=y1_train, batch_size=batch_sizes, epochs=epochs, validation_split=0.1,callbacks=[tb_callback])
    #y1_predict = m.predict([xCnn_eval,x1_eval])


    batch_sizes = 4096#5632 #4096#2048#1024
    epochs = 300#64
    m2.fit(x=x1_train, y=y1_train, batch_size=batch_sizes, epochs=epochs, validation_split=0.1,callbacks=[tb_callback]) #, shuffle=True
    y1_predict = m2.predict(x1_eval)


    print("\nShape: \tevaluate:{} \tpredicted:{}".format(y1_eval.shape,y1_predict.shape))
    #print(y1_predict)

    k = 0
    tTracker = 1
    for i in range(0,y1_eval.shape[1]):
        print("\n========[Show Evalution metrics] =====")
        print("explained_variance_score: (best:1)\t{:.4f}".format(skMet.explained_variance_score(y1_eval[0:,i], y1_predict[0:,i])))
        print("max_error: \t\t{:.4f}".format(skMet.max_error(y1_eval[0:,i], y1_predict[0:,i])))
        print("mean_absolute_error: \t{:.4f}".format(skMet.mean_absolute_error(y1_eval[0:,i], y1_predict[0:,i])))
        print("mean_squared_error: \t{:.4f}".format(skMet.mean_squared_error(y1_eval[0:,i], y1_predict[0:,i])))

        if (i-k) == len(LabelList):
            k = i
            tTracker += 1
            print("time is now: t+" + str(tTracker))

        print(LabelList[i-k])


    Nplots = 8
    fig, axs = plt.subplots(Nplots)
    plotXValues = xNonTrain[LabelList].values
    lenXplot = len(plotXValues) - y1_eval.shape[0]
    plotXValues = plotXValues[lenXplot:]
    for i in range(0,Nplots):
        plotY0 = np.array(plotXValues)
        plotY1 = np.array(y1_eval[:,i])
        plotY2 = np.array(y1_predict[:,i])
        plotX = np.linspace(0,100,num=len(plotY1))

        axs[i].plot(plotX,plotY0,'k-')
        axs[i].plot(plotX,plotY1,'b-')
        axs[i].plot(plotX,plotY2,'g-')

        axs[i].grid()


    plt.show()

    for i in range(0,Nplots):
        plotY0 = np.array(y1_predict[:,i])
        plotX = np.linspace(0,100,num=len(plotY1))
        plt.plot(plotX,plotY(0))

    plt.show()

    #data seems to be linear shifted this is just an intermideat hot fix
    y1_predict = hotfixLinearShift(x1_train,y1_train,y1_predict)


    Nplots = 8
    fig, axs = plt.subplots(Nplots)
    for i in range(0,Nplots):
        plotY1 = np.array(y1_eval[i,0:])
        plotY2 = np.array(y1_predict[i,0:])
        plotX = np.linspace(0,100,num=len(plotY1))
        print("Len Y1: {}".format(len(plotY1)))

        axs[i].plot(plotX,plotY1,'b-')
        axs[i].plot(plotX,plotY2,'g-')

        axs[i].grid()


    plt.show()

    return

    '''
    from PredictionModel import ltsm_model as model_

    model_.tf.keras.backend.clear_session()
    model_.tf.random.set_seed(4)

    m = model_.modelLtsm(x1_train.shape,y1_train.shape[1])
    #opt = tf.keras.optimizers.Adam(learning_rate=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,  name='Adam', **kwargs)
    opt = model_.tf.keras.optimizers.SGD(learning_rate=0.05, momentum=0.9, nesterov=True, name='SGD') #not sure if its a good idea to use nesterov momentum
    m.compile(opt, loss='mse')


    #creat a model callback
    ltsmSavePath = "./PredictionModel/checkpoints/"
    ltsmCallBack = ltsmSavePath + "ltsm.ckpt"
    #cp_callback = model_.tf.keras.callbacks.ModelCheckpoint(filepath=ltsmSavePath, save_weights_only=True, verbose=1)


    # Train
    m.fit(x=x1_train, y=y1_train, batch_size=batch_sizes, epochs=epochs, shuffle=True, validation_split=0.1)#, callbacks=[cp_callback] #shuffle=True, validation_split=0.1


    #[loss, acc] = m.evaluate(x1_eval, y1_eval)
    score = m.evaluate(x1_eval, y1_eval)
    y1_predict = m.predict(x1_eval)
    print("\n=========\nSCORE: \t{}%".format(score))
    #print("\n=========\nLOSS: \t{}\nACC: \t{}%".format(loss,acc*100))
    #print("\n=========\nX1 evaluation \n{}".format(x1_eval))


    print("\nShape: \tevaluate:{} \tpredicted:{}".format(y1_eval.shape,y1_predict.shape))

    k = 0
    tTracker = 1
    for i in range(0,y1_eval.shape[1]):
        m.showMetric(y1_eval[0:,i], y1_predict[0:,i])

        if (i-k) == len(LabelList):
            k = i
            tTracker += 1
            print("time is now: t+" + str(tTracker))

        print(LabelList[i-k])


    Nplots = 6
    fig, axs = plt.subplots(Nplots)
    plotXValues = xNonTrain[LabelList].values
    lenXplot = len(plotXValues) - y1_eval.shape[0]
    plotXValues = plotXValues[lenXplot:]
    for i in range(0,Nplots):
        plotY0 = np.array(plotXValues)
        plotY1 = np.array(y1_eval[:,i])
        plotY2 = np.array(y1_predict[:,i])
        plotX = np.linspace(0,100,num=len(plotY1))

        axs[i].plot(plotX,plotY0,'k-')
        axs[i].plot(plotX,plotY1,'b-')
        axs[i].plot(plotX,plotY2,'g-')

        axs[i].grid()


    plt.show()


    Nplots = 6
    fig, axs = plt.subplots(Nplots)
    for i in range(0,Nplots):
        plotY1 = np.array(y1_eval[i,0:])
        plotY2 = np.array(y1_predict[i,0:])
        plotX = np.linspace(0,100,num=len(plotY1))
        print("Len Y1: {}".format(len(plotY1)))

        axs[i].plot(plotX,plotY1,'b-')
        axs[i].plot(plotX,plotY2,'g-')

        axs[i].grid()


    plt.show()

    return

    y1_evalUpscaled = pp.upscaleMultiData(y1_eval,yNonTrain)
    y1_predictUpscaled = pp.upscaleMultiData(y1_predict,yNonTrain)


    y1_evalUpscaledNumer = y1_evalUpscaled.values
    y1_predictUpscaledNumer = y1_predictUpscaled.values

    Nplots = 4
    fig, axs = plt.subplots(Nplots)
    for i in range(0,Nplots):
        plotY1 = np.array(y1_evalUpscaledNumer[i,1:])
        plotY2 = np.array(y1_predictUpscaledNumer[i,1:])
        plotX = np.linspace(0,TicksIntoFuture,num=len(plotY1))

        axs[i].plot(plotX,plotY1)
        axs[i].plot(plotX,plotY2)
        axs[i].set_title("T+{}: with Ticks:{}".format(i,TicksIntoFuture))
        axs[i].grid()


    plt.show()


    # Display the model's architecture
    m.summary()

    #save the trained weights
    #ltsmSaveWeights = ltsmSavePath + "weights"
    #m.save_weights(ltsmSaveWeights)
    #ps to load the weigths
    #m.load_weights(ltsmSavePath)

    ltsmModelSave = ltsmSavePath + "ltsmModel"
    print("Save the entiered Model")
    m.save(ltsmModelSave)
    #To load saved model
    #new_model = tf.keras.models.load_model('saved_model/my_model')


    print("\n===========================================================")
    #print("No lets calculate the average max provit per\nDay \t{}\nHour \t{}\n15min \t{}".format())

    print("\n===========================================================")
    print("Upscale data")
    #print("\n++++++++[upscale x1_eval]+++++")
    #print("and add Date")
    #tCurren = x1_eval.shape[1] - 1
    #xUpscaled = pp.upscaleData(x1_eval[:,tCurren],xNonTrain)
    #print(xUpscaled)



    print("\n++++++++[test upscale y1_eval]+++++")
    yUpScaledReal = pp.upscaleData(y1_eval,yNonTrain)
    tempColum = ['date']
    tempColum.extend(yUpScaledReal.columns.tolist())

    print("\n=========")
    lenNonTrain = yNonTrain.shape[0]
    lenY1 = y1_eval.shape[0]
    dateColum = yNonTrain['date'][lenNonTrain-lenY1:]
    print("yNonTrain.shape: {}".format(yNonTrain.shape))
    print("y1_eval.shape: {}".format(y1_eval.shape))
    print("dateColum: {}".format(dateColum.shape))
    #add date column
    yUpScaledReal['date'] = dateColum.values
    print("\nreorder columns")
    yUpScaledReal = yUpScaledReal.reindex(columns=tempColum)
    print(yUpScaledReal)

    print("\n++++++++[test upscale y1_predict]+++++")
    yUpScaledPredict = pp.upscaleData(y1_predict,yNonTrain)
    print("\n=========")
    lenNonTrain = yNonTrain.shape[0]
    lenY1 = y1_predict.shape[0]
    dateColum = yNonTrain['date'][lenNonTrain-lenY1:]
    print("yNonTrain.shape: {}".format(yNonTrain.shape))
    print("y1_predict.shape: {}".format(y1_predict.shape))
    print("dateColum: {}".format(dateColum.shape))
    #add date column
    yUpScaledPredict['date'] = dateColum.values
    print("\nreorder columns")
    yUpScaledPredict = yUpScaledPredict.reindex(columns=tempColum)
    print(yUpScaledPredict)


    print("\n===========================================================")
    print("Printing the predicted values to a csv file")
    outData = pd.DataFrame(yUpScaledPredict)#,columns=LabelList
    outDataEval = pd.DataFrame(yUpScaledReal)#,columns=LabelList

    outData.to_csv(outputFile,index=False)
    outDataEval.to_csv(outputFileEval,index=False)
    '''

    #plt.addCandle(dataUpscaled,"Shifted")


    #plt.plotNow()

if __name__ == "__main__":
    main()
