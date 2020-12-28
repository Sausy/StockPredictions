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

def main():
    import matplotlib.pyplot as plt
    import sys

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
    TicksIntoFuture = 4
    TicksIntoPast = 54 #8days => 8[day]*24[std/day] = 192[std]
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
    print("\n++++++++[Down Scaled X]+++++")
    print(data)

    #dataUpscaled = pp.upscaleData(data)
    #plt.addPlot(dataUpscaled,'high',"Shifted")
    #plt.addCandle(dataUpscaled,"Shifted")

    lfoo = data.shape[0]
    fooOut = data[['date','open']][lfoo-246832:]#,columns=LabelList
    fooOut.to_csv(outputPath + "dbg_DownScaledX.csv",index=False)


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

    #for i in range(0,3):
    #    print(LabelList[i])

    #return

    [data,y] = pp.genForcastY(data, LabelList=LabelList) #, includeAllFuturDays=False

    '''
    lfoo = data.shape[0]
    fooOut = data[['date','open']][lfoo-246832:]#,columns=LabelList
    fooOut.to_csv(outputPath + "dbg_GenY_X.csv",index=False)
    lfoo = y.shape[0]
    fooOut = y[['date','open']][lfoo-246832:]#,columns=LabelList
    fooOut.to_csv(outputPath + "dbg_GenY_Y.csv",index=False)
    #return
    '''

    print("\n++++++++[y simple forcast ]+++++")
    print(y)

    print("\n===========================================================")
    print("remove non Trainable Columns in data\n")

    try:
        xNonTrain = copy.deepcopy(data)
        data = data.drop(columns=['date'])
        data = data.values

        print("\n++++++++[Resized X]+++++")
        print(data)
    except:
        print("[ERROR] X no label name date found")


    try:
        yNonTrain = copy.deepcopy(y)
        y = y.drop(columns=['date'])
        y = y.values

        print("\n++++++++[Down Scaled Y]+++++")
        print(y)

    except:
        print("[ERROR] Y no label name date found")



    #x = x[featureList][:y.shape[0]].values
    print("\n++++++++[xNonTrain]+++++")
    print(xNonTrain)
    print("\n++++++++[yNonTrain]+++++")
    print(yNonTrain)

    '''
    print("\n++++++++[test upscale X]+++++")
    xUpscaled = pp.upscaleData(data,xNonTrain)
    print(xUpscaled)

    print("\n++++++++[test upscale Y]+++++")
    yUpscaled = pp.upscaleData(y,yNonTrain)
    print(yUpscaled)
    '''

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
    batch_sizes = 4096#2048#1024

    epochs = 300#64

    dataSize = int(data.shape[0] * splitVal)
    x1_train = data[:dataSize]
    x1_eval = data[dataSize:]
    y1_train = y[:dataSize]
    y1_eval = y[dataSize:]


    #==========================================================================
    #=====[TENSORFLOW]===================================
    #==========================================================================
    from PredictionModel import ltsm_model as model_

    m = model_.modelLtsm(x1_train.shape,y1_train.shape[1])
    #opt = tf.keras.optimizers.Adam(learning_rate=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,  name='Adam', **kwargs)
    opt = model_.tf.keras.optimizers.SGD(learning_rate=0.05, momentum=0.9, nesterov=True, name='SGD') #not sure if its a good idea to use nesterov momentum
    m.compile(opt, loss='mse')
    m.fit(x=x1_train, y=y1_train, batch_size=batch_sizes, epochs=epochs, shuffle=True, validation_split=0.1) #shuffle=True, validation_split=0.1

    scores = m.evaluate(x1_eval, y1_eval)
    y1_predict = m.predict(x1_eval)
    print("\n=========\nScores \n{}".format(scores))
    #print("\n=========\nX1 evaluation \n{}".format(x1_eval))

    print("\nShape: \tevaluate:{} \tpredicted:{}".format(y1_eval.shape,y1_predict.shape))

    for i in range(0,y1_eval.shape[1]):
        m.showMetric(y1_eval[0:,i], y1_predict[0:,i])
        print(LabelList[i])


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


    #plt.addCandle(dataUpscaled,"Shifted")


    #plt.plotNow()

if __name__ == "__main__":
    main()
