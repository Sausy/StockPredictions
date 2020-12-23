#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dez  8 10:53:41 2020

@author: sausy
"""

import numpy as np
import pandas as pd

import plotly.graph_objects as go

#from getData import cryptocurrency
#from StockIndicators import indicators
#from sklearn import preprocessing
#import tensorflow as tf

from preprocessing import preProcessRawData as prepro
from preprocessing import StockIndicators as preStock
from plot2D import stockPlot as plt_modul
from PredictionModel import ltsm_model as model_

def main():
    import matplotlib.pyplot as plt
    import sys

    if len(sys.argv) < 2:
        print("usage: ./programm <inputFile>")
        print("will proceed with ./CSV/rdyCSV/rdy.csv")
        inputFile = "./CSV/rdyCSV/rdy.csv"
        outputFile= "unneed.csv"
    else:
        inputFile = sys.argv[1]
        outputFile= "unneed.csv"

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
    plt = plt_modul.stockPlot(data)


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
    print(data)

    dataUpscaled = pp.upscaleData(data)
    #plt.addPlot(dataUpscaled,'high',"Shifted")
    #plt.addCandle(dataUpscaled,"Shifted")

    print("\n===========================================================")
    print("Create Y\n")
    print("The output equals X(t+n) so x as seen t+n times in the future")
    #because oure model needs knowlege into the past
    #it needs to be defined what features shall be "time shifted"
    LabelList = ["Open",
                "High",
                "Low",
                "Close",
                "sma10",
                'ema10']
    [data,y] = pp.genForcastY(data, LabelList=LabelList, includeAllFuturDays=False)

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
    batch_sizes = 2048#1024
    epochs = 64

    dataSize = int(data.shape[0] * splitVal)
    x1_train = data[:dataSize]
    x1_eval = data[dataSize:]
    y1_train = y[:dataSize]
    y1_eval = y[dataSize:]


    m = model_.modelLtsm(x1_train.shape,y1_train.shape[1])
    #opt = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,  name='Adam', **kwargs)
    opt = model_.tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9, nesterov=True, name='SGD') #not sure if its a good idea to use nesterov momentum
    m.compile(opt, loss='mse')
    m.fit(x=x1_train, y=y1_train, batch_size=batch_sizes, epochs=epochs, shuffle=True, validation_split=0.1) #shuffle=True, validation_split=0.1

    scores = m.evaluate(x1_eval, y1_eval)
    y1_predict = m.predict(x1_eval)
    print("\n=========\nScores \n{}".format(scores))
    #print("\n=========\nX1 evaluation \n{}".format(x1_eval))

    print("\nShape: \tevaluate:{} \tpredicted:{}".format(y1_eval.shape,y1_predict.shape))

    for i in range(0,y1_eval.shape[1]):
        m.showMetric(y1_eval[0:,i], y1_predict[0:,i])


    print("\n===========================================================")
    print("No lets calculate the average max provit per\nDay \t{}\nHour \t{}\n15min \t{}".format())


    #plt.addCandle(dataUpscaled,"Shifted")


    #plt.plotNow()

if __name__ == "__main__":
    main()
