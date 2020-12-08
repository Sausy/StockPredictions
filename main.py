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

def main():
    import matplotlib.pyplot as plt
    import sys

    if len(sys.argv) < 3:
        print("usage: ./programm <inputFile> <OutputFile>")

    inputFile = sys.argv[1]
    outputFile= sys.argv[2]

    print("Sys INPUTFILE \t<{}>\nOUTPUT FILE: \t<{}>".format(inputFile,outputFile))



    #===========================================================
    # PreProcessData
    # Basic Parameters
    #===========================================================
    TicksIntoFuture = 1
    TicksIntoPast = 4#54 #8days => 8[day]*24[std/day] = 192[std]
    ##the present is not included into this value hence TicksIntoPast can be 0
    #and the batch size is TicksIntoPast+1
    pp = prepro.preprocessing(ticksIntoPast=TicksIntoPast,ticksIntoFuture=TicksIntoFuture, debug=True)

    #===========================================================
    # Feture select
    # this section defines what features are
    # *) relevant
    # *) need scaling
    # *) need to be generated
    #===========================================================


    print("\n===========================================================")
    print("Data pulled")
    #In the Raw CSV File we should have the following features
    #Date,Symbol,Open,High,Low,Close,Volume BTC,Volume USD
    #with CsvFeatureList_Raw ... we are able to chose what feature to use
    CsvFeatureList_Raw = ["Date","Open","High","Low","Close"] #Should always include Date ....Todo:needs to be more sofisticated
    data = pp.pullData(path=inputFile, initRow=0, featureList=CsvFeatureList_Raw, maxTicks=0)
    plt = plt_modul.stockPlot(data)
    print(data)


    print("\n===========================================================")
    print("Add Features")
    print("Disclaimer: this requires at least the column \'Date\'")
    #TODO: add region depending holidays
    #... and figure out if i can just simply add binary classifier to a non
    # binary dataset ????
    #we also want to add a couple of features like
    #DaySin   #DayCos   #WeekSin     #WeekCos
    CsvFeatureList_additional = ["DaySin","DayCos","WeekSin","WeekCos","YearSin","YearCos"]
    data = pp.addTimeFeatures(data,CsvFeatureList_additional)
    print(data)



    print("\n===========================================================")
    print("Scale Data")
    #due to preprocessing, certain data needs scaling
    CsvFeatureList_needsScaling = ["Open","High","Low","Close"]
    #TODO: I'm not sure if that makes sense
    CsvFeatureList_needsScaling.extend(CsvFeatureList_additional)
    data = pp.scaleData(data, CsvFeatureList_needsScaling, method='normalize')
    print(data)

    print("\n===========================================================")
    print("Add Trading Features")
    #we also want to add a couple of trading features like
    #"ema"
    CsvFeatureList_trading = ["average","sma","ema","rsi","adx"]
    stock = preStock.indicators()
    data = stock.addStockFeatures(data,CsvFeatureList_trading)
    print(data)

    #==To plot the data proberly ... they also need to be rescaled
    dataUpscaled = pp.upscaleData(data)
    plt.addPlot(dataUpscaled,'High',"TradingFeatures")
    plt.addCandle(dataUpscaled,"TradingFeatures")




    print("\n===========================================================")
    print("Create Time Shifted data\n")
    #because oure model needs knowlege into the past
    #it needs to be defined what features shall be "time shifted"
    LabelList = ["Open","High","Low","Close"]
    [data,y] = pp.genForcastY(data, LabelList=LabelList, includeAllFuturDays=False)
    [data,y] = pp.genTimeSeries(data,y)

    print("\nY Data is \n")
    print(y)
    print("\nTime Shifted data is  \n")
    print(data)

    print("\nY Shape \t{}".format(y.shape))
    print("\ndata Shape\t{}".format(data.shape))
    #dataUpscaled = pp.upscaleData(data)
    #plt.addPlot(dataUpscaled,'High',"Shifted")
    #plt.addCandle(dataUpscaled,"Shifted")


    #===========================================================
    # Datashaping
    #===========================================================
    print("\n===========================================================")
    print("Data will be shaped acording to tensor flows [batch, time, features] ... windows")


    #===========================================================
    # model parameters
    #===========================================================
    splitVal = 0.99
    batch_sizes = 32
    epochs = 45


    plt.plotNow()

if __name__ == "__main__":
    main()
