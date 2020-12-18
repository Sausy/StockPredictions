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
#from plot2D import stockPlot as plt_modul

class gdriverScrapper:
    def __init__(self):
        foo = 0
        print("Download Raw Data: ")
        print("Download incremental Data:")
        print("merge two Both CSV files to ")
        print("File1: \t{}".format(foo))
        print("File2: \t{}".format(foo))
        print("Fileout: \t{}".format(foo))




def main():
    import matplotlib.pyplot as plt
    import sys

    '''
    Big todo:
    the main of kaken.py needs to be included here

    also it should only take the following arguments as input

    clean   ....    should create a clean new csv file from the kraken raw data
                    this will create CSV-Files for clean CSV
    download=[]...  download raw data from gDrive (kraken) and create clean CSV files
                    requires attributes ... if kept empty all posible coins will be computed
                    eg download=["XBT"] ... Still downloads all Zipp folders but only uses
                    !!EURO and XBT will always be the base Value
                    XBT-TO-EURO
    <none> ...      takes the all the csv files in ./CSV/rdyToTrainCsv and processes it
    '''
    print("!!!!!! READ THE CoMMent \nkraken.py needs to be included here")


    if len(sys.argv) < 3:
        print("usage: ./programm <inputFile> <OutputFile>")

    inputFile = sys.argv[1]
    outputFile= sys.argv[2]

    print("Sys INPUTFILE \t<{}>\nOUTPUT FILE: \t<{}>".format(inputFile,outputFile))



    #===========================================================
    # PreProcessData
    # Basic Parameters
    pp = prepro.preprocessing(debug=True)

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
    #plt = plt_modul.stockPlot(data)
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
    print("Add Trading Features")
    #we also want to add a couple of trading features like
    #"ema"
    CsvFeatureList_trading = ["average","sma","ema","rsi","adx"]
    stock = preStock.indicators()
    data = stock.addStockFeatures(data,CsvFeatureList_trading)
    print(data)

    #==To plot the data proberly ... they also need to be rescaled
    #dataUpscaled = pp.upscaleData(data)
    #plt.addPlot(dataUpscaled,'High',"TradingFeatures")
    #plt.addCandle(dataUpscaled,"TradingFeatures")


    print("\n===========================================================")
    print("remove white spaces and convert to lower case")
    # Column names: remove white spaces and convert to lower case
    data.columns = data.columns.str.strip().str.lower()

    print("\n===========================================================")
    print("Save to CSV file")
    data.to_csv(outputFile,index=False)




if __name__ == "__main__":
    main()
