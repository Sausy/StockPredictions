#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 15:09:11 2020

@author: sausy
"""
import numpy as np
import copy
import datetime

import pandas as pd

#import ../CSV/invert

if __name__ == "__main__":
    from common import preTools
else:
    from .common import preTools

class preprocessing:
    def __init__(self,ticksIntoPast=50,ticksIntoFuture=1, debug=False, scaling='minmax'):
        #self.x = x
        self.debug = debug
        self.dbgPrint("Init Preproccing")
        self.ticksIntoPast=ticksIntoPast
        self.ticksIntoFuture=ticksIntoFuture
        #self.bc = cryptocurrency()
        #self.bc.csv_read()

        self.method=scaling

        self.rescale_K = {}
        self.rescale_D = {}

    #===========================================================
    def dbgPrint(self,outData):
        txt = "[PreProcessData]"
        if self.debug == True:
            txt = txt + "[DBG]"

        txt = txt + " "
        print(txt,end='')
        print(outData)

    #===========================================================
    def pullData(self,path='in.csv',initRow=0,featureList=["Open","High","Close"],maxTicks=0):
        #read the raw CSV file and drops all unwanted features

        data = pd.read_csv(path,header=initRow) #names=featureList

        # removing null values to avoid errors
        data.dropna(inplace = True)
        data = data.reset_index(drop=True)

        #to remove the case sensitvity
        featS = pd.Series(featureList)
        originS = pd.Series(data.columns)

        featListLower = featS.str.lower()
        originListLower = originS.str.lower()
        #print(featListLower.values)

        if maxTicks > 0:
            data = data[:][0:maxTicks]

        #find columns that are not part of the feature List
        #and remove them
        for col in originListLower.values:#data.columns:
            notfound = True
            for feat in featListLower.values: #featureList:
                if feat == col:
                    notfound = False
                    break
            if notfound:
                print("\n")
                self.dbgPrint("Removing Column: [{}]".format(col))
                data = data.drop(col,axis=1)

        self.dbgPrint("print out Data")
        self.dbgPrint(data)

        return data



    #===========================================================
    def addTimeFeatures(self,x,featureList):
        #takes the given data:x  and addes TimeLabel Features
        #Disclaimer: THIS requires a column with the name Date
        # Available Features
        #Day   #Hour
        #DaySin   #DayCos   #WeekSin     #WeekCos
        #YearSin   #YearCos

        try:
            databuffer = x["Date"].values
        except:
            self.dbgPrint("There is no Date Colum")
            self.dbgPrint("!!!! The function addTimeFeatures")
            self.dbgPrint("requires a Column called \'Date\'")
            return 0

        #figure out if the format is already known
        try:
            databuffer = pd.to_datetime(databuffer,format='%d-%m-%Y %H:%M:%S')
        except:

            try:
                databuffer = pd.to_datetime(databuffer)
            except:
                try:
                    databuffer = pd.to_datetime(databuffer,format='%Y-%m-%d %I-%p')
                except:
                    self.dbgPrint("timestamp not known")
                    return 0




        #self.dbgPrint("Databuffer Type {}".format(type(featureList)))
        self.dbgPrint("len of featureList {}".format(len(featureList)))
        day = 24*60*60
        week = 7*day
        year = (365.2425)*day

        timestamp_s = databuffer.map(datetime.datetime.timestamp)
        timeSer = databuffer.to_series()

        #add features that are listed in featureList
        for feat in featureList:
            if feat == "Day":
                x["Day"] = timeSer.dt.dayofweek.values #data["Date"]
            elif feat == "Hour":
                x["Hour"] = timeSer.dt.hour.values  #data["Date"]
            elif feat == "DaySin":
                x['DaySin'] = np.sin(timestamp_s * (2 * np.pi / day))
            elif feat == "DayCos":
                x['DayCos'] = np.cos(timestamp_s * (2 * np.pi / day))
            elif feat == "WeekSin":
                x['WeekSin'] = np.sin(timestamp_s * (2 * np.pi / week))
            elif feat == "WeekCos":
                x['WeekCos'] = np.cos(timestamp_s * (2 * np.pi / week))
            elif feat == "YearSin":
                x['YearSin'] = np.sin(timestamp_s * (2 * np.pi / year))
            elif feat == "YearCos":
                x['YearCos'] = np.cos(timestamp_s * (2 * np.pi / year))

        return x



    #===========================================================
    def genTimeSeries(self,x,y):
        #====== Get Time Shifted DATA ====
        print("GenTimeSeries")
        #print(x)

        x = np.array([x[i:i + self.ticksIntoPast+1].copy() for i in range(len(x) - self.ticksIntoPast)])

        print("\n\nTimeBatches of\n==============")
        print("X shape: {}".format(x.shape))
        print("Y shape: {}\n".format(y.shape))
        #print(x)

        resizeFactor = x.shape[0] - (x.shape[0]-y.shape[0])
        x = x[:][0:resizeFactor]

        print("\n\nreshaped x\n==============")
        print(x.shape)

        #resizeFactor = x.shape[0] - (x.shape[0]-y.shape[0])
        print("\n\nTicks into past\n==============")
        print(self.ticksIntoPast)

        print("\n\nreshaped y\n==============")
        print("this is necesary because the new x at t=0 is self.ticksIntoPast shiftet")
        y = y[self.ticksIntoPast:]
        print(y.shape)


        #y = tool.timeShiftData(x[featureList],ticksIntoPast=self.ticksIntoPast,ticksIntoFuture=self.ticksIntoFuture)

        #  Features that need to be predicted in "ticks = currentTime + i"
        #  Features that are just there to give an additional Input
        #dataY = data[featureList]
        #x = x[]
        return [x,y]


    #===========================================================
    def genForcastY(self,x,LabelList=["Open","High","Close"],featureList=["Open","High","Close"],includeAllFuturDays=False):
        #get rid of case sensitvity
        labelS = pd.Series(LabelList)
        labelListLower = labelS.str.lower()
        LabelList = labelListLower.values

        #get rid of case sensitvity
        featureS = pd.Series(LabelList)
        featureListLower = featureS.str.lower()
        featureList = featureListLower.values

        #====== Get Time Shifted DATA ====
        print("\n==============\ngenerate timeshifted Y outputs\n")
        #print(len(LabelList))
        deltaT = self.ticksIntoPast + self.ticksIntoFuture

        if self.ticksIntoFuture < 1:
            print("ERROR ticksIntoFuture must be greater than 0")
            self.ticksIntoFuture = 1

        yLength = x.shape[0] - 1
        print("yLength..{}".format(yLength))

        print("\n==============\nY init with X data but t+1 shift\n")
        y = x[LabelList][1:].values
        #print("\n==============\nY one tick ahead\n")
        #print(y)

        if self.ticksIntoFuture > 1:
            for i in range(self.ticksIntoFuture-1): # -1 because y_init has already one shift
                yLength = yLength - 1
                y = np.append(y[:yLength],y[1:(yLength+1)],axis=1)

        #print("\n==============\nY loop\n")
        #print(y)

                #y = y[:yLength]

        print("\n==============\nOptional:\nInclude all t+n events as additional label")
        print("or only [t+{}] feature".format(self.ticksIntoFuture))
        print("[OptionalFeature]: {}".format(includeAllFuturDays))

        if includeAllFuturDays == False:
            y = y[:,y.shape[1]-len(LabelList):]


        print("\n==============\nResulting Y based on time shifted X\n")
        print(y)

        #Reduction of X is necesary, because
        #Y(t) = X(t+ticksIntoFuture)
        print("\n\nreduction of X...\nbecause Y(t) = X(t+ticksIntoFuture)\n")
        x = x[featureList][:y.shape[0]].values
        print("Size of x {}".format(x.shape))
        #print(x)


        return [x,y]

    #===========================================================
    def scaleData(self,x,featList, method='normalize'):
        self.method = method

        #normalize/standardize
        print(x.columns)
        tool = preTools()

        self.rescale_K = {}
        self.rescale_D = {}

        #to remove the case sensitvity
        featS = pd.Series(featList)
        featListLower = featS.str.lower()

        self.upScaleList = featListLower.values

        for feat in featListLower.values:
            print("\nScale feature: \t{}".format(feat))

            num_data = x[feat].to_numpy()

            if method == 'normalize':
                [x[feat],self.rescale_K[feat],self.rescale_D[feat]] = tool.normalize(num_data)
            else:
                [x[feat],self.rescale_K[feat],self.rescale_D[feat]]  = tool.standardize(num_data)


        '''
        print(data)
        print("\n\nscaleData\n==============")


        #print(data.shape)
        #print(data[:,0])
        #print(data[:,6])


        #for i in range(data.shape[1]):
        #    data[:,i] = tool.normalize(data[:,i])

        print("\n\nscaleData done\n==============")
        print(data)
        x = pd.DataFrame(data,columns=col)
        print(x)
        '''
        return x

    #===========================================================
    def upscaleData(self,x):
        xBuffer = copy.deepcopy(x)
        ret = copy.deepcopy(x)

        for feat in self.upScaleList:
            if self.method == 'normalize':
                ret[feat] = xBuffer[feat] * self.rescale_K[feat] + self.rescale_D[feat]
            else:
                print("==========\nTODO: till now only normalize upscale Available")
                pass

        return ret


    #===========================================================
    def setUP(self, splitVal = 0.9, HistoryExpan = 50, PredictDayIntoFuture = 1):
        print("Transfer Data with len:{}".format(len(self.bc.HistorData)))

        print(self.normalize(self.bc.HistorData))
        print(self.standardize(self.bc.HistorData))

        xNorm = self.normalize(self.bc.HistorData)
        print("Size of Data {}".format(xNorm.shape))
        #xScaled = pp.standardize(bc.HistorData)

        #====== LABEL DATA ======
        #y = xNorm[:,0][Y_TakenAfter_x_Days:]
        deltaT = HistoryExpan + (PredictDayIntoFuture-1);

        ##y = xNorm[:,0][deltaT:]
        y = xNorm[deltaT:]

        y = np.expand_dims(y, -1)
        print("y size {}|{}".format(y.shape,y))


        #====== EXPAND FEATURES um History Dimension =====
        #We basicly transform a Set that consisted of
        #input = [x1, x2, x3 .... xk] ... k Feautures to
        #input = [[x11...x1h], [x21...] ... [...xkh]] ... To an 2D input with k*h features
        data = np.array([xNorm[i:i + HistoryExpan].copy() for i in range(len(xNorm) - HistoryExpan)])

        #data = np.array([xNorm[:, 0][i + HistoryExpan].copy() for i in range(len(xNorm) - HistoryExpan)])
        #data = np.expand_dims(self.evalDataNorm, -1)

        #====== Resize DATA ======
        resizeFactor = data.shape[0] - (data.shape[0]-y.shape[0])
        x = data[:][0:resizeFactor]
        print("Size of data {}|{}".format(data.shape,data))
        print("Size of xnew {}|{}".format(x.shape,x))

        #====== Split it ======
        [Xtrain,Xeval] = self.splitData(x,splitVal)
        [Ytrain,Yeval] = self.splitData(y,splitVal)

        return [Xtrain,Xeval,Ytrain,Yeval]

        #[self.Xtrain,self.Xeval] = self.splitData(x,3)


    #===========================================================
    def createTimeLabel(self,x):
        print("create Time label for data")
        print("basicly we read out the Date colum and replace/generate additional columns")
        print("this allows us to identify weeday/holidays etc.")

        print("testing out Labeling")
        print("*)daytime")
        print("*)current day weekend")
        print("*)was past day weekend")
        print("*)will the next one be weekend")


    #===========================================================
    def findBuySellPoint(self,x):
        print("create a model that searches for the best buy and sell points in a given modell")
        print("import add the transaction fee and a time windows of 3min till an transaction is placed")
        print("also consider amount of transactions per day and month")


def main():
    print('=============\nSandbox: preProcessRawData.py')

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
    TicksIntoPast = 4 #8days => 8[day]*24[std/day] = 192[std]
    ##the present is not included into this value hence TicksIntoPast can be 0
    #and the batch size is TicksIntoPast+1
    pp = preprocessing(ticksIntoPast=TicksIntoPast,ticksIntoFuture=TicksIntoFuture, debug=True)

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
    print(data)


    print("\n===========================================================")
    print("Add Features")
    print("Disclaimer: this requires at least the column \'Date\'")
    #we also want to add a couple of features like
    #DaySin   #DayCos   #WeekSin     #WeekCos
    CsvFeatureList_additional = ["DaySin","DayCos","WeekSin","WeekCos","YearSin","YearCos"]
    data = pp.addTimeFeatures(data,CsvFeatureList_additional)
    print(data)

    print("\n===========================================================")
    print("Add Traiding Features")
    #we also want to add a couple of trading features like
    #"ema"
    CsvFeatureList_traiding = ["ema"]

    #because oure model needs knowlege into the past
    #it needs to be defined what features shall be "time shifted"
    CsvFeatureList_timeShift = ["Open","High","Low","Close"]
    #[data,y] = pp.genForcastY(data, LabelList=labelListTimeShifted, featureList=featureList, includeAllFuturDays=False)

    #due to preprocessing, certain data needs scaling
    CsvFeatureList_needsScaling = ["Open","High","Low","Close"]
    CsvFeatureList_needsScaling.append(CsvFeatureList_traiding)




    print("\n===========================================================")
    print("Create Time Shifted data\n")
    #because oure model needs knowlege into the past
    #it needs to be defined what features shall be "time shifted"
    LabelList = ["Open","High","Low","Close"]
    [data,y] = pp.genForcastY(data, LabelList=LabelList, includeAllFuturDays=False)
    print(data)
    [data,y] = pp.genTimeSeries(data,y)

    print (data)
    print (y)


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



if __name__ == "__main__":
    main()
