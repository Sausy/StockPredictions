#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 14:54:54 2020

@author: sausy
"""

import numpy as np 
import pandas as pd

import plotly.graph_objects as go

import datetime

from getData import cryptocurrency
from StockIndicators import indicators

from sklearn import preprocessing

import tensorflow as tf 

class indicators:
    def __init__(self):
        print("10 Available Stock Indicators")
        
    def MovingAverage(self):
        print("MA")
        
    
    def calc_ema(values, time_period):
        # https://www.investopedia.com/ask/answers/122314/what-exponential-moving-average-ema-formula-and-how-ema-calculated.asp
        sma = np.mean(values[:, 3])
        ema_values = [sma]
        k = 2 / (1 + time_period)
        for i in range(len(his) - time_period, len(his)):
            close = his[i][3]
            ema_values.append(close * k + ema_values[-1] * (1 - k))
        return ema_values[-1]
    
    def EMA(self,x):
        out = []
        for his in x:
            # note since we are using his[3] we are taking the SMA of the closing price
            sma = np.mean(his[:, 3])
            macd = self.calc_ema(his, 12) - self.calc_ema(his, 26)
            out.append(np.array([sma]))
            out.append(np.array([sma,macd,]))
    
        out = np.array(out)
    
        outScaler = preprocessing.MinMaxScaler()
        outNorm = outScaler.fit_transform(out)
        
class preTools:
    def __init__(self):
        print("[preTools]")

    def normalize(self,x):
        return self.minmax(x,0,1)
        
    def minmax(self, x, minVal, maxVal):
        print("MinMax Scaling between [{},{}]".format(minVal,maxVal))
        if np.amin(x) == np.amax(x) :
            print("error min = max")
            return 0
        
        a = (x - np.amin(x)) * (maxVal - minVal)
        b = np.amax(x) - np.amin(x)
        return(minVal + a/b)
    
    def standardize(self, x):
        a = np.average(x)
        b = np.amax(x) - np.amin(x)
        return((x-a)/b)
    
    def splitData(self,x,split=0.9):
        #if AmountIndepSets >
        #split = AmountIndepSets
        n = int(x.shape[0] * split)
        print("x Shape {}".format(x.shape[0]))
        #eVec = np.ones((x.shape[0],1))
        #x = []
        #np.dot(x,eVec)
        #self.testDataNorm = np.array([self.dataNorm[i:i + AmountOfSets].copy() for i in range(len(self.dataNorm) - AmountOfSets)])
        return [x[:n],x[n:]]

    
class preprocessing:
    
    def __init__(self,ticksIntoPast=50,ticksIntoFuture=1, debug=False, scaling='minmax'):
        #self.x = x
        self.debug = debug
        self.dbgPrint("Init Preproccing")
        self.ticksIntoPast=ticksIntoPast
        self.ticksIntoFuture=ticksIntoFuture
                
    
        #self.bc = cryptocurrency()
        #self.bc.csv_read()
        
    def dbgPrint(self,outData):
        txt = "[PreProcessData]"
        if self.debug == True:
            txt = txt + "[DBG]"
            
        txt = txt + " "
        txt = txt + outData
        print(txt)
    
    def pullData(self,path='BTC-USD.csv',initRow=0,featureList=["Open","High","Close"],maxTicks=0):
        data = pd.read_csv(path,header=initRow) #names=featureList
        #data = data.drop(0, axis=0)
        
        if maxTicks > 0:
            data = data[:][0:maxTicks]
        databuffer = data["Date"]
        #s[0] = '2017-07-29 03-PM'
        #s[1] = '2017-07-30 01-PM'
        try:
            databuffer = pd.to_datetime(databuffer.values).to_series()
        except:
            try:
                databuffer = pd.to_datetime(databuffer.values,format='%Y-%m-%d %I-%p').to_series()    
            except:
                print("timestamp not known")
                

        
        data["Day"] = databuffer.dt.dayofweek.values #data["Date"]
        data["Hour"] = databuffer.dt.hour.values  #data["Date"]
        
        '''
        try:
            data = data.drop('Date', axis=1)
        except:
            print("no column with DATE")
        '''
        
        try:
            data = data.drop('Symbol', axis=1)
        except:
            print("no column with Symbol")
            
        #data["Hour"] = databuffer.dt.hour.values  #data["Date"]
        print(data)
        
        return data
        
        #print(data.values)
        #data = data.values
        #print(data)
    
    def genTimeSeries(self,x,y):
        #====== Get Time Shifted DATA ==== 
        print("GenTimeSeries")
        #print(x)
        
        x = np.array([x[i:i + self.ticksIntoPast+1].copy() for i in range(len(x) - self.ticksIntoPast)])
        
        print("\n\nTimeBatches of\n==============")
        print("X shape: {}".format(x.shape))
        print("Y shape: {}\n".format(y.shape))
        print(x)
        
        resizeFactor = x.shape[0] - (x.shape[0]-y.shape[0])
        x = x[:][0:resizeFactor] 
        
        print("\n\nreshaped x\n==============")
        print(x.shape)
        
        #resizeFactor = x.shape[0] - (x.shape[0]-y.shape[0])
        print("\n\nTicks into past y\n==============")
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
    
    def genForcastY(self,x,LabelList=["Open","High","Close"],featureList=["Open","High","Close"],includeAllFuturDays=False):
        #====== Get Time Shifted DATA ==== 
        print("\n\ngenerate timeshifted Y outputs\n==============")
        #print(len(LabelList))
        deltaT = self.ticksIntoPast + self.ticksIntoFuture
        
        if self.ticksIntoFuture < 1:
            print("ERROR ticksIntoFuture must be greater than 0")
            self.ticksIntoFuture = 1
            
        yLength = x.shape[0] - 1
        print("yLength..{}".format(yLength))
        
        print("\n\nY init with X data but t+1 shift\n==============")
        y = x[LabelList][1:].values
        #print("\n\nY one tick ahead\n==============")
        #print(y)
        
        if self.ticksIntoFuture > 1:
            for i in range(self.ticksIntoFuture-1): # -1 because y_init has already one shift                
                yLength = yLength - 1
                y = np.append(y[:yLength],y[1:(yLength+1)],axis=1)
                
        #print("\n\nY loop\n==============")
        #print(y)

                #y = y[:yLength]
        
        print("Optional Label Reduction to only forcast event {}Ticks into the future".format(self.ticksIntoFuture))
        if includeAllFuturDays == False:
            y = y[:,y.shape[1]-len(LabelList):]
                
                
        print("resulting Y ...")
        print(y)
        
        print("reduction of X...")
        #print(x.shape)
        #print(y.shape)
        #print(x[featureList][0:5])
        #print(yLength)
        x = x[featureList][:y.shape[0]].values
        print("Size of x {}".format(x.shape))
        print(x)
        
        
        return [x,y]
    
    
    def scaleData(self,x,method='normalize'):
        #normalize/standardize
        print(x.columns)
        col = x.columns 
        data = x.to_numpy()
        print("\n\nscaleData\n==============")
        
        tool = preTools()
        #print(data.shape)
        #print(data[:,0])
        #print(data[:,6])
        
        if method == 'normalize':
            data = tool.normalize(x.values)
        else:
            data = tool.standardize(x.values)
        #for i in range(data.shape[1]):
        #    data[:,i] = tool.normalize(data[:,i])
        
        print("\n\nscaleData done\n==============")
        print(data)
        x = pd.DataFrame(data,columns=col)
        print(x)
        return x
        
    
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
    
    def createTimeLabel(self,x):
        print("create Time label for data")
        print("basicly we read out the Date colum and replace/generate additional columns")
        print("this allows us to identify weeday/holidays etc.")
        
        print("testing out Labeling")
        print("*)daytime")
        print("*)current day weekend")
        print("*)was past day weekend")
        print("*)will the next one be weekend")
        
        
    def findBuySellPoint(self,x):
        print("create a model that searches for the best buy and sell points in a given modell")
        print("import add the transaction fee and a time windows of 3min till an transaction is placed")
        print("also consider amount of transactions per day and month")
        
    
class model1(tf.keras.Model):
    def __init__(self,inputShape, outputShape):
        tf.keras.backend.clear_session()
        tf.random.set_seed(4)
        
        self.N = inputShape[0]
        self.AmountParallelSeries = inputShape[1]
        self.AmountFeatures = inputShape[2] #outputShape #inputShape[2]
        self.outputSize = outputShape
        
        print("\ninit Model 1")
        print(inputShape)
        print(outputShape)
        
        
        self.UnitCountHiddenLayer1 = int(self.AmountParallelSeries*1.6) #int(self.AmountFeatures*1.6)
        
        print("UnitCountHiddenLayer1: {}\n==============".format(self.UnitCountHiddenLayer1))
        
        super(model1, self).__init__()
        
        
        #inputs: A 3D tensor with shape [batch, timesteps, feature]  .... for LSTM
        self.lin = tf.keras.layers.InputLayer(input_shape=inputShape)
        self.l1 = tf.keras.layers.LSTM(inputShape[1], activation='tanh', return_sequences=True) #input_shape=(self.AmountParallelSeries, self.AmountFeatures)
        self.dropout = tf.keras.layers.Dropout(0.2)
        #Conv2D(filters, kernelsize, ...)
        #self.conv = tf.keras.layers.Conv2D(10, self.AmountFeatures, activation='tanh', input_shape=input_shape[1:])
        self.bn = tf.keras.layers.BatchNormalization()
        
        self.l2 = tf.keras.layers.LSTM(inputShape[1], activation='tanh')
        self.dropout2 = tf.keras.layers.Dropout(0.2)
        
        
        
        #self.l3 = tf.keras.layers.Dense(self.AmountFeatures+5, activation='relu')
        self.l3 = tf.keras.layers.Dense(self.UnitCountHiddenLayer1, activation='relu')
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
        #if training:
        #    x = self.dropout(x, training=training)
        #x = self.bn(x, training=training)
        x = self.l2(x)
        
        #if training:
        #    x = self.dropout2(x, training=training)
        #x = self.bn(x)
        x = self.l3(x)
        
        #if training:
        #    x = self.dropout3(x, training=training)
        #x = self.l4(x)
        #x = self.l4(x)
        
        return self.lout(x)

    

def main():     
    import matplotlib.pyplot as plt 

    splitVal = 0.99
    TicksIntoFuture = 5
    TicksIntoPast = 54 #8days => 8[day]*24[std/day] = 192[std]
    ##the present is not included into this value hence TicksIntoPast can be 0
    #and the batch size is TicksIntoPast+1
    
    batch_sizes = 128
    epochs = 45
    
    
    print("Data will be shaped acording to tensor flows [batch, time, features] ... windows")
    
    featureListRaw = ["Date","Open","High","Low","Close"]
    #labelListTimeShifted = ["High","Low","Close"]
    labelListTimeShifted = ["Open","High","Low","Close"]
    #labelListTimeShifted = ["Open","High","Low","Close"]
    
    featListUnScale = ["Date"]
    featListScale = ["Open","High","Low","Close"]
    featureList = ["DaySin","DayCos","Open","High","Low","Close"]
    #featList.append("Day")
    #featList.append("Hour")
    
    pp = preprocessing(ticksIntoPast=TicksIntoPast,ticksIntoFuture=TicksIntoFuture)
    data = pp.pullData('BTC-h.csv',0,featureListRaw,0)
    #data = pp.pullData('dbg.csv',0,featureListRaw,0)
    
    print("\n\nShrink Data \n==============")
    UnscaleData = data[featListUnScale]
    data = data[featListScale]
    print(data)
    
        
    data = pp.scaleData(data,'standardize')
    
    
    print("\n\nTODO: ADD Additional Features that are not scaled \n==============")
    data[featListUnScale] = UnscaleData
    print(data)
    
    print("\n\nAdding Date\n==============")
    databuffer = data["Date"]
        #s[0] = '2017-07-29 03-PM'
        #s[1] = '2017-07-30 01-PM'
    try:
        databuffer = pd.to_datetime(databuffer.values).to_series()
    except:
        try:
            databuffer = pd.to_datetime(databuffer.values,format='%Y-%m-%d %I-%p').to_series()    
        except:
            print("timestamp not known")
            

    
    data["Day"] = databuffer.dt.dayofweek.values #data["Date"]
    data["Hour"] = databuffer.dt.hour.values 
    
    
    
    
    print("==============")
    
    
    #timestamp_s = date_time.map(datetime.datetime.timestamp)
    print("timestamp_s defines how many ticks per day\nBTC-h.csv has 1h tick\n==============")
    #timestamp_s = np.linspace(0, 1, num=24)
    #date_time = pd.to_datetime(df.pop('Date Time'), format='%d.%m.%Y %H:%M:%S')
    
    databuffer = data["Date"].values
    try:
        databuffer = pd.to_datetime(databuffer)
    except:
        try:
            databuffer = pd.to_datetime(databuffer,format='%Y-%m-%d %I-%p')  
        except:
            print("timestamp not known")
        
    databuffer = pd.to_datetime(databuffer, format='%d.%m.%Y %H:%M:%S')
            
    print("\n\nDataBuffer\n==============")
    print(databuffer)
    
    timestamp_s = databuffer.map(datetime.datetime.timestamp)
    print(timestamp_s)
    
    day = 24*60*60
    week = 7*day
    year = (365.2425)*day
    
    data['DaySin'] = np.sin(timestamp_s * (2 * np.pi / day))
    data['DayCos'] = np.cos(timestamp_s * (2 * np.pi / day))
    
    data['WeekSin'] = np.sin(timestamp_s * (2 * np.pi / week))
    data['WeekCos'] = np.cos(timestamp_s * (2 * np.pi / week))
    

    
    print("==============")
    print(data['WeekSin'][0:25])
    #print(data['Date'][0:25])
    print("==============\n")
    
    #print(data)
    
    
    
    
    [data,y] = pp.genForcastY(data, LabelList=labelListTimeShifted, featureList=featureList, includeAllFuturDays=False)
    [data,y] = pp.genTimeSeries(data,y)
    
    print(y)
    
    print("\n\nAmount of ouputs \n==============")
    N_outPutFeatures = y.shape[1]
    y = np.expand_dims(y, -1)
    print(y.shape)
    
    
    dataSize = int(data.shape[0] * splitVal) 
    x1_train = data[:dataSize]
    x1_eval = data[dataSize:]
    
    y1_train = y[:dataSize]
    y1_eval = y[dataSize:]
    
    print(data.shape)
    print(y.shape)
    
    print(x1_train.shape)
    print(y1_train.shape)
    print(x1_eval.shape)
    print(y1_eval.shape)
    
    print("\n\nHook up Models \n==============")
    
    m1 = model1(x1_train.shape,y1_train.shape[1])
    #m1 = m.call(x1_train,y1_train) #batch_sizes,epochs
    
    m1.compile(optimizer='adam', loss='mse')
    m1.fit(x=x1_train, y=y1_train, batch_size=batch_sizes, epochs=epochs, shuffle=True, validation_split=0.1) #shuffle=True, validation_split=0.1
    
    scores = m1.evaluate(x1_eval, y1_eval)
    print(scores)

    
    
    print("\n\nPlot it\n==============")
    fig, axs = plt.subplots(N_outPutFeatures,3, figsize=(25,14))
    
    y1_predict = m1.predict(x1_eval)
    y1_predict2 = m1.predict(x1_train)
    
    plotX = np.linspace(0, 10, y1_eval.shape[0])
    plotX2 = np.linspace(0, 10, y1_train.shape[0])
    
    print("\n\ndebg \n==============")
    print(y1_eval.shape)
    print(y1_predict.shape)
    print(y1_train.shape)
    print(y1_predict2.shape)
    
    y1_eval[:,]
    Variance = np.array([[]])
    
    for n in range(N_outPutFeatures):
        yDiv = y1_eval[:,n,0] - y1_predict[:,n]
        yDiv = np.abs(yDiv) 
        #print("shape yDiv")
        #print(yDiv.shape)
        #ydum = np.where(y1_eval[:,n,0] == 0, 10000, y1_eval[:,n,0])
        
        #ydum = y1_eval[:,n,0] + 1
        #ydum2 = yDiv + 1
        #ERROR_ = (100/ydum) * ydum2
        ERROR_ = yDiv
        #print("shape ERROR_")
        #print(ERROR_.shape)
        yDiv = np.sum(yDiv)/len(y1_predict[:,n])
        
        Variance = np.append(Variance, np.array([yDiv,yDiv]))
        
        axs[n,0].plot(plotX,y1_eval[:,n], 'k')
        axs[n,0].plot(plotX,y1_predict[:,n], 'g')
        
        axs[n,1].plot(plotX,ERROR_, 'r')
        
        axs[n,2].plot(plotX2,y1_train[:,n], 'k')
        axs[n,2].plot(plotX2,y1_predict2[:,n], 'g')
        
    
    plt.show()
    
    fig2 = go.Figure(data=[go.Candlestick(x=plotX,
                open=y1_eval[:,0],
                high=y1_eval[:,1],
                low=y1_eval[:,2],
                close=y1_eval[:,3])])

    fig2.show()
    
    fig3 = go.Figure(data=[go.Candlestick(x=plotX,
                open=y1_predict[:,0],
                high=y1_predict[:,1],
                low=y1_predict[:,2],
                close=y1_predict[:,3])])

    fig3.show()
    
    
    print("\n\ndebg \n==============")
    print("Variance list {}".format(Variance))
    
    
    
    
    '''
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
    
    '''
    
    
    
    

if __name__ == "__main__":
    main()