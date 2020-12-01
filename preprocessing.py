#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 16:23:01 2020

@author: sausy
"""
import numpy as np 
from getData import cryptocurrency
from StockIndicators import indicators

class preprocessing:
    
    def __init__(self, debug=False, scaling='minmax'):
        #self.x = x
        self.debug = debug
        self.dbgPrint("Init Preproccing")
    
        self.bc = cryptocurrency()
        self.bc.csv_read()
        
    
    
    def dbgPrint(self,outData):
        txt = "[PreProcessData]"
        if self.debug == True:
            txt = txt + "[DBG]"
            
        txt = txt + " "
        txt = txt + outData
        print(txt)
    
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
        print("\n\nY one tick ahead\n==============")
        print(deltaT)
        print(y)
        ##ymax = xNorm[:,1][deltaT:]
        #ymax = np.expand_dims(ymax, -1)
        ##ymin = xNorm[:,2][deltaT:]
        #ymin = np.expand_dims(ymin, -1)
        
        #y = (ymax+ymin)/2
        #print("x {}".format(xNorm[0:4]))
        #print("x {}|{}".format(xNorm[49:51],y[0:4]))
        #print("x {}|{}".format(xNorm,y))
        #y = self.normalize(y)
        #next day average
        
        #y = xNorm[:,1][Y_TakenAfter_x_Days:]
        #y = xNorm[:,3][Y_TakenAfter_x_Days:]
        #print("y size {}".format(y.shape))
        y = np.expand_dims(y, -1)
        #print("Size of Data {}".format(y.shape))
        print("y size {}|{}".format(y.shape,y))
        
        #y_alter = np.array([xNorm[:, 0][i + HistoryExpan].copy() for i in range(len(xNorm) - HistoryExpan)])
        #y_alter = np.expand_dims(y_alter, -1)
        #print("y size {}|{}".format(y_alter.shape,y_alter))
        
        
        #====== EXPAND FEATURES um History Dimension ===== 
        #We basicly transform a Set that consisted of 
        #input = [x1, x2, x3 .... xk] ... k Feautures to 
        #input = [[x11...x1h], [x21...] ... [...xkh]] ... To an 2D input with k*h features
        data = np.array([xNorm[i:i + HistoryExpan].copy() for i in range(len(xNorm) - HistoryExpan)])
        
       # for i in range(len(data)):
       #     print(data[i],y[i])
        
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
        
    def labelData(self,x):
        print("WUT")
        
    
    def normalize(self,x):
        return self.minmax(x,0,1)
        
    def minmax(self, x, minVal, maxVal):
        self.dbgPrint("MinMax Scaling between [{},{}]".format(minVal,maxVal))
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
        self.dbgPrint("x Shape {}".format(x.shape[0]))
        #eVec = np.ones((x.shape[0],1))
        #x = []
        #np.dot(x,eVec)
        #self.testDataNorm = np.array([self.dataNorm[i:i + AmountOfSets].copy() for i in range(len(self.dataNorm) - AmountOfSets)])
        return [x[:n],x[n:]]

def main():     
    import matplotlib.pyplot as plt 
    
    #==== Definitions ==== 
    splitVal = 0.9
    Y_TakenAfter_x_Days = 50
    #==== Parameters of Modell 
    
    bc = cryptocurrency()
    bc.csv_read()
    
    print("Transfer Data with len:{}".format(len(bc.HistorData)))
    
    pp = preprocessing()
    
    print(pp.normalize(bc.HistorData))
    print(pp.standardize(bc.HistorData))
    
    xNorm = pp.normalize(bc.HistorData)
    print("Size of Data {}".format(xNorm.shape))
    #xScaled = pp.standardize(bc.HistorData)
    
    
    #====== LABEL DATA ======
    y = xNorm[:,0][Y_TakenAfter_x_Days:]
    print("Size of Data {}".format(y.shape))
    y = np.expand_dims(y, -1)
    print("Size of Data {}".format(y.shape))
    
    #====== Resize DATA ======
    resizeFactor = xNorm.shape[0] - (xNorm.shape[0]-y.shape[0])
    x = xNorm[:][0:resizeFactor]    
    print("Size of xNorm {}|{}".format(xNorm.shape,xNorm[:,0]))
    print("Size of xnew {}|{}".format(x.shape,x[:,0]))
    
    #====== Split it ======
    [Xtrain,Xeval] = pp.splitData(x,splitVal)
    
    #====== LABEL DATA ======
    #print("in this set we take the Starting Value of Bitcoins of the {}. Day after the current data".format(Y_TakenAfter_x_Days))
    #print("to evaluate if it was worth it to invest ")
    
    
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
    
    
    fig, axs = plt.subplots(plotTrainData.shape[1])
    fig.suptitle('different Features')
    for i in range(plotTrainData.shape[1]):
        axs[i].plot(XPlot1 , np.append(plotTrainData[:,i],Xtrain[:,i],axis=0)) #'r+'
        axs[i].plot(XPlot2 , np.append(plotEvalData[:,i],Xeval[:,i],axis=0))
        
    plt.show()
    

    '''
    print("Test Preprocessing")
    testData = np.array([[10, 0.1, 13],
                [20, 0.2, 16],
                [30, 0.3, 11],
                [40, 0.4, 15]])
    #Where [[x1,x2,x3],[x1,x2,x3]]
    '''

if __name__ == "__main__":
    main()