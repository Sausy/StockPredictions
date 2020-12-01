#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 14:27:19 2020

@author: sausy
"""

import numpy as np 
from sklearn import preprocessing

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

