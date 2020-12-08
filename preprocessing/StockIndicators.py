#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 14:27:19 2020

@author: sausy

Gleitende Durchschnitte der Kurse:
1. #sma
    Einfacher gleitender Durchschnitt (SMA)
2. #ema
    Exponentieller gleitender Durchschnitt (EMA)

Oszillatoren:
3. #StochOszi
    Stochastik-Oszillator
4. #rsi
    Relative Strength Index (RSI)
5. #cci
    Commodity Channel Index (CCI)

Trendindikatoren:
6. #adx
    Average Directional Index (ADX)
7. #macd
    Moving Average Convergence/Divergence (MACD)

Kurs - / Preis-Kanäle:
8. Bollinger Bänder
9. Donchian Channel
10. Keltner Channel

Disclaimer:
A lot of code was copied from
https://www.investopedia.com/ask/answers/122314/what-exponential-moving-average-ema-formula-and-how-ema-calculated.asp
https://docs.anychart.com/Stock_Charts/Technical_Indicators/
https://towardsdatascience.com/trading-toolbox-02-wma-ema-62c22205e2a9
"""

import numpy as np
#from sklearn import preprocessing
import pandas as pd
import copy

#from preProcessRawData import preprocessing as prepo

class indicators:
    def __init__(self):
        print("10 Available Stock Indicators")
        print("be awaire due to the nature of tech indicators")
        print("some indicators only be valid after 10time ticks (sma10) or even 100")


    def addStockFeatures(self,x,featureList):
        #add technical indicators
        #supportet features
        #average
        #sma  .. simple moving average
        #ema  ... this returs with 3 values ema10, ema21, ema100
        ret = copy.deepcopy(x)
        price = self.average(ret)
        #sma10 = self.SimpleMovingAverage(x,10)

        for feat in featureList:
            if feat == "ema":
                #EMA ... Exponential Moving Average
                print("\n========\nCalc EMA\n")
                ema10 = self.calc_ema(price,10)
                ema21 = self.calc_ema(price,21)
                ema100 = self.calc_ema(price,100)
                ret["ema10"] = ema10
                ret["ema21"] = ema21
                ret["ema100"] = ema100
            elif feat == "average":
                print("\n========\nCalc Average\n")
                ret["avg"] = price
            elif feat == "sma":
                print("\n========\nCalc SMA\n")
                ret["sma10"] = self.SimpleMovingAverage(price,10)
                ret["sma100"] = self.SimpleMovingAverage(price,100)
                #print x["sma10"]
            elif feat == "StochOszi":
                pass
            elif feat == "rsi":
                pass
            elif feat == "cci":
                pass
            elif feat == "adx":
                pass
            elif feat == "macd":
                pass

        return ret

    def average(self,x):
        max = x["High"]
        min = x["Low"]

        avg = max + min
        avg = avg / 2

        return avg

    def SimpleMovingAverage(self, x, numOfHistoryDays):
        #calculated the moving average
        #valid after 'numOfHistoryDays' time ticks

        ret = []

        for i in range(0,len(x)):
            if (i - numOfHistoryDays) < 0 :
                nonExistingDays = numOfHistoryDays - i
                startPos = 0
                stopPos = numOfHistoryDays - nonExistingDays
                k = stopPos + 1
            else:
                startPos = i - numOfHistoryDays + 1
                stopPos = i
                k = numOfHistoryDays

            sum_buffer = 0

            for j in range(startPos,stopPos+1):
                sum_buffer = x[j] + sum_buffer

            ret.append([])
            ret[i] = sum_buffer / k

        return ret

    def calc_ema(self, x, numOfHistoryDays):
        #EMA ... Exponential Moving Average
        #calculated the Exponential Moving Average
        #valid after 'numOfHistoryDays' time ticks + 1
        #+1 because the ema needs one more observation
        # than the sma(simple moving average)

        len = x.shape[0]

        n = numOfHistoryDays

        #Smoothing factor alpha
        a = 2/(n + 1)

        # wee need at least num+1 observation till
        # the ema gets valid
        # till then the ema equals sma
        sma = self.SimpleMovingAverage(x[0:numOfHistoryDays],numOfHistoryDays)
        ema = sma

        #push all ema values onto the ema array
        for t in range(numOfHistoryDays,len):
            ema.append([])
            ema[t] = a * x[t] + (1-a)*ema[t-1]

        return ema




def main():
    print('=============\nSandbox: StockIndicators.py')


if __name__ == "__main__":
    main()
