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

from tqdm import tqdm
#from preProcessRawData import preprocessing as prepo

class indicators:
    def __init__(self):
        print("10 Available Stock Indicators")
        print("be awaire due to the nature of tech indicators")
        print("some indicators only be valid after 10time ticks (sma10) or even 100")

    def getStockFeatures(self,x,featureList):
        #if features are already present
        #they need to be removed
        if (set(x.columns) & set(featureList)) != set():
            x = x[set(x.columns) ^ set(featureList)]

        colList = x.columns
        ret = self.addStockFeatures(x,featureList)
        allCol = ret.columns

        retList = set(colList) ^ set(allCol)

        return ret[retList]


    def addStockFeatures(self,x,featureList):
        #add technical indicators
        #supportet features
        #average
        #sma  .. simple moving average
        #ema  ... this returs with 3 values ema10, ema21, ema100

        ret = copy.deepcopy(x)
        price = self.average(ret)
        closingPrice = x["close"]
        #sma10 = self.SimpleMovingAverage(price,10)
        #print(sma10)
        print("Price len: {}".format(len(price)))
        print(price[3832:3836])

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
                #print (x["sma10"])
            elif feat == "StochOszi":
                pass
            elif feat == "rsi":
                print("\n========\nCalc RSI\n")
                ret["rsi"] = self.rsi(closingPrice,14)
            elif feat == "cci":
                pass
            elif feat == "adx":
                pass
            elif feat == "macd":
                pass

        return ret

    def average(self,x):
        max = x["high"]
        min = x["low"]

        avg = max + min
        avg = avg / 2

        return avg

    def rsi(self,x,numOfHistoryDays):
        #x is a vector of the closing prices
        ret = [0.5] #value in percentage [0,100%]
        buffUp = []
        buffDo = []
        avgUp = 0.0
        avgDo = 0.0

        '''
        RSI = 100 – 100 / ( 1 + RS )
        RS = Relative Strength = AvgU / AvgD
        AvgU = average of all up moves in the last N price bars
        AvgD = average of all down moves in the last N price bars
        N = the period of RSI
        There are 3 different commonly used methods for the exact calculation of AvgU and AvgD (see details below)
        '''
        print("calculating RSI{}:".format(numOfHistoryDays))

        foo = 0

        pbar = tqdm(total=len(x))

        #first calculate the AverageUp/down amount
        for i in range(1,len(x)):
            change = x[i]-x[i-1]
            if change > 0:
                buffUp.append(change)
                buffDo.append(0)
            elif change == 0:
                buffUp.append(0)
                buffDo.append(0)
            else:
                buffUp.append(0)
                buffDo.append(-1 * change) #because the change needs to be absolute

            if len(buffUp) >= numOfHistoryDays:
                N = len(buffUp)
                if N <= 0:
                    print("[ERROR] WHAT THE FUCK")
                    N = 1
                #print("[DBG] {}".format(buffUp))

                avgUp = sum(buffUp)/N
                avgDo = sum(buffDo)/N

                if avgDo == 0.0:
                    avgDo = 0.0001
                #calculate the relativ strength
                RS =  avgUp/avgDo

                #RSI
                RSI = 1.0 - 1.0/(1.0+RS) #[0,1] range hence in precentage
                ret.append(RSI)

                #now clear the first element of each buffer
                buffUp.pop(0)
                buffDo.pop(0)


            else:
                #it takes t=t0+numOfHistoryDays+1 steps into the future to get the
                #first valid value
                ret.append(0.5)

            pbar.update(1)


        return ret


    def SimpleMovingAverage(self, x, numOfHistoryDays):
        #calculated the moving average
        #valid after 'numOfHistoryDays' time ticks

        ret = []

        print("calculating SMA{}:".format(numOfHistoryDays))
        with tqdm(total=len(x)) as pbar:
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

                pbar.update(1)

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
