#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dez  30 10:53:41 2020

@author: sausy
"""

import numpy as np
import pandas as pd

import copy

import sys

import matplotlib.pyplot as plt

import datetime
from dateutil.relativedelta import relativedelta

import random as rd


'''
###### TODO: LIST######
*)Kraken interface to actually pull data
and set trades
*)bevor trading wait till we get the best value for the trade
e.g.: wait till rsi actully goes down again ... or something like that

'''




class wallet(object):
    """docstring for wallet."""

    def __init__(self, baseAmount=100.0, currencyTag="xbt"):
        super(wallet, self).__init__()
        self.StartingBalance = baseAmount
        self.balance = baseAmount
        self.bHistory = 10000#0.00001
        self.currencyTag = currencyTag
        self.r = False
        self.lastValue = 0.0

    def canTrade(self,tag):
        #access granted if current currency is the tagname
        if self.currencyTag == tag:
            return True
        return False

    def updateAfterTrade(self,balance,tag):
        self.bHistory = self.balance
        self.balance = balance
        self.currencyTag = tag


class walletHdl(object):
    """docstring for walletHdl."""
    '''
    The wallet Handl should keep track of multiple wallets
    and ensure that manuel tradiding ist still doable without major
    interference of the bot
    '''

    def __init__(self,tradingFee=0.26):
        super(walletHdl, self).__init__()

        #temp_wObj = wallet()
        self.wObjList = []
        self.tradingFee = tradingFee

        #define convRate as dictionary
        self.convRate = {}
        self.convRate["eureur"] = 1.0
        self.convRate["xbtxbt"] = 1.0

        self.lastValue = 0.0

        self.dbgFlag = False

    def addWallet(self, baseAmount, currencyTag):
        wObj = wallet(baseAmount,currencyTag)
        self.wObjList.append(wObj)

    def resetWallets(self,baseAmount, currencyTag):
        for cnt,obj in enumerate(self.wObjList):
            obj.updateAfterTrade(baseAmount,currencyTag)
            self.bHistory = 10000#0.00001


    def printBalance(self):
        print("======[Wallet BALANCE]======")
        for cnt,obj in enumerate(self.wObjList):
            print("Wallet[{}]: ID={}".format(cnt,obj.currencyTag))
            print("Wallet[{}]: VA={}".format(cnt,obj.balance))
            print()

    def printBalanceEuro(self):
        for cnt,obj in enumerate(self.wObjList):
            if obj.canTrade('xbt'):
                [newValue,newCurrency] = self.trade(obj, 'xbt', 'eur')
                obj.updateAfterTrade(newValue,newCurrency)

        print("======[Wallet BALANCE]======")
        for cnt,obj in enumerate(self.wObjList):
            print("Wallet[{}]: ID={}".format(cnt,obj.currencyTag))
            print("Wallet[{}]: VA={}".format(cnt,obj.balance))
            print()

        print("======[Total BALANCE]======")
        sumBalance = 0.0
        baseValue = 0.0
        for cnt,obj in enumerate(self.wObjList):
            sumBalance += obj.balance
            baseValue += obj.StartingBalance
        print("Total Value: {}|{}".format(sumBalance,baseValue))
        print()

        ret = sumBalance/baseValue * 100.0
        print("Total Percentage: {}%".format(ret))

        print()



        return ret


    def pushRawData(self,rate,cFirst,cSec):
        tradePair = str(cFirst) + str(cSec)
        self.convRate[tradePair] = copy.deepcopy(rate)
        tradePair = str(cSec) + str(cFirst)
        self.convRate[tradePair] = copy.deepcopy(1/rate)


    def trade(self,wObj,cFirst,cSec):
        tradePair = str(cFirst) + str(cSec)

        value = copy.deepcopy(wObj.balance)

        value = value - value * self.tradingFee/100.0
        value = self.convRate[tradePair] * value

        #print("Value after Traid = {}".format(value))
        return [value,cSec]

    def inspectTrade2(self,data):

        for cnt,obj in enumerate(self.wObjList):
            if obj.canTrade('xbt'):
                #[newValue,newCurrency] = self.trade(obj, 'xbt', 'eur')

                if obj.r == True:
                    if data/self.lastValue < 1.0: #(1+(cnt+1)/100)
                        obj.r = False
                        self.lastValue = data
                        return True
                if data/self.lastValue >= 1.0:
                    obj.r = True
                else:
                    obj.r = False

        self.lastValue = data

        return False

    def inspectTrade(self,cFirst, cSec, action='b', profit=1.1):
        if action == 'b':
            return True

        for cnt,obj in enumerate(self.wObjList):
            if obj.canTrade(cFirst):
                [newValue,newCurrency] = self.trade(obj, cFirst, cSec)

                if newValue/obj.bHistory > profit: #(1+(cnt+1)/100)
                    return True

        return False

    def setTrade(self,cFirst, cSec, profit=0.0):
        for cnt,obj in enumerate(self.wObjList):
            if obj.canTrade(cFirst):
                [newValue,newCurrency] = self.trade(obj, cFirst, cSec)
                #if newValue/obj.bHistory >= 0.89: #(1+(cnt+1)/100
                #    print("TradeProfit: {}".format(newValue/obj.bHistory))
                #    print("t= -1: {}".format(obj.bHistory))
                #    print("t=  0: {}".format(newValue))
                if newValue/obj.bHistory >= profit: #(1+(cnt+1)/100)
                    obj.updateAfterTrade(newValue,newCurrency)
                    return True
        return False

    def stoppLoss(self,cFirst, cSec, profit=0.5):
        for cnt,obj in enumerate(self.wObjList):
            if obj.canTrade(cFirst):
                [newValue,newCurrency] = self.trade(obj, cFirst, cSec)

                if newValue/obj.bHistory < profit: #(1+(cnt+1)/100)
                    obj.updateAfterTrade(newValue,newCurrency)
                    return True
        return False


class timeIterator(object):
    """docstring for timeIterator."""

    def __init__(self,start=[2016,1,1] ,min=0,day=0,month=0,year=0):
        super(timeIterator, self).__init__()
        self.yearCounter = start[0]
        self.monthCount = 0
        self.dayCounter = 0

        self.yearOffset = year
        self.monthOffset = month
        self.dayOffset = day

        self.start = str(datetime.datetime(start[0], start[1], start[2], 0,0))
        self.stop = str(datetime.datetime(start[0], start[1], start[2], 0,0))
        self.startNext = datetime.datetime(start[0], start[1], start[2], 0,0)

        if year > 0:
            self.incTime = self.addYear
        elif month > 0:
            self.incTime = self.addMonth
        elif day > 0:
            self.incTime = self.addDay
        elif min > 0:
            self.incTime = self.addMin
            self.dayOffset = 1
        else:
            self.incTime = self.addYear

    def addMin(self):
        start_date = self.startNext
        delta = datetime.timedelta(days=self.dayOffset)
        mDelta = datetime.timedelta(minutes=15)
        self.start = str(start_date)
        #print(start_date)
        #print(mDelta)
        self.startNext = start_date+mDelta
        #print(self.startNext)
        self.stop = str(start_date+delta)
        #print(start_date+delta)

    def addYear(self):
        start_date = self.startNext
        self.startNext = datetime.date(start_date.year+self.yearOffset, start_date.month, start_date.day)
        self.start = str(start_date)
        self.stop = str(self.startNext)

    def addMonth(self):
        start_date = self.startNext
        self.startNext = start_date+relativedelta(months=+self.monthOffset)
        self.start = str(start_date)
        self.stop = str(self.startNext)


    def addDay(self):
        start_date = self.startNext
        delta = datetime.timedelta(days=self.dayOffset)
        self.start = str(start_date)
        self.startNext = start_date+delta
        self.stop = str(start_date+delta)


    def next(self):
        self.incTime()



def differentiate_scalar(x):
    #print(x)
    #print("\ndx:")
    dx = np.diff(x)
    a = np.zeros(1)
    dx = np.append(a,dx)

    #print(dx)

    #print("\nshape x: {}".format(len(x)))
    #print("shape dx: {}\n".format(len(dx)))
    return dx


def delteRowsBeforTime(data,dateTime):
    ret = data[~(data['date'] < dateTime)]
    return ret

def delteRowsAfterTime(data,dateTime):
    ret = data[~(data['date'] >= dateTime)]
    return ret

def minmax(x, minVal, maxVal):
    #print("MinMax Scaling between [{},{}]".format(minVal,maxVal))
    min = np.amin(x)
    max = np.amax(x)
    if min == max :
        print("error min = max")
        return 0

    divVal = maxVal - minVal
    a = (x - min) * (divVal)
    b = max - min
    retMatrix = minVal + a/b

    return retMatrix


def monthOverflow(monthNum, yearNum):
    for i in range(1,4):
        #print("loop: {}|{}|{}".format(i,monthNum,yearNum))
        if monthNum > (12 * i):
            monthNum -= 12*i
            yearNum = yearNum + i
            #print(yearNum)

    return [monthNum, yearNum]





def SimpleMovingAverage(x, numOfHistoryDays):
    #calculated the moving average
    #valid after 'numOfHistoryDays' time ticks

    ret = []

    print("calculating SMA{}:".format(numOfHistoryDays))
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

def addBuySellPredictor(data):
    dataLen = data.shape[0]
    ge = np.ones(dataLen) * 0.80
    le = np.ones(dataLen) * 0.20

    data['s'] = np.greater_equal(data["rsi"],ge)
    data['b'] = np.less_equal(data["rsi"],le)

    ret = data[['date','rsi','close','ema21','ema10','sma10','sma100','open','s','b']]

    #print(ret)

    return ret

def divRawData(data,colTag):
    df = copy.deepcopy(data)
    #colList = list(df.columns)

    ret = pd.DataFrame()

    a = df[colTag].values

    #create data shifted by 1 iteration
    shiftData = df.shift(1,fill_value=a[0])
    #print("Div")
    #print(shiftData[colTag])
    #print(df[colTag])
    #divide current value with last one
    df_r = df
    df_r[colTag] = df[colTag].div(shiftData[colTag])

    #because the first row was just div with 1
    #ret = df_r.drop(df_r.index[0])

    #now get the log of it
    #ret[colList] = np.log10(ret[colList])

    return df_r[colTag]

def main():
    inputFile60 = "../CSV/rdyCSV/rdy60.csv"
    inputFile15 = "../CSV/rdyCSV/rdy15.csv"
    inputFile1 = "../CSV/rdyCSV/rdy1.csv"

    inputFile = inputFile15
    #inputFile = inputFile60


    #===[SetUp Multiple Wallets] ===
    wHdl = walletHdl(tradingFee = 0.26)
    startingValue = 10.0
    for i in range(0,10):
        wHdl.addWallet(startingValue,"eur")


    dataRaw = pd.read_csv(inputFile)
    print(dataRaw)

    anualRev = []

    metricIndicator = 'div2'
    YDay_div = []
    YDay_div2 = []
    YDay_rel = []
    YDay_data = []
    YDay_rsi = []

    tI = timeIterator([2019,1,1],min=1)

    Nplots = 7
    fig, axs = plt.subplots(Nplots)
    pltCnt = 0

    waitForBuy = False

    TotalRev = []
    lastValue = [0.0,0.0,0.0]
    for i in range(0,96*7*4):

        #wHdl.resetWallets(startingValue,"eur")
        tI.next() #Iterate to next Date
        data = delteRowsBeforTime(dataRaw,tI.start)
        data = delteRowsAfterTime(data,tI.stop)

        if len(data) <= 1:
            print("\n===============================")
            print("What The FUCK INCONSISTENT DATA")
            print("===============================")
            continue

        #RSI buy sell predictor
        #data = addBuySellPredictor(data)

        wHdl.pushRawData(data['close'][data.shape[0]-1:].values,'xbt','eur')

        #add diverential value
        #data['div'] = differentiate_scalar(data['ema21'].values)
        #data['div2'] = differentiate_scalar(data['div'].values)
        #data['div'] = SimpleMovingAverage(data['div'].values,5)
        #data['div'] = SimpleMovingAverage(data['div'].values,20)
        #data['div'] = SimpleMovingAverage(data['div'].values,20)
        #data['div'] = SimpleMovingAverage(data['div'].values,20)
        #data['div'] = SimpleMovingAverage(data['div'].values,20)

        #add scale Data
        data['rel'] = divRawData(data,'close')

        Xplot = np.linspace(0,10,num=data.shape[0])
        YplotEma = data['ema21'].values
        mymodel = np.poly1d(np.polyfit(Xplot, YplotEma, 4))
        yModell = mymodel(Xplot)
        yModell = YplotEma
        data['div'] = differentiate_scalar(yModell)
        data['div2'] = differentiate_scalar(data['div'])


        #y = minmax(data['div'].values, -1,1)
        YDay_div.append(data['div'][data.shape[0]-1:].values)
        YDay_div2.append(data['div2'][data.shape[0]-1:].values)
        YDay_data.append(data['close'][data.shape[0]-1:].values)
        YDay_rel.append(data['rel'][data.shape[0]-1:].values)
        y_rsi = data['rsi'][data.shape[0]-1:].values
        YDay_rsi.append(y_rsi)

        if wHdl.setTrade('xbt','eur',1.01):
            print("sold")

        #if y_rsi > 0.8:
        #    wHdl.setTrade('xbt','eur',1.05)
        #if y_rsi < 0.2:

        y = data['close'][data.shape[0]-1:].values[0]

        if y < lastValue[0]:
            waitForBuy = True
            #wHdl.setTrade('eur','xbt',0.0)

        if waitForBuy == True:
            if y > lastValue[0] :
                wHdl.setTrade('eur','xbt',0.0)
                waitForBuy = False


        lastValue[0] = data['close'][data.shape[0]-1:].values[0]
        lastValue[1] = lastValue[0]
        lastValue[2] = lastValue[1]
        lastSlope = data['div'][data.shape[0]-1:].values[0]


        #print(YDay_div)

        #print("Out")
        #print(YDay_div)
        #print(len(YDay_div))
        #return
        if len(YDay_div) < 96:
            continue
        else:
            print("Done")
            #p = wHdl.printBalanceEuro()
            #anualRev.append(p)
            wHdl.printBalance()
            #wHdl.resetWallets(startingValue,"eur")

            '''
            y_data = minmax(YDay_data, -1,1)
            #y_rel = minmax(YDay_rel, -1,1)
            y_rel = np.array(YDay_rel)
            y_rel = y_rel - np.ones(len(y_rel))
            y_rel = y_rel * 100
            y_div = minmax(YDay_div, -1,1)
            #y_div = YDay_div
            #y_div2 = minmax(YDay_div2, -1,1)
            y_div2 = np.array(YDay_div2)*3

            y_rsi = minmax(YDay_rsi, -1,1)

            X = np.linspace(0,10,num=len(y_data))
            #axs[pltCnt].hlines(0.0,0,10,colors='k')
            #axs[pltCnt].hlines(1.0,0,10,colors='k')
            #axs[pltCnt].hlines(-1.0,0,10,colors='k')
            ##axs[pltCnt].plot(X,y_data,'k-',X,y_rel,'m-',X,y_div,'y-',X,y_div2,'b-')
            #axs[pltCnt].plot(X,y_data,'k-',X,y_rsi,'g-',X,y_div,'y-')
            ##axs[pltCnt].show()
            '''

            YDay_div = []
            YDay_div2 = []
            YDay_rel = []
            YDay_data = []
            YDay_rsi = []

            pltCnt+=1
            #wHdl.resetWallets(startingValue,"eur")

            if pltCnt >= Nplots:
                #wHdl.printBalance()

                #plt.show()
                #fig, axs = plt.subplots(Nplots)
                #wHdl.resetWallets(startingValue,"eur")
                pltCnt = 0

            continue



        #======[PLOT DATA]====
        Xplot = np.linspace(0,10,num=data.shape[0])
        YplotDiv = data['div'].values
        Yplot = data['close'].values
        Yrel = data['rel'].values
        Ysmooth = data['div2'].values

        YplotSc = minmax(Yplot, np.amin(YplotDiv), np.amax(YplotDiv))
        YplotRelSc = minmax(Yrel, np.amin(YplotDiv), np.amax(YplotDiv))
        YplotEma = data['ema21'].values
        YplotEmaSc = minmax(YplotEma, np.amin(YplotDiv), np.amax(YplotDiv))
        YplotSmoothSc = minmax(Ysmooth, np.amin(YplotDiv), np.amax(YplotDiv))


        mymodel = np.poly1d(np.polyfit(Xplot, YplotEma, 4))
        Ymod = minmax(mymodel(Xplot), np.amin(YplotDiv), np.amax(YplotDiv))

        x = mymodel(Xplot)
        data['div'] = differentiate_scalar(x)
        print(x)
        print(data['div'])

        YplotDiv = data['div'].values

        print(YplotDiv)



        #plt.plot(Xplot,YplotDiv,'b-',Xplot,YplotEmaSc,'y-',Xplot,YplotSc,'k-')
        #plt.plot(Xplot,YplotDiv,'b-',Xplot,YplotEmaSc,'y-',Xplot,YplotSc,'k-',Xplot,YplotSmoothSc,'m-',Xplot,YplotRelSc,'g-')
        plt.plot(Xplot,YplotSc,'k-',Xplot,Ymod,'b-',Xplot,YplotEmaSc,'y-',Xplot,YplotDiv,'m-')
        plt.hlines(0.0,xmin=0,xmax=10,colors='k')

        rowCnt = 0
        slopPusher = 2
        slopLimitBase = 0.8
        slopLimit = 0.8
        for index, row in data.iterrows():
            wHdl.pushRawData(row['close'],'xbt','eur')

            if wHdl.setTrade('xbt','eur',1.01):
                slopLimit = slopLimit * slopPusher
                plt.vlines(Xplot[rowCnt],ymin=YplotSmoothSc[rowCnt],ymax=np.amax(YplotDiv),colors='g')

            if slopLimit > slopLimitBase:
                slopLimit = slopLimitBase

            #if wHdl.setTrade('eur','xbt',slopLimit):
            #    plt.vlines(Xplot[rowCnt],ymin=YplotSmoothSc[rowCnt],ymax=np.amax(YplotDiv),colors='r')
            if wHdl.stoppLoss('eur','xbt',slopLimit):
                slopLimit = slopLimit / slopPusher
                plt.vlines(Xplot[rowCnt],ymin=YplotSmoothSc[rowCnt],ymax=np.amax(YplotDiv),colors='r')

            lastValue = row['close']
            lastSlope = row[metricIndicator]
            rowCnt += 1


        anualRev.append(wHdl.printBalanceEuro())

        #print("\nMetric: \t{}\t{}".format(zeroLevelCnt,maxLen))

        print("\nPrinting starting and stopping time")
        print(tI.start)
        print(tI.stop)

        a = 0
        TotalRev = 0.0
        for i in anualRev:
            print("Max: {}".format(i))
            a += i
            if TotalRev == 0.0:
                TotalRev = i/100
            else:
                TotalRev *= i/100

        a = a/len(anualRev)
        print("\nAVG REV: {:.2f}".format(a))

        plt.show()

        #===TotalRev====
        print()
        print("Total rev: {:.2f}%".format(TotalRev * 100))
        print("e.g.: starting with 100euro => {}".format(TotalRev*100))



        continue
        return

    wHdl.printBalanceEuro()

    for rev in p:
        print("Rev: {}".format(rev))

    return


    #==========================================================================

    for i in range(0,40):
        wHdl.resetWallets(startingValue,"eur")
        tI.next()

        #==============

        print("\nStartDate: {}".format(dataStr))
        print("EndDate: {}".format(dataMax))

        data = delteRowsBeforTime(dataRaw,dataStr)
        data = addBuySellPredictor(data)

        data = delteRowsAfterTime(data,dataMax)
        print(data)

        #data['div'] = differentiate_scalar(data['ema21'].values)
        #data['div'] = differentiate_scalar(data['ema21'].values)
        data['div'] = differentiate_scalar(data['close'].values)
        #data['div'] = differentiate_scalar(data['sma100'].values)
        #data['div2'] = differentiate_scalar(data['ema21'].values)
        #data['div'] = np.log10(data['div'])
        data['div'] = SimpleMovingAverage(data['div'].values,20)
        data['div'] = SimpleMovingAverage(data['div'].values,20)
        data['div'] = SimpleMovingAverage(data['div'].values,20)
        data['div'] = SimpleMovingAverage(data['div'].values,20)
        data['div'] = SimpleMovingAverage(data['div'].values,20)
        #data['div'] = SimpleMovingAverage(data['div'].values,10)
        #data['div'] = SimpleMovingAverage(data['div'].values,30)
        #data['div'] = SimpleMovingAverage(data['div'].values,40)
        data['divSmooth1'] = SimpleMovingAverage(data['div'].values,10)
        data['div2'] = differentiate_scalar(data['div'].values)

        data['divSmooth1'] = SimpleMovingAverage(data['div2'].values,10)


        '''
        X = np.linspace(0,100,num=data.shape[0])
        plt.plot(X,data['div'].values)
        plt.show()

        X = np.linspace(0,100,num=100)
        plt.plot(X,data['div'][data.shape[0]-100:].values)
        plt.show()
        '''
        #return

        #====[Plot close Values]=======
        dataPlot = delteRowsAfterTime(data,dataMax)
        if len(dataPlot) <= 1:
            print("\n===done======")
            break

        Xplot = np.linspace(0,10,num=dataPlot.shape[0])
        YplotDiv = dataPlot['div'].values
        Yplot = dataPlot['close'].values

        Ysmooth = dataPlot['div2'].values

        YplotSc = minmax(Yplot, np.amin(YplotDiv), np.amax(YplotDiv))
        YplotEma = dataPlot['sma100'].values
        YplotEmaSc = minmax(YplotEma, np.amin(YplotDiv), np.amax(YplotDiv))

        #YplotSmoothSc = minmax(Ysmooth, np.amin(YplotDiv), np.amax(YplotDiv))
        YplotSmoothSc = Ysmooth
        #plt.plot(Xplot,YplotDiv,'b-',Xplot,YplotEmaSc,'y-',Xplot,YplotSc,'k-')
        plt.plot(Xplot,YplotDiv,'b-',Xplot,YplotEmaSc,'y-',Xplot,YplotSc,'k-',Xplot,YplotSmoothSc,'m-')
        plt.hlines(0.0,xmin=0,xmax=10,colors='k')
        #plt.show()
        #plt.plot(Xplot,Yplot,Xplot,YplotEma)

        printOutCnt = 0
        maxLen = data.shape[0]
        zeroLevelCnt = 0
        rowCnt = 0
        lastValue = 0.0
        lastSlope = 0.0
        metricIndicator = 'div2'
        slopPusher = 2
        slopLimitBase = 0.8
        slopLimit = 0.8

        for index, row in data.iterrows():
            wHdl.pushRawData(row['close'],'xbt','eur')

            if wHdl.setTrade('xbt','eur',1.001):
                slopLimit = slopLimit * slopPusher
                plt.vlines(Xplot[rowCnt],ymin=YplotSmoothSc[rowCnt],ymax=np.amax(YplotDiv),colors='g')

            if slopLimit > slopLimitBase:
                slopLimit = slopLimitBase

            #if wHdl.setTrade('eur','xbt',slopLimit):
            #    plt.vlines(Xplot[rowCnt],ymin=YplotSmoothSc[rowCnt],ymax=np.amax(YplotDiv),colors='r')
            if wHdl.stoppLoss('eur','xbt',slopLimit):
                slopLimit = slopLimit / slopPusher
                plt.vlines(Xplot[rowCnt],ymin=YplotSmoothSc[rowCnt],ymax=np.amax(YplotDiv),colors='r')




            #if rd.random() > 0.989:
            #    if wHdl.setTrade("eur","xbt",0.0):
            #        plt.vlines(Xplot[rowCnt],ymin=YplotSmoothSc[rowCnt],ymax=np.amax(YplotDiv),colors='r')


            '''
            #To identity a Zero crossing
            if (lastSlope > 0.1) and (row[metricIndicator] < -0.1):
                #this is Sell indicator
                #for i in range(0,5):
                suc = wHdl.setTrade('xbt','eur',1.001)#1.001
                if suc:
                    slopLimit = slopLimit / slopPusher
                    print("====XBT TO EUR===")
                    wHdl.printBalance()
                    #plt.vlines(Xplot[rowCnt],ymin=YplotSc[rowCnt],ymax=YplotSc[rowCnt]*1.2,colors='g')
                    plt.vlines(Xplot[rowCnt],ymin=YplotSmoothSc[rowCnt],ymax=np.amax(YplotDiv),colors='g')


            if (lastSlope < -0.01) and (row[metricIndicator] >  0.01):
                #this is buy indicator
                suc = wHdl.setTrade("eur","xbt",0.0)
                if suc:
                    print("+++ EUR to XBT +++ ")
                    wHdl.printBalance()
                    #plt.vlines(Xplot[rowCnt],ymin=YplotSc[rowCnt],ymax=YplotSc[rowCnt]-YplotSc[rowCnt]*0.2,colors='r')
                    plt.vlines(Xplot[rowCnt],ymin=YplotSmoothSc[rowCnt],ymax=np.amax(YplotDiv),colors='r')

            '''


            lastValue = row['close']
            lastSlope = row[metricIndicator]
            rowCnt += 1

            if (row['date'] > dataMax):
                break

        anualRev.append(wHdl.printBalanceEuro())

        print("\nMetric: \t{}\t{}".format(zeroLevelCnt,maxLen))

        a = 0
        TotalRev = 0.0
        for i in anualRev:
            print("Max: {}".format(i))
            a += i
            if TotalRev == 0.0:
                TotalRev = i/100
            else:
                TotalRev *= i/100

        a = a/len(anualRev)
        print("\nAVG REV: {:.2f}".format(a))

        plt.show()
        continue
        return




        for index, row in data.iterrows():
            wHdl.pushRawData(row['close'],'xbt','eur')



            #if (row['div'] > -0.05) and (row['div'] < 0.05):
            #    zeroLevelCnt += 1

            #for i in range(0,5):
            '''
            suc = wHdl.setTrade('xbt','eur',1.1)#1.001
            if suc:
                slopLimit = slopLimit / slopPusher
                plt.vlines(Xplot[rowCnt],ymin=YplotSmoothSc[rowCnt],ymax=YplotSc[rowCnt],colors='g')

            '''


            #To identity a Zero crossing
            if (lastSlope > 0.01) and (row[metricIndicator] < -0.01):
                #this is Sell indicator
                #for i in range(0,5):
                suc = wHdl.setTrade('xbt','eur',1.01)#1.001
                if suc:
                    slopLimit = slopLimit / slopPusher
                    print("====XBT TO EUR===")
                    wHdl.printBalance()
                    #plt.vlines(Xplot[rowCnt],ymin=YplotSc[rowCnt],ymax=YplotSc[rowCnt]*1.2,colors='g')
                    plt.vlines(Xplot[rowCnt],ymin=YplotSmoothSc[rowCnt],ymax=YplotSc[rowCnt],colors='g')


            if (lastSlope < 0.0) and (row[metricIndicator] > 0.0):
                #this is buy indicator
                suc = wHdl.setTrade("eur","xbt",0.0)
                if suc:
                    print("+++ EUR to XBT +++ ")
                    wHdl.printBalance()
                    #plt.vlines(Xplot[rowCnt],ymin=YplotSc[rowCnt],ymax=YplotSc[rowCnt]-YplotSc[rowCnt]*0.2,colors='r')
                    plt.vlines(Xplot[rowCnt],ymin=YplotSmoothSc[rowCnt],ymax=YplotSc[rowCnt],colors='r')

            '''
            if slopLimit > slopLimitBase:
                slopLimit = slopLimitBase

            if row[metricIndicator] < slopLimit:
                slopLimit = slopLimit * slopPusher
                suc = wHdl.setTrade("eur","xbt",0.0)
                if suc:
                    #plt.vlines(Xplot[rowCnt],ymin=YplotSc[rowCnt],ymax=YplotSc[rowCnt]-YplotSc[rowCnt]*0.2,colors='r')
                    plt.vlines(Xplot[rowCnt],ymin=YplotSmoothSc[rowCnt],ymax=YplotSc[rowCnt],colors='r')
            '''
            '''
            if row['div'] == 0.0:
                zeroLevelCnt += 1
                if lastValue > row['close']:
                    suc = wHdl.setTrade("eur","xbt",0.0)
                elif lastValue < row['close']:
                    wHdl.setTrade('xbt','eur',1.0)#1.001
            '''

            '''
            #if its a high Sell
            if (row['div'] < 0.008) and (row['div'] >= 0) and (lastValue > row['close']):
                suc = wHdl.setTrade('xbt','eur',1.001)#1.001
                if suc:
                    plt.vlines(Xplot[rowCnt],ymin=YplotSc[rowCnt],ymax=YplotSc[rowCnt]*1.2,colors='g')

            #if its a low Buy:
            elif (row['div'] > -0.008) and (row['div'] <= 0)and (lastValue < row['close']):
                suc = wHdl.setTrade("eur","xbt",0.0)
                if suc:
                    plt.vlines(Xplot[rowCnt],ymin=YplotSc[rowCnt],ymax=YplotSc[rowCnt]-YplotSc[rowCnt]*0.2,colors='r')
            '''


            if (row['date'] > dataMax):
                break

            rowCnt += 1

            lastValue = row['close']
            lastSlope = row[metricIndicator]


            #printOutCnt += 1
            #if printOutCnt >= 1000:
            #    print("Date: {}".format(row['date']))
            #    wHdl.printBalance()
            #    printOutCnt = 0


        anualRev.append(wHdl.printBalanceEuro())

        print("\nMetric: \t{}\t{}".format(zeroLevelCnt,maxLen))

        a = 0
        TotalRev = 0.0
        for i in anualRev:
            print("Max: {}".format(i))
            a += i
            if TotalRev == 0.0:
                TotalRev = i/100
            else:
                TotalRev *= i/100

        a = a/len(anualRev)
        print("\nAVG REV: {:.2f}".format(a))

        plt.show()
        return

    #===TotalRev====
    print()
    print("Total rev: {:.2f}%".format(TotalRev * 100))
    print("e.g.: starting with 100euro => {}".format(TotalRev*100))

    #=============================================================================

    return

    #=============================================================================



    printOutCnt = 0
    dbg_count = 0
    wait_buy = False
    wait_sell = False
    momentumCount = 0

    rdyToSell = False
    rdyToBuy = False
    for index, row in data.iterrows():
        wHdl.pushRawData(row['close'],'xbt','eur')#this is static for now TODO:

        #proz = 1.001#(1 + (4*i+1)/1000)

        #if wHdl.inspectTrade('xbt','eur','s',proz):
        wHdl.setTrade('xbt','eur',1.9)

        #wHdl.stoppLoss('xbt','eur',0.5)

        if row['s'] == True:
            for i in range(0,10):
                proz = 0.97#(1 + (4*i+1)/1000)
            #if wHdl.inspectTrade('xbt','eur','s',proz):
                #wHdl.setTrade('xbt','eur',proz)


        if row['b'] == True:
            if not wait_buy:
                wHdl.setTrade("eur","xbt",0.0)
                #wHdl.stoppLoss('eur','xbt',0.9)
                #wait_buy = True
        else:
            wait_buy = False

        '''
        momentumCount += 1
        for i in range(0,10):
            proz = (1 + (2*i+1)/1000)
            if wHdl.inspectTrade('xbt','eur','s',proz):
            #if wHdl.inspectTrade2(row['close']):
                wHdl.pushRawData(row['close'],'xbt','eur') #this is static for now TODO:
                wHdl.setTrade('xbt','eur')
                momentumCount = 0
                wait_buy = False
        '''
        #for i in range(0,10):




        printOutCnt += 1
        if printOutCnt >= 1000:
            print("Date: {}".format(row['date']))
            wHdl.printBalance()
            printOutCnt = 0



    print("\n=======================")
    print("Total Money in wallet: ")
    #wHdl.printBalance()
    wHdl.printBalanceEuro()

    '''
    if (row['date'] > '2017-03-13'):
        print("==== XBT TO EURO === ")
        print("Date: {}".format(row['date']))
        print("Value: {}".format(row['close']))
        print("RSI: {}".format(row['rsi']))
        wHdl.printBalance()
        sys.stdout.flush()
    '''


if __name__ == "__main__":
    main()
