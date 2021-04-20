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
                    return


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
    ret = data[~(data['date'] > dateTime)]
    return ret

def addBuySellPredictor(data):
    dataLen = data.shape[0]
    ge = np.ones(dataLen) * 0.80
    le = np.ones(dataLen) * 0.20

    data['s'] = np.greater_equal(data["rsi"],ge)
    data['b'] = np.less_equal(data["rsi"],le)

    ret = data[['date','rsi','close','ema21','sma10','open','s','b']]

    #print(ret)

    return ret

def dateIterator():
    pass

def main():
    inputFile60 = "../CSV/rdyCSV/rdy60.csv"
    inputFile15 = "../CSV/rdyCSV/rdy15.csv"
    inputFile1 = "../CSV/rdyCSV/rdy1.csv"

    #inputFile = inputFile1
    #inputFile = inputFile15
    inputFile = inputFile60


    #===[SetUp Multiple Wallets] ===
    wHdl = walletHdl(tradingFee = 0.26)
    startingValue = 10
    for i in range(0,10):
        wHdl.addWallet(startingValue,"eur")


    dataRaw = pd.read_csv(inputFile)
    print(dataRaw)

    anualRev = []

    yearCounter = 16
    monthNum = 1
    monthCount = 0
    strMonth = '01'
    monthOffset = 12

    TotalRev = 0.0

    #16,21
    for i in range(0,40):
        #dataStr = "20" + str(i) + "-01-01"
        #dataMax = "20" + str(i+1) + "-01-01"
        wHdl.resetWallets(startingValue,"eur")


        monthNum = monthCount * monthOffset + 1
        if monthNum < 10:
            strMonth = "0" + str(monthNum)
        else:
            strMonth = str(monthNum)

        dataStr = "20" + str(yearCounter) + "-" + strMonth + "-01"

        monthNum += (monthOffset)
        if monthNum < 10:
            strMonth = "0" + str(monthNum)
        else:
            strMonth = str(monthNum)


        dataMax = "20" + str(yearCounter) + "-" + strMonth + "-01"

        monthCount += 1
        if monthCount >= 12/monthOffset:
            yearCounter += 1
            monthCount = 0

        data = delteRowsBeforTime(dataRaw,dataStr)
        data = addBuySellPredictor(data)

        dataPlot = delteRowsAfterTime(data,dataMax)

        print("\n====[Processing Data] ==== ")
        print(dataPlot)
        if len(dataPlot) < 1:
            print("\n===done======")
            break

        #data['div'] = differentiate_scalar(data['ema21'].values)
        data['div'] = differentiate_scalar(data['ema21'].values)
        #data['div2'] = differentiate_scalar(data['ema21'].values)
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
        #dataPlot = delteRowsAfterTime(dataRaw,dataMax)
        #Xplot = np.linspace(0,10,num=dataPlot.shape[0])
        #Yplot = dataPlot['close'].values
        #YplotEma = dataPlot['ema21'].values
        #plt.plot(Xplot,Yplot,Xplot,YplotEma)


        printOutCnt = 0
        maxLen = data.shape[0]
        zeroLevelCnt = 0
        rowCnt = 0
        for index, row in data.iterrows():
            wHdl.pushRawData(row['close'],'xbt','eur')

            if (row['div'] > -0.05) and (row['div'] < 0.05):
                zeroLevelCnt += 1

            #for i in range(0,10):
            #    wHdl.setTrade('xbt','eur',5.0)


            #if its a high Sell
            if (row['div'] < 0.01) and (row['div'] >= 0):
                wHdl.setTrade('xbt','eur',1.0)#1.001

            #if its a low Buy:
            elif (row['div'] > -0.01) and (row['div'] <= 0):
                wHdl.setTrade("eur","xbt",0.0)

            if row['date'] > dataMax:
                break

            rowCnt += 1


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

        #plt.show()

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
