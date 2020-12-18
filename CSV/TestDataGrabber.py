#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 15:09:11 2020

@author: sausy
"""
import getData
dataGrab = getData.cryptocurrency()

import matplotlib.pyplot as plt 

def main():
    print('TestDataGrabber')
    
    #if we didn't store historical data yet 
    #dataGrab.getHistoricalDataFromDatabase()
    dataGrab.setUp()
    
    plt.plot(dataGrab.HistorData[1][:],dataGrab.HistorData[0][:])
    plt.show()
    plt.plot(dataGrab.trainingData[1][:],dataGrab.trainingData[0][:])
    plt.show()
    plt.plot(dataGrab.evalData[1][:],dataGrab.evalData[0][:])
    plt.show()

    dataGrab.getCurrentPrice()
    #print(bc.data[1][:])
    plt.plot(dataGrab.data[1][:],dataGrab.data[0][:])
    
    '''
    while True:
        bc.updatePrice()
        
        plt.plot([1, 2, 3, 4])
          
        if price != last_price:
          print('Bitcoin price: ',price)
          last_price = price
    '''

if __name__ == "__main__":
    main()