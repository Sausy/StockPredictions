#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 15:09:11 2020

@author: sausy
"""

import numpy as np

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

        #to rescale the data back up
        retMatrix = minVal + a/b
        retK = b/(maxVal - minVal)
        retD = np.amin(x)
        
        return [retMatrix, retK, retD]

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


def main():
    print('=============\nSandbox: common.py')

if __name__ == "__main__":
    main()
