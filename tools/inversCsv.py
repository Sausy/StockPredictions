#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: sausy
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


class InputData:
    def __init__(self,path='in.csv',initRow=0):
        #self.data = pd.read_csv(path,header=4) #names=featureList
        self.path = path
        self.data = pd.read_csv(path,header=initRow) #names=featureList

        print(self.data)

        self.outData = 'out.csv'
        self.firstLine = initRow
        self.lastLine = initRow

        #self.MaxSamples = 512
        #print(path)
        #data = data.drop(0, axis=0)

        #plt.plot(self.data.values[:300,0],self.data.values[:300,2])
        #plt.show()

    def invert(self):
        print("\n\nInvert Time\n==============")
        print(self.path)

        self.firstLine = -1
        self.MaxColumn = -1

        f= open(self.path,"r")
        outFile = open(self.outData,"w+")

        f_line = f.readlines()
        #for rowData in f_line:
        #print("=====\nf_line\n")
        #print(f_line)

        foo = np.array(f_line[0].replace('\n', ''))

        #print("=====\nFirst Line:")
        #print(foo)

        #outFile.write(f_line[0])
        for i in range(len(f_line)-1,0,-1):
            rowData = f_line[i]
            rowData = rowData.replace('\n', '')

            if rowData.find(',') != -1:
                foo = np.append(foo,rowData)
                #print("=====\nLine {}:".format(i))
                #print(foo)
            #outFile.write(f_line[i])


        print("=====\nfoo\n")
        print(foo)

        for i in range(foo.shape[0]):
            outFile.write(foo[i])
            outFile.write("\n")

        outFile.close()
        f.close()



def main():
    #iData = InputData(path="in.csv")
    #iData = InputData(path="BTC-USD.csv")
    iData = InputData(path="BTC-USD-h.csv")
    iData.invert()


if __name__ == "__main__":
    main()
