#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dez 9  09:08:41 2020

@author: sausy
"""

import numpy as np
import pandas as pd

import sys
import time
import urllib.request
import json

from datetime import datetime
from tqdm import tqdm #progress bar

import os

class kraken:
    def __init__(self,currencySymbol="BTCUSD",api_start="0",api_end="9999999999",tradingFee=0.26):
        self.api_domain = "https://api.kraken.com"
        self.api_path = "/0/public/"
        self.api_method = "Trades"
        self.api_data = ""

        #=== need to figure out shit
        self.api_symbol = currencySymbol.upper()
        self.api_start = api_start #str(int(api_start) - 1) + "999999999"
        self.api_end = api_end
        self.tradingFee = tradingFee
        self.isRaw = True

        self.CHUNK_SIZE = 10000

        #self.FileName = "./CSV/cleanCSV/XXBTZEUR_5m.csv"
        self.ApiTradeName = "XXBTZEUR"

    def getBalance(self):
        pass

    def placeOrder(self):
        pass


    def grabHistoryIntoCSV(self,outFileName="out.csv",fullPull=False):
        if fullPull == True:
            #first we clear the log csv file
            #and write the Identifier on top
            fs = open(outFileName, 'w')
            #fs.write("Date,Price,Volume\n")
            fs.close()
        else:
            #self.api_start = self.api_end
            self.api_end = int(time.time())

        fs = open(outFileName, 'a')

        download_amount = int(self.api_end) - int(self.api_start)
        download_fac = 100.0/download_amount

        print("Amount to Download \t{}\ndownload factor: \t{}".format(download_amount,download_fac))

        try:
            #while True:
            with tqdm(total=100) as pbar:
                while True:
                    self.api_data = "?pair=%(pair)s&since=%(since)s" % {"pair":self.api_symbol, "since":self.api_start}
                    api_request = urllib.request.Request(self.api_domain + self.api_path + self.api_method + self.api_data)

                    try:
                        self.api_data = urllib.request.urlopen(api_request).read()
                    except Exception:
                        time.sleep(3)
                        continue


                    self.api_data = json.loads(self.api_data)

                    if len(self.api_data["error"]) != 0:
                        time.sleep(3)
                        continue

                    #TODO:
                    #Implement the easier approache
                    #np.array([1368431149, 1368431150]).astype('datetime64[s]')
                    ## array([2013-05-13 07:45:49, 2013-05-13 07:45:50], dtype=datetime64[s])

                    for trade in self.api_data["result"][self.api_symbol]:
                        if int(trade[2]) < int(self.api_end):
                            price = trade[0]
                            volume = trade[1]
                            #format="%Y-%m-%d %I-%p"
                            currentTime = datetime.utcfromtimestamp(trade[2]).strftime('%d-%m-%Y %H:%M:%S')
                            currentTime = str(currentTime)
                            #currentTime = time.strftime('%Y-%m-%d', str(trade[2]))
                            #print("[DATA:]%(datetime)s,%(price)s,%(volume)s" % {"datetime":currentTime, "price":trade[0], "volume":trade[1]})
                            #print("%(datetime)d,%(price)s,%(volume)s" % {"datetime":currentTime, "price":trade[0], "volume":trade[1]})

                            pushStr = currentTime + ","
                            pushStr = pushStr + str(price) + ","
                            pushStr = pushStr + str(volume) + "\n"
                            fs.write(pushStr)
                            pbar.update(download_fac)

                        else:
                            sys.exit(0)


                    self.api_start = self.api_data["result"]["last"]

        except KeyboardInterrupt:
        	None

        fs.close()

    def callBackTimeStamp(self,x):
        #Converts the UnixTimeStamp to datetimeFormat
        self.isRaw = True
        if "-" in x:
            #print("Assuming DataFormat: 2020-07-01 00:00:00 ")
            self.isRaw = False
            ret = pd.to_datetime(x)
        else:
            #print("Assuming DataFormat: UnixTimeStamp")
            ret = pd.to_datetime(x, unit='s')

        return ret

    def preProcessToOhlc(self,timeInterval,fileName="XXBTZEUR.csv",OutName="XXBTZEUR_5m.csv"):
        #self.CHUNK_SIZE=10000
        #self.CHUNK_SIZE=2

        #timeInterval="10S"
        #fileName = OutName
        #OutName = "foo.csv"
        barName = "processOHLC" + OutName
        pbar = tqdm(desc=barName)



        print("processToOhlc\nDisclaimer: This might take a while\nFileName: {}".format(fileName))
        print("outFile: {}".format(OutName))


        '''
        #to test out with a small dataset
        dbg_df = pd.read_csv(fileName, parse_dates=True, index_col=0, date_parser=self.callBackTimeStamp)
        #dbg_d = dbg_df["Price"].resample(timeInterval).ohlc()
        dbg_d = dbg_df.resample(timeInterval).agg({'open':'first', 'high':'max','low':'min','close':'last'})


        #Drop NaN rows
        dbg_d = dbg_d.dropna()
        print(dbg_d)

        return
        '''

        #read in the csv file but with chunksize
        df_chunk = pd.read_csv(fileName, parse_dates=True, index_col=0, date_parser=self.callBackTimeStamp, iterator=True, chunksize=self.CHUNK_SIZE)


        print("\nTrying to detect if data format is raw:")
        print("This is currently pretty limited")
        print("The decission is made in the callback function from the timestamp")
        print("and the only check that is done, is if a : is in the date I assume that")
        print("its an already cleanCSV file ")

        prevLastLine = pd.DataFrame()
        initDone = False
        while True:
            try:
                df = df_chunk.get_chunk(self.CHUNK_SIZE)
            except:
                print("end of file reached")
                break


            if self.isRaw == True:
                resampled = df["Price"].resample(timeInterval).ohlc()
            else:
                resampled = df.resample(timeInterval).agg({'open':'first', 'high':'max','low':'min','close':'last'})

            if initDone:
                #df = prevLastLine.append(df)
                resampled = pd.concat([prevLastLine,resampled])
                resampled = resampled.resample(timeInterval).agg({'open':'first', 'high':'max','low':'min','close':'last'})

            #Drop NaN rows
            resampled = resampled.dropna()

            #save last line into a temp variable
            prevLastLine = resampled.tail(1)
            #drop last line
            resampled.drop(resampled.tail(1).index,inplace=True)

            #in the first loop we want to clear the OutName file
            #and write the labels in the first line
            if initDone == False:
                fs = open(OutName, 'w')
                #gather the colum labels
                firstCol = str(resampled.index.name)
                for j in resampled.columns:
                    firstCol += "," + j

                firstCol += "\n" #end the string with a chReturn
                fs.write(firstCol)
                fs.close()

            resampled.to_csv(OutName, mode="a",header=False)
            initDone = True
            pbar.update(1)

        prevLastLine.to_csv(OutName, mode="a",header=False)




    def processToOhlc(self,timeInterval,fileName="XXBTZEUR.csv",OutName="XXBTZEUR_5m.csv"):
        #self.CHUNK_SIZE=100000
        print("processToOhlc\nDisclaimer: This might take a while\nFileName: {}".format(fileName))
        print("outFile: {}".format(OutName))

        #dateparse = lambda x: datetime.strptime(x, '%d-%m-%Y %H:%M:%S')

        columsDataFrame=[]
        tempIndex=0
        #data_frame = pd.DataFrame()

        data_frame = pd.read_csv(fileName, parse_dates=True, index_col=0)

        ret = data_frame.resample(timeInterval).agg({
                                    'open':'first',
                                    'high':'max',
                                    'low':'min',
                                    'close':'last'
                                })

        ret.to_csv(OutName,index=True)



    def resetStartTime_viaCSV(self,fileName):
        last = ""
        with open(fileName, "r") as f:
            for last in f:
                pass

        print("\nLastRow: \t{}".format(last))
        splitData = last.split(",")
        pTime = int(splitData[0])

        #databuffer = datetime.strptime(splitData[0], '%d-%m-%Y %H:%M:%S')
        #databuffer = datetime.strptime(splitData[0], '%Y-%m-%d %H:%M:%S')
        #databuffer = pd.to_datetime(splitData[0], unit='s')
        #databuffer = pd.Timestamp(splitData[0], unit='s')
        #print("TimeTuple: {}".format(databuffer.timetuple()))
        #pTime = int(time.mktime(databuffer.timetuple()))
        #self.api_start = str(pTime)

        print("\nNew Starting time: \t{} \t{}".format(splitData[0],pTime))



    def useOrigianlCsv(self,startTime,fileName="XXBTZEUR_original.csv",OutName="XXBTZEUR.csv"):

        print("Disclaimer: This might take a while")
        #startTime must be in unix timestamp format
        f = open(fileName,'r')

        f2 = open(OutName,'w')
        string = "Date,Price,Volume\n"
        f2.write(string)


        for lineData in f:
            splitData = lineData.split(',')
            timeData = int(splitData[0])

            if timeData < startTime :
                pass
            else:
                writeRow = str(datetime.utcfromtimestamp(timeData).strftime('%d-%m-%Y %H:%M:%S'))
                writeRow = writeRow + ","
                writeRow = writeRow + splitData[1] + ","
                writeRow = writeRow + splitData[2]

                f2.write(writeRow)

        f.close()
        f2.close()

    def mergeFiles(self,csvList,outFile):
        print("\n===========================================================")
        print("Merge base Data with incremental Data")
        print("Create clean File:")
        fs = open(outFile, 'w')
        fs.write("Date,Price,Volume\n")
        fs.close()

        print("write merged Files\n")
        pbar = tqdm(desc="merge", position=1)

        for f in csvList:
            chunk_container = pd.read_csv(f, chunksize=self.CHUNK_SIZE)
            for chunk in chunk_container:
                #if the csv file has a header the following parameter
                #"header=False ... needs to be added
                pbar.update(1)
                chunk.to_csv(outFile, mode="a", index=False, encoding='utf-8-sig')


    def processKrakenData(self,CoinName="XBT",BaseName="EUR"):
        #TOD: sellingFee 16% and buyingFee 26%
        tradingFee = 0.26 #there is a selling and a buying fee

        #TODO: must be replaced with find realName funktion from bublebuy
        rawFile_base = "./kraken/" + CoinName + BaseName + "_base.csv"
        rawFile_inc = "./kraken/" + CoinName + BaseName + "_inc.csv"
        csvList = [rawFile_base,rawFile_inc]
        FileName = "./CSV/rawCSV/" + CoinName + BaseName + ".csv"
        ApiTradeName = "X" + CoinName + "Z" + BaseName
        self.ApiTradeName = ApiTradeName

        outPath = './CSV/cleanCSV/'

        self.CHUNK_SIZE = 100000

        #check if base and incremental data is available
        if os.path.exists(rawFile_base) and os.path.exists(rawFile_inc):
            print("Base files found")
        else:
            print("original File from Kraken not found")
            print("please download the files from the Krakens gdrive")
            print("name the base data: <coindouble>_base.csv")
            print("name the incremental data: <coindouble>_inc.csv")

            return False

        #check if data is already merged
        if os.path.exists(FileName):
            print("\nMerged File found will proceed to create")
            print("time discreed files 1minut 15minute 60minutes....")
        else:
            print("File needs merging")
            self.mergeFiles(csvList,FileName)

        print("\n=========================================================")
        print("Preprocess OHLC")
        timeInterval = "1min"
        outFilePRE = outPath + ApiTradeName + "_" + timeInterval + ".csv"
        if os.path.exists(outFilePRE) == False:
            print("Start processing Data \t{}".format(timeInterval))
            self.preProcessToOhlc(timeInterval, fileName=FileName,OutName=outFilePRE)


        print("\n=========================================================")
        timeInterval = "15min"
        outFile = outPath + ApiTradeName + "_" + timeInterval + ".csv"
        print("Start processing Data \t{}".format(timeInterval))
        if os.path.exists(outFile) == False:
            self.preProcessToOhlc(timeInterval, fileName=outFilePRE,OutName=outFile)
        outFilePRE = outFile

        print("\n=========================================================")
        timeInterval = "60min"
        outFile = outPath + ApiTradeName + "_" + timeInterval + ".csv"
        print("Start processing Data \t{}".format(timeInterval))
        if os.path.exists(outFile) == False:
            self.preProcessToOhlc(timeInterval, fileName=outFilePRE,OutName=outFile)
        outFilePRE = outFile


        print("\n=========================================================")
        timeInterval = "1D"
        outFile = outPath + ApiTradeName + "_" + timeInterval + ".csv"
        print("Start processing Data \t{}".format(timeInterval))
        if os.path.exists(outFile) == False:
            self.preProcessToOhlc(timeInterval, fileName=outFilePRE,OutName=outFile)
        outFilePRE = outFile


        print("\n=========================================================")
        timeInterval = "3D"
        outFile = outPath + ApiTradeName + "_" + timeInterval + ".csv"
        print("Start processing Data \t{}".format(timeInterval))
        if os.path.exists(outFile) == False:
            self.preProcessToOhlc(timeInterval, fileName=outFilePRE,OutName=outFile)
        outFilePRE = outFile


        print("\n=========================================================")
        timeInterval = "10D"
        outFile = outPath + ApiTradeName + "_" + timeInterval + ".csv"
        print("Start processing Data \t{}".format(timeInterval))
        if os.path.exists(outFile) == False:
            self.preProcessToOhlc(timeInterval, fileName=outFilePRE,OutName=outFile)


        print("\n=====IMPORTANT==========\n")
        print("https://support.kraken.com/hc/en-us/articles/201893638-How-trading-fees-work-on-Kraken")
        print("ther are two fees one for buy and one for sell")
        print("also there is a 30day dependencie on the fees")
        print("if one would trade more than 50k within those 30days, the fee would drop")
        #how to calc the fee value
        #https://support.kraken.com/hc/en-us/articles/216784328-How-are-fee-conversion-rates-calculated-

        return True



def main():
    #TOD: sellingFee 16% and buyingFee 26%
    tradingFee = 0.26 #there is a selling and a buying fee
    CoinName = "XBT"
    BaseName = "EUR"

    #TODO: must be replaced with find realName funktion from bublebuy
    rawFile_base = CoinName + BaseName + "_base.csv"
    rawFile_inc = CoinName + BaseName + "_inc.csv"
    csvList = [rawFile_base,rawFile_inc]
    rawFile = CoinName + BaseName + ".csv"
    ApiTradeName = "X" + CoinName + "Z" + BaseName


    outPath = '../CSV/cleanCSV/'

    CHUNK_SIZE = 100000

    '''
    print("\n===========================================================")
    print("Merge base Data with incremental Data")
    print("Create clean File:")
    fs = open(rawFile, 'w')
    fs.write("Date,Price,Volume\n")
    fs.close()

    print("write merged Files")
    pbar = tqdm(desc="merge", position=1)

    for f in csvList:
        #chunk_container = pd.read_csv(f)
        #hunk_container.to_csv(rawFile, mode="a", index=False)
        chunk_container = pd.read_csv(f, chunksize=CHUNK_SIZE)
        for chunk in chunk_container:
            #if the csv file has a header the following parameter
            #"header=False ... needs to be added
            pbar.update(1)
            chunk.to_csv(rawFile, mode="a", index=False, encoding='utf-8-sig')

    '''

    FileName = rawFile#ApiTradeName + ".csv"


    #class datetime.datetime(year, month, day, hour=0, minute=0, second=0, microsecond=0, tzinfo=None, *, fold=0)
    #startTime = datetime(2017,2,1)

    #time.struct_time(
    #tm_year=2018, tm_mon=12, tm_mday=27,
    #tm_hour=6, tm_min=35, tm_sec=17,
    #tm_wday=3, tm_yday=361, tm_isdst=0)

    #the 1.1.2017 was a sunday
    #and tm_yday=1 because it was the first day of the year
    startTime_ = (2017, 1, 1, 1, 0, 0, 6, 1, -1)
    #at one o'clock because the kraken api somehow has a one hour shift

    #time needs to be converted into unix timestamp format
    startTime = int(time.mktime(startTime_))
    endTime = int(time.time())

    print(startTime)
    print(endTime)
    print("pull InitTime: \t{}".format(time.ctime(startTime)))
    print("pull EndTime: \t{}".format(time.ctime(endTime)))


    webApi = kraken(ApiTradeName,str(startTime),str(endTime))
    #webApi.useOrigianlCsv(fileName="XXBTZEUR_original.csv",OutName=FileName)

    '''
    print("=====================")
    print("Loading original CSV")
    if os.path.exists(FileName):
        print("File Already exist ... assuming original csv was already loaded")
    else:
        webApi.useOrigianlCsv(startTime, fileName=rawFile,OutName=FileName)
    print("\n[DONE]")

    '''
    print("=====================")
    print("Get the new end time form the last entery of the csv file")
    webApi.resetStartTime_viaCSV(fileName=FileName)

    '''
    print("=====================")
    print("Start pulling Data")
    webApi.grabHistoryIntoCSV(FileName,fullPull=False)
    fileSize = os.path.getsize(FileName)
    fileSize = fileSize/1000.0 #Kb
    fileSize = fileSize/1000.0 #Mb
    fileSize = fileSize/1000.0 #Gb
    print("\n[DONE] FileSize {}Gb".format(fileSize))

    return
    print("=====================")
    print("Add more recent Data")
    endTime = int(time.time())
    webApi.api_end = str(endTime)
    webApi.grabHistoryIntoCSV(FileName,fullPull=False)
    print("\n[DONE] FileSize {}Mb".format(fileSize))
    '''



    print("\n=========================================================")
    print("Preprocess OHLC")
    timeInterval = "1min"
    outFilePRE = outPath + ApiTradeName + "_" + timeInterval + ".csv"
    print("Start processing Data \t{}".format(timeInterval))
    webApi.preProcessToOhlc(timeInterval, fileName=FileName,OutName=outFilePRE)


    print("\n=========================================================")
    timeInterval = "15min"
    outFile = outPath + ApiTradeName + "_" + timeInterval + ".csv"
    print("Start processing Data \t{}".format(timeInterval))

    webApi.preProcessToOhlc(timeInterval, fileName=outFilePRE,OutName=outFile)

    return

    print("\n=========================================================")
    timeInterval = "15min"
    outFile = outPath + ApiTradeName + "_" + timeInterval + ".csv"
    print("Start processing Data \t{}".format(timeInterval))

    '''
    if os.path.exists(outFile):
        print("File Already exist ... assuming original csv was already loaded")
    else:
        webApi.processToOhlc(timeInterval, fileName=FileName,OutName=outFile)
    '''

    webApi.processToOhlc(timeInterval, fileName=outFilePRE,OutName=outFile)

    print("\n=========================================================")
    timeInterval = "60min"
    outFile = outPath + ApiTradeName + "_" + timeInterval + ".csv"
    print("Start processing Data \t{}".format(timeInterval))
    webApi.processToOhlc(timeInterval, fileName=outFilePRE,OutName=outFile)

    print("\n=========================================================")
    timeInterval = "1D"
    outFile = outPath + ApiTradeName + "_" + timeInterval + ".csv"
    print("Start processing Data \t{}".format(timeInterval))
    webApi.processToOhlc(timeInterval, fileName=outFilePRE,OutName=outFile)

    print("\n=========================================================")
    timeInterval = "3D"
    outFile = outPath + ApiTradeName + "_" + timeInterval + ".csv"
    print("Start processing Data \t{}".format(timeInterval))
    webApi.processToOhlc(timeInterval, fileName=outFilePRE,OutName=outFile)

    print("\n=========================================================")
    timeInterval = "10D"
    outFile = outPath + ApiTradeName + "_" + timeInterval + ".csv"
    print("Start processing Data \t{}".format(timeInterval))
    webApi.processToOhlc(timeInterval, fileName=outFilePRE,OutName=outFile)

    print("\n=====IMPORTANT==========\n")
    print("https://support.kraken.com/hc/en-us/articles/201893638-How-trading-fees-work-on-Kraken")
    print("ther are two fees one for buy and one for sell")
    print("also there is a 30day dependencie on the fees")
    print("if one would trade more than 50k within those 30days, the fee would drop")
    #how to calc the fee value
    #https://support.kraken.com/hc/en-us/articles/216784328-How-are-fee-conversion-rates-calculated-



if __name__ == "__main__":
    main()
