# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#for import purpose
from requests import Request, Session
from requests.exceptions import ConnectionError, Timeout, TooManyRedirects
import json
import pandas as pd

import numpy as np 
#from sklearn import datasets, model_selection 

#For testing purpose
import matplotlib.pyplot as plt 

#from requests import Request, Session
#from requests.exceptions import ConnectionError, Timeout, TooManyRedirects
#import json

from datetime import datetime, timedelta
#import datetime
#from datetime import timedelta
#import dateutil.parser

#To scale and split the data
#from sklearn import preprocessing


class cryptocurrency:
    datetime_Format = '%Y-%m-%dT%H:%M:%S.%fZ'
    SamplingInterval = 15 #5minutes
    SamplingIntervalDomain = 'm'
    
    
    def __init__(self, coinName = 'BTC', MAX_BUFFER_SIZE=512, debug = False):
        self.debug = debug
        
        self.dbgPrint("Init Data Grabber")
        
        self.errorUseLocal = False
        
        self.defaultCoin = coinName#'bitcoin'
        self.price = []
        #self.MAX_BUFFER_SIZE = MAX_BUFFER_SIZE
        self.AmountOfSamples = 200
        
        self.HistoricalMaxTimeFrame = []
        
        #Feature 0 price in USD
        #Feature 1 timestamp in 2019-08-30T18:09:02.000Z
        
        
        # All the data that is currently available at the time of training
        self.HistorData = [[],[]] 
        #scaled historData between 0-1 to improve training converg time
        self.dataScaled = [[],[]] 
        # curret Data that got availiable after the modell was trained 
        self.data = [[],[]]
        
        #scaled training and eval Data  (between  0-1)
        self.trainingData = [[],[]]
        self.evalData = [[],[]]
        
        #self.data = []
        print(datetime.now())
        current_time = datetime.strftime(datetime.now(), format="%Y-%m-%dT%H:%M:%S.000Z")
        print(current_time)
    
        self.url = 'https://sandbox-api.coinmarketcap.com/v1/cryptocurrency/quotes/historical'
        self.parameters = {
          'interval':str(self.SamplingInterval) + str(self.SamplingIntervalDomain),
          'count':str(self.AmountOfSamples),
          'symbol':str(coinName),
          'convert':'USD'
        }
        #'time_end':'2020-07-15T12:15:18.988Z',
        #self.parameters['symbol'] = coinName
        
        self.headers = {
          'Accepts': 'application/json',
          'X-CMC_PRO_API_KEY': '4a9753c1-544c-4729-9af8-f6d78a5a8311',
        }
    
    def dbgPrint(self,outData):
        txt = "[DataGrabber]"
        if self.debug == True:
            txt = txt + "[DBG]"
            
        txt = txt + " "
        txt = txt + outData
        print(txt)
        
    def setUp(self):
        print("Get Info From DataBase")
        self.getDataBaseInfo()
        print("Get Data")
        self.getHistoricalDataFromDatabase()
        if self.errorUseLocal == False :
            self.splitData()
        
        #loadHistoricalData = ... TODO save Historical data to pc .... 
    def getDataBaseInfo(self):
        url = 'https://sandbox-api.coinmarketcap.com/v1/cryptocurrency/map'
        parameters = {
          'symbol':str(self.defaultCoin)
        }
        
        session = Session()
        session.headers.update(self.headers)
        
        try:
          response = session.get(url, params=parameters)
          data = json.loads(response.text)
        except (ConnectionError, Timeout, TooManyRedirects) as e:
          print(e)
        
        try:
            self.HistoricalMaxTimeFrame.append(data['data'][0]['first_historical_data'])
            self.HistoricalMaxTimeFrame.append(data['data'][0]['last_historical_data'])
        except:
            print("NonValid History Data")
            
        print("Min Max Time from data in Database THIS INFORMATION IS UNUSED YET")
        print(self.HistoricalMaxTimeFrame)
            
        
    def getRawData(self):
        session = Session()
        session.headers.update(self.headers)
        
        #https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest?sort=market_cap&start=1&limit=10&cryptocurrency_type=tokens&convert=USD,BTC
        #parametersbuffer = self.parameters.copy()
        #parametersbuffer['convert']  = parametersbuffer['convert'] + ',' + str(self.defaultCoin)
        #print(parametersbuffer)
        
        try:
          response = session.get(self.url, params=self.parameters)
          data = json.loads(response.text)   
          #print(data)
        except (ConnectionError, Timeout, TooManyRedirects) as e:
          print(e)
          return None
      
        try:
            return data['data']['quotes']
        except:
            print("ERROR server didn't give you the data")
            self.errorUseLocal = True
            print(data)
            return None
      
        
        
    def getHistoricalDataFromDatabase(self):   
        session = Session()
        session.headers.update(self.headers)
        #print(datetime.now())
        current_time = datetime.strftime(datetime.now(), format="%Y-%m-%dT%H:%M:%S.000Z")
        current_time = datetime.strptime(current_time, self.datetime_Format)
        #print(current_time)
        #time_interval = dateutil.parser.parse("2200-01-01T00:05:00.000Z") 
        #time_interval = datetime.strptime(time_interval, datetime_Format)
        
        
        #self.parameters['count'] = str(1)
        self.parameters['count'] = str(self.AmountOfSamples)
        #self.parameters['time_end'] = str(datetime.strftime(current_time, format="%Y-%m-%dT%H:%M:%S.000Z"))  
        self.parameters['interval'] = str(self.SamplingInterval) + str(self.SamplingIntervalDomain),
        
        #HistorData = [[9558.27076135],
        #                 [datetime.strptime('2019-08-30T18:04:00.000Z', self.datetime_Format)]]
       
        
        d = timedelta(minutes=self.SamplingInterval)
        
                
        for subIterationCnt in range(10):
            
            data_ = self.getRawData()
            if data_ == None:
                print(len(self.HistorData[1]))
                break
            data_buffer = []
            data_buffer = [rowdata['quote']['USD']['price'] for rowdata in data_]
            self.HistorData[0] = data_buffer[:] + self.HistorData[0][:]
            data_buffer = [datetime.strptime(rowdata['timestamp'], self.datetime_Format) for rowdata in data_]
            self.HistorData[1] = data_buffer[:] + self.HistorData[1][:]
            #print(HistorData[0])
            #HistorData[1].extend()
            #print(HistorData)
            current_time = self.HistorData[1][0] - d
            #print(current_time)
            
            self.parameters['time_end'] = str(datetime.strftime(current_time, format="%Y-%m-%dT%H:%M:%S.000Z"))  
            
            #print(HistorData[1][len(HistorData[1])-1])
            #(AmountOfSamples*)
            #buffer = datetime.strptime(current_time, datetime_Format) - d
            #d = timedelta(minutes=self.SamplingInterval*self.AmountOfSamples)
            #current_time = current_time - d
            #buffer = current_time - d
            #print(current_time)
            #time_buf = dateutil.parser.parse(current_time) + d
            #current_time = buffer
            #current_time = datetime.strftime(time_buff, format="%Y-%m-%dT%H:%M:%S.000Z")
        
    def getCurrentPrice(self):  
        del self.parameters['time_end'] 
        
        data_ = self.getRawData()
        if data_ == None:
            return 
        #self.data[0] = [rowdata['quote']['USD']['price'] for rowdata in data['data']['quotes']]
        #self.data[1] = [datetime.strptime(rowdata['timestamp'], datetime_Format) for rowdata in data['data']['quotes']]
        self.data[0] = [rowdata['quote']['USD']['price'] for rowdata in data_]
        self.data[1] = [datetime.strptime(rowdata['timestamp'], self.datetime_Format) for rowdata in data_]
        #self.data = [[rowdata['quote']['USD']['price'],rowdata['timestamp']] for rowdata in data['data']['quotes']]
        '''
        foo = [] 
        for i in range(len(data['data']['quotes'])): #len(data['data']['quotes'])
            foo.join([[data['data']['quotes'][i]['quote']['USD']['price']], [data['data']['quotes'][i]['timestamp']]])
            self.data.append([data['data']['quotes'][i]['quote']['USD']['price'], data['data']['quotes'][i]['timestamp']])
        ''' 
          
    
    def updatePrice(self):
        #TODO: ... get data every 5min with historical 1min samples with those 5min
        self.price.append(self.getCurrentPrice())
        if len(self.price) >= self.MAX_BUFFER_SIZE:
            self.price.pop()
    
    def scaleData(self):
        #This is importent to ensure a faster conerging of the model ... due to ? no clue 
        #preprocessing
        #self.dataScaled = preprocessing.MinMaxScaler()
        #self.dataScaled = dataScaled.fit_transform(self.HistorData)
        
        
        
        '''
        print("amount of CrossValidation Sets")
        AmountOfSets = 3
        
        norm = preprocessing.MinMaxScaler()
        self.dataNorm = norm.fit_transform(self.HistorData)
        print("======={}".format(self.dataNorm.shape))
        print(self.dataNorm[:,0])
        print(self.dataNorm[:5])

        # using the last {history_points} open close high low volume data points, predict the next open value
        self.testDataNorm = np.array([self.dataNorm[i:i + AmountOfSets].copy() for i in range(len(self.dataNorm) - AmountOfSets)])
        #print("=======")
        #print(self.dataNorm[:10])
        #print("=======")
        #print(self.testDataNorm)
        #print("=======")
        #print(self.testDataNorm[:4])
        
        print("======={}".format(self.testDataNorm.shape))
        print(self.testDataNorm[:5])
        self.evalDataNorm = np.array([self.dataNorm[:, 0][i + AmountOfSets].copy() for i in range(len(self.dataNorm) - AmountOfSets)])
        print("======={}".format(self.evalDataNorm.shape))
        print(self.evalDataNorm)
        self.evalDataNorm = np.expand_dims(self.evalDataNorm, -1)
        print("======={}".format(self.evalDataNorm.shape))
        print(self.evalDataNorm)
        

        self.evalData = np.array([self.HistorData[:, 0][i + AmountOfSets].copy() for i in range(len(self.HistorData) - AmountOfSets)])
        self.evalData = np.expand_dims(self.evalData, -1)

        self.evalFit = preprocessing.MinMaxScaler()
        self.evalFit.fit(self.evalData)
        
        
        
        '''
        
        
        
        #self.dataNorm = preprocessing.MinMaxScaler()
        #self.dataNorm = self.dataNorm.fit_transform(self.HistorData)
        '''
        self.trainingDataNorm = 
        self.evalDataNorm = 
        
        
        # using the last {history_points} open close high low volume data points, predict the next open value
        ohlcv_histories_normalised = np.array([data_normalised[i:i + history_points].copy() for i in range(len(data_normalised) - history_points)])
        next_day_open_values_normalised = np.array([data_normalised[:, 0][i + history_points].copy() for i in range(len(data_normalised) - history_points)])
        next_day_open_values_normalised = np.expand_dims(next_day_open_values_normalised, -1)
    
        next_day_open_values = np.array([data[:, 0][i + history_points].copy() for i in range(len(data) - history_points)])
        next_day_open_values = np.expand_dims(next_day_open_values, -1)
    
        y_normaliser = preprocessing.MinMaxScaler()
        y_normaliser.fit(next_day_open_values)
        '''

            
    def splitData(self):
        #Split data to 10 sets
                
        print("Split Data")        
        len_data = len(self.HistorData[0])
        setOnePice_len = int(len_data/10)
        training_len = len_data - setOnePice_len
        print("len:{}|{}|{}".format(len_data,setOnePice_len,training_len))
        cnt = 0
        for row in self.HistorData:
            self.trainingData[cnt] = self.HistorData[cnt][0:training_len].copy()
            self.evalData[cnt] = self.HistorData[cnt][training_len+1:len_data].copy()
            cnt = cnt + 1
            #self.trainingData[row] = self.HistorData[row][0:500].copy()
        
        
        print("Scale Data")
        self.scaleData()

        #self.evalData = self.HistorData[:][training_len+1:len_data]
    
    def csv_read(self, dataselect=0):
        #CSV DATA
        #Date//Open//High//Low//Close//Adj Close//Volume
        if dataselect == 0:    
            self.HistorData = pd.read_csv('BTC-USD.csv')
            print("size loaded HistorData (shape={})".format(self.HistorData.shape))
            print(self.HistorData)
    
            print("DropUnused Data (Colums)")
            print("For First Try we only use MaxValue and openValue ")
            self.HistorData = self.HistorData.drop('Date', axis=1)
            #self.HistorData = self.HistorData.drop('Open', axis=1)
            #self.HistorData = self.HistorData.drop('High', axis=1)
            #self.HistorData = self.HistorData.drop('Low', axis=1)
            #self.HistorData = self.HistorData.drop('Close', axis=1)
            self.HistorData = self.HistorData.drop('Adj Close', axis=1)
            self.HistorData = self.HistorData.drop('Volume', axis=1)
            
            #self.HistorData = self.HistorData.drop(, axis=1)
            #print(self.HistorData.values[0])
    
            print("Delete First Row, that Names")
            self.HistorData = self.HistorData.drop(0, axis=0)
            print("Setting Data As numpy Array")
            print(self.HistorData)
            #print(self.HistorData.values[:])
            #print(self.HistorData.values[:][0])
            #print(self.HistorData.values[0][:])
            #self.HistorData = [self.HistorData.values[:][0],self.HistorData.values[:][1]]
            self.HistorData = self.HistorData.values
            #print(self.HistorData)
            
            #print(self.HistorData[:,0])
    
            #self.data_normaliser = preprocessing.MinMaxScaler()
            #self.data_normaliser = self.data_normaliser.fit_transform(self.HistorData)
        else:
            self.HistorData = pd.read_csv('BTC-USD-h.csv',header=1) 
            #self.HistorData = self.HistorData.drop(0, axis=0)
            
            print("size loaded HistorData (shape={})".format(self.HistorData.shape))
            print(self.HistorData)
            
            
            
            print("DropUnused Data (Colums)")
            print("For First Try we only use MaxValue and openValue ")
            self.HistorData = self.HistorData.drop('Date', axis=1)
            self.HistorData = self.HistorData.drop('Symbol', axis=1)
            self.HistorData = self.HistorData.drop('Volume BTC', axis=1)
            self.HistorData = self.HistorData.drop('Volume USD', axis=1)
            #self.HistorData = self.HistorData.drop('Open', axis=1)
            #self.HistorData = self.HistorData.drop('High', axis=1)
            #self.HistorData = self.HistorData.drop('Low', axis=1)
            #self.HistorData = self.HistorData.drop('Close', axis=1)
            #self.HistorData = self.HistorData.drop('Adj Close', axis=1)
            #self.HistorData = self.HistorData.drop('Volume', axis=1)
            print(self.HistorData)
            
            self.HistorData = self.HistorData.drop(0, axis=0)
            self.HistorData = self.HistorData.values
       
        
        '''        
        csvTimeStamp = datetime.datetime.fromtimestamp(1325317920)        
        t = datetime.datetime.fromtimestamp(1325317920)
        print(t.strftime("%Y-%m-%dT%H:%M:%S.000Z")
        t = datetime.datetime.fromtimestamp(1325317980)
        print(t.strftime("%Y-%m-%dT%H:%M:%S.000Z"))


        print(t.strftime("%Y-%m-%dT%H:%M:%S.000Z"))
        '''
    

def main():        
    print("Init Data")   
    bc = cryptocurrency()
    
    #if we didn't store historical data yet 
    #bc.getHistoricalDataFromDatabase()
    bc.setUp()
    
    if bc.errorUseLocal == True :
        plt.plot(bc.HistorData[1][:],bc.HistorData[0][:])
        plt.show()
        plt.plot(bc.trainingData[1][:],bc.trainingData[0][:])
        plt.show()
        plt.plot(bc.evalData[1][:],bc.evalData[0][:])
        plt.show()
    
        #print("getCurrent price")    
        #bc.getCurrentPrice()
        #print(bc.data[1][:])
        #plt.plot(bc.data[1][:],bc.data[0][:])
    
    #bcLocal = localData()
    bc.csv_read()
    bc.scaleData()
    
    plt.plot(np.linspace(0, 10, len(bc.HistorData)),bc.HistorData)
    plt.show()
    print("y min max between 0-1")
    plt.plot(np.linspace(0, 10, len(bc.dataNorm)),bc.dataNorm)
    plt.show()
    print(bc.testDataNorm.shape)
    #print(bc.testDataNorm)
    plt.plot(np.linspace(0, 10, len(bc.testDataNorm[:,0,0])),bc.testDataNorm[:,0,0])
    plt.show()
    plt.plot(np.linspace(0, 10, len(bc.testDataNorm[:,49,0])),bc.testDataNorm[:,49,0])
    plt.show()
    print(bc.evalDataNorm.shape)
    plt.plot(np.linspace(0, 10, len(bc.evalDataNorm[:,0])),bc.evalDataNorm[:,0])
    plt.show()
    
    
    
    
    
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
