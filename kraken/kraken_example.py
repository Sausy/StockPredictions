#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dez 9  09:08:41 2020

@author: sausy
"""

import numpy as np

import sys
import time
import urllib.request
import json

class kraken:
    def __init__(self,currencySymbol="BTCUSD",api_start="0",api_end="9999999999",tradingFee=0.26):
        self.api_domain = "https://api.kraken.com"
        self.api_path = "/0/public/"
        self.api_method = "Trades"
        self.api_data = ""

        #=== need to figure out shit
        self.api_symbol = currencySymbol.upper()
        self.api_start = str(int(api_start) - 1) + "999999999"
        self.api_end = api_end #"9999999999"

    def grabHistory(self):
        try:
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
        		for trade in self.api_data["result"][self.api_symbol]:
        			if int(trade[2]) < int(self.api_end):
        				print("%(datetime)d,%(price)s,%(volume)s" % {"datetime":trade[2], "price":trade[0], "volume":trade[1]})
        			else:
        				sys.exit(0)
        		self.api_start = self.api_data["result"]["last"]
        except KeyboardInterrupt:
        	None

def main():
    tradingFee = 0.26 #in percent
    #class datetime.datetime(year, month, day, hour=0, minute=0, second=0, microsecond=0, tzinfo=None, *, fold=0)
    #startTime = datetime.datetime(2017,2,1)

    #time.struct_time(
    #tm_year=2018, tm_mon=12, tm_mday=27,
    #tm_hour=6, tm_min=35, tm_sec=17,
    #tm_wday=3, tm_yday=361, tm_isdst=0)

    #the 1.1.2017 was a sunday
    #and tm_yday=1 because it was the first day of the year
    startTime_ = (2017, 1, 1, 0, 0, 0, 6, 1, 0)
    startTime = int(time.mktime(startTime_))
    endTime = int(time.time())

    print(startTime)
    print(endTime)
    print("pull StartTime: \t{}".format(time.ctime(startTime)))
    print("pull EndTime: \t{}".format(time.ctime(endTime)))

    webApi = kraken("XXBTZUSD",str(startTime),str(endTime))
    webApi.grabHistory()


if __name__ == "__main__":
    main()
