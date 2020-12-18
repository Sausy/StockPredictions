#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dez  8 10:53:41 2020

@author: sausy
"""

import numpy as np
import pandas as pd

import sys
import platform
import time
import base64
import hashlib
import hmac

import json
from tqdm import tqdm #progress bar

if int(platform.python_version_tuple()[0]) > 2:
	import urllib.request as urllib2
else:
	import urllib2

tqdm_counter = 0

class krakentrading:
	def __init__(self):
		self.tradingFee = 0.26 #in percent
		self.CoinName = "XBT"
		self.BaseName = "EUR"
		self.api_path_private = "/0/private/"
		self.api_path_public = "/0/public/"


		self.api_domain = "https://api.kraken.com"
		#self.api_data = ""

		self.multiOrderPrvent = 0

		#to work on windows as well
		#pip install colorama
		global tqdm_counter
		self.pbar = tqdm(desc=str(tqdm_counter),ncols = 80, position=tqdm_counter)
		tqdm_counter += 1


	def basicSetup(self):
		#The asset pair has a spezial name
		#TODO: this needs to be more sofisticated
		#first one need to pull all the assets and find the prober coin
		#with this its designated Symbol
		self.ApiTraidPair = "X" + self.CoinName + "Z" + self.BaseName

		try:
		    self.api_key = open("API_Public_Key").read().strip()
		    self.api_secret = base64.b64decode(open("API_Private_Key").read().strip())
		except:
		    print("API public key and API private (secret) key must be in text files called API_Public_Key and API_Private_Key")
		    sys.exit(1)

	def findTradingName(self):
		api_request = urllib2.Request(self.api_domain + self.api_path_public + "Assets" + '?')

		try:
			api_reply = urllib2.urlopen(api_request).read().decode()
		except Exception as error:
			print("API call failed (%s)" % error)
			sys.exit(1)

		splitbuffer = api_reply.replace("}","{")
		splitbuffer = splitbuffer.replace(":","")
		splitbuffer = splitbuffer.replace(",","")
		splitbuffer = splitbuffer.replace(" ","")
		splitbuffer = splitbuffer.split("{")
		outbuffer = []

		for cnt in splitbuffer:
		    if len(cnt) < 15:
		        outbuffer.append(cnt)

		self.pbar.set_description("COIN: %s" % self.CoinName)
		self.pbar.update(0)

		#remove whitespaces
		#AND update new coin name
		splitbuffer = outbuffer
		i = 0
		for cnt in splitbuffer:
			cnt = cnt.replace('"','')
			if cnt == "":
				outbuffer.pop(i)
				i == i - 1
			else:
				outbuffer[i] = cnt

				if cnt.find(self.CoinName) != -1:
					if cnt.find(".M") == -1 and cnt.find(".S") == -1:
						print("found the coin")
						print(outbuffer[i])
						self.CoinName = outbuffer[i]
						self.ApiTraidPair = self.CoinName + "Z" + self.BaseName

						self.pbar.set_description("COIN: %s ... FOUND" % self.ApiTraidPair)

						return True

			i += 1

		#print(outbuffer)
		self.pbar.update(1)
		#print("\n[ERROR] COIN NOT FOUND")
		return False

	def getValue(self):
		api_data = "pair=" + self.ApiTraidPair
		api_request = urllib2.Request(self.api_domain + self.api_path_public + "Ticker" + '?' + api_data)

		try:
			api_reply = urllib2.urlopen(api_request).read().decode()
		except Exception as error:
			print("API call failed (%s)" % error)
			sys.exit(1)

		return api_reply


	def isRdy(self):
		if self.findTradingName() == False:
		    return False

		reply = self.getValue()

		print(reply)
		if '"error":[]' in reply:
			print("Asset {}.. is rdy".format(self.CoinName))
			print("and is tradeable")
			return True
		else:
			return False


	def placeOrder(self,BuyValue):
		if self.multiOrderPrvent > 0:
		    return False

		if self.isRdy() == False:
			return False

		reply = self.getValue()

		if '"error":[]' in reply:
		    pass
		else:
		    return False

		obj = json.loads(reply)
		assetObj = obj["result"][self.ApiTraidPair]

		if '"error":[]' in reply:
			pass
		else:
			return False

		return True
		'''
		a = ask array(<price>, <whole lot volume>, <lot volume>),
		b = bid array(<price>, <whole lot volume>, <lot volume>),
		c = last trade closed array(<price>, <lot volume>),
		v = volume array(<today>, <last 24 hours>),
		p = volume weighted average price array(<today>, <last 24 hours>),
		t = number of trades array(<today>, <last 24 hours>),
		l = low array(<today>, <last 24 hours>),
		h = high array(<today>, <last 24 hours>),
		o = today's opening price
		'''

		print("\n=======\nPlace order: {}\nCurrentValue:".format(self.CoinName))
		price = float(assetObj["c"][0])
		print("last Trade: {}".format(price))

		print("High 24h: {}".format(assetObj["h"][0]))
		print("Low 24h: {}".format(assetObj["l"][0]))
		print("Open 24h: {}".format(assetObj["o"][0]))
		print("Close 24h: {}".format(assetObj["c"][0]))
		print("Volum 24h {}".format(assetObj["v"][0]))

		print("we place an order of: ")
		print(BuyValue, "Euro")

		#Volume in XRP = 50 / 0.21245 = 235.34949399
		volum = BuyValue / price

		traidpair_buffer = str(self.ApiTraidPair)
		traidpair_buffer = traidpair_buffer.lower()

		api_method = "AddOrder"
		api_data = "pair=" + traidpair_buffer
		api_data += " type=" + "buy"
		api_data += " ordertype=" + "market"
		api_data += " volume=" + str(volum)

		#self.sendRequest(api_method,api_data)

		self.multiOrderPrvent +=  1

		#Set multiple Sell Limits
		#self.setSellLimit(price, 1.0, BuyValue, 1.02) 	#sell 25% after 20%profit
		#self.setSellLimit(price, 0.25, BuyValue, 1.60) 	#sell 25% after 30%profit
		#self.setSellLimit(price, 0.25, BuyValue, 2) 		#sell 25% after 200%profit
		#self.setSellLimit(price, 0.25, BuyValue, 10) 	#sell 25% after 1000%profit

		return True

	def setSellLimit(self, AssetPrice, AssetAmount, SellValue, SellProvit):
		volumeCoins = AssetPrice * AssetAmount
		PriceEuro = SellValue * SellProvit

		print("\n========\nSet Sell Limit: {}\nAmount of Asset: {}|{}".format(PriceEuro,volumeCoins,AssetAmount))

		traidpair_buffer = str(self.ApiTraidPair)
		traidpair_buffer = traidpair_buffer.lower()

		api_method = "AddOrder"
		api_data = "pair=" + traidpair_buffer
		api_data += " type=" + "sell"
		api_data += " ordertype=" + "take-profit"
		api_data += " price=" + str(PriceEuro)
		api_data += " volume=" + str(volumeCoins)

		self.sendRequest(api_method,api_data)




	def sendRequest(self, api_method, api_data):
		api_nonce = str(int(time.time()*1000))

		print("\nAPIdata: {}".format(api_data))

		api_postdata = api_data + "&nonce=" + api_nonce
		api_postdata = api_postdata.encode('utf-8')
		api_sha256 = hashlib.sha256(api_nonce.encode('utf-8') + api_postdata).digest()
		api_hmacsha512 = hmac.new(self.api_secret, self.api_path_private.encode('utf-8') + api_method.encode('utf-8') + api_sha256, hashlib.sha512)

		api_request = urllib2.Request(self.api_domain + self.api_path_private + api_method, api_postdata)
		api_request.add_header("API-Key", self.api_key)
		api_request.add_header("API-Sign", base64.b64encode(api_hmacsha512.digest()))

		#print("\nSys arv: ")
		#print(sys.argv[0])
		#api_request.add_header("User-Agent", "Kraken REST API - %s" % sys.argv[0])
		api_request.add_header("User-Agent", "Kraken REST API - bublebuy")

		#Send the request
		print("\nBuy it now")

		try:
			api_reply = urllib2.urlopen(api_request).read().decode()
		except Exception as error:
			print("API call failed (%s)" % error)
			return False

		print(api_reply)
		if '"error":[]' in api_reply:
			print("\nSucess:")
			return True
		else:
			print("\nERROR:")
			return False



def main():
	k1 = krakentrading()
	k2 = krakentrading()
	k1.CoinName = "MANA"
	k2.CoinName = "AAVE"
	#k2.CoinName = "XBT"
	k1.basicSetup()
	k2.basicSetup()

	amountOfPlacedOrders = 0
	try:
		while amountOfPlacedOrders < 2:
			amountOfPlacedOrders += 1
			if k1.multiOrderPrvent <= 0:
				#k1.isRdy()
				k1.placeOrder(20)
				amountOfPlacedOrders = 0

			amountOfPlacedOrders += 1
			if k2.multiOrderPrvent <= 0:
				#k2.isRdy()
				k2.placeOrder(20)
				amountOfPlacedOrders = 0

			time.sleep(1)


	except KeyboardInterrupt:
		pass



if __name__ == "__main__":
    main()
