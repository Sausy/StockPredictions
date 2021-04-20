#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dez  30 10:53:41 2020

@author: sausy
"""

import numpy as np
import pandas as pd

import copy
import time

import sys

import matplotlib.pyplot as plt


class wallet(object):
    """docstring for wallet."""

    def __init__(self, baseAmount=10.0, baseTag="eur", minDelay=0):
        super(wallet, self).__init__()
        if baseTag == "euro":
            baseTag = "eur"
        #self.StartingBalance = baseAmount

        #============= [define Type] =======
        #basic config
        self.balance = 0.0 # the correct value will be set at the end of init
        self.tag = "na" # the correct value will be set at the end of init
        self.setTime = time.time()# the correct value will be set at the end of init

        self.minDelay = minDelay #min timeDelay after each trade in seconds

        #history data
        self.History = {
            "balance":[],
            "tag":[],
            "value":[],
            "time":[],
            "info":[]
        }

        #============= [init defaults] =======
        self.baseTag = baseTag #basetag is the base currency, which also can be displayed
        self.updateAfterTrade(baseAmount,baseTag)

    def canUpdate(self,tag):
        #access granted if current currency is the tagname
        #prevents trading with non existing currency
        if self.tag != tag:
            return False

        #check timing if min time delay passed to allow next trade
        if (time.time() - self.setTime) < self.minDelay:
            return False

        return True

    def updateWallet(self,balance,tag,exchangeFactor):
        if self.canUpdate(tag) != True:
            return False

        if tag == self.baseTag:
            value = self.balance * exchangeFactor
        else:
            value = self.balance / exchangeFactor

        #if data not present we can't push a hisory value
        if self.tag != "na":
            self.History["balance"].append(self.balance)
            self.History["tag"].append(self.tag)
            self.History["value"].append(value)
            self.History["time"].append(self.setTime)
            self.History["info"].append("n/a")

        self.balance = balance
        self.tag = tag
        self.setTime = time.time()

        return True


    def getBalance(self,exchangeFactor=-1.0):
        #the output is based on the initial baseTag
        if self.tag == self.baseTag:
            return self.balance

        if exchangeFactor < 0.0:
            return -1.0 #ERROR

        return self.balance * exchangeFactor
