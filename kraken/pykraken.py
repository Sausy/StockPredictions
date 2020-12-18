#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dez  8 10:53:41 2020

@author: sausy
"""

import numpy as np
import pandas as pd
import krakenex
from pykrakenapi import KrakenAPI


def main():
    import sys

    api = krakenex.API()
    k = KrakenAPI(api)

    ohlc, last = k.get_ohlc_data("MANAEUR")
    print(ohlc)
    ohlc, last = k.get_ohlc_data("AAVEEUR")
    print(ohlc)


if __name__ == "__main__":
    main()
