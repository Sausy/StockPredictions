#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 15:09:11 2020

@author: sausy

plots stuff in the web browser
http://localhost:8050/
"""

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as plt_go

import copy

import pandas as pd

import os
import glob #to get files with wildcard

class stockPlot:
    def __init__(self,debug=False):
        self.debug = debug


        self.app = dash.Dash(__name__)
        #self.df = StockData
        self.df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv')

        #self.fig = px.line(self.df, x='Date', y='High', title='Bitcoin to USD')
        self.fig = px.line(self.df, x='Date', y='AAPL.High', title='Bitcoin to USD')
        self.fig.update_xaxes(
        rangeslider_visible=True,
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            )
        )


        self.app.layout = html.Div([
            dcc.Graph(
                id='raw',
                figure=self.fig
            )
        ])



    def addCandle(self,data_,title):
        data = copy.deepcopy(data_)
        title = title + "Candl"

        fig_ = plt_go.Figure(data=[plt_go.Candlestick(x=data['date'],
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'])])

        #ps count e.g. 15min .... we want to observe 7days
        #count=  7days * 24hr * 60min = 10080
        fig_.update_xaxes(
        rangeslider_visible=True,
            rangeselector=dict(
                buttons=list([
                    dict(count=4320, label="1min", step="minute", stepmode="backward"),
                    dict(count=4320, label="15min", step="minute", stepmode="backward"),
                    dict(count=4320, label="60min", step="minute", stepmode="backward"),
                    dict(count=168, label="1hour", step="hour", stepmode="backward"),
                    dict(count=20, label="1day", step="day", stepmode="backward"),
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            )
        )

        temp = html.Div([
            dcc.Graph(
                id=title,
                figure=fig_
            )
        ])

        layoutBuffer = self.app.layout.children
        layoutBuffer.append(temp)
        self.app.layout = html.Div(layoutBuffer)


    def addPlot(self,data_,ColumIdent,title):
        df2 = copy.deepcopy(data_)

        fig_ = px.line(df2, x='date', y=ColumIdent, title=title)
        fig_.update_xaxes(
        rangeslider_visible=True,
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            )
        )

        temp = html.Div([
            dcc.Graph(
                id=title,
                figure=fig_
            )
        ])

        layoutBuffer = self.app.layout.children
        layoutBuffer.append(temp)
        self.app.layout = html.Div(layoutBuffer)

    def includeLinGraphToCandle(self):
        pass

    #===========================================================
    def dbgPrint(self,outData):
        txt = "[PreProcessData]"
        if self.debug == True:
            txt = txt + "[DBG]"

        txt = txt + " "
        print(txt,end='')
        print(outData)

    #===========================================================
    def plotNow(self):
        self.app.run_server(debug=True)




#==============================================================
def main():
    print('=============\nSandbox: stockPlot.py')
    splt = stockPlot()

    basePath = "../CSV/"
    rdyCSV = basePath + "rdyCSV/"
    listPredictedCsv = glob.glob(rdyCSV + "predicted*.csv")
    listEvalCsv = glob.glob(rdyCSV + "eval*.csv")
    #if it needs to be more complicated
    #files = [f for f in os.listdir(rdyCSV) if re.match(r'[0-9]+.*\.csv', f)]

    candlData = ['date','open','high','low']

    for pred in listPredictedCsv:
        df = pd.read_csv(pred)
        splt.addCandle(df, str(pred))

    for ev in listEvalCsv:
        df = pd.read_csv(ev)
        splt.addCandle(df, str(ev))



    splt.plotNow()

if __name__ == "__main__":
    main()
