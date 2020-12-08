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

import pandas as pd

df = px.data.tips()
df2 = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv')

app = dash.Dash(__name__)

app.layout = html.Div([
    html.P("x-axis:"),
    dcc.Checklist(
        id='x-axis',
        options=[{'value': x, 'label': x}
                 for x in ['smoker', 'day', 'time', 'sex']],
        value=['time'],
        labelStyle={'display': 'inline-block'}
    ),
    html.P("y-axis:"),
    dcc.RadioItems(
        id='y-axis',
        options=[{'value': x, 'label': x}
                 for x in ['total_bill', 'tip', 'size']],
        value='total_bill',
        labelStyle={'display': 'inline-block'}
    ),
    dcc.Graph(id="box-plot"),
    dcc.Graph(id="graph-plot"),
])

@app.callback(
    Output("box-plot", "figure"),
    [Input("x-axis", "value"),
     Input("y-axis", "value")])
def generate_chart(x, y):
    fig = px.box(df, x=x, y=y)
    return fig

@app.callback(
    Output("graph-plot", "figure"),
    [Input("x-axis", "value"),
     Input("y-axis", "value")])
def generate_chart(x, y):
    fig = px.line(df2, x='Date', y='AAPL.High', title='Time Series with Range Slider and Selectors')
    fig.update_xaxes(
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
    return fig

#==============================================================
def main():
    print('=============\nSandbox: stockPlot.py')
    app.run_server(debug=True)

if __name__ == "__main__":
    main()
