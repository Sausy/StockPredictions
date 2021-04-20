# StockPredictions

## setup
I got major issues running tensorflow on windows, the only way to proberly run it
was in the Anaconda promt

It would only work with python3.6
```
conda create -n tf-gpu tensorflow-gpu python3.6
conda activate tf-gpu
pip install sklearn plotly dash matplotlib tqdm pandas numpy
```

## hotfix tqdm
to fix the position bug of tqdm on windows
```
pip install colorama
```

## usage
### create a usable csv file
could ether be
1min, 15min, 60min, 1D, 3D, 10D
```
python3.6 createCSV.py XBT EUR 15min
```
### Train the system
python3.6 main.py ./CSV/rdyCSV/rdy.CSV


## Kraken Spezific fee calculation
fee 0.16/0.26 means buy/sell fee

we want to buy 182.25 with a leverage of x2 at a price of 24900.0
```
%-----Buying
fee = buyValue * leverage * ((1-buyFee)+(1-margingFee))
TradingValue = (buyValue - fee) * leverage
```
Assuming we want at least 3.0% win  
```
minWin = (100.0+3.0)/100.0
```
Selling 
```
%-----Selling
%rateAtWin = transValue * minWin
rateAtWin = transValue * minWin

SellLimit = buyValue * leverage * minWin
SellLimit - SellLimit * margingFee
SellFeeValue = SellLimit * (1 -  sellFee)

AfterTradeBalance = SellLimit - ( buyValue * leverage - buyValue )
AfterTradeBalance - SellFeeValue/2 - fee

%-----Selling
winValue = AfterTradeBalance - buyValue
```
