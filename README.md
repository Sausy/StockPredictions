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
