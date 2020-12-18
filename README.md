# StockPredictions

## setup
I got major issues running tensorflow on windows, the only way to proberly run it
was in the Anaconda promt

It would only work with python3.6
'''
conda create -n tf-gpu tensorflow-gpu python3.6
conda activate tf-gpu
pip install sklearn plotly dash matplotlib tqdm pandas
'''

## hotfix tqdm
to fix the position bug of tqdm on windows
'''
pip install colorama
'''
