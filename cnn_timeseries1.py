#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 22:10:13 2024
@author: javi
"""

import warnings
warnings.filterwarnings('ignore')

from talib import (RSI, BBANDS, MACD,
                   NATR, WILLR, WMA,
                   EMA, SMA, CCI, CMO,
                   MACD, PPO, ROC,
                   ADOSC, ADX, MOM)
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.regression.rolling import RollingOLS
import statsmodels.api as sm
import pandas_datareader.data as web
import pandas as pd
import numpy as np
from pathlib import Path
#%matplotlib inline

DATA_STORE = '/Users/javi/Desktop/ML/assets/assets01.h5'

MONTH = 21
YEAR = 12 * MONTH

START = '2000-01-01'
END = '2017-12-31'

sns.set_style('whitegrid')
idx = pd.IndexSlice

T = [1, 5, 10, 21, 42, 63]

results_path = Path('results', 'cnn_for_trading')
if not results_path.exists():
    results_path.mkdir(parents=True)
    

adj_ohlcv = ['adj_open', 'adj_close', 'adj_low', 'adj_high', 'adj_volume']

with pd.HDFStore(DATA_STORE) as store:
    prices = (store['quandl/wiki/prices']
              .loc[idx[START:END, :], adj_ohlcv]
              .rename(columns=lambda x: x.replace('adj_', ''))
              .swaplevel()
              .sort_index()
             .dropna())
    metadata = (store['us_equities/stocks'].loc[:, ['marketcap', 'sector']])
ohlcv = prices.columns.tolist()

print("Primeros registros de la variable prices:")
print(prices.head())
print("Datos de la variable metadata:")
print(metadata.head())

prices.volume /= 1e3
prices.index.names = ['symbol', 'date']
metadata.index.name = 'symbol'


dollar_vol = prices.close.mul(prices.volume).unstack('symbol').sort_index()

years = sorted(np.unique([d.year for d in prices.index.get_level_values('date').unique()]))

print("Lista de años:")
print(years)

train_window = 5 # years
universe_size = 500

universe = []
for i, year in enumerate(years[5:], 5):
    start = str(years[i-5])
    end = str(years[i])
    most_traded = (dollar_vol.loc[start:end, :]
                   .dropna(thresh=1000, axis=1)
                   .median()
                   .nlargest(universe_size)
                   .index)
    universe.append(prices.loc[idx[most_traded, start:end], :])
universe = pd.concat(universe)

universe = universe.loc[~universe.index.duplicated()]

universe.groupby('symbol').size().describe()

print("DataFrame 'universe':")
print(universe)
print(universe.index)
universe.to_hdf('data.h5', 'universe')

T = list(range(6, 21))

for t in T:
    universe[f'{t:02}_WILLR'] = (universe.groupby(level='symbol', group_keys=False)
     .apply(lambda x: WILLR(x.high, x.low, x.close, timeperiod=t)))
    
for t in T:
    universe[f'{t:02}_EMA'] = universe.groupby(level='symbol').close.apply(EMA, timeperiod=t)
    
print(universe)


