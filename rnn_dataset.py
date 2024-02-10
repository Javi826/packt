#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 21:13:55 2024

@author: javi
"""

import warnings
warnings.filterwarnings('ignore')

from pathlib import Path

import numpy as np
import pandas as pd

np.random.seed(42)
idx = pd.IndexSlice

DATA_DIR = Path('/Users/javi/Desktop/ML/assets/')

prices = (pd.read_hdf(DATA_DIR / 'assets00.h5', 'quandl/wiki/prices')
          .loc[idx['2010':'2017', :], ['adj_close', 'adj_volume']])
prices.info()


n_dates = len(prices.index.unique('date'))
dollar_vol = (prices.adj_close.mul(prices.adj_volume)
              .unstack('ticker')
              .dropna(thresh=int(.95 * n_dates), axis=1)
              .rank(ascending=False, axis=1)
              .stack('ticker'))

most_traded = dollar_vol.groupby(level='ticker').mean().nsmallest(500).index

#print(dollar_vol.head())

#print(list(most_traded))

returns = (prices.loc[idx[:, most_traded], 'adj_close']
           .unstack('ticker')
           .pct_change()
           .sort_index(ascending=False))
returns.info()

#print(returns.head())

n = len(returns)
T = 21 # days
tcols = list(range(T))
tickers = returns.columns

data = pd.DataFrame()
#print(data.head())

for i in range(n-T-1):
    df = returns.iloc[i:i+T+1]
    date = df.index.max()
    data = pd.concat([data, 
                      df.reset_index(drop=True).T
                      .assign(date=date, ticker=tickers)
                      .set_index(['ticker', 'date'])])
data = data.rename(columns={0: 'label'}).sort_index().dropna()
data.loc[:, tcols[1:]] = (data.loc[:, tcols[1:]].apply(lambda x: x.clip(lower=x.quantile(.01),
                                                  upper=x.quantile(.99))))
data.info()

data.shape

data.to_hdf('data.h5', 'returns_daily')

prices = (pd.read_hdf(DATA_DIR / 'assets00.h5', 'quandl/wiki/prices')
          .adj_close
          .unstack().loc['2007':])
prices.info()

returns = (prices
           .resample('W')
           .last()
           .pct_change()
           .loc['2008': '2017']
           .dropna(axis=1)
           .sort_index(ascending=False))
returns.info()

#print(returns.head())

n = len(returns)
T = 52 # weeks
tcols = list(range(T))
tickers = returns.columns

data = pd.DataFrame()
for i in range(n-T-1):
    df = returns.iloc[i:i+T+1]
    date = df.index.max()    
    data = pd.concat([data, (df.reset_index(drop=True).T
                             .assign(date=date, ticker=tickers)
                             .set_index(['ticker', 'date']))])
data.info()

data[tcols] = (data[tcols].apply(lambda x: x.clip(lower=x.quantile(.01),
                                                  upper=x.quantile(.99))))

data = data.rename(columns={0: 'fwd_returns'})


data['label'] = (data['fwd_returns'] > 0).astype(int)

data.shape

data.sort_index().to_hdf('data.h5', 'returns_weekly')
