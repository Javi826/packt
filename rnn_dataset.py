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
          .adj_close
          .unstack().loc['2007':])
#prices.info()

prices.head(1000).to_excel('prices_rnn_data.xlsx')

#WEEKLY------------------------------------------------------------------------
#WEEKLY------------------------------------------------------------------------

returns = (prices
           .resample('W')
           .last()
           .pct_change()
           .loc['2008': '2017']
           .dropna(axis=1)
           .sort_index(ascending=False))
returns.info()


#returns.head(1000).reset_index().to_excel('returns_rnn_data.xlsx', index=True)
print(returns.head())

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
#data.info()

data[tcols] = (data[tcols].apply(lambda x: x.clip(lower=x.quantile(.01),
                                                  upper=x.quantile(.99))))

data = data.rename(columns={0: 'fwd_returns'})

#print(data)
data['label'] = (data['fwd_returns'] > 0).astype(int)

data.shape

data.sort_index().to_hdf('data.h5', 'returns_weekly')

data.head(1000).to_excel('returns_weekly_rnn_data.xlsx')

print(data.head())
