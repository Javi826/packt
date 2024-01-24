    
import warnings
warnings.filterwarnings('ignore')

#%matplotlib inline

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import pearsonr, spearmanr
from talib import RSI, BBANDS, MACD, ATR

MONTH = 21
YEAR = 12 * MONTH
START = '2013-01-01'
END = '2017-12-31'
 
sns.set_style('whitegrid')
idx = pd.IndexSlice   
    
ohlcv = ['adj_open', 'adj_close', 'adj_low', 'adj_high', 'adj_volume']
    
DATA_STORE = '../data/assets01.h5'  


with pd.HDFStore(DATA_STORE) as store:
    prices = (store['quandl/wiki/prices']
              .loc[idx[START:END, :], ohlcv]
              .rename(columns=lambda x: x.replace('adj_', ''))
              .assign(volume=lambda x: x.volume.div(1000))
              .swaplevel()
              .sort_index())


    stocks = (store['us_equities/stocks']
              .loc[:, ['marketcap', 'ipoyear', 'sector']])
       
# want at least 2 years of data
min_obs = 2 * YEAR

# have this much per ticker 
nobs = prices.groupby(level='ticker').size()
#print(nobs)
#print(stocks)

# keep those that exceed the limit
keep = nobs[nobs > min_obs].index

prices = prices.loc[idx[keep, :], :]

stocks = stocks[~stocks.index.duplicated() & stocks.sector.notnull()]
stocks.sector = stocks.sector.str.lower().str.replace(' ', '_')
stocks.index.name = 'ticker'
#print(stocks)
#prices.info(show_counts=True)

#print(prices.index.get_level_values('ticker').unique())
#print(stocks.index.get_level_values('ticker').unique())

print("Índice de 'prices':")
print(prices.index)

print("\nÍndice de 'stocks':")
print(stocks.index)

shared = (prices.index.get_level_values('ticker').unique()
          .intersection(stocks.index))

print("Símbolos compartidos:")
print(shared)
stocks = stocks.loc[shared, :]
prices = prices.loc[idx[shared, :], :]
#print("DataFrame 'prices' después de cargar los datos:")
#print(prices.head())
#print("\nDataFrame 'stocks' después de cargar los datos:")
#print(stocks.head())
#prices.info(show_counts=True)

#stocks.info(show_counts=True)
    
    
    
    
    