#%matplotlib inline

from mod_init import *
from paths import path_assets01,path_assets00

MONTH = 21
YEAR = 12 * MONTH
START = '2013-01-01'
END = '2017-12-31'
   
    
ohlcv = ['adj_open', 'adj_close', 'adj_low', 'adj_high', 'adj_volume']
    
DATA_STORE = (path_assets01)  


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

# keep those that exceed the limit
keep = nobs[nobs > min_obs].index

prices = prices.loc[idx[keep, :], :]

#print(prices)
stocks = stocks[~stocks.index.duplicated() & stocks.sector.notnull()]
stocks.sector = stocks.sector.str.lower().str.replace(' ', '_')
stocks.index.name = 'ticker'
#print(stocks)

#shared = (prices.index.get_level_values('ticker').unique()
#          .intersection(stocks.index))
#print(shared)
#stocks = stocks.loc[shared, :]
#prices = prices.loc[idx[shared, :], :]


# compute dollar volume to determine universe
prices['dollar_vol'] = prices[['close', 'volume']].prod(axis=1)
prices['dollar_vol_1m'] = (prices.dollar_vol.groupby('ticker')
                           .rolling(window=21, level='date')
                           .mean()).values

prices.info(show_counts=True)



    
    
    
    
    