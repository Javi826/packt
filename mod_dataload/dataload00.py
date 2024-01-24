#!/usr/bin/env python3
"""
Created on Sat Jan 20 15:21:21 2024
@author: javi
"""

from mod_init import *
from paths import path_wiki_prices,path_wiki_stocks,path_assets00


DATA_STORE = Path(path_assets00)

df_wiki = pd.read_csv(path_wiki_prices, parse_dates=['date'], index_col=['date', 'ticker'], infer_datetime_format=True).sort_index()

df_wiki.info(show_counts=True)
with pd.HDFStore(DATA_STORE) as store:
    store.put('quandl/wiki/prices', df_wiki)

df_stocks = pd.read_csv(path_wiki_stocks)

df_stocks.info(show_counts=True)
with pd.HDFStore(DATA_STORE) as store:
    store.put('quandl/wiki/stocks', df_stocks)