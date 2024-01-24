#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 15:47:56 2024
@author: javi
"""

from mod_init import *
from paths import path_wiki_prices,path_wiki_us,path_assets01


DATA_STORE = Path(path_assets01)

df_wiki = pd.read_csv(path_wiki_prices, parse_dates=['date'], index_col=['date', 'ticker'], infer_datetime_format=True).sort_index()

df_wiki.info(show_counts=True)
with pd.HDFStore(DATA_STORE) as store:
    store.put('quandl/wiki/prices', df_wiki)

df_us = pd.read_csv(path_wiki_us)

df_us.info(show_counts=True)
with pd.HDFStore(DATA_STORE) as store:
    store.put('us_equities/stocks', df_us)