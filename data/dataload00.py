#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 15:21:21 2024

@author: javi
"""
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import requests
from io import BytesIO
from zipfile import ZipFile, BadZipFile

import numpy as np
import pandas as pd
import pandas_datareader.data as web
from sklearn.datasets import fetch_openml

pd.set_option('display.expand_frame_repr', False)
DATA_STORE = Path('assets00.h5')

df = (pd.read_csv('wiki_prices.csv',
                 parse_dates=['date'],
                 index_col=['date', 'ticker'],
                 infer_datetime_format=True)
     .sort_index())

df.info(show_counts=True)
with pd.HDFStore(DATA_STORE) as store:
    store.put('quandl/wiki/prices', df)

df = pd.read_csv('wiki_stocks.csv')

df.info(show_counts=True)
with pd.HDFStore(DATA_STORE) as store:
    store.put('quandl/wiki/stocks', df)