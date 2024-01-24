#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 15:47:56 2024
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
DATA_STORE = Path('assets01.h5')

df = (pd.read_csv('wiki_prices.csv',
                 parse_dates=['date'],
                 index_col=['date', 'ticker'],
                 infer_datetime_format=True)
     .sort_index())

print(df.info)
with pd.HDFStore(DATA_STORE) as store:
    store.put('quandl/wiki/prices', df)

df = pd.read_csv('us_equities_meta_data.csv')

print(df.info)
with pd.HDFStore(DATA_STORE) as store:
    store.put('us_equities/stocks', df)