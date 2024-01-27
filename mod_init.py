#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 22:19:20 2024
@author: javi
"""
import pandas as pd
#VISUALIZATION PRINTS
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

#IGNORE PRINTS
import warnings 
warnings.filterwarnings('ignore')

import seaborn as sns
sns.set_style('whitegrid')
idx = pd.IndexSlice 

import os
import pandas as pd
import numpy as np
from numpy.linalg import LinAlgError
import pandas_datareader.data as web
from sklearn.datasets import fetch_openml

from pathlib import Path
import requests
from io import BytesIO
from zipfile import ZipFile, BadZipFile

import matplotlib.pyplot as plt

from scipy.stats import pearsonr, spearmanr
from talib import RSI, BBANDS, MACD, ATR


#ARIMA
from tqdm import tqdm
from itertools import product


import statsmodels.tsa.api as tsa
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, q_stat, adfuller
from scipy.stats import probplot, moment
from sklearn.metrics import mean_squared_error
