#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 22:25:29 2024
@author: javi
"""

import warnings
warnings.filterwarnings('ignore')

#%matplotlib inline

from pathlib import Path

import numpy as np
import pandas as pd
import pandas_datareader.data as web
from scipy.stats import spearmanr

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow import keras

import matplotlib.pyplot as plt
import seaborn as sns

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
if gpu_devices:
    print('Using GPU')
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)
else:
    print('Using CPU')
    
sns.set_style('whitegrid')
np.random.seed(42)

results_path = Path('results', 'univariate_time_series')
if not results_path.exists():
    results_path.mkdir(parents=True)
    
sp500 = web.DataReader('SP500', 'fred', start='2010', end='2020').dropna()
#ax = sp500.plot(title='S&P 500',
 #          legend=False,
  #         figsize=(14, 4),
   #        rot=0)
#ax.set_xlabel('')
#sns.despine()

print(sp500.head())
print(sp500.tail())

scaler = MinMaxScaler()

sp500_scaled = pd.Series(scaler.fit_transform(sp500).squeeze(), 
                         index=sp500.index)
sp500_scaled.describe()
#print(sp500_scaled.head())

#summary = sp500_scaled.describe().to_string()
#print(summary)

def create_univariate_rnn_data(data, window_size):
    n = len(data)
    y = data[window_size:]
    data = data.values.reshape(-1, 1) # make 2D
    X = np.hstack(tuple([data[i: n-j, :] for i, j in enumerate(range(window_size, 0, -1))]))
    return pd.DataFrame(X, index=y.index), y

window_size = 63

X, y = create_univariate_rnn_data(sp500_scaled, window_size=window_size)
X.to_excel('datos_X.xlsx')
#y.to_excel('datos_y.xlsx')

#print(X.head())

#print(y.head())
#print(y.tail())

axS = sp500_scaled.plot(lw=2, figsize=(14, 4), rot=0)
axS.set_xlabel('')
sns.despine()

