#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 10:51:32 2024
@author: javi
"""

import sys
from time import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_regression

import tensorflow as tf
tf.autograph.set_verbosity(0, True)
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (Dense, 
                                     Flatten, 
                                     Conv1D, 
                                     MaxPooling1D, 
                                     Dropout, 
                                     BatchNormalization)

import matplotlib.pyplot as plt
import seaborn as sns

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
if gpu_devices:
    print('Using GPU')
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)
else:
    print('Using CPU')
    
sys.path.insert(1, Path(sys.path[0], '..').as_posix())
#from utils import MultipleTimeSeriesCV, format_time


np.random.seed(1)
tf.random.set_seed(1)

sns.set_style('whitegrid')

results_path = Path('results', 'time_series')
if not results_path.exists():
    results_path.mkdir(parents=True)
    
prices = (pd.read_hdf('/Users/javi/Desktop/ML/assets/assets01.h5', 'quandl/wiki/prices')
          .adj_close
          .unstack().loc['2000':])
prices.info()

print(prices.head())







