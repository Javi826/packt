#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 12:30:04 2024

@author: javi
"""

import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# spacy, textblob and nltk for language processing
from textblob import TextBlob

# sklearn for feature extraction & modeling
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score

sns.set_style('white')

data_path = Path('..', 'data', 'sentiment140')
if not data_path.exists():
    data_path.mkdir(parents=True)
    
names = ['polarity', 'id', 'date', 'query', 'user', 'text']

def load_train_data():
    parquet_file = data_path / 'train.parquet'
    if not parquet_file.exists():
        df = (pd.read_csv(data_path / 'train.csv',
                          low_memory=False,
                          encoding='latin1',
                          header=None,
                          names=names,
                          parse_dates=['date'])
              .drop(['id', 'query'], axis=1)
              .drop_duplicates(subset=['polarity', 'text']))
        df = df[df.text.str.len() <= 140]
        df.polarity = (df.polarity > 0).astype(int)
        df.to_parquet(parquet_file)
        return df
    else:
        return pd.read_parquet(parquet_file)
    
train = load_train_data()
train.info(null_counts=True)