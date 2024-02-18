#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 21:08:20 2024
@author: javi
"""

#%matplotlib inline

from pathlib import Path
import numpy as np
import pandas as pd

from scipy.stats import spearmanr

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Input, concatenate, Embedding, Reshape, BatchNormalization
import tensorflow.keras.backend as K

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns


gpu_devices = tf.config.experimental.list_physical_devices('GPU')
if gpu_devices:
    print('Using GPU')
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)
else:
    print('Using CPU')
    
idx = pd.IndexSlice
sns.set_style('whitegrid')
np.random.seed(42)

results_path = Path('results', 'lstm_embeddings')
if not results_path.exists():
    results_path.mkdir(parents=True)
    
data = pd.read_hdf('data.h5', 'returns_weekly')

#print(data.head(100))
#print(data.tail())

data['ticker'] = pd.factorize(data.index.get_level_values('ticker'))[0]

data['month'] = data.index.get_level_values('date').month
data = pd.get_dummies(data, columns=['month'], prefix='month')

#print(data.head(100))
#data.info()

window_size=52
sequence = list(range(1, window_size+1))
ticker = 1
months = 12
n_tickers = data.ticker.nunique()

#print(n_tickers)

train_data = data.drop('fwd_returns', axis=1).loc[idx[:, :'2016'], :]
test_data = data.drop('fwd_returns', axis=1).loc[idx[:, '2017'],:]

X_train = [
    train_data.loc[:, sequence].values.reshape(-1, window_size , 1),
    train_data.ticker,
    train_data.filter(like='month')
]
y_train = train_data.label
[x.shape for x in X_train], y_train.shape


# keep the last year for testing
X_test = [
    test_data.loc[:, list(range(1, window_size+1))].values.reshape(-1, window_size , 1),
    test_data.ticker,
    test_data.filter(like='month')
]
y_test = test_data.label
[x.shape for x in X_test], y_test.shape


K.clear_session()

n_features = 1

returns = Input(shape=(window_size, n_features),
                name='Returns')

tickers = Input(shape=(1,),
                name='Tickers')

months = Input(shape=(12,),
               name='Months')

lstm1_units = 25
lstm2_units = 10

lstm1 = LSTM(units=lstm1_units, 
             input_shape=(window_size, 
                          n_features), 
             name='LSTM1', 
             dropout=.2,
             return_sequences=True)(returns)

lstm_model = LSTM(units=lstm2_units, 
             dropout=.2,
             name='LSTM2')(lstm1)

ticker_embedding = Embedding(input_dim=n_tickers, 
                             output_dim=5, 
                             input_length=1)(tickers)
ticker_embedding = Reshape(target_shape=(5,))(ticker_embedding)

merged = concatenate([lstm_model, 
                      ticker_embedding, 
                      months], name='Merged')

bn = BatchNormalization()(merged)
hidden_dense = Dense(10, name='FC1')(bn)

output = Dense(1, name='Output', activation='sigmoid')(hidden_dense)

rnn = Model(inputs=[returns, tickers, months], outputs=output)

rnn.summary()

optimizer =tf.keras.optimizers.Adam()

rnn.compile(loss='mse',
            optimizer=optimizer)

lstm_path = (results_path / 'lstm.regression.h5').as_posix()


checkpointer = ModelCheckpoint(filepath=lstm_path,
                               verbose=1,
                               monitor='val_loss',
                               mode='min',
                               save_best_only=True)

early_stopping = EarlyStopping(monitor='val_loss', 
                              patience=5,
                              restore_best_weights=True)

training = rnn.fit(X_train,
                   y_train,
                   epochs=50,
                   batch_size=64,
                   validation_data=(X_test, y_test),
                   callbacks=[early_stopping, checkpointer],
                   verbose=1)

loss_history = pd.DataFrame(training.history)

test_predict = pd.Series(rnn.predict(X_test).squeeze(), index=y_test.index)

df = y_test.to_frame('ret').assign(y_pred=test_predict)

by_date = df.groupby(level='date')
df['deciles'] = by_date.y_pred.apply(pd.qcut, q=5, labels=False, duplicates='drop')

ic = by_date.apply(lambda x: spearmanr(x.ret, x.y_pred)[0]).mul(100)

df.info()

test_predict = test_predict.to_frame('prediction')
test_predict.index.names = ['symbol', 'date']
test_predict.to_hdf(results_path / 'predictions.h5', 'predictions')

rho, p = spearmanr(df.ret, df.y_pred)
print(f'{rho*100:.2f} ({p:.2%})')

fig, axes = plt.subplots(ncols=2, figsize=(14,4))
sns.barplot(x='deciles', y='ret', data=df, ax=axes[0])
axes[0].set_title('Weekly Fwd Returns by Predicted Quintile')
axes[0].yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.2%}'.format(y))) 
axes[0].set_ylabel('Weekly Returns')
axes[0].set_xlabel('Quintiles')

avg_ic = ic.mean()
title = f'4-Week Rolling IC | Weekly avg: {avg_ic:.2f} | Overall: {rho*100:.2f}'
ic.rolling(4).mean().dropna().plot(ax=axes[1], title=title)
axes[1].axhline(avg_ic, ls='--', c='k', lw=1)
axes[1].axhline(0, c='k', lw=1)
axes[1].set_ylabel('IC')
axes[1].set_xlabel('Date')

sns.despine()
fig.tight_layout()
fig.savefig(results_path / 'lstm_reg');