#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 21:08:20 2024
@author: javi
"""

import warnings
warnings.filterwarnings('ignore')

#%matplotlib inline

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Input, concatenate, Embedding, Reshape, BatchNormalization
import tensorflow.keras.backend as K

import matplotlib.pyplot as plt
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

optimizer = tf.keras.optimizers.legacy.RMSprop(learning_rate=0.001,
                                               rho=0.9,
                                               epsilon=1e-08,
                                               decay=0.0)

rnn.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy', 
                     tf.keras.metrics.AUC(name='AUC')])

lstm_path = (results_path / 'lstm.classification.h5').as_posix()

checkpointer = ModelCheckpoint(filepath=lstm_path,
                               verbose=1,
                               monitor='val_AUC',
                               mode='max',
                               save_best_only=True)

early_stopping = EarlyStopping(monitor='val_AUC', 
                              patience=5,
                              restore_best_weights=True,
                              mode='max')

training = rnn.fit(X_train,
                   y_train,
                   epochs=50,
                   batch_size=32,
                   validation_data=(X_test, y_test),
                   callbacks=[early_stopping, checkpointer],
                   verbose=1)

loss_history = pd.DataFrame(training.history)

def which_metric(m):
    return m.split('_')[-1]

fig, axes = plt.subplots(ncols=3, figsize=(18,4))
for i, (metric, hist) in enumerate(loss_history.groupby(which_metric, axis=1)):
    hist.plot(ax=axes[i], title=metric)
    axes[i].legend(['Training', 'Validation'])

sns.despine()
fig.tight_layout()
fig.savefig(results_path / 'lstm_stacked_classification', dpi=300);

test_predict = pd.Series(rnn.predict(X_test).squeeze(), index=y_test.index)

roc_auc_score(y_score=test_predict, y_true=y_test)

((test_predict>.5) == y_test).astype(int).mean()

spearmanr(test_predict, y_test)[0]


