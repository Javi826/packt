#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 20:02:30 2024
@author: javi
"""

import warnings
warnings.filterwarnings('ignore')

#%matplotlib inline

from pathlib import Path
from copy import deepcopy
import numpy as np
import pandas as pd

import sklearn
from sklearn.datasets import make_circles # To generate the dataset

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import TensorBoard

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D  # 3D plots

import seaborn as sns

# plotting style
sns.set_style('white')
# for reproducibility
np.random.seed(seed=42)

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
if gpu_devices:
    print('Using GPU')
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)
else:
    print('Using CPU')
    
    
results_path = Path('results')
if not results_path.exists():
    results_path.mkdir()

# Lista de valores de batch_size a probar
batch_size_values = [32, 64, 128, 256, 1024,2048,4000]

# Almacenar las métricas de precisión para cada configuración de batch_size
accuracy_history = []

for batch_size in batch_size_values:

    # generate data for the current batch_size
    X, y = make_circles(
        n_samples=50000,
        shuffle=True,
        factor=0.1,
        noise=0.1)

    # define outcome matrix
    Y = np.zeros((50000, 2))
    for c in [0, 1]:
        Y[y == c, c] = 1
    
    f'Shape of: X: {X.shape} | Y: {Y.shape} | y: {y.shape}'

    model = Sequential([
        Dense(units=3, input_shape=(2,), name='hidden'),
        Activation('sigmoid', name='logistic'),
        Dense(2, name='output'),
        Activation('softmax', name='softmax'),
    ])
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Entrenar el modelo con el número de epochs actual
    np.random.seed(42)
    tf.random.set_seed(42)
    training = model.fit(X, 
                         Y, 
                         epochs=50,
                         validation_split=0.2,
                         batch_size=batch_size, 
                         verbose=0)  # No imprimir detalles durante el entrenamiento

    # Almacenar la última precisión en la historia
    accuracy_history.append(training.history['accuracy'][-1])
    print(f'Última precisión para batch_size={batch_size}: {accuracy_history[-1]}')

# Visualizar la precisión en función de los valores de batch_size
plt.figure(figsize=(10, 6))
plt.plot(batch_size_values, accuracy_history, marker='o')
plt.xlabel('Batch Size')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Batch Size')
plt.grid(True)
plt.show()




    
    