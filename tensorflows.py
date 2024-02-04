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


# dataset params
N = 50000
factor = 0.1
noise = 0.1

# generate data
X, y = make_circles(
    n_samples=N,
    shuffle=True,
    factor=factor,
    noise=noise)

# define outcome matrix
Y = np.zeros((N, 2))
for c in [0, 1]:
    Y[y == c, c] = 1
    
f'Shape of: X: {X.shape} | Y: {Y.shape} | y: {y.shape}'


#sns.scatterplot(x=X[:, 0], 
#                y=X[:, 1], 
#                hue=y,
#               style=y,
#               markers=['_', '+']);

# Definir valores de epochs a probar
epochs_values = [1,2,3,4,5,6,7,8,9,10, 25, 50]

model = Sequential([
    Dense(units=3, input_shape=(2,), name='hidden'),
    Activation('sigmoid', name='logistic'),
    Dense(2, name='output'),
    Activation('softmax', name='softmax'),
])
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])


# Almacenar las métricas de precisión para cada configuración de epochs
accuracy_history = []

for epochs in epochs_values:

    # Entrenar el modelo con el número de epochs actual
    np.random.seed(42)
    tf.random.set_seed(42)
    training = model.fit(X, 
                         Y, 
                         epochs=epochs,
                         validation_split=0.2,
                         batch_size=128, 
                         verbose=0)  # No imprimir detalles durante el entrenamiento

    # Almacenar la última precisión en la historia
    accuracy_history.append(training.history['accuracy'][-1])
    print(f'Última precisión después de {epochs} epochs: {accuracy_history[-1]}')

# Visualizar la precisión en función del número de epochs
plt.figure(figsize=(10, 6))
plt.plot(epochs_values, accuracy_history, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Number of Epochs')
plt.grid(True)
plt.show()




    
    