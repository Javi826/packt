#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 19:25:05 2024
@author: javi
"""
import numpy as np
from tensorflow.keras.layers import Embedding

# Supongamos que tienes 5 tickers únicos representados por números enteros
num_unique_tickers = 5

# Supongamos que cada ticker se representa en un espacio vectorial de dimensión 5
embedding_dim = 5

# Creamos un ejemplo de secuencia de tickers (números enteros)
ticker_sequence = [2, 0, 3, 1, 4, 2]

# Definimos la capa de embedding
embedding_layer = Embedding(input_dim=num_unique_tickers, output_dim=embedding_dim)

# Aplicamos la capa de embedding a la secuencia de tickers
embedded_sequence = embedding_layer(np.array(ticker_sequence))

# Visualizamos la secuencia original y la secuencia después de aplicar la capa de embedding
print("Secuencia original de tickers:", ticker_sequence)
print("Secuencia después de la capa de embedding:\n", embedded_sequence.numpy())
