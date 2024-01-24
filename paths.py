#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 17:14:40 2024

@author: javi
"""


import os

## Directorio para almacenar archivos CSV y assets
path_base = "/Users/javi/Desktop/ML/"
path_assets00 = '/Users/javi/Desktop/ML/assets/assets00.h5'
path_assets01 = '/Users/javi/Desktop/ML/assets/assets01.h5'


file_df_wiki = "wiki_prices.csv"
folder_wiki = "csv"
path_wiki_prices = os.path.join(path_base, folder_wiki, file_df_wiki)

file_df_stocks = "wiki_stocks.csv"
folder_stocks = "csv"
path_wiki_stocks = os.path.join(path_base, folder_stocks, file_df_stocks)

file_df_us = "us_equities_meta_data.csv"
folder_us = "csv"
path_wiki_us = os.path.join(path_base, folder_us, file_df_us)

