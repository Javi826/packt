#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 14:06:26 2024
@author: javi
"""

from mod_init import *

def plot_correlogram(x, lags=None, title=None):
    lags = min(10, int(len(x)/5)) if lags is None else lags
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 8))
    
    # Plotting Time Series Data
    axes[0][0].set_title('Residuals')
    x.plot(ax=axes[0][0])
    x.rolling(21).mean().plot(ax=axes[0][0], c='k', lw=1)
    axes[0][0].set_ylim(-0.05, 0.05)  # Ajuste del rango del eje y
    q_p = np.max(q_stat(acf(x, nlags=lags), len(x))[1])
    stats = f'Q-Stat: {np.max(q_p):>8.2f}\nADF: {adfuller(x)[1]:>11.2f}'
    axes[0][0].text(x=.02, y=.85, s=stats, transform=axes[0][0].transAxes)
    
    # Plotting Probability Plot
    probplot(x, plot=axes[0][1])
    axes[0][1].set_ylim(-0.05, 0.05)  # Ajuste del rango del eje y
    
    # Plotting Moments Statistics
    mean, var, skew, kurtosis = moment(x, moment=[1, 2, 3, 4])
    s = f'Mean: {mean:>12.2f}\nSD: {np.sqrt(var):>16.2f}\nSkew: {skew:12.2f}\nKurtosis:{kurtosis:9.2f}'
    axes[0][1].text(x=.02, y=.75, s=s, transform=axes[0][1].transAxes)
    
    # Plotting Autocorrelation Function (ACF)
    plot_acf(x=x, lags=lags, zero=False, ax=axes[1][0])
    axes[1][0].set_xlabel('Lag')
    axes[1][0].set_ylim(-0.4, 0.4)  # Ajuste del rango del eje y
    
    # Plotting Partial Autocorrelation Function (PACF)
    plot_pacf(x, lags=lags, zero=False, ax=axes[1][1])
    axes[1][1].set_xlabel('Lag')
    axes[1][1].set_ylim(-0.4, 0.4)  # Ajuste del rango del eje y
    
    fig.suptitle(title, fontsize=14)
    sns.despine()
    fig.tight_layout()
    fig.subplots_adjust(top=.9)