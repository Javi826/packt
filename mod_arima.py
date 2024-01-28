#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 13:03:49 2024
@author: javi
"""
from mod_init import *
import statsmodels as sm
from statsmodels.tsa.arima.model import ARIMA
from def_functions import plot_correlogram

industrial_production = web.DataReader('IPGMFN', 'fred', '1988', '2017-12').squeeze().dropna()
nasdaq = web.DataReader('NASDAQCOM', 'fred', '1990', '2017-12-31').squeeze().dropna()

#LOG transform
nasdaq_log = np.log(nasdaq)
industrial_production_log = np.log(industrial_production)

#DIFFERENCING
nasdaq_log_diff = nasdaq_log.diff().dropna()
industrial_production_log_diff = industrial_production_log.diff(12).dropna()

#model1 = tsa.ARIMA(endog=nasdaq_log_diff, order=(2,0,2)).fit()
model2 = tsa.ARIMA(endog=nasdaq_log, order=(2,1,2)).fit()

model2.params.sort_index() == model2.params.sort_index().values

components = tsa.seasonal_decompose(industrial_production, model='additive')

ts = (industrial_production.to_frame('Original')
      .assign(Trend=components.trend)
      .assign(Seasonality=components.seasonal)
      .assign(Residual=components.resid))
with sns.axes_style('white'):
    ts.plot(subplots=True, figsize=(14, 8), title=['Original Series', 'Trend Component', 'Seasonal Component','Residuals'], legend=False)
    plt.suptitle('Seasonal Decomposition', fontsize=14)
    sns.despine()
    plt.tight_layout()
    plt.subplots_adjust(top=.91);
    
(nasdaq == 0).any(), (industrial_production==0).any()
    
plot_correlogram(nasdaq_log_diff, lags=100, title='NASDAQ Composite (Log, Diff)')

train_size = 120
results = {}
y_true = industrial_production_log_diff.iloc[train_size:]

for p in range(5):
    for q in range(5):
        aic, bic = [], []
        if p == 0 and q == 0:
            continue
        print(p, q)
        convergence_error = stationarity_error = 0
        y_pred = []

        for T in range(train_size, len(industrial_production_log_diff)):
            train_set = industrial_production_log_diff.iloc[T - train_size:T]

            try:
                # Use ARIMA with differencing (d=1)
                model = ARIMA(train_set, order=(p, 1, q)).fit()

                # Forecast only one step ahead
                forecast = model.get_forecast(steps=1)

                # Extract the predicted value after differencing
                y_pred.append(forecast.predicted_mean.iloc[0])

                aic.append(model.aic)
                bic.append(model.bic)

            except LinAlgError:
                convergence_error += 1
            except ValueError:
                stationarity_error += 1

        result = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})

        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(result['y_true'], result['y_pred']))

        # Calculate mean AIC and BIC
        avg_aic = np.mean(aic)
        avg_bic = np.mean(bic)

        results[(p, q)] = [rmse, avg_aic, avg_bic, convergence_error, stationarity_error]

# Print or use the results as needed
for key, value in results.items():
    print(f"Order: {key}, RMSE: {value[0]}, Avg AIC: {value[1]}, Avg BIC: {value[2]}, Conv. Errors: {value[3]}, Stat. Errors: {value[4]}")


arima_results = pd.DataFrame(results).T
arima_results.columns = ['RMSE', 'AIC', 'BIC', 'convergence', 'stationarity']
arima_results.index.names = ['p', 'q']
arima_results.info()

with pd.HDFStore('arima.h5') as store:
    store.put('arima', arima_results)
    
fig, axes = plt.subplots(ncols=2, figsize=(10,4), sharex=True, sharey=True)
sns.heatmap(arima_results[arima_results.RMSE<.5].RMSE.unstack().mul(10), fmt='.3f', annot=True, cmap='Blues', ax=axes[0], cbar=False);
sns.heatmap(arima_results.BIC.unstack(), fmt='.2f', annot=True, cmap='Blues', ax=axes[1], cbar=False)
axes[0].set_title('Root Mean Squared Error')
axes[1].set_title('Bayesian Information Criterion')
fig.tight_layout();


arima_results.rank().loc[:, ['RMSE', 'BIC']].mean(1).nsmallest(5)

# Calculate the mean rank of RMSE and BIC and find the index of the minimum value
best_p, best_q = arima_results.rank().loc[:, ['RMSE', 'BIC']].mean(1).idxmin()

# Fit the best ARIMA model (with differencing, d=1)
best_arima_model = ARIMA(endog=industrial_production_log_diff, order=(best_p, 1, best_q)).fit()

# Print the summary of the best ARIMA model
print(best_arima_model.summary())

plot_correlogram(best_arima_model.resid)

    

    
  





















