import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing



file_path = 'data/PAS_with_crime.csv'
data = pd.read_csv(file_path)


data['Date'] = pd.to_datetime(data['Date'])
data = data.set_index('Date')

# Define the trust and confidence columns based on the provided image
trust_columns = ['Treat everyone fairly', 'Listen to concerns', 'Relied on to be there', 'Trust MPS']
confidence_columns = ['Understand issues', 'Informed local', 'Contact ward officer', 'Good Job local']

# Weights will be adjusted
trust_weights = [0.25, 0.25, 0.20, 0.30]
confidence_weights = [0.15, 0.15, 0.15, 0.55]

# Ensure that we are working only with numeric data
numeric_columns = trust_columns + confidence_columns
data_numeric = data[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Calculate the weighted scores for trust and confidence
data['Trust'] = np.dot(data_numeric[trust_columns], trust_weights)
data['Confidence'] = np.dot(data_numeric[confidence_columns], confidence_weights)

# Convert the data to quarterly frequency using 'Q' for quarter end
data_quarterly = data[['Trust', 'Confidence']].resample('Q').mean() 


def fit_ets_model(time_series):
    model = ExponentialSmoothing(time_series, seasonal='additive', seasonal_periods=4)
    result = model.fit()
    return result

def fit_sarima_model(time_series):
    # Decompose the series to understand its components
    decomposition = seasonal_decompose(time_series, model='additive', period=4)
    decomposition.plot()
    plt.show()

    # Fit the SARIMA model
    model = SARIMAX(time_series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 4))
    result = model.fit(disp=False)
    return result


results_ets = {}
results_sarima = {}
for column in ['Trust', 'Confidence']:
    results_ets[column] = fit_ets_model(data_quarterly[column])
    results_sarima[column] = fit_sarima_model(data_quarterly[column])

# Display the summary of the trust models as examples
print("ETS Model Summary for Trust:")
print(results_ets['Trust'].summary())
print("\nSARIMA Model Summary for Trust:")
print(results_sarima['Trust'].summary())


forecast_steps = 8  # for next 8 quarters
for column in ['Trust', 'Confidence']:
    # ETS Forecast
    ets_forecast = results_ets[column].forecast(steps=forecast_steps)
    ets_forecast_index = pd.date_range(start=data_quarterly.index[-1], periods=forecast_steps + 1, freq='Q')[1:]
    ets_forecast_series = pd.Series(ets_forecast, index=ets_forecast_index)

    # SARIMA Forecast
    sarima_forecast = results_sarima[column].get_forecast(steps=forecast_steps)
    sarima_forecast_index = pd.date_range(start=data_quarterly.index[-1], periods=forecast_steps + 1, freq='Q')[1:]
    sarima_forecast_series = pd.Series(sarima_forecast.predicted_mean, index=sarima_forecast_index)

    plt.figure(figsize=(10, 6))
    plt.plot(data_quarterly[column], label='Observed')
    plt.plot(ets_forecast_series, label='ETS Forecast', color='blue')
    plt.plot(sarima_forecast_series, label='SARIMA Forecast', color='red')
    plt.title(f'Forecast for {column}')
    plt.legend()
    plt.show()