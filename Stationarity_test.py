import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('latest_pas_with_sas_with_street.csv')

# Create a datetime index
data['Date'] = pd.to_datetime(data['year'].astype(str) + 'Q' + data['quarter'].astype(str), errors='coerce')
data.set_index('Date', inplace=True)

# Drop duplicates in the index
data = data[~data.index.duplicated(keep='first')]


# Perform Dickey-Fuller test
def adf_test(series, title=''):
    """
    Pass in a time series and an optional title, returns an ADF report
    """
    print(f'Augmented Dickey-Fuller Test: {title}')
    result = adfuller(series.dropna(), autolag='AIC')
    labels = ['ADF Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used']
    out = pd.Series(result[0:4], index=labels)

    for key, value in result[4].items():
        out[f'Critical Value ({key})'] = value

    print(out.to_string())
    if result[1] <= 0.05:
        print("Strong evidence against the null hypothesis (reject H0), the series is stationary.")
    else:
        print("Weak evidence against the null hypothesis (fail to reject H0), the series is non-stationary.")
    print()


# Check stationarity of 'Trust MPS' and '"Good Job" local'
adf_test(data['Trust MPS'], 'Trust MPS')
adf_test(data['"Good Job" local'], '"Good Job" local')

# Plot the time series
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Trust MPS'], label='Trust MPS')
plt.plot(data.index, data['"Good Job" local'], label='"Good Job" local')
plt.title('Time Series of Trust MPS and "Good Job" local')
plt.legend()
plt.show()

# Perform Dickey-Fuller test on the first differences to check for stationarity
data['Trust MPS Diff'] = data['Trust MPS'].diff().dropna()
data['"Good Job" local Diff'] = data['"Good Job" local'].diff().dropna()

adf_test(data['Trust MPS Diff'], 'Trust MPS (Differenced)')
adf_test(data['"Good Job" local Diff'], '"Good Job" local (Differenced)')
