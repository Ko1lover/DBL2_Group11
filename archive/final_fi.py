import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load data
data = pd.read_csv('latest_pas_with_sas_with_street.csv')

# Create a datetime index
data['Date'] = pd.to_datetime(data['year'].astype(str) + 'Q' + data['quarter'].astype(str))
data.set_index('Date', inplace=True)

# Select numeric features and exclude 'year', 'quarter'
numeric_features = data.select_dtypes(include=[np.number]).columns.tolist()
numeric_features.remove('year')
numeric_features.remove('quarter')

# Initialize MinMaxScaler and scale numeric features
scaler = MinMaxScaler()
data[numeric_features] = scaler.fit_transform(data[numeric_features])

# Split the data for predictions (to distinguish training data)
train_data = data[:'2019']
test_data = data['2020':]

# Get unique boroughs
boroughs = data['borough'].unique()

# Analysis for each borough
for borough in boroughs:
    print(f"Processing {borough}...")
    borough_data = data[data['borough'] == borough]

    # Define features
    features = borough_data.columns.drop(['borough', 'Trust MPS', '"Good Job" local'])
    # Train RandomForest models for Trust MPS and calculate feature importances
    rf_trust = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_trust.fit(borough_data[features][:'2019'], borough_data['Trust MPS'][:'2019'])
    trust_importances = rf_trust.feature_importances_
    indices_trust = np.argsort(trust_importances)[-10:]  # Top 10 features

    plt.figure(figsize=(12, 6))
    plt.title(f'Top 10 Feature Importances for Trust MPS in {borough}')
    plt.barh(np.arange(10), trust_importances[indices_trust], align='center', color='blue')
    plt.yticks(np.arange(10), [features[i] for i in indices_trust])
    plt.xlabel('Feature Importance')
    plt.show()

    # Train RandomForest models for "Good Job" local and calculate feature importances
    rf_confidence = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_confidence.fit(borough_data[features][:'2019'], borough_data['"Good Job" local'][:'2019'])
    confidence_importances = rf_confidence.feature_importances_
    indices_confidence = np.argsort(confidence_importances)[-10:]  # Top 10 features

    plt.figure(figsize=(12, 6))
    plt.title(f'Top 10 Feature Importances for "Good Job" local in {borough}')
    plt.barh(np.arange(10), confidence_importances[indices_confidence], align='center', color='red')
    plt.yticks(np.arange(10), [features[i] for i in indices_confidence])
    plt.xlabel('Feature Importance')
    plt.show()
    # SARIMA models trained on data until 2019
    sarima_trust_until_2019 = SARIMAX(train_data['Trust MPS'][train_data['borough'] == borough],
                                      exog=train_data[features][train_data['borough'] == borough],
                                      order=(1, 1, 1),
                                      seasonal_order=(1, 1, 1, 4))
    sarima_trust_fit_until_2019 = sarima_trust_until_2019.fit(disp=False)
    sarima_confidence_until_2019 = SARIMAX(train_data['"Good Job" local'][train_data['borough'] == borough],
                                          exog=train_data[features][train_data['borough'] == borough],
                                          order=(1, 1, 1),
                                          seasonal_order=(1, 1, 1, 4))
    sarima_confidence_fit_until_2019 = sarima_confidence_until_2019.fit(disp=False)

    # SARIMA models trained on the full data
    sarima_trust_full = SARIMAX(borough_data['Trust MPS'],
                                exog=borough_data[features],
                                order=(1, 1, 1),
                                seasonal_order=(1, 1, 1, 4))
    sarima_trust_fit_full = sarima_trust_full.fit(disp=False)
    sarima_confidence_full = SARIMAX(borough_data['"Good Job" local'],
                                     exog=borough_data[features],
                                     order=(1, 1, 1),
                                     seasonal_order=(1, 1, 1, 4))
    sarima_confidence_fit_full = sarima_confidence_full.fit(disp=False)
    # Validation predictions
    test_exog = test_data[features][test_data['borough'] == borough]
    trust_predictions = sarima_trust_fit_until_2019.get_forecast(steps=len(test_exog), exog=test_exog).predicted_mean
    confidence_predictions = sarima_confidence_fit_until_2019.get_forecast(steps=len(test_exog),
                                                                           exog=test_exog).predicted_mean

    # Calculate validation metrics
    mse_trust = mean_squared_error(test_data['Trust MPS'][test_data['borough'] == borough], trust_predictions)
    mae_trust = mean_absolute_error(test_data['Trust MPS'][test_data['borough'] == borough], trust_predictions)
    rmse_trust = np.sqrt(mse_trust)

    mse_confidence = mean_squared_error(test_data['"Good Job" local'][test_data['borough'] == borough],
                                        confidence_predictions)
    mae_confidence = mean_absolute_error(test_data['"Good Job" local'][test_data['borough'] == borough],
                                         confidence_predictions)
    rmse_confidence = np.sqrt(mse_confidence)

    print(f"Validation Metrics for Trust MPS in {borough}: MAE={mae_trust}, MSE={mse_trust}, RMSE={rmse_trust}")
    print(
        f"Validation Metrics for 'Good Job' Local in {borough}: MAE={mae_confidence}, MSE={mse_confidence}, RMSE={rmse_confidence}")

    # Plotting validation results
    plt.figure(figsize=(14, 6))
    plt.plot(test_data.index[test_data['borough'] == borough], test_data['Trust MPS'][test_data['borough'] == borough],
             label='Actual Trust MPS')
    plt.plot(test_exog.index, trust_predictions, label='Predicted Trust MPS', color='red')
    plt.title(f'Validation of Trust MPS Predictions for {borough}')
    plt.legend()
    plt.show()

    plt.figure(figsize=(14, 6))
    plt.plot(test_data.index[test_data['borough'] == borough],
             test_data['"Good Job" local'][test_data['borough'] == borough], label='Actual "Good Job" Local')
    plt.plot(test_exog.index, confidence_predictions, label='Predicted "Good Job" Local', color='red')
    plt.title(f'Validation of "Good Job" Local Predictions for {borough}')
    plt.legend()
    plt.show()

    # Predictions for future data from 2020 to one year into the future using data up to 2019
    future_dates_until_2019 = pd.date_range(start='2020-01-01', periods=20, freq='Q')
    future_exog_until_2019 = pd.DataFrame(np.tile(train_data[features][train_data['borough'] == borough].iloc[-1].values, (len(future_dates_until_2019), 1)), columns=features)
    future_trust_predictions_until_2019 = sarima_trust_fit_until_2019.get_forecast(steps=len(future_dates_until_2019), exog=future_exog_until_2019).predicted_mean
    future_confidence_predictions_until_2019 = sarima_confidence_fit_until_2019.get_forecast(steps=len(future_dates_until_2019), exog=future_exog_until_2019).predicted_mean

    # Predictions for future data using the full dataset
    future_dates_full = pd.date_range(start=data.index[-1], periods=5, freq='Q')[1:]
    future_exog_full = pd.DataFrame(np.tile(test_data[features].iloc[-1].values, (len(future_dates_full), 1)), columns=features)
    future_trust_predictions_full = sarima_trust_fit_full.get_forecast(steps=len(future_dates_full), exog=future_exog_full).predicted_mean
    future_confidence_predictions_full = sarima_confidence_fit_full.get_forecast(steps=len(future_dates_full), exog=future_exog_full).predicted_mean

    # Plot Trust Predictions including future forecast from both models
    plt.figure(figsize=(14, 6))
    plt.plot(borough_data.index, borough_data['Trust MPS'], label='Actual Trust MPS')
    plt.plot(future_dates_until_2019, future_trust_predictions_until_2019, label='Future Predicted Trust MPS (Up to 2019)', color='red')
    plt.plot(future_dates_full, future_trust_predictions_full, label='Future Predicted Trust MPS (Full Data)', color='green')
    plt.title(f'Trust MPS Predictions and Future Forecast for {borough}')
    plt.ylim(0, 1)
    plt.legend()
    plt.show()

    # Plot Confidence Predictions including future forecast from both models
    plt.figure(figsize=(14, 6))
    plt.plot(borough_data.index, borough_data['"Good Job" local'], label='Actual "Good Job" local')
    plt.plot(future_dates_until_2019, future_confidence_predictions_until_2019, label='Future Predicted "Good Job" Local (Up to 2019)', color='red')
    plt.plot(future_dates_full, future_confidence_predictions_full, label='Future Predicted "Good Job" Local (Full Data)', color='green')
    plt.title(f'"Good Job" local Predictions and Future Forecast for {borough}')
    plt.ylim(0, 1)
    plt.legend()
    plt.show()
