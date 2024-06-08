import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Load data
data = pd.read_csv('data/latest_pas_with_sas_with_street.csv')

# Create a datetime index
data['Date'] = pd.to_datetime(data['year'].astype(str) + 'Q' + data['quarter'].astype(str))
data.set_index('Date', inplace=True)

# Normalize the numerical values
numeric_features = data.select_dtypes(include=[np.number]).columns.tolist()
numeric_features.remove('year')
numeric_features.remove('quarter')
scaler = StandardScaler()
data[numeric_features] = scaler.fit_transform(data[numeric_features])

# Split the data for predictions
train_data = data[:'2019']
test_data = data['2020':]

boroughs = data['borough'].unique()

# Function to get feature importances
def get_feature_importances(train_features, train_target):
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(train_features, train_target)
    return rf.feature_importances_

# Function to make weighted sum predictions
def make_weighted_predictions(features, weights):
    return np.dot(features, weights)

# Analysis for each borough
for borough in boroughs:
    borough_data = data[data['borough'] == borough]
    borough_train_data = train_data[train_data['borough'] == borough]
    borough_test_data = test_data[test_data['borough'] == borough]

    # Feature Importance for Trust MPS
    rf_trust = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_trust.fit(borough_train_data.drop(columns=['borough', 'Trust MPS', '"Good Job" local']), borough_train_data['Trust MPS'])
    trust_importances = rf_trust.feature_importances_
    top_trust_features_indices = np.argsort(trust_importances)[-10:]
    top_trust_features = borough_train_data.drop(columns=['borough', 'Trust MPS', '"Good Job" local']).columns[top_trust_features_indices]

    # Plotting top 10 feature importances for Trust
    plt.figure(figsize=(12, 6))
    plt.barh(top_trust_features, trust_importances[top_trust_features_indices])
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.title(f'Top 10 Feature Importances for Trust MPS in {borough}')
    plt.show()

    # Feature Importance for "Good Job" local
    rf_confidence = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_confidence.fit(borough_train_data.drop(columns=['borough', 'Trust MPS', '"Good Job" local']), borough_train_data['"Good Job" local'])
    confidence_importances = rf_confidence.feature_importances_
    top_confidence_features_indices = np.argsort(confidence_importances)[-10:]
    top_confidence_features = borough_train_data.drop(columns=['borough', 'Trust MPS', '"Good Job" local']).columns[top_confidence_features_indices]

    # Plotting top 10 feature importances for Confidence
    plt.figure(figsize=(12, 6))
    plt.barh(top_confidence_features, confidence_importances[top_confidence_features_indices])
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.title(f'Top 10 Feature Importances for "Good Job" local in {borough}')
    plt.show()

    # Trust Predictions and Feature Importances
    weighted_trust_predictions = make_weighted_predictions(borough_test_data[top_trust_features], trust_importances[top_trust_features_indices])

    # SARIMA models for forecasting from 2020 to 2023 for Trust using data until 2019
    sarima_model_trust_until_2019 = SARIMAX(borough_train_data['Trust MPS'], exog=borough_train_data[top_trust_features], order=(1, 1, 1), seasonal_order=(1, 1, 1, 4))
    sarima_model_trust_fit_until_2019 = sarima_model_trust_until_2019.fit(disp=False)
    future_trust_exog_until_2019 = borough_test_data[top_trust_features]
    future_trust_until_2019 = sarima_model_trust_fit_until_2019.get_forecast(steps=len(future_trust_exog_until_2019), exog=future_trust_exog_until_2019)
    future_trust_predictions_until_2019 = future_trust_until_2019.predicted_mean

    # SARIMA models for forecasting one year beyond 2024 for Trust using total data
    sarima_model_trust_total = SARIMAX(borough_data['Trust MPS'], exog=borough_data[top_trust_features], order=(1, 1, 1), seasonal_order=(1, 1, 1, 4))
    sarima_model_trust_fit_total = sarima_model_trust_total.fit(disp=False)
    future_trust_exog_total = pd.DataFrame(np.zeros((4, len(top_trust_features))), columns=top_trust_features)
    future_trust_total = sarima_model_trust_fit_total.get_forecast(steps=4, exog=future_trust_exog_total)
    future_trust_predictions_total = future_trust_total.predicted_mean

    # Confidence Predictions and Feature Importances
    weighted_confidence_predictions = make_weighted_predictions(borough_test_data[top_confidence_features], confidence_importances[top_confidence_features_indices])

    # SARIMA models for forecasting from 2020 to 2023 for Confidence using data until 2019
    sarima_model_confidence_until_2019 = SARIMAX(borough_train_data['"Good Job" local'], exog=borough_train_data[top_confidence_features], order=(1, 1, 1), seasonal_order=(1, 1, 1, 4))
    sarima_model_confidence_fit_until_2019 = sarima_model_confidence_until_2019.fit(disp=False)
    future_confidence_exog_until_2019 = borough_test_data[top_confidence_features]
    future_confidence_until_2019 = sarima_model_confidence_fit_until_2019.get_forecast(steps=len(future_confidence_exog_until_2019), exog=future_confidence_exog_until_2019)
    future_confidence_predictions_until_2019 = future_confidence_until_2019.predicted_mean

    # SARIMA models for forecasting one year beyond 2024 for Confidence using total data
    sarima_model_confidence_total = SARIMAX(borough_data['"Good Job" local'], exog=borough_data[top_confidence_features], order=(1, 1, 1), seasonal_order=(1, 1, 1, 4))
    sarima_model_confidence_fit_total = sarima_model_confidence_total.fit(disp=False)
    future_confidence_exog_total = pd.DataFrame(np.zeros((4, len(top_confidence_features))), columns=top_confidence_features)
    future_confidence_total = sarima_model_confidence_fit_total.get_forecast(steps=4, exog=future_confidence_exog_total)
    future_confidence_predictions_total = future_confidence_total.predicted_mean

    # Plotting Trust Predictions
    plt.figure(figsize=(14, 6))
    plt.plot(borough_data.index, borough_data['Trust MPS'], label='Actual Trust')
    plt.plot(borough_test_data.index, weighted_trust_predictions, label='Weighted Trust Predictions (2020-2023)', color='orange')
    future_index_until_2019 = pd.date_range(borough_test_data.index[0], periods=len(future_trust_predictions_until_2019), freq='Q')
    plt.plot(future_index_until_2019, future_trust_predictions_until_2019, label='Future Predicted Trust (Post-2019)', color='green')
    future_index_total = pd.date_range(borough_data.index[-1], periods=5, freq='Q')[1:]  # Skip the last date in original data
    plt.plot(future_index_total, future_trust_predictions_total, label='Future Predicted Trust (Total Data)', color='blue')
    plt.title(f'Future Trust Predictions for {borough}')
    plt.legend()
    plt.ylim(-4, 4)
    plt.show()

    # Plotting Confidence Predictions
    plt.figure(figsize=(14, 6))
    plt.plot(borough_data.index, borough_data['"Good Job" local'], label='Actual Confidence')
    plt.plot(borough_test_data.index, weighted_confidence_predictions, label='Weighted Confidence Predictions (2020-2023)', color='orange')
    future_index_until_2019 = pd.date_range(borough_test_data.index[0], periods=len(future_confidence_predictions_until_2019), freq='Q')
    plt.plot(future_index_until_2019, future_confidence_predictions_until_2019, label='Future Predicted Confidence (Post-2019)', color='green')
    future_index_total = pd.date_range(borough_data.index[-1], periods=5, freq='Q')[1:]  # Skip the last date in original data
    plt.plot(future_index_total, future_confidence_predictions_total, label='Future Predicted Confidence (Total Data)', color='blue')
    plt.title(f'Future Confidence Predictions for {borough}')
    plt.legend()
    plt.ylim(-4, 4)
    plt.show()
