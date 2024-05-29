import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder


# Load data
data = pd.read_csv('data/PAS_with_crime.csv')

# Combine 'year' and 'month' to create a proper datetime index for quarterly data
data['Date'] = pd.to_datetime(data[['year', 'month']].astype(str).agg('-'.join, axis=1) + '-01')
data.set_index('Date', inplace=True)


numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
data_numeric = data[numeric_cols].groupby(data.index).mean()

data['Borough'] = data['Borough'].groupby(data.index).agg(lambda x: pd.Series.mode(x)[0])

# Combine numeric data with the 'Borough' column
data = pd.concat([data_numeric, data[['Borough']].drop_duplicates()], axis=1)

# Calculate weighted trust
trust_weights = {
    'Treat everyone fairly': 0.292,
    'Listen to concerns': 0.258,
    'Relied on to be there': 0.24,
    'Trust MPS': 0.31
}

data['Trust_Weighted'] = sum(data[col] * weight for col, weight in trust_weights.items())

# Calculate weighted confidence
confidence_weights = {
    'Understand issues': 0.50,
    'Informed local': 0.07,
    'Contact ward officer': 0.078,
    'Good Job local': 0.35
}

data['Confidence_Weighted'] = sum(data[col] * weight for col, weight in confidence_weights.items())

#Both weights will be adjusted after each run of feature importance 



# Function to train and evaluate the SARIMA model
def train_evaluate_sarima(data, column, order=(1, 1, 1), seasonal_order=(1, 1, 1, 4)):
    model = SARIMAX(data[column], order=order, seasonal_order=seasonal_order)
    model_fit = model.fit()
    forecast_steps = 12  # Forecast for 3 years (4 quarters per year)
    forecast_index = pd.date_range(start=data.index[-1] + pd.DateOffset(months=3), periods=forecast_steps, freq='Q')
    forecast = model_fit.get_forecast(steps=forecast_steps)
    forecast_df = forecast.conf_int(alpha=0.05)  # 95% confidence interval
    forecast_df['forecast'] = forecast.summary_frame()['mean']
    forecast_df.index = forecast_index

    return model_fit, forecast_df

# Train and evaluate the SARIMA model using all data for Trust and Confidence
model_fit_trust_all, forecast_df_trust_all = train_evaluate_sarima(data, 'Trust_Weighted')
model_fit_confidence_all, forecast_df_confidence_all = train_evaluate_sarima(data, 'Confidence_Weighted')

# Train and evaluate using data up to 2019 for Trust and Confidence
data_until_2019 = data[data.index < '2020-01-01']
model_fit_trust_until_2019, forecast_df_trust_until_2019 = train_evaluate_sarima(data_until_2019, 'Trust_Weighted')
model_fit_confidence_until_2019, forecast_df_confidence_until_2019 = train_evaluate_sarima(data_until_2019, 'Confidence_Weighted')

# Plotting forecasts for Trust
plt.figure(figsize=(12, 8))
plt.plot(data['Trust_Weighted'], label='Actual Weighted Trust')
plt.plot(forecast_df_trust_all['forecast'], label='Forecasted Weighted Trust (All Data)', color='red')
plt.fill_between(forecast_df_trust_all.index, forecast_df_trust_all.iloc[:, 0], forecast_df_trust_all.iloc[:, 1], color='pink', alpha=0.3)
plt.plot(forecast_df_trust_until_2019['forecast'], label='Forecasted Weighted Trust (After 2019)', color='blue')
plt.fill_between(forecast_df_trust_until_2019.index, forecast_df_trust_until_2019.iloc[:, 0], forecast_df_trust_until_2019.iloc[:, 1], color='lightblue', alpha=0.3)
plt.title('Weighted Trust Forecast')
plt.xlabel('Date')
plt.ylabel('Weighted Trust')
plt.legend()
plt.show()

# Plotting forecasts for Confidence
plt.figure(figsize=(12, 8))
plt.plot(data['Confidence_Weighted'], label='Actual Weighted Confidence')
plt.plot(forecast_df_confidence_all['forecast'], label='Forecasted Weighted Confidence (All Data)', color='red')
plt.fill_between(forecast_df_confidence_all.index, forecast_df_confidence_all.iloc[:, 0], forecast_df_confidence_all.iloc[:, 1], color='pink', alpha=0.3)
plt.plot(forecast_df_confidence_until_2019['forecast'], label='Forecasted Weighted Confidence (After 2019)', color='blue')
plt.fill_between(forecast_df_confidence_until_2019.index, forecast_df_confidence_until_2019.iloc[:, 0], forecast_df_confidence_until_2019.iloc[:, 1], color='lightblue', alpha=0.3)
plt.title('Weighted Confidence Forecast')
plt.xlabel('Date')
plt.ylabel('Weighted Confidence')
plt.legend()
plt.show()



####################



# Correlation Matrix
correlation_data = data[
    ['Trust_Weighted', 'Confidence_Weighted', 'property_q_crimes', 'violent_q_crimes', 'public_order_q_crimes',
     'drug_or_weapon_q_crimes', 'total_q_crimes']]
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Weighted Trust, Confidence, and Crime Data')
plt.show()

# Scatterplot Matrix
sns.pairplot(correlation_data)
plt.show()


####################



# Model Validation: Calculating RMSE for Trust
forecast_period_trust = data.loc[data.index[-48:]]
common_index_trust = forecast_period_trust.index.intersection(forecast_df_trust_all.index)
if not common_index_trust.empty:
    rmse_trust_all = np.sqrt(mean_squared_error(forecast_period_trust.loc[common_index_trust, 'Trust_Weighted'],
                                                forecast_df_trust_all.loc[common_index_trust, 'forecast']))
    print(f'Root Mean Squared Error for Trust (All Data): {rmse_trust_all}')
else:
    print('No common index for Trust RMSE calculation (All Data).')

forecast_period_trust_until_2019 = data_until_2019.loc[data_until_2019.index[-48:]]
common_index_trust_until_2019 = forecast_period_trust.index.intersection(forecast_df_trust_until_2019.index)
if not common_index_trust_until_2019.empty:
    rmse_trust_until_2019 = np.sqrt(
        mean_squared_error(forecast_period_trust.loc[common_index_trust_until_2019, 'Trust_Weighted'],
                           forecast_df_trust_until_2019.loc[common_index_trust_until_2019, 'forecast']))
    print(f'Root Mean Squared Error for Trust (Until 2019): {rmse_trust_until_2019}')
else:
    print('No common index for Trust RMSE calculation (Until 2019).')

# Model Validation: Calculating RMSE for Confidence
forecast_period_confidence = data.loc[data.index[-48:]]
common_index_confidence = forecast_period_confidence.index.intersection(forecast_df_confidence_all.index)
if not common_index_confidence.empty:
    rmse_confidence_all = np.sqrt(
        mean_squared_error(forecast_period_confidence.loc[common_index_confidence, 'Confidence_Weighted'],
                           forecast_df_confidence_all.loc[common_index_confidence, 'forecast']))
    print(f'Root Mean Squared Error for Confidence (All Data): {rmse_confidence_all}')
else:
    print('No common index for Confidence RMSE calculation (All Data).')

forecast_period_confidence_until_2019 = data_until_2019.loc[data_until_2019.index[-48:]]
common_index_confidence_until_2019 = forecast_period_confidence.index.intersection(
    forecast_df_confidence_until_2019.index)
if not common_index_confidence_until_2019.empty:
    rmse_confidence_until_2019 = np.sqrt(
        mean_squared_error(forecast_period_confidence.loc[common_index_confidence_until_2019, 'Confidence_Weighted'],
                           forecast_df_confidence_until_2019.loc[common_index_confidence_until_2019, 'forecast']))
    print(f'Root Mean Squared Error for Confidence (Until 2019): {rmse_confidence_until_2019}')
else:
    print('No common index for Confidence RMSE calculation (Until 2019).')

# Feature Importance with RandomForestRegressor
# For Trust
features_trust = ['Treat everyone fairly', 'Listen to concerns', 'Relied on to be there', 'Trust MPS']
X_trust = data[features_trust]
y_trust = data['Trust_Weighted']

# Split the data into training and testing sets
X_train_trust, X_test_trust, y_train_trust, y_test_trust = train_test_split(X_trust, y_trust, test_size=0.3,
                                                                            random_state=42)

# Train the model
rf_trust = RandomForestRegressor(n_estimators=100, random_state=42)
rf_trust.fit(X_train_trust, y_train_trust)

# Evaluate the model
y_pred_trust = rf_trust.predict(X_test_trust)
rmse_rf_trust = np.sqrt(mean_squared_error(y_test_trust, y_pred_trust))
print(f'Random Forest RMSE for Trust: {rmse_rf_trust}')

# Feature importance
importance_trust = rf_trust.feature_importances_
feature_importance_trust = pd.DataFrame({'Feature': features_trust, 'Importance': importance_trust})
feature_importance_trust.sort_values(by='Importance', ascending=False, inplace=True)
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_trust)
plt.title('Feature Importance for Trust')
plt.show()

# For Confidence
features_confidence = ['Understand issues', 'Informed local', 'Contact ward officer', 'Good Job local']
X_confidence = data[features_confidence]
y_confidence = data['Confidence_Weighted']

# Split the data into training and testing sets
X_train_confidence, X_test_confidence, y_train_confidence, y_test_confidence = train_test_split(X_confidence,
                                                                                                y_confidence,
                                                                                                test_size=0.3,
                                                                                                random_state=42)

# Train the model
rf_confidence = RandomForestRegressor(n_estimators=100, random_state=42)
rf_confidence.fit(X_train_confidence, y_train_confidence)

# Evaluate the model
y_pred_confidence = rf_confidence.predict(X_test_confidence)
rmse_rf_confidence = np.sqrt(mean_squared_error(y_test_confidence, y_pred_confidence))
print(f'Random Forest RMSE for Confidence: {rmse_rf_confidence}')

# Feature importance
importance_confidence = rf_confidence.feature_importances_
feature_importance_confidence = pd.DataFrame({'Feature': features_confidence, 'Importance': importance_confidence})
feature_importance_confidence.sort_values(by='Importance', ascending=False, inplace=True)
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_confidence)
plt.title('Feature Importance for Confidence')
plt.show()