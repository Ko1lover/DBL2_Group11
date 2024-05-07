import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

def load_data(csv_path, excel_path):
    data_csv = pd.read_csv(csv_path)
    data_excel = pd.read_excel(excel_path, sheet_name='Borough')

    data = pd.concat([data_csv, data_excel], ignore_index=True)

    data['Date'] = pd.to_datetime(data['Date'])
    data['Year'] = data['Date'].dt.year
    data.drop('Date', axis=1, inplace=True)

    for column in data.columns:
        if data[column].dtype == 'object':
            data[column] = LabelEncoder().fit_transform(data[column])

    return data

data = load_data('/Users/kaanncc/Downloads/PAS_borough (1).csv', '/Users/kaanncc/Downloads/PAS_T&Cdashboard_to Q3 23-24.xlsx')

X = data.drop('Proportion', axis=1)
y = data['Proportion']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

importances = model.feature_importances_
feat_importances = pd.Series(importances, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.title('Feature Importance')
plt.show()
