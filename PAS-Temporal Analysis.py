import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import statsmodels.api as sm

data = pd.read_csv('data/PAS_borough.csv')

data['Measure'] = data['Measure'].str.strip('"').str.strip()

data['Date'] = pd.to_datetime(data['Date'])
data['Year'] = data['Date'].dt.year

label_encoder = LabelEncoder()
data['Borough'] = label_encoder.fit_transform(data['Borough'])

print("Unique Measures after stripping quotes:", data['Measure'].unique())

data['Measure'] = label_encoder.fit_transform(data['Measure'])

data['Borough_Measure_Interaction'] = data['Borough'] * data['Measure']

X = data[['Borough', 'Measure', 'Borough_Measure_Interaction', 'Year']]
X = sm.add_constant(X)
y = data['Proportion']

model = sm.OLS(y, X).fit()
print(model.summary())

print("Transformed label for 'Good Job local':", label_encoder.transform(['Good Job local']))

good_job_data = data[data['Measure'] == label_encoder.transform(['Good Job local'])[0]]

plt.figure(figsize=(12, 8))
for label in np.unique(good_job_data['Borough']):
    subset = good_job_data[good_job_data['Borough'] == label]
    plt.plot(subset['Year'], subset['Proportion'], marker='o', label=label_encoder.inverse_transform([label])[0])

plt.title('Perception Over Time by Borough for "Good Job Local"')
plt.xlabel('Year')
plt.ylabel('Proportion Rating')
plt.legend(title='Borough')
plt.show()

sns.boxplot(x='Borough', y='Proportion', data=good_job_data)
plt.xticks(rotation=45, labels=label_encoder.inverse_transform(good_job_data['Borough'].unique()))
plt.title('Variation of "Good Job Local" Perception by Borough')
plt.xlabel('Borough')
plt.ylabel('Proportion')
plt.show()
