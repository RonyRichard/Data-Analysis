import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming the data is stored in a CSV file named 'heart_disease_data.csv'
data = pd.read_csv('Heart_Disease_Prediction.csv')

# Check for missing values and data types
print(data.info())
print(data.head())

# Check for missing values
print(data.isnull().sum())


# Relationship between cholesterol levels and heart disease


data['Heart Disease'] = data['Heart Disease'].astype('category')

 
data['Age Group'] = pd.cut(data['Age'], bins=[29, 40, 50, 60, 70, 80], labels=['30s', '40s', '50s', '60s', '70s'])

 
plt.figure(figsize=(12, 8))
sns.violinplot(x='Age Group', y='Cholesterol', hue='Heart Disease', data=data, palette='muted', split=True, inner=None)
sns.swarmplot(x='Age Group', y='Cholesterol', hue='Heart Disease', data=data, color='k', alpha=0.5, dodge=True)
plt.title('Cholesterol Levels Across Different Age Groups by Heart Disease Status')
plt.xlabel('Age Group')
plt.ylabel('Cholesterol')
plt.legend(title='Heart Disease', loc='upper right')

# Display the plot
plt.show()


# Impact of exercise-induced angina on heart disease


angina_comparison = data.groupby('Exercise angina')['Heart Disease'].value_counts(normalize=True).unstack()


angina_comparison *= 100


ax = angina_comparison.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Impact of Exercise-Induced Angina on Heart Disease')
plt.ylabel('Percentage')
plt.xticks([0, 1], ['No Angina', 'Angina'], rotation=0)
plt.legend(title='Heart Disease', labels=['Absence', 'Presence'])

ax.set_yticklabels(['{:.0f}%'.format(y) for y in ax.get_yticks()])

plt.show()