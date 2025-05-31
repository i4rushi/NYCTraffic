import pandas
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

''' 
The following is the starting code for path2 for data reading to make your first step easier.
'dataset_2' is the clean data for path1.
'''
dataset_2 = pandas.read_csv('NYC_Bicycle_Counts_2016.csv')
dataset_2['Brooklyn Bridge']      = pandas.to_numeric(dataset_2['Brooklyn Bridge'].replace(',','', regex=True))
dataset_2['Manhattan Bridge']     = pandas.to_numeric(dataset_2['Manhattan Bridge'].replace(',','', regex=True))
dataset_2['Queensboro Bridge']    = pandas.to_numeric(dataset_2['Queensboro Bridge'].replace(',','', regex=True))
dataset_2['Williamsburg Bridge']  = pandas.to_numeric(dataset_2['Williamsburg Bridge'].replace(',','', regex=True))
dataset_2['Williamsburg Bridge']  = pandas.to_numeric(dataset_2['Williamsburg Bridge'].replace(',','', regex=True))
# print(dataset_2.to_string()) #This line will print out your data

# Rename 'dataset_2' to 'bike_data' for consistency with new analyses
bike_data = dataset_2

# Convert columns with numeric values formatted as strings, ensuring they are treated as strings
columns_to_convert = ['Brooklyn Bridge', 'Manhattan Bridge', 'Williamsburg Bridge', 'Queensboro Bridge', 'Total']
for column in columns_to_convert:
    if bike_data[column].dtype == 'object':
        bike_data[column] = bike_data[column].str.replace(',', '').astype(int)
    # else:
        # print(f"No conversion needed for {column}, current type: {bike_data[column].dtype}")

# Correlation Analysis
correlation_matrix = bike_data[['Brooklyn Bridge', 'Manhattan Bridge', 'Williamsburg Bridge', 'Queensboro Bridge', 'Total']].corr()
plt.figure(figsize=(10, 8))
heatmap = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation between Bridges and Total Traffic')
plt.show()

# Weather Predictability of Traffic
X = bike_data[['High Temp', 'Low Temp', 'Precipitation']]
y = bike_data['Total']
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())

# Predicting the Day of the Week
X_class = bike_data[['Brooklyn Bridge', 'Manhattan Bridge', 'Williamsburg Bridge', 'Queensboro Bridge']]
y_class = bike_data['Day']
days_of_week = {day: idx for idx, day in enumerate(y_class.unique())}
y_class = y_class.map(days_of_week)
X_train, X_test, y_train, y_test = train_test_split(X_class, y_class, test_size=0.2, random_state=42)
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy of day of the week prediction: {accuracy:.2f}')
