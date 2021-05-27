import pandas

print("Dataset collected\n\n")
ds = pandas.read_csv('SalaryData.csv')

x = ds['YearsExperience']
y = ds['Salary']

x = ds['YearsExperience'].values.reshape(30,1)

from sklearn.linear_model import LinearRegression

model = LinearRegression()

print("Model has been trained\n")

model.fit(x,y)

print("The predicted value of salary with x=2.5 is",model.predict([[2.5]]))

import joblib

joblib.dump(model,'SalaryData.pk1')

print("model has been saved successfully")


