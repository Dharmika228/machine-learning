import pandas

print("Dataset collected\n\n")
ds = pandas.read_csv('SalaryData.csv')
print("\n")
print("the dataset that I collected here\n")
print(ds)

x = ds['YearsExperience']
y = ds['Salary']

x = ds['YearsExperience'].values.reshape(30,1)

from sklearn.linear_model import LinearRegression

model = LinearRegression()

print("Model has been trained\n")

model.fit(x,y)

print("The predicted value of salary with x=2.5 is",model.predict([[2.5]]))

print("""
    If you want to save the model:
         Y for yes
         N for no
      """)

choice = input("enter your choice")

if "Y" in choice:
  import joblib
  joblib.dump(model,'SalaryData.pk1')
  print("model has been saved successfully")
  
if "N" in choice:
  break


