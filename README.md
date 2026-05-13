# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load and preprocess the dataset by handling missing values and encoding categorical features (salary) using LabelEncoder.
2. Split the dataset into input features (X) and target variable (y), then divide it into training and testing sets.
3. Train a Decision Tree Classifier using the entropy criterion on the training data.
4. Make predictions on the test data, evaluate model accuracy, and predict churn for new employee input data.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Krithika V
RegisterNumber: 212225040194 
*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
data = {
    'Position': ['Business Analyst', 'Junior Consultant', 'Senior Consultant',
                 'Manager', 'Country Manager', 'Region Manager',
                 'Partner', 'Senior Partner', 'C-level', 'CEO'],
    'Level': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Salary': [45000, 50000, 60000, 80000, 110000, 150000, 200000, 300000, 500000, 1000000]
}
df = pd.DataFrame(data)
X = df[['Level']]     
y = df['Salary']      
regressor = DecisionTreeRegressor(random_state=42)
regressor.fit(X, y)
y_pred = regressor.predict(X)
print("Predicted salaries:", y_pred)
level = np.array([[6.5]])
predicted_salary = regressor.predict(level)
print(f"Predicted Salary for level {level[0][0]}: {predicted_salary[0]}")
X_grid = np.arange(min(X.values), max(X.values)+0.01, 0.01) 
X_grid = X_grid.reshape(-1, 1)
plt.scatter(X, y, color='red', label='Actual Salary')
plt.plot(X_grid, regressor.predict(X_grid), color='blue', label='Decision Tree Prediction')
plt.title('Decision Tree Regression: Level vs Salary')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.legend()
plt.show()
```

## Output:
<img width="943" height="80" alt="image" src="https://github.com/user-attachments/assets/f7badbd0-fc8b-42d7-9954-3cd3eed9d9b2" />
<img width="893" height="616" alt="image" src="https://github.com/user-attachments/assets/a478fe59-84ee-4e11-8043-a8ce25b3611a" />




## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
