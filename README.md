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
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import accuracy_score, classification_report
data = pd.read_csv("Salary.csv")
X = data.drop("Salary", axis=1)
y = data["Salary"]
X = pd.get_dummies(X, drop_first=True)
#X_train, X_test, y_train, y_test = train_test_split(    X, y, test_size=0.2, random_state=42)
model = DecisionTreeRegressor(random_state=42)
model.fit(X, y)
y_pred = model.predict(X)
# Accuracy
print("\nAccuracy:", accuracy_score(y, y_pred)*100)

# Classification Report
print("\nClassification Report:")
print(classification_report(y, y_pred))

plt.figure(figsize=(25,12))
plot_tree(
    model,
    feature_names=X.columns,
    filled=True
)

plt.title("Decision Tree Regressor")
plt.show()
```

## Output:
<img width="1226" height="815" alt="mlpivc1" src="https://github.com/user-attachments/assets/a7652521-9424-41e9-af44-d936a7e36455" />
<img width="1507" height="882" alt="mlpic" src="https://github.com/user-attachments/assets/28dd7e0d-7ac7-4df6-8a70-77ac8cccbfbd" />
<img width="1600" height="831" alt="mlpic3" src="https://github.com/user-attachments/assets/be588fff-2374-485d-9361-6a5ec7ef1bac" />



## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
