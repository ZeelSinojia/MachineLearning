# -*- coding: utf-8 -*-
"""
Created on Sat May 16 02:15:00 2020

@author: Sinojia Zeel
"""


#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importinf datasets
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, 2].values

#Splitting dataset in training and testing
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# #Feature Scaling
# from sklearn.preprocessing import StandardScaler
# sc_X = StandardScaler()
# X_train = sc_X.fit_transform(X_train)
# X_test = sc_X.transform(X_test)

#Fitting Decision Tree Regression Model to dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)


#Predicting a new result with Decision Tree Regression 
y_pred = regressor.predict([[6.5]])

#Visualizing the Decision Tree Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title("TRUTH or BLUFF(Decision tree Regression)")
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#Visualizing the Decision Tree Regression results 
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title("TRUTH or BLUFF(Decision tree Regression)")
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()