# -*- coding: utf-8 -*-
"""
Created on Fri May 15 17:05:17 2020

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

#Fitting Regression Model to dataset


#Predicting a new result with Regression 
y_pred = regressor.predict([[6.5]])

#Visualizing the Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title("TRUTH or BLUFF(Regression)")
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#Visualizing the Regression results 
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title("TRUTH or BLUFF(Regression)")
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()