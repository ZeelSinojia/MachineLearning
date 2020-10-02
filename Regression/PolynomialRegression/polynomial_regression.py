# -*- coding: utf-8 -*-
"""
Created on Thu May 14 15:38:52 2020

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

# #Taking care of missing data
# from sklearn.impute import SimpleImputer
# imputer = SimpleImputer(missing_values = np.nan, strategy = "mean")
# imputer = imputer.fit(X[:, 1:3])
# X[:, 1:3] = imputer.transform(X[:, 1:3])

# #Encoding categorical data
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# from sklearn.compose import ColumnTransformer

# # Country column
# ct = ColumnTransformer([("Country", OneHotEncoder(), [0])], remainder = 'passthrough')
# X = ct.fit_transform(X)

# #Encdoing the dependant variable
# # Yes/No
# labelencoder_X = LabelEncoder()
# y = labelencoder_X.fit_transform(y)

#Splitting dataset in training and testing
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# #Feature Scaling
# from sklearn.preprocessing import StandardScaler
# sc_X = StandardScaler()
# X_train = sc_X.fit_transform(X_train)
# X_test = sc_X.transform(X_test)

#Fitting Linear regression model to dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

#Fitting Polynomial regression model to dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)

regressor = LinearRegression()
regressor.fit(X_poly, y)

#Visualizing the Linear Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title("TRUTH or BLUFF(Linear Regression)")
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show() 


#Visualizing the Polynomial Regression results
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(poly_reg.fit_transform(X_grid)), 
         color = 'blue')
plt.title("TRUTH or BLUFF(Polynomial Regression)")
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#Predicting a new result with Linear Regression 
lin_reg.predict([[6.5]])

#Predicting a new result with Polynomial Regression 
regressor.predict(poly_reg.fit_transform([[6.5]]))