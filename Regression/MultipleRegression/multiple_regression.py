# -*- coding: utf-8 -*-
"""
Created on Tue May 12 16:14:50 2020

@author: Sinojia Zeel
"""


#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importinf datasets
dataset = pd.read_csv('50_Startups.csv')


curated_dataset = pd.get_dummies(dataset, drop_first=True)


y = curated_dataset.iloc[:, 3].values
X = curated_dataset.iloc[:, :]
X.drop('Profit', inplace=True, axis=1)

# #Taking care of missing data
# from sklearn.impute import SimpleImputer
# imputer = SimpleImputer(missing_values = np.nan, strategy = "mean")
# imputer = imputer.fit(X[:, 1:3])
# X[:, 1:3] = imputer.transform(X[:, 1:3])

# #Encoding categorical data
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.compose import ColumnTransformer

# # State column
# ct = ColumnTransformer([("State", OneHotEncoder(), [3])], 
#                        remainder = 'passthrough')
# X = ct.fit_transform(X)

#Avoiding the Dummy Variable Trap
# X = X[:, 1:]

#Splitting dataset in training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, 
                                                    random_state = 0)

# #Feature Scaling
# from sklearn.preprocessing import StandardScaler
# sc_X = StandardScaler()
# X_train = sc_X.fit_transform(X_train)
# X_test = sc_X.transform(X_test)

# Fitting multiple Linear regression into the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the test set
y_pred = regressor.predict(X_test)


#Buidling optimal model using Backward Elimination
import statsmodels.api as sm
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)
X_opt = X[:, :]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 1, 2, 3, 4]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 1, 2, 3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 1, 3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 1]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()


X_train, X_test, y_train, y_test = train_test_split(X_opt, y, test_size = 0.2, 
                                                    random_state = 0)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the test set
y_pred = regressor.predict(X_test)