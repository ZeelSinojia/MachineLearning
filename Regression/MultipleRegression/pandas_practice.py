# -*- coding: utf-8 -*-
"""
Created on Tue May 12 18:33:06 2020

@author: Sinojia Zeel
"""



import pandas as pd

#Importinf datasets
dataset = pd.read_csv('50_Startups.csv')


curated_dataset = pd.get_dummies(dataset, drop_first=True)

y = curated_dataset.iloc[:, 3]

X = curated_dataset.iloc[:, :]

X.drop('Profit', inplace=True, axis=1)