# -*- coding: utf-8 -*-
"""
@author: Pachimatla Rajesh

-Random Forest algorithm for weather classification

"""
### Import the libraries ####
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

from sklearn.tree import export_graphviz
#import pydot

df = pd.read_csv('temps.csv')

df.head(5)

df.columns

df.drop(['forecast_noaa', 'forecast_acc','forecast_under'], axis=1, inplace=True)

df.shape

#Check for null values
df.isnull().sum()
#No null values

#create dummies from categorical variables
df = pd.get_dummies(df)

#Lables are the values we want to predict
labels = df['actual']

#remove the labels from the features
df = df.drop('actual', axis=1)

#saving feature names for later usage
feature_list = list(df.columns)

columns_1 = df.columns

#
from sklearn.model_selection import train_test_split

#Split the data training and testing data

train_features, test_features, train_labels, test_labels = train_test_split(df,
                                                                            labels,
                                                                            test_size=0.2,
                                                                            random_state=0)

print('training feature shape', train_features.shape)
print('training labels shape', train_labels.shape)
print('testing feature shape', test_features.shape)
print('testing labels shape', test_labels.shape)

#Import the random forest model  from sci-kit learn
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=1000, random_state=42)

#train the model on training data
rf.fit(train_features, train_labels)

#Ude the forest predict method on the test data
predictions = rf.predict(test_features)

#Calculate the absolute error
errors = abs(predictions-test_labels)

#printout the main absolute error
print('Mean absolute error', round(np.mean(errors),3), 'degrees.')

#calculate mean absolute percentage error
mape = 100*(errors/test_labels)

#Calculate and display accuracy
accuaracy = 100 - np.mean(mape)
print(accuaracy)
############## End of Script #####







