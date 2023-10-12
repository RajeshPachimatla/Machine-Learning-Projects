# -*- coding: utf-8 -*-
"""
@author: Pachimatla Rajesh

Diabetes Predictions

"""
### import libraries ###

import pandas as pd
import numpy as np
import math

from sklearn.model_selection import train_test_split
#for splitting the data
from sklearn.preprocessing import StandardScaler
#for standardization of features
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
#to check the accuracy, recall etc.
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

dataset = pd.read_csv('diabetes.csv')
#read the data set

len_dataset = len(dataset)
dataset.head()

# Replace zeroes in some of the column since which doesnt having any meaning
zero_not_accepted = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'Insulin']

for column in zero_not_accepted:
    dataset[column] = dataset[column].replace(0, np.NaN)
    #first replace the zero to nan
    mean = int(dataset[column].mean(skipna=True))
    #then calculate the mean by skiping nan and replace the this value with nan
    dataset[column] = dataset[column].replace(np.NaN, mean)

#dataset.columns
#['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
#       'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
# split dataset
X = dataset.iloc[:, 0:8]
x2 = dataset.values[:, 0:8]
#here X is dataframe whereas X2 is an array of floats
#here ..iloc is used to split the dataset in column wise first 8 columns as X input

y = dataset.iloc[:, 8]
#here y is single column sliced out from dataframe, hence it is a series
#9th column 'Outcome' as y which target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)
#random_state = 0 means every time you run the code, same random split it will be done


print(len(X_train))
print(len(y_train))
print(len(X_test))
print(len(y_test))


#Feature scaling or standardized

sc_X = StandardScaler()
#creates an instance of snadardScaler()
X_train = sc_X.fit_transform(X_train)
#This calculates the mean and standard deviation, then tranform the data / scaling
X_test = sc_X.transform(X_test)
#It is important to know, same mean and standard deviation from train data should be used
# to standardize test data

#How to chose k value
k_value = math.sqrt(len(X_test))
#k value is 12.4, so either take 11 or 13

#k = 11
# Define the model: Init K-NN
classifier1 = KNeighborsClassifier(n_neighbors=11, p=2,metric='euclidean')
#here p = 2 is the power parameter which specifies what type of distance metric is used
# p = 2 means 'euclidean' distance

# Fit Model
classifier1.fit(X_train, y_train)

# Predict the test set results
y_pred1 = classifier1.predict(X_test)

# Evaluate Model
cm1 = confusion_matrix(y_test, y_pred1)
f1_score1 = f1_score(y_test, y_pred1)
# F1 score will tell the prob of false positives
accuracy_score1 = accuracy_score(y_test, y_pred1)

#k=13
classifier2 = KNeighborsClassifier(n_neighbors=13, p=2,metric='euclidean')
#here p = 2 is the power parameter which specifies what type of distance metric is used
# p = 2 means 'euclidean' distance

# Fit Model
classifier2.fit(X_train, y_train)

# Predict the test set results
y_pred2 = classifier2.predict(X_test)

# Evaluate Model
cm2 = confusion_matrix(y_test, y_pred2)
f1_score2 = f1_score(y_test, y_pred2)
accuracy_score2 = accuracy_score(y_test, y_pred2)

############### End of Script ##########



