# -*- coding: utf-8 -*-
"""

@author: Pachimatla Rajesh

Support Vector Machine for classification

Case Study: Breast Cancer prediction

Goal: To create a classification model that looks at predicts if the cancer 
diagnosis is benign or malignant based on several features.
"""
### Import required libraries and modules

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.svm import SVC

########## 1. DATA EXPLORATION ##################
#Lets import the dataset which is from kaggle
df_cancer = pd.read_csv('Breast_cancer_data.csv')
df_cancer.head()

#get some information about our Data-Set
df_cancer.info()
#Lets see the descriptive information about data
df_cancer.describe()

#visualizing data
sns.pairplot(df_cancer, hue = 'diagnosis')
#pairplot, which is a grid of scatterplots showing relationships between 
#pairs of variables in a DataFrame.
#The hue parameter is used to color the data points in the pairplot based on 
#the values in the 'diagnosis' column of the DataFrame

plt.figure(figsize=(7,7))
sns.heatmap(df_cancer['mean_radius mean_texture mean_perimeter mean_area mean_smoothness diagnosis'.split()].corr(), annot=True)
#selects specific columns from the dataframe
# The .split() method is used to split the string into a list of column names.
#.corr(): This function calculates the correlation matrix for the selected columns. 
#The correlation matrix shows how each pair of columns is correlated with each othe.
#annot = True means the values will be displayed in the cells

plt.figure(figsize=(6,6))
sns.scatterplot(x = 'mean_texture', y = 'mean_perimeter', hue = 'diagnosis', data = df_cancer)

#Visulalising feature correlations
palette ={0 : 'orange', 1 : 'blue'}
#"palette" refers to a color scheme or a set of colors used to represent different
#groups or categories of data points in the plot.
edgecolor = 'grey'

fig = plt.figure(figsize=(12,12))
plt.subplot(2,2,1)
ax1 = sns.scatterplot(x = df_cancer['mean_radius'], y = df_cancer['mean_texture'], hue = "diagnosis",
data = df_cancer, palette =palette, edgecolor=edgecolor)
plt.title('mean_radius vs mean_texture')

plt.subplot(2,2,2)
ax2 = sns.scatterplot(x = df_cancer['mean_radius'], y = df_cancer['mean_perimeter'], hue = "diagnosis",
data = df_cancer, palette =palette, edgecolor=edgecolor)
plt.title('mean_radius vs mean_perimeter')

plt.subplot(2,2,3)
ax3 = sns.scatterplot(x = df_cancer['mean_radius'], y = df_cancer['mean_area'], hue = "diagnosis",
data = df_cancer, palette =palette, edgecolor=edgecolor)
plt.title('mean_radius vs mean_area')

plt.subplot(2,2,4)
ax4 = sns.scatterplot(x = df_cancer['mean_radius'], y = df_cancer['mean_smoothness'], hue = "diagnosis",
data = df_cancer, palette =palette, edgecolor=edgecolor)
plt.title('mean_radius vs mean_smoothness')

fig.suptitle('Features Correlation', fontsize = 20)

plt.savefig('2')

plt.show()

################ 2. Dealing with missing information ##############

#check how many values are missing (NaN)
df_cancer.isnull().sum()
#no missing data in 6 columns/features
#handling categorical data
df_cancer['diagnosis'].unique()
#df_cancer['diagnosis'] = df_cancer['diagnosis'].map({'benign':0,'malignant':1})
df_cancer.head()

#visualizing diagnosis column >>> 'benign':0,'malignant':1
sns.countplot(x='diagnosis',data = df_cancer)
plt.title('number of Benign_0 vs Malignan_1')
# correlation between features
df_cancer.corr()['diagnosis'][:-1].sort_values().plot(kind ='bar')
#['diagnosis'][:-1]: This part of the code selects the correlation values 
#between the 'diagnosis' column and all other columns except for the last one. 
#It's excluding the last value because the 'diagnosis' column is also included 
#in the correlation matrix
plt.title('Corr. between features and target')

########### 3. Data splitting and modeling of SVM ##########
#define X variables and our target(y)
X = df_cancer.drop(['diagnosis'],axis=1).values
y = df_cancer['diagnosis'].values
#split Train and Test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101) 


#Support Vector Classification model
from sklearn.svm import SVC
svc_model = SVC()
svc_model.fit(X_train, y_train)


############# 4. Model Evaluations ###########

from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
y_predict = svc_model.predict(X_test)
cm = confusion_matrix(y_test, y_predict)
precision, recall, fscore,_= precision_recall_fscore_support(y_test, y_predict, average='binary')
sns.heatmap(cm, annot=True)   
print('precion, recll and f1score are', round(precision,2), round(recall,2), round(fscore,2))
print(classification_report(y_test, y_predict))

########## 5. Improving model efficiency by nomralization ############

#normalized scaler - fit&transform on train, fit only on test
from sklearn.preprocessing import MinMaxScaler

n_scaler = MinMaxScaler()
#After applying the MinMaxScaler, your data will be transformed in such a way
# that the minimum value becomes 0, and the maximum value becomes 1, with all 
#other values scaled proportionally between these two bounds. 
#This scaling is useful when working with machine learning algorithms that are 
#sensitive to the scale of the input features.

X_train_scaled = n_scaler.fit_transform(X_train.astype(np.float))
#X_train.astype(np.float): This converts the data in X_train to a NumPy array of 
#type float. It's important to work with float data when using scalers like 
#MinMaxScaler.
X_test_scaled = n_scaler.transform(X_test.astype(np.float))

#Support Vector Classification model -  apply on scaled data
from sklearn.svm import SVC
svc_model = SVC()
svc_model.fit(X_train_scaled, y_train)


from sklearn.metrics import classification_report, confusion_matrix
y_predict_scaled = svc_model.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_predict_scaled)
sns.heatmap(cm, annot=True)
print(classification_report(y_test, y_predict_scaled))

############# 6. SVM parameter optimization ############3
#C parameter — as we said, it controls the cost of misclassification on Train data.
#Smaller C: Lower variance but higher bias (soft margin) and reduce the cost of 
#miss-classification (less penalty).
#Larger C: Lower bias and higher variance (hard margin) and increase the cost 
#of miss-classification (more strict).
#Smaller Gamma: Large variance, far reach, and more generalized solution.
#Larger Gamma: High variance and low bias, close reach, and also closer data 
#points have a higher weight.

#find best hyper parameters
from sklearn.model_selection import GridSearchCV
#GridSearchCV will take care of training and evaluating the model with different
# hyperparameter combinations and provide you with the best-performing model 
#according to your specified evaluation metric.

param_grid = {'C':[0.1,1,10,100,1000],'gamma':[1,0.1,0.01,0.001,0.001], 'kernel':['rbf']}
#C: The regularization parameter. It controls the trade-off between maximizing 
#the margin and minimizing the classification error
#gamma: The kernel coefficient for 'rbf' (Radial Basis Function) kernel. 
#It defines how much influence a single training example has. Small values of 
#gamma mean a large influence, and large values mean a small influence. 
#A small gamma value tends to create a smoother decision boundary, 
#while a large gamma value creates a more complex, tightly fitted boundary.
grid = GridSearchCV(SVC(),param_grid,verbose = 4)
#The verbose parameter controls the verbosity of the output during the grid search. 
#A higher value, like 4, will provide more detailed output about the progress of 
#the grid search, including the fit and scoring of each parameter combination.
grid.fit(X_train_scaled,y_train)
grid.best_params_
grid.best_estimator_

grid_predictions = grid.predict(X_test_scaled)

cmG = confusion_matrix(y_test,grid_predictions)
sns.heatmap(cmG, annot=True)
print(classification_report(y_test,grid_predictions))

## End of the script ####
