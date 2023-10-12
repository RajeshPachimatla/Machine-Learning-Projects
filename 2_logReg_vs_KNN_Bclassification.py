"""Author: Pachimatla Rajesh

Case study: Income Classification for subsidy

LOGISTIC REGRESSION VS KNN

"""

#########Section 1: Importing section required packages#############

# To work with data frames
import pandas as pd

# To Work with numerical operations
import numpy as np

# To visualise
import seaborn as sns

# To partition the data
from sklearn.model_selection import train_test_split

#importing library for logistic regression
from sklearn.linear_model import LogisticRegression

#importing performance metrics - accuracy and confusion matrix
from sklearn.metrics import accuracy_score, confusion_matrix

import matplotlib.pyplot as plt

########## Section 2: Importing DATA ##################

data_income = pd.read_csv('income.csv')

#Creating a duplicate of data

data = data_income.copy()

######### Section 3: Exploratory Data Analysis ################

"""
First wee will do the follwing
1. Getting to know the data
2. Identify missing values (Data preprocessing)
3  Cross tables and data visualisation

"""

########## 3.1 Getting to know the data #######

print(data.info()) #We check data types

print('Data columns with missing values :\n', data.isnull().sum())
# /no missing value found in data column, but in real cases, we can see...
#lot of missing values

# to know summary of numerical variables
data.describe() 

#to know the summary of categorical variable
data.describe(include = "O")

## Lets see frequency of each categories
data['JobType'].value_counts()
data['occupation'].value_counts() 
#It found that, there are missing values, but special mark like question mark is there
#hence it is not showing has missing value

#lets check for unique classes
print(np.unique(data['JobType']))
print(np.unique(data['occupation']))
#it found that as said earlier, there exist ' ?' class instead of nan

#We can read the ' ?' as nan while importing the data itself as show below
data = pd.read_csv('income.csv',na_values=[" ?"])


### Section 3.2: Data Pre-Processing #####

data.isnull().sum()
#JobType and occupation have missing values 1809 nd 1816
 
missing = data[data.isnull().any(axis=1)]
#data.isnull().any(axis=1) computes whether any of the columns for each row 
#contains True (i.e., any missing values) by applying the .any(axis=1) method 
#along the rows.
#It will return a DataFrame missing containing only the rows from data where at 
#least one column has missing data.

"""
Summary of learning till now
1. Missing values in JobType = 1809
2. Missing values in  occupation  = 1816
3. There are 1809 columns have missing value for both columns
4. For 7 people occupation is filled since there job type is 'never worked'
"""

data2 = data.dropna(axis=0)

### Section 3.3: Relationship between independent variables ####

correlation = data2.corr(numeric_only=True)
print(correlation)

### section 3.4: Cross tables and data visualisations

#extracting column names
data2.columns

# Gender proportion table
gender = pd.crosstab(index = data2["gender"],
                     columns = 'count',
                     normalize = True)
print(gender)

# Gender vs Salary
gender_salstat = pd.crosstab(index = data2["gender"],
                             columns = data2["SalStat"],
                             margins = True,
                             normalize= 'index')

print(gender_salstat)

#Frequency distrubution of 'Salary Status'
sns.countplot(y=data2.SalStat)

## histogram of age ##
sns.distplot(data2['age'], bins=10, kde=False)

##  Box plot - Age vs Salary ###
sns.boxplot(x='SalStat', y='age', data=data2)
data2.groupby('SalStat')['age'].median()

#sns.barplot(x='count', y='JobType', hue='SalStat',data=data2)
sns.countplot(y=data2.JobType)

############### 4. Logistic Regression ################
#Reindexing the salary status names to 0 or 1
data2['SalStat'] = data2['SalStat'].replace({' less than or equal to 50,000':0, ' greater than 50,000':1})

print(data2['SalStat'])

new_data = pd.get_dummies(data2, drop_first = True)

#Storing the column names
columns_list = list(new_data.columns)
print(columns_list)

#seperating the input names from data
features = list(set(columns_list)-set(['SalStat']))
#The code then uses the set() function to convert columns_list into a set and 
#does the same for ['SalStat']. This is done to ensure that both lists are 
#treated as sets, which have unique elements and don't allow duplicates.
print(features)

#Storing the output values in y
y = new_data['SalStat'].values
print(y)

#Storing the values from the input features
x = new_data[features].values
print(x)


#Splitting the data into train and test set
train_x, test_x, train_y, test_y = train_test_split(x,y,test_size=0.3, random_state=0)

#make an instance of the model
logistic = LogisticRegression()

#Fitting the values for x and y
logistic.fit(train_x,train_y)
logistic.coef_
logistic.intercept_

#Prediction from test data
prediction = logistic.predict(test_x)
print(prediction)

#Confusion matrix gives true_pos, true_neg, False_pos, False_neg
confusion_matrix1 = confusion_matrix(test_y, prediction)
print(confusion_matrix1)

#Calculate the accuracy
accuracy_score1 = accuracy_score(test_y, prediction)
print(accuracy_score1)

#Printing misclassified values from the predictions
print('misclassied sample number :%d' % (test_y != prediction).sum())

################################################
# Removing Insignificant variables and re-doing Logistic regression #
################################################

#Reindexing the salary status names to 0 or 1
data2['SalStat'] = data2['SalStat'].replace({' less than or equal to 50,000':0, ' greater than 50,000':1})

print(data2['SalStat'])

cols = ['gender','nativecountry','race','JobType']
new_data = data2.drop(cols, axis=1)
new_data = pd.get_dummies(data2, drop_first = True)

#Storing the column names
columns_list = list(new_data.columns)
print(columns_list)

#seperating the input names from data
features = list(set(columns_list)-set(['SalStat']))
print(features)

#Storing the output values in y
y = new_data['SalStat'].values
print(y)

#Storing the values from the input features
x = new_data[features].values
print(x)


#Splitting the data into train and test set
train_x, test_x, train_y, test_y = train_test_split(x,y,test_size=0.3, random_state=0)

#make an instance of the model
logistic = LogisticRegression()

#Fitting the values for x and y
logistic.fit(train_x,train_y)
logistic.coef_
logistic.intercept_

#Prediction from test data
prediction = logistic.predict(test_x)
print(prediction)

#Confusion matrix gives true_pos, true_neg, False_pos, False_neg
confusion_matrix2 = confusion_matrix(test_y, prediction)
print(confusion_matrix2)

#Calculate the accuracy
accuracy_score2 = accuracy_score(test_y, prediction)
print(accuracy_score2)
## After reducing the variables,there is not improvment in the accuracy

###################################
################# 4. KNN Models ###################

#importing the library of KNN
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

#Storing the K nearest neighbors classifier
KNN_classifier = KNeighborsClassifier(n_neighbors=5)

#Fitting the values of x an Y
KNN_classifier.fit(train_x, train_y)

#Predicting the test values with model
prediction = KNN_classifier.predict(test_x)

#Performance metrics check
confusion_matrix3 = confusion_matrix(test_y, prediction)
print("\t", "Predicted values")
print("Original Values", "\n", confusion_matrix3)
 
#Calculate the accuracy
accuracy_score3 = accuracy_score(test_y, prediction)
print(accuracy_score3)

#Printing misclassified values from the predictions
print('misclassied sample number :%d' % (test_y != prediction).sum())

###################
# Effect of K value on classifier

misclassied_sample = []

#calculating the error for k value from 1 to 20
for i in range(1, 20):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(train_x,train_y)
    pred_i = knn.predict(test_x)
    misclassied_sample.append((test_y!=pred_i).sum())
    print(i)

print(misclassied_sample)

#So finally chose the k value for which misclassified number is small


######### End of Script #################










