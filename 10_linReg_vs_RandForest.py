# -*- coding: utf-8 -*-
"""
@author: Pachimatla Rajesh

Predicting Price of Pre-owned cars by

Linear Regression and Random Forest

"""
#### 1. Importing necessary Libraries ###
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

#Setting dimensions for plot

sns.set(rc={'figure.figsize':(11.7, 8.27)})

######### 2. Explore the data given ########
cars_data = pd.read_csv('cars_sampled.csv')

#creating a copy of data
cars = cars_data.copy()

cars.info()

cars.describe()

pd.set_option('display.float_format', lambda x: '%0.3f' % x)
cars.describe()

#To display maximum number of columns in console
pd.set_option('display.max_columns', 500)
cars.describe()

cars.columns

#Dropping unwanted columns
col = ['name', 'dateCrawled', 'dateCreated','postalCode','lastSeen']
cars = cars.drop(columns=col, axis=1)

#removing duplicate records
cars.drop_duplicates(keep='first', inplace=True)

### check for null values ###
cars.isnull().sum()

####Lets check the variable wise and fine outliers by plotting #####
# Variable 1: yearOfRegistration
yearwise_count = cars['yearOfRegistration'].value_counts().sort_index()
sum(cars['yearOfRegistration'] > 2018)
sum(cars['yearOfRegistration'] < 1950)

sns.regplot(x='yearOfRegistration', y='price', scatter=True, fit_reg=False, data=cars)

#lets chop off the data below 1950 and above 2018

# variable 2: price
price_count = cars['price'].value_counts().sort_index()

sns.distplot(cars['price'])

cars['price'].describe()

sns.boxplot(y=cars['price'])
sum(cars['price']>150000)
sum(cars['price']<100)
#after looking at the price data, it is better to fix the price range between 100 and 150000

# variable 3: powerPS
power_count = cars['powerPS'].value_counts().sort_index()
sns.distplot(cars['powerPS'])
cars['powerPS'].describe()
sns.boxplot(y=cars['powerPS'])
sns.regplot(x='powerPS', y='price', scatter=True, fit_reg=False, data=cars)
sum(cars['powerPS']>500)
sum(cars['powerPS']<10)

## Working range of data ##

cars = cars[(cars.yearOfRegistration <=2018)
            & (cars.yearOfRegistration >= 1950)
            & (cars.price >= 100)
            & (cars.price <= 150000)
            & (cars.powerPS >= 10)
            & (cars.powerPS <= 500)]
#by doing above, -6700 records are dropped

#Further reducing the variables
#combine yearofregistration and monthofregistration

cars['monthOfRegistration']/=12

#creating new variable Age by year and month of registration
#Current year 2018

cars['Age']=(2018-cars['yearOfRegistration'])+cars['monthOfRegistration']
cars['Age'] = round(cars['Age'],2)
cars['Age'].describe()

#Droping year of registration and monthofregitration

cars=cars.drop(columns=['yearOfRegistration','monthOfRegistration'], axis=1)

#Visualising Parameters
#Age
sns.distplot(cars['Age'])
sns.boxplot(y=cars['Age'])

#Price
sns.distplot(cars['price'])
sns.boxplot(y=cars['price'])

#powerPS
sns.distplot(cars['powerPS'])
sns.boxplot(y=cars['powerPS'])

#Check for Relationship between independent variables, if they are co-related

correlation = cars.corr(numeric_only=True)
print(correlation)

#Visualising parameters after reducing the working range

#Age vs Price
sns.regplot(x='Age', y='price', scatter=True,fit_reg=False,data=cars)
#New cars are having relatively higher price
#However some cars are prices higher with age

#powerPs vs Price
sns.regplot(x='powerPS', y='price', scatter=True, fit_reg=False,data=cars)

#Variable 4: seller
cars['seller'].value_counts()
pd.crosstab(cars['seller'], columns='count',normalize=True)
sns.countplot(x='seller',data=cars)
#Commercial type sellers are very insigificant ==> so insignificant variable

#Variable 5: offer type
cars['offerType'].value_counts()
pd.crosstab(cars['offerType'], columns='count',normalize=True)
sns.countplot(x='offerType',data=cars)
#All cars have offer ==> insignificant variable

#Variable 6: abtest
cars['abtest'].value_counts()
pd.crosstab(cars['abtest'], columns='count',normalize=True)
sns.countplot(x='abtest',data=cars)

#Equally distributed test and control abtest

sns.boxplot(data=cars,x='abtest',y='price')
#Since for every price value there is 50-50 distribution, so we consider
#it will not effect price ==> insignificant

#Variable 7: Vehicle Type
cars['vehicleType'].value_counts()
pd.crosstab(cars['vehicleType'], columns='count',normalize=True)
sns.countplot(x='vehicleType',data=cars)
sns.boxplot(data=cars,x='vehicleType',y='price')

#Variable 8: gearbox
cars['gearbox'].value_counts()
pd.crosstab(cars['gearbox'], columns='count',normalize=True)
sns.countplot(x='gearbox',data=cars)
sns.boxplot(data=cars,x='gearbox',y='price')
#gear box effects the price

#variable 9: model
cars['model'].value_counts()
pd.crosstab(cars['model'], columns='count',normalize=True)
sns.countplot(x='model',data=cars)
sns.boxplot(data=cars,x='model',y='price')
#Cars are distributed over many models
#hence, considering in modeling

#variable 10: Kilometer
cars['kilometer'].value_counts()
pd.crosstab(cars['kilometer'], columns='count',normalize=True)
sns.countplot(x='kilometer',data=cars)
sns.boxplot(data=cars,x='kilometer',y='price')
cars['kilometer'].describe()

#variable 11: FuelType
cars['fuelType'].value_counts()
pd.crosstab(cars['fuelType'], columns='count',normalize=True)
sns.countplot(x='fuelType',data=cars)
sns.boxplot(data=cars,x='fuelType',y='price')
# fueltype is affects price

#Variable 12: brand
cars['brand'].value_counts()
pd.crosstab(cars['brand'], columns='count',normalize=True)
sns.countplot(x='brand',data=cars)
sns.boxplot(data=cars,x='brand',y='price')
#Cars are distributed over many cars
#considering for modeling

#Variable 13: notRepairedDamage
# yes - car damaged but not repaired
# no - car damaged but repaired
cars['notRepairedDamage'].value_counts()
pd.crosstab(cars['notRepairedDamage'], columns='count',normalize=True)
sns.countplot(x='notRepairedDamage',data=cars)
sns.boxplot(data=cars,x='notRepairedDamage',y='price')
#Where cars have not been rectified falls under lower price range


########### removing insigficant variables #########3

col = ['seller', 'offerType', 'abtest']
cars = cars.drop(columns=col,axis=1)
cars_copy = cars.copy()

##### Correlation ####
cars_select1 = cars.select_dtypes(exclude=[object])
correlation = round(cars_select1.corr(),3)
print(correlation)

cars_select1.corr().loc[:,'price'].abs().sort_values(ascending=False)[1:]
#The above will display the first column of corr with absolute value excluding 
#price value
"""
We are going to build a linear regression and Random forest model on two datasets
1. data obtained by ommitting rows with any missing value
2. data obtained by imputing the missing values
"""

### Ommitting Missing Rows ###

cars_omit = cars.dropna(axis=0)
#here axis = 0 indicates omit rows

#Converting categorical variables to dummy variables
cars_omit = pd.get_dummies(cars_omit, drop_first=True)

###############2. Model Building with omitted data ###############

#seperating input and output features
x1 = cars_omit.drop(['price'],axis='columns',inplace=False)
y1 = cars_omit['price']

#Plotting the variable price
prices = pd.DataFrame({"1. Before":y1, "2. After":np.log(y1)})
prices.hist()

#Transforming price as a logarithmic value
y1 = np.log(y1)

#Splitting data into test and train
x_train, x_test, y_train, y_test = train_test_split(x1,y1,test_size=0.3, random_state=3)

print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)

### Baseline model for omitted data ###
"""
Lets make a base line using test data mean value
this is to set a benchmark and to compare with our regression model
"""

#finding the mean for test data value
base_pred = np.mean(y_test)
print(base_pred)

#regenerating the same value till the length of the testdata
base_pred = np.repeat(base_pred, len(y_test))

#Finding the RMSE root mean square error
base_root_mean_square_error1 = np.sqrt(mean_squared_error(y_test, base_pred))
                                      
print(base_root_mean_square_error1)

####Model 1: Linear regression with omitted data ###

#Setting intercept as true
lgr = LinearRegression(fit_intercept=True)

#Model
model_lin1 = lgr.fit(x_train, y_train)

#Predicting the model on test set
cars_predictions_lin1 = lgr.predict(x_test)

#Computing MSE and RMSE
lin_mse1 = mean_squared_error(y_test, cars_predictions_lin1)
lin_rmse1 = np.sqrt(lin_mse1)
print(lin_rmse1)

# R squared value
r2_lin_test1 = model_lin1.score(x_test,y_test)
r2_lin_train1 = model_lin1.score(x_train,y_train)
print(r2_lin_test1,r2_lin_train1)

#Regression diagnostics - residual plot analysis
residual1 = y_test - cars_predictions_lin1
sns.regplot(x=cars_predictions_lin1, y=residual1, scatter=True, fit_reg=False)
residual1.describe()
#residuals are close to zero as we can also see from mean value

##### Model 2: Random Forest with omitted data ####

#Model Parameter

rf = RandomForestRegressor(n_estimators=100, max_features='auto',
                           max_depth=100,min_samples_split=10,
                           min_samples_leaf=4,random_state=1)
#n_estimators: This parameter specifies the number of decision trees 
#max_features: This parameter controls the number of features each decision 
#tree is allowed to consider when making a split. 
#'auto' means it will consider all features for each split.
#max_depth=100, which means no tree in the forest will have more than 100 
#levels of nodes from the root to a leaf.
#min_samples_leaf=10,which means a node won't split further if it contains 
#fewer than 10 samples.
#min_samples_leaf: This parameter sets the minimum number of samples required 
#to be at a leaf node.

#Model
model_rf1 = rf.fit(x_train,y_train)

#predicting the model on test set
cars_predictions_rf1 = rf.predict(x_test)

#Computing MSE and RMSE
rf_mse1 = mean_squared_error(y_test, cars_predictions_rf1)
rf_rmse1 = np.sqrt(rf_mse1)
print(rf_rmse1)

# R squared value
r2_rf_test1 = model_rf1.score(x_test,y_test)
r2_rf_train1 = model_rf1.score(x_train,y_train)
print(r2_lin_test1,r2_lin_train1)


############ 4. Model Building with Imputed data ##########

cars_imputed = cars.apply(lambda x:x.fillna(x.median())\
                          if x.dtype == 'float' else \
                              x.fillna(x.value_counts().index[0]))
#If the column is of type 'float', it fills missing values with the median of that
#column (x.median()). This is a common strategy for imputing missing values in numeric columns.

#If the column is not of type 'float' (assumed to be categorical), it fills missing
# values with the most frequent value in that column (x.value_counts().index[0]), 
#which is essentially the mode of the column. This is a common strategy for imputing missing values in categorical columns.

cars_imputed.isnull().sum()

#Converinf the categorical variables to dummy variable
cars_imputed = pd.get_dummies(cars_imputed,drop_first=True)

## Model Building with imputed data ###

#seperative input and output variables
x2 = cars_imputed.drop(['price'],axis='columns',inplace=False)
y2 = cars_imputed['price']

#Plotting the variable price
prices = pd.DataFrame({"1. Before":y2, "2. After":np.log(y2)})
prices.hist()

#Transforming price as a logarithmic value
y2 = np.log(y2)

#Splitting data into test and train
x_train1, x_test1, y_train1, y_test1 = train_test_split(x2,y2,test_size=0.3, random_state=3)

print(x_train1.shape,x_test1.shape,y_train1.shape,y_test1.shape)

### Baseline model for omitted data ###
"""
After imput
Lets make a base line using test data mean value
this is to set a benchmark and to compare with our regression model
"""

#finding the mean for test data value
base_pred = np.mean(y_test1)
print(base_pred)

#regenerating the same value till the length of the testdata
base_pred = np.repeat(base_pred, len(y_test1))

#Finding the RMSE root mean square error
base_root_mean_square_error2 = np.sqrt(mean_squared_error(y_test1, base_pred))
                                      
print(base_root_mean_square_error2)

####### Model 3: Linear regression with omitted data #####333

#Setting intercept as true
lgr2 = LinearRegression(fit_intercept=True)

#Model
model_lin2 = lgr.fit(x_train1, y_train1)

#Predicting the model on test set
cars_predictions_lin2 = lgr.predict(x_test1)

#Computing MSE and RMSE
lin_mse2 = mean_squared_error(y_test1, cars_predictions_lin2)
lin_rmse2 = np.sqrt(lin_mse2)
print(lin_rmse2)

# R squared value
r2_lin_test2 = model_lin2.score(x_test1,y_test1)
r2_lin_train2 = model_lin2.score(x_train1,y_train1)
print(r2_lin_test2,r2_lin_train2)

######## Model4: Random forest with Imputed data ####

#Model Parameter
rf2 = RandomForestRegressor(n_estimators=100, max_features='auto',
                           max_depth=100,min_samples_split=10,
                           min_samples_leaf=4,random_state=1)

#Model
model_rf2 = rf2.fit(x_train1,y_train1)

#predicting the model on test set
cars_predictions_rf2 = rf2.predict(x_test1)

#Computing MSE and RMSE
rf_mse2 = mean_squared_error(y_test1, cars_predictions_rf2)
rf_rmse2 = np.sqrt(rf_mse2)
print(rf_rmse2)

# R squared value
r2_rf_test2 = model_rf2.score(x_test1,y_test1)
r2_rf_train2 = model_rf2.score(x_train1,y_train1)
print(r2_lin_test2,r2_lin_train2)

##########################################################33
#Final Output

print("Metrics for module built from data where missing values were omitted")
print("R squared value from train from linear regression = %s" % r2_lin_train1)
#print("R squared value from train from linear regression = %.3f" % r2_lin_train1)
#print("R squared value from train from linear regression = {}".format(round(r2_lin_train1,3)))
print("R squared value from test from linear regression = %s" % r2_lin_test1)
print("R squared value from train from random forest = %s" % r2_rf_train1)
print("R squared value from test from random forest = %s" % r2_rf_test1)
print("Base RMSE of model built from data where missing values are omitted = %s" % base_root_mean_square_error1)
print("RMSE value for test from linear regression = %s" % lin_rmse1)
print("RMSE value for test from random forest = %s" % rf_rmse1)
print("\n\n")
print("Metrics for module built from data where missing values were imputed")
print("R squared value from train from linear regression = %s" % r2_lin_train2)
#print("R squared value from train from linear regression = %.3f" % r2_lin_train2)
#print("R squared value from train from linear regression = {}".format(round(r2_lin_train2,3)))
print("R squared value from test from linear regression = %s" % r2_lin_test2)
print("R squared value from train from random forest = %s" % r2_rf_train2)
print("R squared value from test from random forest = %s" % r2_rf_test2)
print("Base RMSE of model built from data where missing values are omitted = %s" % base_root_mean_square_error2)
print("RMSE value for test from linear regression = %s" % lin_rmse2)
print("RMSE value for test from random forest = %s" % rf_rmse2)

# Random forest is better than linear regression in this case
# Model with missing is better than imputed data

################ END OF SCRIPT ##################











































