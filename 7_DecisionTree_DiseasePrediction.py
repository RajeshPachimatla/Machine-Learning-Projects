# -*- coding: utf-8 -*-
"""
@author: Pachimatla Rajesh

-Machine Learning Algorithm
-Decision Tree for deisease prediction

"""
## import relevant libraries ###

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

#import the training data, here it is given as seperately
df = pd.read_csv('Training.csv')
df.fillna(0, inplace=True)
#df.isnull().sum()
df.describe()

df.shape

df.columns

df['prognosis'].value_counts()
#Gives the count of all the disease variety

#Lets split the data for train and test

#df.dropna(axis=0, inplace=True)

x = df.drop('prognosis', axis=1)

y = df['prognosis']

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=42)

#Lets create the instance of DecisionTreeclassifier
tree = DecisionTreeClassifier()

#first the fit the train data x vs y
tree.fit(x_train, y_train)

predictions = tree.predict(x_test)

Accuracy = tree.score(x_test, y_test)

print('accuracy of test set {:.2f}%'.format(Accuracy*100))

#create the summery of the results
print(tree.feature_importances_)
#Random forests are ensembles of decision trees. In this context, tree.feature_importances_ 
#provides the feature importances averaged over all the decision trees in the 
#ensemble. It helps identify which features are most influential in the ensemble's 
#predictions.
feature_imp = pd.DataFrame(tree.feature_importances_*100, x_test.columns, columns=['Importance'])

feature_imp.sort_values(by='Importance', ascending=False, inplace=True)

print(feature_imp.iloc[0])

############ End of the script #######




