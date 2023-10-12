# -*- coding: utf-8 -*-
"""
@author: Pachimatla Rajesh

-Decision Tree for policy prediction

"""
#### import libraries #####
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree

balance_data = pd.read_csv('Decision_Tree_ Dataset.csv',sep= ',', header= 0)

print ("Dataset Length:: ", len(balance_data))
print ("Dataset Shape:: ", balance_data.shape)

print ("Dataset:: ")
balance_data.head()

X = balance_data.values[:, 1:5]
Y = balance_data.values[:,5]

X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 100)

clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,
 max_depth=3, min_samples_leaf=5)
#The above step creates the instance of Decission tree
#criteria: can be entropy, gini or mse for spliting the nodes
#random_state: the parameter sets the random seed for reproducibility
#max_depth: indicates the depth of decision tree, this can be used to prevent overfitting
#min_samples_leaf: atleast 5 samples at the final node

clf_entropy.fit(X_train, y_train)

y_pred = clf_entropy.predict(X_test)
y_pred

print ("Accuracy is ", accuracy_score(y_test,y_pred)*100)


######### End of Script ######




