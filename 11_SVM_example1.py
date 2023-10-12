# -*- coding: utf-8 -*-
"""
@author: Pachimatla Rajesh

-Support Vector Machine
-
"""
## import the libraries ###
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs

#Lets create 40 seperable points
X, y = make_blobs(n_samples = 40, centers = 2, random_state = 20)

#fit to the model without regularization
clf = svm.SVC(kernel= 'linear', C=1)
#Create an SVM classifier: You create an SVM classifier object clf using the svm.
#SVC constructor. In this example, you specify a linear kernel by setting 
#kernel='linear', and you set the regularization parameter C to 1. 
clf.fit(X,y)

#Lets see the data in scatter plot
plt.scatter(X[:,0],X[:,1], c=y, s=30, cmap=plt.cm.Paired)
plt.xlabel('feature x')
plt.ylabel('feature y')
plt.colorbar()
plt.show()
#s=30: This parameter sets the size of the markers representing the data points.
#cmap=plt.cm.Paired: This parameter specifies the colormap to be used for coloring 
#the points. plt.cm.Paired is a predefined colormap.

newdata = [[3,4],[5,6]]
print(clf.predict(newdata))
## End of the script ##


