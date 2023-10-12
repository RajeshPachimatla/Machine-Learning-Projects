# -*- coding: utf-8 -*-
"""
@author: Pachimatla Rajesh

-clustering by K-means using SKlearn

"""
#### import the necessary libraries ###
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Set the styles to Seaborn
sns.set()
# Import the KMeans module so we can perform k-means clustering with sklearn
from sklearn.cluster import KMeans

# Load the country clusters data
data = pd.read_csv('Country_clusters.csv')

# Use the simplest code possible to create a scatter plot using the longitude and latitude
# Note that in order to reach a result resembling the world map, we must use the longitude as y, and the latitude as x
plt.scatter(data['Longitude'],data['Latitude'])
# Set limits of the axes, again to resemble the world map
plt.xlim(-180,180)
plt.ylim(-90,90)
plt.show

# iloc is a method used to 'slice' data 
# 'slice' is not technically correct as there are methods 'slice' which are a bit different
# The term used by pandas is 'selection by position'
# The first argument of identifies the rows we want to keep
# The second - the columns
# When choosing the columns, e.g. a:b, we will keep columns a,a+1,a+2,...,b-1 ; so column b is excluded
x = data.iloc[:,1:3]
# for this particular case, we are choosing columns 1 and 2
# Note column indices in Python start from 0

# Create an object (which we would call kmeans)
# The number in the brackets is K, or the number of clusters we are aiming for
#If K = 2
kmeans = KMeans(2)

# Fit the input data, i.e. cluster the data in X in K clusters
kmeans.fit(x)

# Create a variable which will contain the predicted clusters for each observation
identified_clusters = kmeans.fit_predict(x)
# Check the result
identified_clusters

# Create a copy of the data
data_with_clusters = data.copy()
# Create a new Series, containing the identified cluster for each observation
data_with_clusters['Cluster'] = identified_clusters
# Check the result
data_with_clusters

# Plot the data using the longitude and the latitude
# c (color) is an argument which could be coded with a variable 
# The variable in this case has values 0,1,2,.. indicating to plt.scatter, that there are three colors (0,1,2)
# All points in cluster 0 will be the same colour, all points in cluster 1 - another one, etc.
# cmap is the color map. Rainbow is a nice one, but you can check others here: https://matplotlib.org/users/colormaps.html
plt.scatter(data_with_clusters['Longitude'],data_with_clusters['Latitude'],c=data_with_clusters['Cluster'],cmap='rainbow')
plt.xlim(-180,180)
plt.ylim(-90,90)
plt.show()

#If K = 3
kmeans1 = KMeans(3)
kmeans1.fit(x)
identified_clusters1 = kmeans1.fit_predict(x)
data_with_clusters1 = data.copy()
data_with_clusters1['cluster']=identified_clusters1

plt.scatter(data_with_clusters1['Longitude'], data_with_clusters1['Latitude'],c=data_with_clusters1['cluster'],cmap='rainbow')
plt.xlim(-180,180)
plt.ylim(-90,90)
plt.show()

#### Selecting number of clusters, Elbow method ########
# Get the WCSS for the current solution
kmeans.inertia_
#The kmeans.inertia_ attribute is used to compute the sum of squared distances 
#from each data point to its assigned cluster center in a k-means clustering model.
#This sum of squared distances is often referred to as the "inertia" or 
#"within-cluster sum of squares." It is a measure of how compact the clusters are. 
#Lower inertia values indicate that the data points are closer to their cluster 
#centers, which is generally desirable in k-means clustering.

# Create an empty list
wcss=[]

# Create all possible cluster solutions with a loop
for i in range(1,7):
    # Cluster solution with i clusters
    kmeans = KMeans(i)
    # Fit the data
    kmeans.fit(x)
    # Find WCSS for the current iteration
    wcss_iter = kmeans.inertia_
    # Append the value to the WCSS list
    wcss.append(wcss_iter)
    
# Create a variable containing the numbers from 1 to 6, so we can use it as X axis of the future plot
number_clusters = range(1,7)
# Plot the number of clusters vs WCSS
plt.plot(number_clusters,wcss)
# Name your graph
plt.title('The Elbow Method')
# Name the x-axis
plt.xlabel('Number of clusters')
# Name the y-axis
plt.ylabel('Within-cluster Sum of Squares')

#As per elbow method, ideal K value is 3


############# End of Script ############




