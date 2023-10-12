# -*- coding: utf-8 -*-
"""
@author: Pachimatla Rajesh

-Principle component analysis
-K-means clustering
-Identifying the countries which needed the aid most

"""
### import the libraries ####
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

country_data = pd.read_csv('Country_data.csv')

country_data.head(5)

country_data.info()

country_data.describe()

#percentage of missing values
print(round(100*(country_data.isnull().sum()/len(country_data.index)),2))

country_data.columns

#Checking for outliers for all the features 

plt.figure(figsize=(20,20), dpi=200)

plt.subplot(4,3,1)
sns.boxplot(x='child_mort',data=country_data)

plt.subplot(4,3,2)
sns.boxplot(x='exports',data=country_data)

plt.subplot(4,3,3)
sns.boxplot(x='health',data=country_data)

plt.subplot(4,3,4)
sns.boxplot(x='imports',data=country_data)

plt.subplot(4,3,5)
sns.boxplot(x='income',data=country_data)

plt.subplot(4,3,6)
sns.boxplot(x='inflation',data=country_data)

plt.subplot(4,3,7)
sns.boxplot(x='life_expec',data=country_data)

plt.subplot(4,3,8)
sns.boxplot(x='total_fer',data=country_data)

plt.subplot(4,3,9)
sns.boxplot(x='gdpp',data=country_data)

#Checking for ooutliers with z score
from scipy import stats

z = np.abs(stats.zscore(country_data[['child_mort', 'exports', 'health', 'imports', 'income',
                                      'inflation', 'life_expec', 'total_fer', 'gdpp']]))

#stats.zscore() is a function from SciPy's stats module used to calculate the 
#Z-scores for the selected columns. Z-scores measure how many standard deviations 
#a data point is away from the mean. It helps in standardizing the data and 
#identifying outliers.


print(z)

print("\n")
print("**********************************************************")
print("\n")

#Select threshold = 3 to identify the outliers
print("Below are the outliers points along with respective column numbers in the second array")
print(np.where(z>3))

country_data_outliers_removed = country_data[(z<3).all(axis=1)]
#country_data[(z < 3).all(axis=1)] filters the rows in the country_data 
#DataFrame based on the condition described in step 1. It keeps only 
#those rows where all Z-scores are less than 3.

print('shape of the dataframe before outlier removal '+str(country_data.shape))
print('Shape of the dataframe after the outlier removal ' + str(country_data_outliers_removed.shape))

X = country_data_outliers_removed.drop('country', axis=1)
y = country_data_outliers_removed['country']

X.shape

y.shape

#Standardization of the dataset before applying PCA
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
#X_scaled is a series here, not a dataframe

X_scaled[:3, :3]

X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

#Lets see the correlation matrix
plt.figure(figsize=(20,20))

xx = X_scaled_df.corr()

sns.heatmap(X_scaled_df.corr(), annot=True)
#here annot = True is to display the correlation strenght values in the cell

#PCA
from sklearn.decomposition import PCA 
#linear dimensionality reduction using singular value decomposition of the data

pca = PCA(random_state=42)
# The random_state parameter is used to set the random seed for reproducibility. 
#By setting it to a specific value (42 in this case), we ensure that the 
#randomness involved in the PCA process is the same every time we run our code 
#with the same dataset.

pca.fit(X_scaled)

print(pca.components_[0])
#pca.components_[0] returns the first principal component from the PCA analysis.
#it is a vector of coefficients that represents the direction in the original 
#feature space along with the data varies the most.

print(pca.explained_variance_ratio_)
#pca.explained_variance_ratio_ is an array where each element corresponds to 
#the proportion of the total variance explained by the respective principal component. 

var_cumu = np.cumsum(pca.explained_variance_ratio_)

fig = plt.figure(figsize=(12,8), dpi=200)
plt.vlines(x=4, ymin=1, ymax=0, colors="r",linestyles="--")
plt.hlines(y=0.95, xmin=0, xmax=30, colors="g", linestyles="--")
plt.plot(var_cumu)
plt.ylabel("Cumulative Variane explained")
plt.show()

#Lets use PCA with 4 components
from sklearn.decomposition import IncrementalPCA
#This can be useful when working with large datasets that don't fit into memory 
#all at once, as Incremental PCA allows you to process the data in smaller 
#chunks or batches.

pca_final = IncrementalPCA(n_components=4)
x_pca_final = pca_final.fit_transform(X_scaled)

print(X.shape)
print(x_pca_final.shape)

######### Clustering ##########

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import cut_tree

#k-means with some orbitrary numbeor of clusters
kmeans = KMeans(n_clusters=5, max_iter=1000)
kmeans.fit(x_pca_final)
kmeans.labels_

#Lets find optimum number of clustes with Elbow method
ssd = []

#range_n_clusters = [2,3,4,5,6,7,8]
for num_clusters in np.arange(2,9,1):
    kmeans = KMeans(n_clusters=num_clusters, max_iter=1000)
    kmeans.fit(x_pca_final)
    
    ssd.append(kmeans.inertia_)

plt.plot(ssd)
#from the Elbow we can say n_clusters can be 4

#Slhoutte analysis    
#the silhouette score of a set of samples, which is a measure of how similar 
#an object is to its own cluster (cohesion) compared to other clusters (separation).
#A higher silhouette score indicates that the clusters are well-separated and 
#that data points within the same cluster are similar to each other.

for num_clusters in np.arange(2,9,1):
    kmeans = KMeans(n_clusters=num_clusters, max_iter=1000)
    kmeans.fit(x_pca_final)
    cluster_labels = kmeans.labels_
    silhoutte_avg = silhouette_score(x_pca_final, cluster_labels)
    print("for n_cluster={0}, the silhoutte score is {1}".format(num_clusters,silhoutte_avg ))
#n_clusters = 3 is giving the highest silhoutte score

#model with n = 3
kmeans = KMeans(n_clusters=4, max_iter=1000, random_state=42)
kmeans.fit(x_pca_final)
kmeans.labels_

country_data_outliers_removed['K-Means_Cluster_ID'] = kmeans.labels_

plt.figure(figsize=(8,4), dpi=200)
sns.boxplot(x='K-Means_Cluster_ID', y='gdpp', data=country_data_outliers_removed)


plt.figure(figsize=(8,4), dpi=200)
sns.boxplot(x='K-Means_Cluster_ID', y='income', data=country_data_outliers_removed)

plt.figure(figsize=(8,4), dpi=200)
sns.boxplot(x='K-Means_Cluster_ID', y='child_mort', data=country_data_outliers_removed)

### Heirarchical Clustering ####
#single Linkage
s1_mergings = linkage(X_scaled_df, method = "single", metric='euclidean' )
dendrogram(s1_mergings)
plt.show()

c1_mergings = linkage(X_scaled_df, method = "complete", metric='euclidean' )
#inkage matrix, which specifies how the clusters are merged during the 
#hierarchical clustering process. You are using the "complete" linkage method,
#which computes the maximum pairwise distance between clusters.
dendrogram(c1_mergings)
plt.show()

#4 clusters using single lankage
s1_cluster_labels = cut_tree(s1_mergings, n_clusters=4).reshape(-1,)
c1_cluster_labels = cut_tree(c1_mergings, n_clusters=4).reshape(-1,)
#The result of cut_tree is initially a 2D array, where each row represents a 
#data point, and the value in each row corresponds to the cluster to which that 
#data point belongs. By calling .reshape(-1,), you reshape this 2D array into a 
#1D array. This 1D array contains the cluster labels for each data point.

country_data_outliers_removed['Hierarchical_cluster_labels'] = c1_cluster_labels

plt.figure(figsize=(20,20), dpi=200)

plt.subplot(3,2,1)
sns.boxplot(x='K-Means_Cluster_ID', y='gdpp', data=country_data_outliers_removed)

plt.subplot(3,2,2)
sns.boxplot(x='Hierarchical_cluster_labels', y='gdpp', data=country_data_outliers_removed)

plt.subplot(3,2,3)
sns.boxplot(x='K-Means_Cluster_ID', y='child_mort', data=country_data_outliers_removed)

plt.subplot(3,2,4)
sns.boxplot(x='Hierarchical_cluster_labels', y='child_mort', data=country_data_outliers_removed)

plt.subplot(3,2,5)
sns.boxplot(x='K-Means_Cluster_ID', y='income', data=country_data_outliers_removed)

plt.subplot(3,2,6)
sns.boxplot(x='Hierarchical_cluster_labels', y='income', data=country_data_outliers_removed)

plt.show()

x_pca_final_df = pd.DataFrame(x_pca_final, columns = ['pc1','pc2','pc3','pc4'])

x_pca_final_df['K_Means_Cluster_ID'] = kmeans.labels_
x_pca_final_df['Hierarchical_Clutser_Labels'] = c1_cluster_labels

x_pca_final_df.head()

#Lets see distribution of first two components to observe the cluster distribution
plt.figure(figsize=(12,6), dpi=200)

plt.subplot(1,2,1)
sns.scatterplot(x = 'pc1', y='pc2', data = x_pca_final_df, hue = 'K_Means_Cluster_ID')

plt.subplot(1,2,2)
sns.scatterplot(x = 'pc1', y='pc2', data = x_pca_final_df, hue = 'Hierarchical_Clutser_Labels')


#Sattter plot using gdpp and child_mort to observe cluster distribution
plt.figure(figsize=(12,6), dpi=200)

plt.subplot(1,2,1)
sns.scatterplot(x = 'gdpp', y='child_mort', data = country_data_outliers_removed, hue = 'K-Means_Cluster_ID')

plt.subplot(1,2,2)
sns.scatterplot(x = 'gdpp', y='child_mort', data = country_data_outliers_removed, hue = 'Hierarchical_cluster_labels')

#Sattter plot using gdpp and income to observe cluster distribution
plt.figure(figsize=(12,6), dpi=200)

plt.subplot(1,2,1)
sns.scatterplot(x = 'gdpp', y='income', data = country_data_outliers_removed, hue = 'K-Means_Cluster_ID')

plt.subplot(1,2,2)
sns.scatterplot(x = 'gdpp', y='income', data = country_data_outliers_removed, hue = 'Hierarchical_cluster_labels')
#We can see near linear relation between gdpp and income

#Sattter plot using child_mort and income to observe cluster distribution
plt.figure(figsize=(12,6), dpi=200)

plt.subplot(1,2,1)
sns.scatterplot(x = 'child_mort', y='income', data = country_data_outliers_removed, hue = 'K-Means_Cluster_ID')

plt.subplot(1,2,2)
sns.scatterplot(x = 'child_mort', y='income', data = country_data_outliers_removed, hue = 'Hierarchical_cluster_labels')
#We can observe from the above figure low income results in higher child mortalitu
#which are in need of dire nead of aid
#from K means clustering we can see cluster 1 is the one needed aid most
K_Means_countries = country_data_outliers_removed[country_data_outliers_removed['K-Means_Cluster_ID']==1]
K_Means_countries.columns

#Similarly from hierarchicla clustering, cluster 0 is need of aid
Hierarchical_countries = country_data_outliers_removed[country_data_outliers_removed['Hierarchical_cluster_labels']==0]
Hierarchical_countries.head()


## Lets find the common countries ###
common_countries = pd.merge(K_Means_countries,Hierarchical_countries,how='inner',on=['country', 'child_mort', 'exports', 'health', 'imports', 'income',
       'inflation', 'life_expec', 'total_fer', 'gdpp', 'K-Means_Cluster_ID',
       'Hierarchical_cluster_labels'])

common_countries.columns

common_countries_final = common_countries[['country','child_mort','gdpp','income']].sort_values(['child_mort','income'],ascending = False)
#sorting the final list with decreasing child mortality rate

#we can select the final countries depening on the budget

Final_countries = common_countries_final[(common_countries_final['child_mort']>80) & (common_countries_final['income']<1200)]
Final_countries = Final_countries.reset_index(drop=True)
print(Final_countries)


############ End of script #########

