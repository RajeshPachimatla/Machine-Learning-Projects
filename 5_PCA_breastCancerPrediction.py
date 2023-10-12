# -*- coding: utf-8 -*-
"""
@author: Pachimatla Rajesh

-Principal component analysis technique
-Breast cancer detection

"""
#### Import the libraries ###

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
#%matplotlib inline

#lets access inbuilt dataset
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

print(cancer.keys())

print(cancer['DESCR'])
#describe the dataset characteristics

print(cancer['target_names'])

df = pd.DataFrame(cancer['data'], columns = cancer['feature_names'] )
df.head()

#PCA visualisation
#   It is difficult to visualise high dimensional data like this, hence we PCA
# to visualise the data in 2 D
# before doing we need standardise the data so that each variable has single 
#unit variance

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(df)

scaled_data = scaler.transform(df)


#first the PCA module, then create an instance, find the PCA with fit method,
#then apply rotation and dimensionality reduction transo=form()
from sklearn.decomposition import PCA

pca = PCA(n_components=2)

pca.fit(scaled_data)

x_pca  = pca.transform(scaled_data)

print(scaled_data.shape)
print(x_pca.shape)

plt.figure(figsize=(8,6))
plt.scatter(x_pca[:,0], x_pca[:,1],c=cancer['target'],cmap = 'plasma')

plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')

print(pca.components_)
#in this numpy array, each column represents the PCA and column represste
#the its original features

df_comp= pd.DataFrame(pca.components_, columns = cancer['feature_names'])


plt.figure(figsize = (8,6))
sns.heatmap(df_comp, cmap='plasma')

#### End of the script ######

