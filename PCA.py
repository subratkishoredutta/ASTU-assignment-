# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 10:27:46 2020

@author: Asus
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
##laoding data
pca_data=pd.read_csv('PCA_practice_dataset.csv')
x=pca_data.to_numpy()
print(x.shape)
scaler = StandardScaler()
x=scaler.fit_transform(x)

##PCA
pca=PCA()

x=pca.fit_transform(x)

cum_var = np.cumsum(pca.explained_variance_ratio_)*100

thresholds = [i for i in range(90,98)]

components = [np.argmax(cum_var>threshold) for threshold in thresholds]

for component,threshold in zip(components,thresholds):
    print("component we got for threshold "+str(threshold)+"% are :",component)
    
##plot corresponding to principle components vs threshold
plt.ylabel("threshold")
plt.xlabel("principle component")
plt.plot(components,range(90,98),'o-')



##dimensionality reduction 
M=x
for component ,threshold in zip(components,thresholds):
    pca=PCA(n_components=component)
    newX=pca.fit_transform(M)
    print("\nthreshold: ",threshold)
    print("shape after dimensionality reduction: ",newX.shape)
    