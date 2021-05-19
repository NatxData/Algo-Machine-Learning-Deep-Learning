# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 18:24:53 2020

@author: Nataniel
"""



import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import sklearn.metrics as sm 

from sklearn import datasets


iris = datasets.load_iris()


x=pd.DataFrame(iris.data)

x.columns=['Sepal_Length','Sepal_width','Petal_Length', 'Petal_width']
y=pd.DataFrame(iris.target)
from sklearn.cluster import KMeans
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++',random_state = 0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)     
    
    plt.plot(range(1, 11), wcss)
    plt.title('La methode elbow')
    plt.xlabel('nbr de clusters ')
    plt.ylabel('Wcss')
    plt.show()
   
    
    
model=KMeans(n_clusters=3)

model.fit(x) 

colormap=np.array(['red','green', 'blue'])
plt.scatter(x.Petal_Length, x.Petal_width, c=colormap[model.labels_],s=40)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,3))
fig.suptitle('Horizontally stacked subplots')
ax1.scatter(x.Petal_Length, x.Petal_width, c=colormap[model.labels_],s=40)
ax2.scatter(x.Petal_Length, x.Petal_width, c=colormap[iris.target],s=40)


