# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 18:24:45 2020

@author: Nataniel
"""


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier 


dataset = pd.read_csv('diabetes.csv')
dataset.head()



X = dataset.drop('Outcome',axis=1).values 
y = dataset['Outcome'].values 
X.shape

X_train,X_test , y_train, y_test = train_test_split(X,y, test_size=0.1,random_state=1,stratify= y)

neighbors= np.arange(1,21)

train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len (neighbors))

for i, k in enumerate(neighbors):
    
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    train_accuracy[i] = knn.score(X_train, y_train)
    test_accuracy[i] = knn.score(X_test, y_test)

    
    print(test_accuracy)
   
    

    
    plt.figure(figsize=(12,6))
    plt.title('K-NN nombre de voisins')
    plt.scatter(neighbors, test_accuracy, label='Test Accuracy')
    plt.scatter(neighbors, train_accuracy, label=' Train Accuracy')
    plt.legend()
    plt.xlabel('Nombre de voisins')
    plt.ylabel('Précision')
    plt.show()
    


    
    Knn1 = KNeighborsClassifier(n_neighbors=13)
    Knn1.fit(X,y)  
    Knn1.score(X,y)
    
    topredict = [0, 137, 40, 35, 170, 43, 2.33,34]
    Knn1.predict([topredict]
                )

    
    
    
