# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 17:00:47 2020

@author: horna
"""
"""     ANN avec des données"""
import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 

from sklearn.preprocessing import StandarScaler 

"""importer le dataset"""

dataset=pd.read_csv('datasets/titanic.csv')
dataset.head()

"prépare les données"
dataset.drop(['PassengerId', 'Name', 'Sibsp', 'Parch', 'Ticket', 'Cabin', 'Embarked'], axis='columns', inplace=True)
"""modifier des variables catégorique"""
dataset.Sex=dataset.Sex.map(('male':0, 'female':1)) 
dataset.head()

"""séparation du dataset"""
target = dataset.Survived.values
dataset.drop(['Survived'], axis='columns', inplace=True)
data = dataset.astype(float)
data.age = data.Age.fillna(data.Age.mean())
data = data.values

scaler= StandarScaler()"""entre -1 et 1"""
data = scaler.fit_transform(data)
data

"""creation du modele"""

#add the layers 

model.add(Dense(4, activation="relu"))

model.add(Dense(256, activation="relu"))
model.add(Dense(129, activation="relu"))
model.add(Dense(2, activation="relu"))

#compile the model
model.compile(
        loss="sparse_categorical_crossentropy"),
        optimizer="agd",
        metrics=["accuracy"]
        )

"""entrainement"""
history = model.fit(data,target, epochs=50, batch_size=10,validation_split=0.2)
topred= scaler.transform([[3,0,22,40]])
prediction = model.predict(topred)
print("dicaprio chance de survie:",pred[0][1])

"""courbe d'apprentisage"""
print(history.history.keys())

"""construction graphique"""
import matplotlib.pyplot as plt 

# summarise history for accuraccy 
plt.plot(history.history)

















