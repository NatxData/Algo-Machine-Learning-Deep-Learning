# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 14:42:53 2020

@author: Nataniel
"""

import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm

dataset = pd.read_csv('titanic.csv')
dataset.head()
dataset.describe()

dataset.fillna(dataset.mean(), inplace=True)

dataset.describe()

dataset.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked'], axis='columns', inplace=True)
dataset.Sex = dataset.Sex.map({'male': 0 , 'female':1})
dataset.head()

X = dataset.drop(['Survived'],axis= 'columns')
y = dataset.Survived

X_train, X_test , y_train, y_test = train_test_split(X,y, test_size = 0.1)

X_train.head()

svclassifier = svm.SVC(kernel= 'linear',random_state = 0)


svclassifier.fit(X_train, y_train)

y_pred= svclassifier.predict(X_test)

svclassifier.score(X_test,y_test)


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,y_pred))
