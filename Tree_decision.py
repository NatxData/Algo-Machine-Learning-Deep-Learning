# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 15:35:27 2020

@author: Nataniel
"""


import os 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.linear_model import LogisticRegression 
from sklearn.preprocessing import StandardScaler


from os import system 


os.environ["PATH"] += os.pathsep + 'C:/Users/horna/OneDrive/Bureau/graphviz-2.38/release/bin'
 
dataset = pd.read_csv('titanic.csv')
dataset.head()

dataset.drop(['PassengerId','Name', 'SibSp', 'Parch','Ticket', 'Fare', 'Cabin', 'Fare','Embarked'], axis='columns',inplace=True)


Features = dataset.drop('Survived', axis='columns')
y = dataset.Survived
Features.head()


Features.Sex = Features.Sex.map({'male': 1, 'female': 2})
Features.Age = Features.Age.fillna(Features.Age.mean())
Features.head()


    
X_train, X_test, y_train, y_test = train_test_split(Features,y,test_size=0.1)
modelTree = tree.DecisionTreeClassifier(random_state=0, criterion='gini',max_depth=6 )
modelTree.fit(X_train,y_train)
accuracyTreeReel = modelTree.score(X_test,y_test)
accuracyTreeTrain = modelTree.score(X_train,y_train)
    
print('Accuracy Arbre x_test: ', accuracyTreeReel)
print('Accuracy Arbre x_train: ', accuracyTreeTrain)


modelRl = LogisticRegression(random_state = 0, solver='newton-cg')
modelRl.fit(X_train,y_train)
accuracyRlReel = modelRl.score(X_test,y_test)
accuracyRlTrain = modelRl.score(X_train,y_train)
print('Accuracy RL x_test: ', accuracyRlReel)
print('Accuracy RL x_train : ', accuracyRlTrain)


 
dotfile = open("test.dot", 'w')
tree.export_graphviz(modelTree, out_file=dotfile, 
                          feature_names=['Pclass','Genre','Age'],
                          class_names =['Mort','Vivant'],
                          filled=True, rounded=True,
                          special_characters=True)  
   
dotfile.close()

system("dot -Tpng test.dot -o dtree7.png")

modelTree.predict([[1,1,54]])









