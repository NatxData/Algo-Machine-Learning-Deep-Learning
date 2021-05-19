# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 15:36:19 2020

@author: Nataniel
"""



import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split


from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV 
acc_scorer = make_scorer(accuracy_score)

df = pd.read_csv('titanic.csv')
df.head()

df.describe()

df.dropna(subset=['Age'],inplace=True)
df.Age = df.Age.astype(int)
df.head()


df.Sex = df.Sex.map({'male': 0, 'female':1})

df['Cabin'].value_counts()


df.drop(['PassengerId','Name','Ticket','Cabin'],axis='columns', inplace=True)
df.describe()

table = pd.crosstab(df.Age,df.Survived)
table.div(table.sum(1).astype(float),axis=0).plot(kind='bar', stacked=True)
plt.title('Age/Survie')
plt.xlabel('Survie')
plt.ylabel('%Survie')



df['AgeBand'] = pd.cut(df['Age'],5)

df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)
df.head()

df.loc[df['Age'] <= 16, 'Age'] = 0
df.loc[(df['Age']>16)& (df['Age']<= 32), 'Age']=1
df.loc[(df['Age']>32)& (df['Age']<= 48), 'Age']=2
df.loc[(df['Age']>48)& (df['Age']<= 64), 'Age']=3
df.loc[(df['Age']>64), 'Age']=4

df.drop(['AgeBand'],axis=1,inplace=True)


df.head()

table= pd.crosstab(df.Age,df.Survived)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar',stacked=True)
plt.title('Age/Survie')
plt.xlabel('Survie')
plt.ylabel('%survie') 

table= pd.crosstab(df.Sex,df.Survived)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar',stacked=True)
plt.title('Sex/Survie')
plt.xlabel('Survie')
plt.ylabel('%survie') 


table= pd.crosstab(df.Pclass,df.Survived)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar',stacked=True)
plt.title('Pclass/Survie')
plt.xlabel('Survie')
plt.ylabel('%survie') 

table= pd.crosstab(df.Fare,df.Survived)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar',stacked=True)
plt.title('Fare/Survie')
plt.xlabel('Survie')
plt.ylabel('%survie') 

df['FareGroup'] = pd.qcut(df['Fare'],3)
df[['FareGroup','Survived']].groupby(['FareGroup'], as_index=False).mean().sort_values(by='FareGroup',ascending=True)

df.loc[df['Fare'] <= 10.462, 'Fare'] = 0
df.loc[(df['Fare'] >10.462) & (df['Fare']<= 26.55), 'Fare']=1
df.loc[df['Fare'] >26.55, 'Fare']= 2
      
df['Fare'] = df['Fare'].astype(int)
df.drop(['FareGroup'],axis=1,inplace=True)

df.head()

table= pd.crosstab(df.Fare,df.Survived)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar',stacked=True)
plt.title('Fare/Survie')
plt.xlabel('Survie')
plt.ylabel('%survie') 



table= pd.crosstab(df.Embarked,df.Survived)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar',stacked=True)
plt.title('Embarked/Survie')
plt.xlabel('Survie')
plt.ylabel('%survie') 

table= pd.crosstab(df.SibSp,df.Survived)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar',stacked=True)
plt.title('Sibsp/Survie')
plt.xlabel('Survie')
plt.ylabel('%survie') 


table= pd.crosstab(df.Parch,df.Survived)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar',stacked=True)
plt.title('Parch/Survie')
plt.xlabel('Survie')
plt.ylabel('%survie') 



df['FamilySize'] = df['SibSp']+df['Parch']+1

table= pd.crosstab(df.FamilySize,df.Survived)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar',stacked=True)
plt.title('Family/Survie')
plt.xlabel('Survie')
plt.ylabel('%survie') 


df.drop(['SibSp','Parch'],axis='columns', inplace=True)

Features = df.drop('Survived',axis='columns')
y = df.Survived
Features.head()
Features.shape


Features.head()


Features.tail()

Features.drop(['Embarked','Pclass'],axis=1,inplace=True)

Features.head()

X_train, X_test, y_train, y_test = train_test_split(Features,y, test_size=0.3, random_state=0)

random_forest = RandomForestClassifier()
parameters = {'n_estimators':[3,4,5,10,15,20,25],'criterion': ['entropy','gini'],'max_depth':[2,3,5,10]}

grid_obj =GridSearchCV(random_forest,parameters, scoring=acc_scorer,cv=3)



grid_obj = grid_obj.fit(X_train, y_train)
clf = grid_obj.best_estimator_
print(grid_obj.best_estimator_)


clf.fit(X_train, y_train)
print(clf.score(X_train, y_train))
print(clf.score(X_test, y_test))
print(clf.score(Features, y))



























