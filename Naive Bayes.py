# -*- coding: utf-8 -*-

"""
Created on Sun Dec 13 19:30:12 2020

@author: Nataniel
"""



import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, CategoricalNB


dataset = pd.read_csv('titanic.csv')
dataset.head()


dataset.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked'], axis='columns', inplace=True)
dataset.head()


X = dataset.drop('Survived',axis= 'columns')
Y = dataset.Survived 

X['Sex']=X['Sex'].apply(lambda x: 1 if x=='female' else 0)
X.head()


X.columns[X.isna().any()]
X.describe()

X.fillna(X.mean(), inplace=True)
X.describe()


X_train, X_test, y_train,y_test = train_test_split(X,Y, test_size=0.2)
model = CategoricalNB()
model.fit(X_train, y_train)
model.score(X_test, y_test)
model.score(X_train, y_train)

data =pd.read_csv('spam.csv.xlsx')
data.head()
from sklearn.feature_extraction.text import CountVectorizer
data.describe()
data.groupby('Category').describe()
data['spam']=data['Category'].apply(lambda x: 1 if x=='spam' else 0)
data.head()
X_train, X_test, y_train, y_test = train_test_split(data.Message, data.spam)

v = CountVectotrizer()
X_train_count = v.fit_transform(X_train.values)
X_train_count.toarray([:2])

model = MultinomialNB
model.fit(X_train_count, y_train)

X_test_count = v.transform(X_test)
model.score(X_test_count, y_test)




from sklearn.pipeline import Pipeline 
clf = Pipeline ([
        ('vectorizer', CountVectorizer()),
        ('nb', MultinomialeNB())
        ])


















