# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 12:26:30 2020

@author: Nataniel
"""



import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report 
from matplotlib.colors import ListedColormap 
from mpl_toolkits.mplot3d import Axes3D 
import seaborn as sns 


dataset = pd.read_csv('clients.csv')

 
dataset.head()


plt.scatter(dataset.EstimatedSalary, dataset.Purchased)




dataset.drop(['User ID'], axis='columns', inplace=True)

dataset.head()


dataset.Gender = dataset.Gender.map({'Male':1, 'Female': 2})
dataset.head()


ax = plt.axes(projection='3d')
ax.scatter(dataset.Gender,dataset.Age, dataset.EstimatedSalary, c=dataset.Purchased) 


count_sub = len(dataset[dataset['Purchased']==1])
count_no_sub =len(dataset[dataset['Purchased']==0])
pct_of_no_sub = count_no_sub/(count_no_sub+count_sub)
print("pourcentage absence d'achat :",  pct_of_no_sub* 100)


table = pd.crosstab(dataset.Gender, dataset.Purchased)
table.div(table.sum(1).astype(float),axis=0).plot(kind='bar', stacked='True')
plt.tilte('Genre / achat')
plt.xlabel('Genre')
plt.ylabel('Pourcentagede client')


dataset.drop(['Gender'], axis='columns', inplace= True)
dataset.head()

table = pd.crosstab(dataset.Age, dataset.Purchased )
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked= True)
plt.title('Age /Achat')
plt.xlabel('Age')
plt.ylabel('Pourcentage de client')
plt.savefig('Age-Achat')


table = pd.crosstab(dataset.EstimatedSalary, dataset.Purchased )
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked= True)
plt.title('Salaire /Achat')
plt.xlabel('Salaire')
plt.ylabel('Pourcentage de client')
plt.savefig('salaire-Achat')







X = dataset.iloc[:, [0,1]].values 
y = dataset.iloc[:,-1].values


plt.scatter(X[:,0], X[:,1], c = y) 
 


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.25, random_state = 0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test) 
X_test



classifier = LogisticRegression(random_state = 0, solver ='liblinear')

classifier.fit(X_train, y_train)




y_pred = classifier.predict(X_test) 
classifier.score(X_test,y_test)




cm = confusion_matrix(y_test, y_pred) 
print(cm)

x_predict = sc.transform([[30,15000]])
classifier.predict(x_predict)

X_set, y_set = X_train, y_train

X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                         np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha = 0.4, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Résultats du Training set')
plt.xlabel('Age')
plt.ylabel('Salaire Estimé')
plt.legend()
plt.show()










































