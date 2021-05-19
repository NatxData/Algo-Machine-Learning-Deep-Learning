

import matplotlib.pyplot as plt 
import pandas as pd 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import r2_score 
from  sklearn.preprocessing import PolynomialFeatures

dataset = pd.read_csv('',sep=';')
X = dataset.iloc[:,:-1].values 


Y = dataset.iloc[:,-1].values 


plt.scatter(X, Y, color = 'green')
regressor = LinearRegression()
regressor.fit(X,y)
y_pred = regressor.predict(X)
r2 = r2_score(y,y_pred)


poly_reg = PolynomialFeatures(degree = 3)

X_poly = poly_reg.fit_transform(X)

regressor2 = LinearRegression()
regressor2.fit(X_poly, y)
y_poly_pred = regressor2.predict(X_poly)

r2_poly = r2_score(y,y_poly_pred)

predictions = poly_reg.fit_transform([[2021]],[[2022]],[[2023]])
regressor2.predict([predictions])
