# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, -2:-1]
y = dataset.iloc[:, 2].values


'''
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
'''
# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

#Fitting Linear Regression 
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

#Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)

lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly,y)

#Visualizing Linear Regression
plt.scatter(X,y, color = "red")
plt.plot(X, lin_reg.predict(X))
plt.title("Truth or Bluff")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()


#Visualizing Polunomial Regression
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X,y, color = "red")
plt.plot(X_grid, lin_reg2.predict(poly_reg.fit_transform(X_grid)))
plt.title("Truth or Bluff")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()