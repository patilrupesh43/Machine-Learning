# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#import file
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1]
y = dataset.iloc[:,1]

#Splitting Test and Train
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

'''
#Fearutre Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train =sc_X.fit_transform(X_train)
X_test =sc_X.transform(X_test)

'''

#Fitting Simple Linear Regression to training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#Predict the test set results
y_pred = regressor.predict(X_test)

#Visualising the training set results
plt.scatter(X_train,y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
    
    #Visualising the test set results
    plt.scatter(X_test ,y_test, color='red')
    plt.plot(X_train, regressor.predict(X_train), color = 'blue')
    plt.title('Salary vs Experience (Test Set)')
    plt.xlabel('Years of Experience')
    plt.ylabel('Salary')
    plt.show()