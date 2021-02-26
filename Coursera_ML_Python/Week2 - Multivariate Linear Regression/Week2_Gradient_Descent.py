# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 23:14:05 2021

@author: KingaKali
"""

'''
Gradient Descent implementation
'''
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

def loss_function(theta,X,y):
    return 1/(2*len(X))*np.sum((np.dot(X,theta)-y)**2)
    
    
def gradient_descent(X,y,theta,alpha,number_of_iterations):
    #Initialize theta with random values
    cost_history=[]
    m=len(y)
           
    for iter in range(number_of_iterations):
         gradients=2/m*X.T.dot(X.dot(theta)-y)
         theta=theta-alpha*gradients
         loss=loss_function(theta,X,y)
         cost_history.append(loss)
        
    return theta,cost_history

def predict(X,theta):
    return np.dot(X,theta)

def mse(predictions,output):
    return ((predictions-output)**2).mean()

def split_train_test(data,test_ratio):
    shuffled_indices=np.random.permutation(len(data))
    test_set_size=int(len(data)*test_ratio)
    test_indices=shuffled_indices[:test_set_size]
    train_indices=shuffled_indices[test_set_size:]
    return data.iloc[train_indices,:],data.iloc[test_indices,:]

class Standard_Scaler(BaseEstimator,TransformerMixin):
    def fit(self,X,y=None):
        return self
    def transform(self,X):
        for i in range(X.shape[1]):
            mu=X.iloc[:,i].mean()
            std=X.iloc[:,i].std()
            X.iloc[:,i]=(X.iloc[:,i]-mu)/std
        return X
    
class Linear_Regressor(BaseEstimator,TransformerMixin):
    def fit(self,X,y,alpha,number_of_iterations):
        self.y=y
        self.alpha=alpha
        self.number_of_iterations=number_of_iterations
        X = np.concatenate([np.ones((len(self.y), 1)), X], axis=1)
        X=pd.DataFrame(X)
        theta=np.zeros((X.shape[1],1))
        theta,loss_history=gradient_descent(X,self.y,theta,self.alpha,self.number_of_iterations)
        self.theta=theta
        self.loss_history=loss_history
    def predict(self,X):
        return predict(X,self.theta)
       
def normal_equation(X,y):
    X = np.concatenate([np.ones((len(y), 1)), X], axis=1)
    X=pd.DataFrame(X)
    return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
  
def R_squared(X,y):
    sum_of_squares=np.sum((X-y)**2)
    mean=y.mean()
    total_sum_of_squares=np.sum((y-mean)**2)
    return 1-sum_of_squares/total_sum_of_squares