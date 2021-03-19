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

def sigmoid(x):
    # Activation function used to map any real value between 0 and 1
    return 1 / (1 + np.exp(-x))


def weighted_inputs(theta, x):
    # Computes the weighted sum of inputs
    return np.dot(x, theta)


def probability(theta, x):
    # Returns the probability after passing through sigmoid
    return sigmoid(weighted_inputs(theta, x))

def loss_function(theta,X,y,lambda_):
    loss=1/(2*len(X))*np.sum(y*np.log(probability(theta,X))+ (1-y)*np.log(1-probability(theta,X)))+lambda_/(2*len(X))*np.sum(theta[1:]**2)
    return -loss[0]
   
def gradient_descent(X,y,theta,alpha,number_of_iterations,lambda_):
    #Initialize theta with random values
    cost_history=[]
    m=len(y)       
    for iter in range(number_of_iterations):
         temp=theta
         gradients=1/m*X.T.dot(sigmoid(weighted_inputs(theta,X))-y)
         temp=theta*(1-lambda_*alpha/m)-alpha*gradients
         temp[0]=theta[0]-alpha*gradients[0]
         theta=temp
         loss=loss_function(theta,X,y,lambda_)
         cost_history.append(loss)
        
    return theta,cost_history

def predict(X,theta):
    predictions=probability(theta,X)
    predictions=[1 if predictions[x]>=0.5 else 0 for x in range(len(predictions))]
    return pd.DataFrame(predictions)

def mae(predictions,output):
    return (np.abs(predictions-output)).mean()

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
    
class Logistic(BaseEstimator,TransformerMixin):
    def fit(self,X,y,alpha,number_of_iterations,lambda_):
        self.y=y
        self.alpha=alpha
        self.number_of_iterations=number_of_iterations
        self.lambda_=lambda_
        X = np.concatenate([np.ones((len(self.y), 1)), X], axis=1)
        theta=np.zeros((X.shape[1],1))
        theta,loss_history=gradient_descent(X,self.y,theta,self.alpha,self.number_of_iterations,self.lambda_)
        self.theta=theta
        self.loss_history=loss_history
    def predict(self,X):
        return predict(X,self.theta)
       


    
    
    