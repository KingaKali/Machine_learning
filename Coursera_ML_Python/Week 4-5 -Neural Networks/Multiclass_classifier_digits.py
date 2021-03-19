# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 22:05:42 2021

@author: Komputer
"""


import mat4py
import numpy as np
import matplotlib.pyplot as pyplot

import os
import pandas as pd
import scipy

import Week3_Gradient_Descent as gd



def displayData(X, example_width=None, figsize=(10, 10)):
    """
    Displays 2D data stored in X in a nice grid.
    """
    # Compute rows, cols
    if X.ndim == 2:
        m, n = X.shape
    elif X.ndim == 1:
        n = X.size
        m = 1
        X = X[None]  # Promote to a 2 dimensional array
    else:
        raise IndexError('Input X should be 1 or 2 dimensional.')

    example_width = example_width or int(np.round(np.sqrt(n)))
    example_height = n / example_width

    # Compute number of items to display
    display_rows = int(np.floor(np.sqrt(m)))
    display_cols = int(np.ceil(m / display_rows))

    fig, ax_array = pyplot.subplots(display_rows, display_cols, figsize=figsize)
    fig.subplots_adjust(wspace=0.025, hspace=0.025)

    ax_array = [ax_array] if m == 1 else ax_array.ravel()

    for i, ax in enumerate(ax_array):
        ax.imshow(X[i].reshape(example_width, example_width, order='F'),
                  cmap='Greys', extent=[0, 1, 0, 1])
        ax.axis('off')


def sigmoid(z):
    """
    Computes the sigmoid of z.
    """
    return 1.0 / (1.0 + np.exp(-z))


data = mat4py.loadmat(os.path.join('Data', 'ex3data1.mat'))
data=pd.DataFrame(data)
data[['y_']]=pd.DataFrame(data.y.tolist(), index= data.index)

data_full = pd.DataFrame(data.y.tolist(), index= data.index,columns=['y'])

dataX = pd.DataFrame(data.X.tolist())
data_full=np.concatenate([dataX,data_full],axis=1)
data_full=pd.DataFrame(data_full)

X, y = data_full.loc[:,0:399], data_full.loc[:,400].ravel()
X=np.array(X)


y[y == 10] = 0

# 20x20 Input Images of Digits
input_layer_size  = 400

# 10 labels, from 1 to 10 (note that we have mapped "0" to label 10)
num_labels = 10

m = y.size

# Randomly select 100 data points to display
rand_indices = np.random.choice(m, 100, replace=False)
sel = X[rand_indices]

displayData(sel)

def weighted_inputs(theta, x):
    # Computes the weighted sum of inputs
    return np.dot(x, theta)


def probability(theta, x):
    # Returns the probability after passing through sigmoid
    return sigmoid(weighted_inputs(theta, x))


def CostFunction(theta,X,y,lambda_):
       gradients=np.zeros(theta.shape[0])
       temp=theta
       temp[0]=0
       gradients=1/m*X.T.dot(sigmoid(weighted_inputs(theta,X))-y)
       gradients=gradients+lambda_/ X.shape[0]*temp
       loss=1/(2*len(X))*np.sum(y*np.log(probability(theta,X))+ (1-y)*np.log(1-probability(theta,X)))+lambda_/(2*len(X))*np.sum(theta[1:]**2)
       return -loss,gradients

def onevsAll(K,X,y,lambda_):
    
    X=np.concatenate([np.ones((X.shape[0],1)),X],axis=1)
    all_theta=np.zeros((K,X.shape[1]))
    
    for k in range(K):
        theta=np.zeros(X.shape[1])
        options = {'maxiter': 50}
        result = scipy.optimize.minimize(CostFunction, 
                                theta, 
                                (X, (y == k), lambda_), 
                                jac=True, 
                                method='CG',
                                options=options) 
        all_theta[k]=result.x 
    return all_theta
   
train,test=gd.split_train_test(data_full,0.3)
X_train, y_train = train.loc[:,0:399], train.loc[:,400].ravel()
X_train=np.array(X_train)

X_test, y_test = test.loc[:,0:399], test.loc[:,400].ravel()
X_test=np.array(X_test)
    
lambda_=0.1
        
onevsAll=onevsAll(10,X_train,y_train,lambda_)

def predict(X,theta):
    predictions=probability(theta,X)
    return predictions

X_train_copy=np.concatenate([np.ones((X_train.shape[0],1)),X_train],axis=1)
predictions=sigmoid(X_train_copy.dot(onevsAll.T))
p = np.argmax(predictions,axis=1)

pd_=pd.DataFrame(predictions)



