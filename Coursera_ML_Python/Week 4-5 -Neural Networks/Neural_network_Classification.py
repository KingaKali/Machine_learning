# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 22:53:52 2021

@author: Komputer
"""

import mat4py
import numpy as np
import matplotlib.pyplot as pyplot

import os
import pandas as pd
import scipy

import Week3_Gradient_Descent as gd



data = mat4py.loadmat(os.path.join('Data', 'ex4data1.mat'))
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

mc.displayData(sel)


# Setup the parameters you will use for this exercise
input_layer_size  = 400  # 20x20 Input Images of Digits
hidden_layer_size = 25   # 25 hidden units
num_labels = 10          # 10 labels, from 0 to 9

def randInitializeWeights(L_in, L_out, epsilon_init=0.12):
     W = np.random.rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init
     return W

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels)

# Unroll parameters
initial_nn_params = np.concatenate([initial_Theta1.ravel(), initial_Theta2.ravel()], axis=0)


def sigmoid(z):
    """
    Computes the sigmoid of z.
    """
    return 1.0 / (1.0 + np.exp(-z))

def sigmoidGradient(z):
    
    return sigmoid(z)*(1-sigmoid(z))


def nnCostFunction(nn_params,
                   input_layer_size,
                   hidden_layer_size,
                   num_labels,
                   X, y, lambda_=0.0):
    Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                        (hidden_layer_size, (input_layer_size + 1)))

    Theta2 = np.reshape(nn_params[(hidden_layer_size * (input_layer_size + 1)):],
                        (num_labels, (hidden_layer_size + 1)))
    
       # You need to return the following variables correctly 
    J = 0
    Theta1_grad = np.zeros(Theta1.shape)
    Theta2_grad = np.zeros(Theta2.shape)
    
    
    a1=np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
    a2=sigmoid(a1.dot(Theta1.T))
    a2 = np.concatenate([np.ones((a2.shape[0], 1)), a2], axis=1)
    a3=sigmoid(a2.dot(Theta2.T))
    
    y = y.reshape(-1).astype(int)
    y[y == 10] = 0
    y = np.eye(num_labels)[y]
    
    temp1 = Theta1
    temp2 = Theta2
    
    # Add regularization term
    
    reg_term = (lambda_ / (2 * m)) * (np.sum((temp1[:, 1:])**2) + np.sum(np.square(temp2[:, 1:])))
    
    J = (-1 / m) * np.sum(np.log(a3)*y + np.log(1 - a3) * (1 - y)) + reg_term
    
    delta_3=a3-y
    
    delta_2 = delta_3.dot(Theta2)[:, 1:] * sigmoidGradient(a1.dot(Theta1.T))
    
    Delta2=delta_3.T.dot(a2)
    Delta1=delta_2.T.dot(a1)
    
    Theta1_grad = (1 / m) * Delta1
    Theta1_grad[:, 1:] = Theta1_grad[:, 1:] + (lambda_ / m) * Theta1[:, 1:]
    
    Theta2_grad = (1 / m) * Delta2
    Theta2_grad[:, 1:] = Theta2_grad[:, 1:] + (lambda_ / m) * Theta2[:, 1:]
    

    
    grad = np.concatenate([Theta1_grad.ravel(), Theta2_grad.ravel()])
    
    return J, grad
    
def computeNumericalGradient(J, theta, e=1e-4):
    numgrad = np.zeros(theta.shape)
    perturb = np.diag(e * np.ones(theta.shape))
    for i in range(theta.size):
        loss1, _ = J(theta - perturb[:, i])
        loss2, _ = J(theta + perturb[:, i])
        numgrad[i] = (loss2 - loss1)/(2*e)
    return numgrad


def checkGradient(nn_params,
                   input_layer_size,
                   hidden_layer_size,
                   num_labels,
                   X, y, lambda_=0.0):

    _ , grad_algorithm = nnCostFunction(nn_params,
                                        input_layer_size,
                                        hidden_layer_size,
                                        num_labels,
                                        X, y, lambda_=0.0)
    
    
 # short hand for cost function
    costFunc = lambda p: nnCostFunction(p, input_layer_size, hidden_layer_size,
                                        num_labels, X, y, lambda_)
    cost, grad = costFunc(nn_params)
    numgrad = computeNumericalGradient(costFunc, nn_params)
    
    diff = np.linalg.norm(numgrad - grad_algorithm)/np.linalg.norm(numgrad + grad_algorithm)

    print('If your backpropagation implementation is correct, then \n'
          'the relative difference will be small (less than 1e-9). \n'
          'Relative Difference: %g' % diff)

def predict(Theta1, Theta2, X):
    """
    Predict the label of an input given a trained neural network
    Outputs the predicted label of X given the trained weights of a neural
    network(Theta1, Theta2)
    """
    # Useful values
    m = X.shape[0]
    num_labels = Theta2.shape[0]

    # You need to return the following variables correctly
    p = np.zeros(m)
    h1 = sigmoid(np.dot(np.concatenate([np.ones((m, 1)), X], axis=1), Theta1.T))
    h2 = sigmoid(np.dot(np.concatenate([np.ones((m, 1)), h1], axis=1), Theta2.T))
    p = np.argmax(h2, axis=1)
    return p




train,test=gd.split_train_test(data_full,0.3)
X_train, y_train = train.loc[:,0:399], train.loc[:,400].ravel()
X_train=np.array(X_train)

X_test, y_test = test.loc[:,0:399], test.loc[:,400].ravel()
X_test=np.array(X_test)


options= {'maxiter': 100}

#  You should also try different values of lambda
lambda_ = 1

# Create "short hand" for the cost function to be minimized
costFunction = lambda p: nnCostFunction(p, input_layer_size,
                                        hidden_layer_size,
                                        num_labels, X_train, y_train, lambda_)

# Now, costFunction is a function that takes in only one argument
# (the neural network parameters)
res = scipy.optimize.minimize(costFunction,
                        initial_nn_params,
                        jac=True,
                        method='TNC',
                        options=options)

# get the solution of the optimization
nn_params = res.x
        
# Obtain Theta1 and Theta2 back from nn_params
Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                    (hidden_layer_size, (input_layer_size + 1)))

Theta2 = np.reshape(nn_params[(hidden_layer_size * (input_layer_size + 1)):],
                    (num_labels, (hidden_layer_size + 1)))

predictions= predict(Theta1,Theta2,X_train)

checkGradient(initial_nn_params,
                   input_layer_size,
                   hidden_layer_size,
                   num_labels,
                   X_train, y_train, lambda_=0.0)