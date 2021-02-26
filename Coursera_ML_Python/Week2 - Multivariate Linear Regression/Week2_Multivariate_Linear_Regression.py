# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 21:26:09 2021

@author: KingaKali
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import Week2_Gradient_Descent as gd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


# Importing data
data = pd.read_csv("data.csv", sep=';')
data=data.reset_index(drop=True)
X=data.loc[:,['size','number_bedrooms']]
y=data.loc[:,['price']]
y=y.rename(columns={'price':0})

# Exploring data
data.info()
data.describe()
data.loc[:,"number_bedrooms"].value_counts()

#Histogram for each numerical attribute
data.hist(bins=50,figsize=(10,10))
plt.show()

#Boxplot
data.boxplot()

#Visualization of data
from pandas.plotting import scatter_matrix
attributes=["size","number_bedrooms","price"]
scatter_matrix(data,figsize=(12,8))

corr_matrix=data.corr()
corr_matrix["price"].sort_values(ascending=True)


#Splitting train,test based on split ratio
train,test=gd.split_train_test(data,0.3)
train=train.reset_index(drop=True)
test=test.reset_index(drop=True)
X_train=train.loc[:,['size','number_bedrooms']]
X_test=test.loc[:,['size','number_bedrooms']]
y_train=train.loc[:,['price']]
y_test=test.loc[:,['price']]

y_train=y_train.rename(columns={'price':0})
y_test=y_test.rename(columns={'price':0})
#Creating pipeline for standard scaling and encoding for categorical variable

num_pipeline=Pipeline([
        ('std_scaler',gd.Standard_Scaler()),
    ])

num_attr=["size"]
cat_attribs=["number_bedrooms"]

full_pipeline=ColumnTransformer([
        ("cat",OrdinalEncoder(),cat_attribs),
        ("num",num_pipeline,num_attr),
        ])

train_data_prepared=full_pipeline.fit_transform(X_train)
train_data_prepared=pd.DataFrame(train_data_prepared)
# Linear Regression with Batch Gradient Descent - own implementation

# Linear Regression with Gradient Descent
alpha=0.1
nb_iteration=400

lin_reg=gd.Linear_Regressor()
lin_reg.fit(train_data_prepared,y_train,alpha,nb_iteration)

# Plot loss vs number of iteration
plt.plot(lin_reg.loss_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")

# Predicting our linear function on test set
test_data_prepared=full_pipeline.fit_transform(X_test)
test_data_prepared = np.concatenate([np.ones((len(y_test), 1)), test_data_prepared], axis=1)
test_data_prepared=pd.DataFrame(test_data_prepared)
predictions=lin_reg.predict(test_data_prepared)

# Calculation of Root Mean Square Error
mse=gd.mse(predictions,y_test)
rsme=float(np.sqrt(mse))

# Calculation of 95% confidence interval for the genaralization error
from scipy import stats
confidence=0.95
squared_errors=(predictions-y_test)**2
interval=np.sqrt(stats.t.interval(confidence,len(squared_errors)-1,loc=squared_errors.mean(),
                         scale=stats.sem(squared_errors)))


m = len(squared_errors)
mean = squared_errors.mean()

# manual implementation of t-scores
tscore = stats.t.ppf((1 + confidence) / 2, df=m - 1)
tmargin = tscore * squared_errors.std(ddof=1) / np.sqrt(m)
np.sqrt(mean - tmargin), np.sqrt(mean + tmargin)

# z-scores rather than t-scores
zscore = stats.norm.ppf((1 + confidence) / 2)
zmargin = zscore * squared_errors.std(ddof=1) / np.sqrt(m)
np.sqrt(mean - zmargin), np.sqrt(mean + zmargin)

# Calculating theta and predictions using Normal Equation
theta_normal_equation=gd.normal_equation(train_data_prepared,y_train)
prediction_NE=gd.predict(test_data_prepared,theta_normal_equation)

mse_NE=gd.mse(prediction_NE,y_test)
rsme_NE=float(np.sqrt(mse_NE))


# Using sklearn implementation with SVD method (Moore-Penrose inverse)
test_data=full_pipeline.fit_transform(X_test)
test_data=pd.DataFrame(test_data)

from sklearn.linear_model import LinearRegression
lin_reg_sklearn = LinearRegression()
lin_reg_sklearn.fit(train_data_prepared,y_train)

predictions_lr_sklearn=lin_reg_sklearn.predict(test_data)

mse_lr_sklearn=gd.mse(predictions_lr_sklearn,y_test)
rsme_lr_sklearn=float(np.sqrt(mse_lr_sklearn))


# Using sklearn implementation with Stochastic Gradient Descent
y_tr=y_train.copy()
y_tr=y_tr.rename_axis('ID').values
from sklearn.linear_model import SGDRegressor
sgd_regressor= SGDRegressor(max_iter=1000,tol=1e-3,penalty=None,eta0=0.1)
sgd_regressor.fit(train_data_prepared,y_tr.ravel())


predictions_sgd=sgd_regressor.predict(test_data)
predictions_sgd=pd.DataFrame(predictions_sgd)
mse_sgd=gd.mse(predictions_sgd,y_test)
rsme_sgd=float(np.sqrt(mse_sgd))


#Calculation of R-squared for sklearn Linear Regression Implementation

X=full_pipeline.fit_transform(X)
predict=lin_reg_sklearn.predict(X)
coefficient=pd.DataFrame(lin_reg_sklearn.coef_)
predict=pd.DataFrame(predict)
gd.R_squared(predict,y)


