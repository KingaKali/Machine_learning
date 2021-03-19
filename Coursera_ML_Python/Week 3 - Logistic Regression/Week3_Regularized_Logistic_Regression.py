# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 21:26:09 2021

@author: KingaKali
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import Week3_Gradient_Descent as gd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import uniform
from sklearn.metrics import classification_report


# Importing data
data = pd.read_csv("data_2.csv", sep=';')
data=data.reset_index(drop=True)
X=data.loc[:,['Test1','Test2']]
y=data.loc[:,['Accept_reject_microchip']]

# Exploring data
data.info()
data.describe()


# Use the 'hue' argument to provide a factor variable
sns.lmplot( x="Test1", y="Test2", data=data, fit_reg=False, hue='Accept_reject_microchip', legend=False)
# Move the legend to an empty part of the plot
plt.legend(loc='lower right')


corr_matrix=data.corr()
corr_matrix["Accept_reject_microchip"].sort_values(ascending=True)


#Splitting train,test based on split ratio
train,test=gd.split_train_test(data,0.3)
train=train.reset_index(drop=True)
test=test.reset_index(drop=True)
X_train=train.loc[:,['Test1','Test2']]
X_test=test.loc[:,['Test1','Test2']]
y_train=train.loc[:,['Accept_reject_microchip']]
y_test=test.loc[:,['Accept_reject_microchip']]

#Creating pipeline for standard scaling and encoding for categorical variable

num_pipeline=Pipeline([
        ('std_scaler',gd.Standard_Scaler()),
    ])


train_data_prepared=num_pipeline.fit_transform(X_train)
train_data_prepared=pd.DataFrame(train_data_prepared)
# Linear Regression with Batch Gradient Descent - own implementation

# Ridge Regression with Gradient Descent
alpha=0.1
nb_iteration=100
lambda_=0.1

lin_reg=gd.Logistic()
lin_reg.fit(train_data_prepared,y_train,alpha,nb_iteration,lambda_)

# Plot loss vs number of iteration
plt.plot(lin_reg.loss_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")

# Predicting our linear function on test set
test_data_prepared=num_pipeline.fit_transform(X_test)
test_data_prepared = np.concatenate([np.ones((len(y_test), 1)), X_test], axis=1)
test_data_prepared=pd.DataFrame(test_data_prepared)
predictions=lin_reg.predict(test_data_prepared)
y_test=y_test.rename(columns={"Accept_reject_microchip":0})

# Calculation of Mean Absolute Error
mae=float(gd.mae(predictions,y_test))



# Using sklearn implementation with Choleskhy factorization for matrix inverse
test_data=num_pipeline.fit_transform(X_test)

y_tr=y_train.copy()
y_tr=y_tr.rename_axis('ID').values
y_te=y_test.copy()
y_te=y_te.rename_axis('ID').values

ridge_reg_sklearn = LogisticRegression(C=3.2486749151019727,solver='lbfgs',penalty="l2")
ridge_reg_sklearn.fit(train_data_prepared,y_tr.ravel())

predict_train=ridge_reg_sklearn.predict(train_data_prepared)
predictions_rr_sklearn=ridge_reg_sklearn.predict(test_data)
predictions_rr_sklearn=pd.DataFrame(predictions_rr_sklearn)
mae_rlr_sklearn=float(gd.mae(predictions_rr_sklearn,y_test))

# Decision boundaries
min1, max1 = X.iloc[:, 0].min()-1, X.iloc[:, 0].max()+1
min2, max2 = X.iloc[:, 1].min()-1, X.iloc[:, 1].max()+1

# define the x and y scale
x1grid = np.arange(min1, max1, 0.1)
x2grid = np.arange(min2, max2, 0.1)

# create all of the lines and rows of the grid
xx, yy = np.meshgrid(x1grid, x2grid)

# flatten each grid to a vector
r1, r2 = xx.flatten(), yy.flatten()
r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))

# horizontal stack vectors to create x1,x2 input for the model
grid = np.hstack((r1,r2))

# make predictions for the grid
yhat = ridge_reg_sklearn.predict(grid)

# reshape the predictions back into a grid
zz = yhat.reshape(xx.shape)
# plot the grid of x, y and z values as a surface
plt.contourf(xx, yy, zz, cmap='Paired')

for class_value in range(2):
    # get row indexes for samples with this class
    row_ix = np.where(y == class_value)
    row_ix=list(row_ix[0])
    # create scatter of these samples
    X=pd.DataFrame(X)
    plt.scatter(X.iloc[row_ix, 0], X.iloc[row_ix, 1], cmap='Paired')

### Evaluation of the model
confusion_matrix(y_test,predictions_rr_sklearn)


print('Precision Score of our model is {}'.format(precision_score(y_test, predictions_rr_sklearn).round(2)))
print('Recall Score of our model is {}'.format(recall_score(y_test, predictions_rr_sklearn).round(2)))



print('AUC score of our model is {}'.format(roc_auc_score(y_test, predictions_rr_sklearn).round(2)))

print('Log loss of our model is {}'.format(log_loss(y_test, predictions_rr_sklearn).round(2)))


report=classification_report(y_test,predictions_rr_sklearn)
print('report:', report, sep='\n')


logistic_reg=LogisticRegression(solver='liblinear',max_iter=200,random_state=0)

#Randomized Search

distributions = dict(C=uniform(loc=0, scale=4),
                      penalty=['l2', 'l1'])

clf = RandomizedSearchCV(logistic_reg, distributions, random_state=0)
search=clf.fit(train_data_prepared,y_tr.ravel())
search.best_params_

model2=search.best_estimator_
model2.fit(test_data,y_te.ravel())
model2_predictions=model2.predict(test_data)
accuracy_grid_search=precision_score(y_test,model2_predictions)

# Grid Search
param_grid=[
        {'C':[0.1,3.2486749151019727]},
        ]

grid_search=GridSearchCV(logistic_reg,param_grid,cv=5,
                         scoring="neg_log_loss",
                         return_train_score=False)

grid_search.fit(train_data_prepared,y_tr.ravel())

model1= grid_search.best_estimator_
model1.fit(test_data,y_te.ravel())
model1_predictions=model1.predict(test_data)
accuracy_grid_search=precision_score(y_test,model1_predictions)

cvres=grid_search.cv_results_

for mean_score, params  in zip(cvres["mean_test_score"],cvres["params"]):
    print( -mean_score, params)
    
