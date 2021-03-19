# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 23:30:40 2021

@author: KingaKalinowska
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplt as plt
from pandas.plotting import scatter_matrix

df=pd.read_csv(r'C:\Users\Komputer\Desktop\ML Projects\house-prices-advanced-regression-techniques\train.csv')
test_data=pd.read_csv(r'C:\Users\Komputer\Desktop\ML Projects\house-prices-advanced-regression-techniques\test.csv')

df.head()

#Basic info and statistics for training data
df.info()
df.describe()
df.hist(bins=50,figsize=(20,15))

attributes=['LotArea','MSSubClass']
scatter_matrix(df[attributes],figsize=(12,8))

df.corr()["SalePrice"].abs().sort_values(ascending = False)
sns.heatmap(df.corr())

sns.regplot(df["GrLivArea"], df["SalePrice"],fit_reg=False)
sns.distplot(df['SalePrice'])

var = 'OverallQual'
data = pd.concat([df['SalePrice'], df[var]], axis=1)
f, ax = plt.subplots(figsize=(14, 8))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000)


total = df.isnull().sum().sort_values(ascending=False)
percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)


total = test_data.isnull().sum().sort_values(ascending=False)
percent = (test_data.isnull().sum()/test_data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)

obj_df_train = df.select_dtypes(include=['object']).copy().reset_index()
obj_df_train

obj_df_train = obj_df_train.fillna("Not Listed")
obj_df_train

obj_df_test = test_data.select_dtypes(include=['object']).copy().reset_index()
obj_df_test

obj_df_test = obj_df_test.fillna("Not Listed")
obj_df_test

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

obj_df_train = obj_df_train.apply(le.fit_transform)
obj_df_train

int_df_train = df.select_dtypes(include=['int64']).copy().reset_index()
int_df_train

int_df_train = int_df_train.fillna(0)
int_df_train

float_df_train = df.select_dtypes(include=['float64']).copy().reset_index()
float_df_train

float_df_train = float_df_train.fillna(0)
float_df_train

train = obj_df_train.merge(int_df_train, on="index").merge(float_df_train, on="index")
train

test = obj_df_test.merge(int_df_test, on="index").merge(float_df_test, on="index")
test

id = test.Id

y = train.SalePrice.values
X = train.drop(['Id','index',  'SalePrice'], axis = 1)
X_test = test.drop(['Id', 'index'], axis = 1)


id = test.Id

y = train.SalePrice.values
X = train.drop(['Id','index',  'SalePrice'], axis = 1)
X_test = test.drop(['Id', 'index'], axis = 1)


from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Define the scaler 
scaler = StandardScaler().fit(X_train)
# Scale the train set
X_train = scaler.transform(X_train)
# Scale the validation set
X_val = scaler.transform(X_val)
# Scale the test set
X_test = scaler.transform(X_test)


from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression


model = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=15),random_state=1,n_estimators=1000, loss='exponential').fit(X_train, y_train)
print(model.score(X_train, y_train))


y_pred = model.predict(X_val)
y_pred = y_pred.astype(int)
print(model.score(X_val, y_val))


from sklearn.model_selection import cross_val_score
scores=cross_val_score(tree_reg,housing_prepared, housing_labels,scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores=np.sqrt(-scores)

def display_scores(scores):
    print("Scores", scores)
    print("Mean", scores.mean())
    print("Standard deviation", scores.std())
    
    
from sklearn.metrics import mean_squared_error

rmse = mean_squared_error(y_val, y_pred, squared=False)
rmse

#Fine-tunning the model
from sklearn.model_selection import GridSearchCV
param_grid=[
        {'n_estimators':[3,10,30], 'max_features': [2,4,6,8]},
        {'bootstrap':[False],'n_estimators':[3,10],'max_features':[2,3,4]},]

forest_req=RandomForestRegressor()
grid_search=GridSearchCV(forest_req,param_grid,cv=5,scoring="neg_mean_squared_error",return_train_score=True)
grid_search.fit(X_train,y_train)

grid_search.best_params_
grid_search.best_estimator_
cvres=grid_search.cv_results_

for mean_score, params in zip(cvres['mean_test_score'],cvres['params']):
    print(np.sqrt(-mean_score),params)
    

final_labels = model.predict(X_test)
final_labels[final_labels < 0] = 0
final_labels = final_labels.astype(int)
final_labels

final_result = pd.DataFrame({'Id': id, 'SalePrice': final_labels})
