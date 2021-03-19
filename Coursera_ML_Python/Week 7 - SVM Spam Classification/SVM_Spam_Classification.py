# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 20:14:17 2021

@author: Komputer
"""

import re
from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize 
import mat4py
import numpy as np
import matplotlib.pyplot as pyplot

import os
import pandas as pd
import scipy
from sklearn.metrics import classification_report
   

  
#  To use an SVM to classify emails into Spam v.s. Non-Spam, you first need
#  to convert each email into a vector of features. In this part, you will
#  implement the preprocessing steps for each email. You should
#  complete the code in processEmail.m to produce a word indices vector
#  for a given email.

def processEmail(email_contents, verbose=True):


    # ========================== Preprocess Email ===========================
    # Find the Headers ( \n\n and remove )
    # Uncomment the following lines if you are working with raw emails with the
    # full headers
    # hdrstart = email_contents.find(chr(10) + chr(10))
    # email_contents = email_contents[hdrstart:]

    # Lower case
    email_contents=email_contents.lower()
    #email_contents =  [x.upper() for x in email_contents]
    
    #fruit_list = ['apple', 'banana', 'peach', 'plum', 'pineapple', 'kiwi']
    #fruit = re.compile(r'\b(?:{0})\b'.format('|'.join(fruit_list))
    # Strip all HTML
    # Looks for any expression that starts with < and ends with > and replace
    # and does not have any < or > in the tag it with a space
    email_contents =re.compile('<[^<>]+>').sub(' ', email_contents)

    # Handle Numbers
    # Look for one or more characters between 0-9
    email_contents = re.compile('[0-9]+').sub(' number ', email_contents)

    # Handle URLS
    # Look for strings starting with http:// or https://
    email_contents = re.compile('(http|https)://[^\s]*').sub(' httpaddr ', email_contents)

    # Handle Email Addresses
    # Look for strings with @ in the middle
    email_contents = re.compile('[^\s]+@[^\s]+').sub(' emailaddr ', email_contents)
    
    # Handle $ sign
    email_contents = re.compile('[$]+').sub(' dollar ', email_contents)
    
    # get rid of any punctuation
    email_contents = re.split('[ @$/#.-:&*+=\[\]?!(){},''">_<;%\n\r]', email_contents)

    # remove any empty word string
    email_contents = [word for word in email_contents if len(word) > 0]
    
    # Stem the email contents word by word
    ps = PorterStemmer() 
    processed_email = []
    for word in email_contents:
        # Remove any remaining non alphanumeric characters in word
        word = re.compile('[^a-zA-Z0-9]').sub('', word).strip()
        word=ps.stem(word)
        processed_email.append(word)
        
        
        if len(word) < 1:
            continue
        
    return processed_email


def extract_features(words,vocabulary): 
    features_matrix=np.zeros(len(vocabulary))
    docID=0
    for i in range(len(words)):
        for j in range(len(vocabulary)):
            if words[i]==vocabulary[j]:
                features_matrix[j]=1
    return features_matrix

import os
# Extract Features
with open(os.path.join('Data', 'emailSample1.txt')) as fid:
    file_contents = fid.read()
    #.split()

vocabList = np.genfromtxt(os.path.join('Data', 'vocab.txt'), dtype=object)
vocabulary=list(vocabList[:, 1].astype('str'))

        
voc=pd.DataFrame(vocabulary)

import collections

word_indices  = processEmail(file_contents)
word_indices=list(set(word_indices))
matrix=extract_features(word_indices,vocabulary)
matrix=pd.DataFrame(matrix).T
matrix.columns=vocabulary

data_train = mat4py.loadmat(os.path.join('Data', 'spamTrain.mat'))
data_train=pd.DataFrame(data_train)

data_train[['y_']]=pd.DataFrame(data_train.y.tolist(), index= data_train.index)

data_full = pd.DataFrame(data_train.y.tolist(), index= data_train.index,columns=['y'])

dataX = pd.DataFrame(data_train.X.tolist())
data_full=np.concatenate([dataX,data_full],axis=1)
data_full=pd.DataFrame(data_full)

X_train, y_train = data_full.loc[:,0:1898], data_full.loc[:,1899].ravel()
X_train=np.array(X_train)

data_test = mat4py.loadmat(os.path.join('Data', 'spamTest.mat'))
data_test=pd.DataFrame(data_test)

data_test[['y_']]=pd.DataFrame(data_test.ytest.tolist(), index= data_test.index)

data_full_test = pd.DataFrame(data_test.ytest.tolist(), index= data_test.index,columns=['y'])

dataX = pd.DataFrame(data_test.Xtest.tolist())
data_full_test=np.concatenate([dataX,data_full_test],axis=1)
data_full_test=pd.DataFrame(data_full_test)

X_test, y_test = data_full_test.loc[:,0:1898], data_full_test.loc[:,1899].ravel()
X_test=np.array(X_test)

from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

svm_clf= Pipeline([
        ("scaler",StandardScaler()),
        ("linear_SVM",LinearSVC(C=1,loss="hinge")),
        ])
        
svm_clf.fit(X_train,y_train)

sc=StandardScaler()
X_test=sc.fit_transform(X_test)
predictions_linear=svm_clf.predict(X_test)  
    
report=classification_report(y_test,predictions_linear)
print('report:', report, sep='\n')

from sklearn.metrics import precision_score,recall_score
precision_score(y_test,predictions_linear)
recall_score(y_test,predictions_linear)

from sklearn.metrics import precision_recall_curve
precisions,recalls,thresholds=precision_recall_curve(y_test,predictions_linear)
import  matplotlib.pyplot as plt

def plot_precision_recall_vs_threshold(precisions,recalls,thresholds):
    plt.plot(thresholds,precisions[:-1], "b--",label="Precision")
    plt.plot(thresholds,recalls[:-1],"--b",label="Recall")
    
plot_precision_recall_vs_threshold(precisions,recalls,thresholds)
plt.show()
    
from sklearn.svm import SVC

svm_clf_kernel= Pipeline([
        ("scaler",StandardScaler()),
        ("svm_clf",SVC(kernel="rbf",gamma=5, C=1)),])
    
svm_clf_kernel.fit(X_train,y_train)


predictions_kernel=svm_clf_kernel.predict(X_test)  
    
report=classification_report(y_test,predictions_kernel)
print('report:', report, sep='\n')

from sklearn.metrics import precision_score,recall_score,confusion_matrix
precision_score(y_test,predictions_kernel)
recall_score(y_test,predictions_kernel)


### Evaluation of the model
confusion_matrix(y_test,predictions_kernel)


from sklearn.metrics import precision_recall_curve
precisions,recalls,thresholds=precision_recall_curve(y_test,predictions_kernel)
import  matplotlib.pyplot as plt

def plot_precision_recall_vs_threshold(precisions,recalls,thresholds):
    plt.plot(thresholds,precisions[:-1], "b--",label="Precision")
    plt.plot(thresholds,recalls[:-1],"--b",label="Recall")
    
plot_precision_recall_vs_threshold(precisions,recalls,thresholds)
plt.show()


from sklearn.model_selection import GridSearchCV 
  
# defining parameter range 
param_grid = {'C': [0.1, 1, 10, 100, 1000],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf','poly','linear']}  
  
grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3) 
  
X_train_gs=sc.fit_transform(X_train)


# fitting the model for grid search 
grid.fit(X_train_gs, y_train) 

final_model=grid.best_estimator_
final_predictions=final_model.predict(X_test)

precision_score(y_test,final_predictions)
recall_score(y_test,final_predictions)

