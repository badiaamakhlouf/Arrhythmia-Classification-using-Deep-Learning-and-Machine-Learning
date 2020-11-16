# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 17:58:43 2018

@author: Badiaa
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
#%matplotlib inline

#%% - Exercise 1 %%%#
#read the content of the provided file and assign ? to Not a Number value (NaN)
ss = pd.read_csv('arrhythmia.data', sep=',', header=None, na_values=['?'])
s=ss.copy()
# drop columns with NaN values 
data1=ss.dropna(axis=1)
data_frames=data1.values

# some columns store only zeros so they carry no information ==> need to exclude them 
# For each columns sum all the lines 
b=np.sum(np.abs(data_frames), axis=0)
# identifying those with zero value 
idx=np.where(b==0)
# delete them 
clean_data_frame=np.delete(data_frames,idx[0], axis=1)
Target=clean_data_frame[:,-1] # the target is the last column of the clean_data_frame matrix

# In this script we are going to implement binary classification on the last column
# Each value greater than 1 will be assigned to value 1 and value 1 will be assigned to value 0
# ==> 0 means healthy person and 1 means person with Arrythmia
Actual_class=0*(Target==1)+1*(Target>1)# class_id 
Y=clean_data_frame[:,0:-1]

# normalizing the data 
mu=np.mean(Y, axis=0) # mean (one valye for each column)
sigma=np.std(Y, axis=0) #standard deviation (one value for each column)
norm_dataset=(Y-mu)/sigma # normalized training data 
# divide the dataset onto two subsets: test and train 
data_train=norm_dataset[0:226,:] # train data
data_test=norm_dataset[226:452,:] # test data

Target_train=Actual_class[0:226] # the target is the last column of the clean_data_frame matrix
Target_test=Actual_class[226:452]


#%% Processing the data with PCA 

print('PCA')
N=len(data_train)
Rx = np.dot(np.transpose(data_train),data_train)/N;
eignenval1, U1= np.linalg.eig(Rx) # eig is the eigendecomposition to compute 
# U1 is the matrix of eigen vectors (257 X 257)
# eignenval1 is the the vector column vector of eigenvalues
Z_train=np.dot(data_train, U1)
R_Z= np.dot(np.transpose(Z_train),Z_train)/len(Z_train)

# Test########
N_tes=len(data_test)
Rx_test = np.dot(np.transpose(data_test),data_test)/N_tes;
eignenval2, U2= np.linalg.eig(Rx_test) # eig is the eigen decomposition to compute 
# U2 is the matrix of eigen vectors (257 X 257)
# eignenval2 is the the vector column vector of eigenvalues

Z_test=np.dot(data_test, U2)
R_Z_test= np.dot(np.transpose(Z_test),Z_test)/len(Z_test)

print('SVM classification ')
# Create a linear SVM classifier 
svclassifier= SVC(kernel='linear', C=10)
# train the classifier
svclassifier.fit(Z_train,Target_train)

 # # Make predictions on unseen test data
Target_pred=svclassifier.predict(Z_test)
# metrics for Evaluating the algorith

print(confusion_matrix(Target_test, Target_pred))
print(classification_report(Target_test, Target_pred))

tn, fp, fn, tp=confusion_matrix(Target_test, Target_pred).ravel()

W=svclassifier.coef_[0]
I=svclassifier.intercept_[0]





