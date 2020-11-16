# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 15:03:19 2018

@author: Badiaa
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
from matplotlib import rcParams
rcParams['axes.titlepad'] = 20 

#%% - Exercise 2%%%#
#read the content of the provided file and assign ? to Not a Number value (NaN)
xx = pd.read_csv('arrhythmia.data', sep=',', header=None, na_values=['?'])
x=xx.copy()
# drop columns with NaN values 
data1=xx.dropna(axis=1)
data_frames=data1.values
# some columns store only zeros so they carry no information ==> need to exclude them 
# For each columns sum all the lines 
b=np.sum(np.abs(data_frames), axis=0)
# identifying those with zero value 
idx=np.where(b==0)
# delete them 
clean_data_frame=np.delete(data_frames,idx[0], axis=1)
#%%  
Y=clean_data_frame[:,0:-1]

# normalizing the data 
# m and s are those of the training matrix
m=np.mean(Y, axis=0) # mean (one valye for each column)
s=np.std(Y, axis=0) #standard deviation (one value for each column)
norm_dataset=(Y-m)/s # normalized training data 
#data_test_norm=(Test_data-m)/s #normalized testing data 
# divide the dataset onto two subsets: test and train 
data_train_norm=norm_dataset[0:226,:] # train data
data_test_norm=norm_dataset[226:452,:] # test data
# In this scrip we are going to implement binary classification on the last column
Target=clean_data_frame[:,-1] # the target is the last column of the clean_data_frame matrix
Target_train=Target[0:226]
Target_test=Target[226:452]

# the training phase :
# initialize the Mean vector that will contain the mean of each row
Mean_train = np.zeros([16,257])
for i in range(1,17):
    if i != 11 and i!=12 and i!=13:
        class_idTrain = np.where(Target_train==i)
        Y_train = data_train_norm[class_idTrain[0],:]
        Mean_train[i-1,:] = Y_train.mean(0)
    else:
        Mean_train[i-1, :] = float('Inf')

# the threshold is the mean the minimum distance between classes 
#%%  Apply the minimum distance criterion to associate each row of y with
#either est class id=1 or est class id=2.
print('minimum distance criterion Train')
DistTrain= np.zeros([226,16])

for j in range(0,17):
    # decision function is ||x-mean||^2
    DistTrain[:,j-1] = (((data_train_norm-Mean_train[j-1,:] ))**2).sum(1) # sum the lines 
    
est_class_MD = np.argmin(DistTrain,axis=1)+1


# Linear Decision Boundaries
# to verify that I am getting the same results as np.where
error_idTrain=np.zeros(226)
jj=1
for k in range(1,226) : 
    if (est_class_MD[k] != Target_train[k]) :
        error_idTrain[jj]=k
        jj= jj +1
#print (error_idTrain)

error = np.zeros(len(Target_train))
index_error = np.where(est_class_MD != Target_train)
error[index_error[0]] = 1
error_probability_MD = error.mean(0)
print('Probability error MD train data:',error_probability_MD)

#to calculate the true/false positive and the true/false negative I had used the confusion matrix 
# In case of multiclass classification, the confusion matrix
# is measured: the element in position i,j of the matrix is the probability
# that the estimated class is j given that the true class is i (the sum of the
# elements in a row must be 1).

# confusion matrix for the train data 
confusion_Matrix_train = np.zeros([16,16])
for i in range(1,17):
    for j in range(1, 17):
        if i != 11 and i != 12 and i != 13:
            indextrue_classTrain = np.where(Target_train == i)
            indexest_classTrain = np.where(est_class_MD[indextrue_classTrain] == j)
            confusion_Matrix_train[j-1,i-1] = len(indexest_classTrain[0])/len(indextrue_classTrain[0])

ax1=[1, 2,3,  4, 5,  6, 7, 8, 9, 10, 11,  12, 13, 14, 15, 16]
plt.pcolormesh(ax1, ax1, confusion_Matrix_train)
#plt.matshow( confusion_Matrix_train)
plt.ylabel('Predicted')
plt.xlabel('Actual')
plt.title('Training Data Confusion Matrix')
plt.colorbar()
plt.show()

tru_predTrain=np.where(est_class_MD == Target_train)
train_accuracy= len(tru_predTrain[0])/ len(est_class_MD)

somme_conf= confusion_Matrix_train.sum(1) 
# Test phase :
# initialize the Mean vector that will contain the mean of each row

# the threshold is the mean the minimum distance between classes 
#%%  Apply the minimum distance criterion to associate each row of y with
#either est class id=1 or est class id=2.
print('minimum distance criterion Test ')
Dist_Test= np.zeros([226,16])
for j in range(0,17):
    # decision function is ||x-mean||^2
    Dist_Test[:,j-1] = (((data_test_norm-Mean_train[j-1,:] ))**2).sum(1) # sum the lines 
error_idTest=np.zeros(226)
ii=1
# Linear Decision Boundaries

est_class_MDTest = np.argmin(Dist_Test,axis=1)+1
for k in range(1,226) : 
    if (est_class_MDTest[k] != Target_test[k]) :
        error_idTest[ii]=k
        ii= ii +1

#print (error_idTest)

errorTest = np.zeros(len(Target_test))

index_errorTest = np.where(est_class_MDTest != Target_test)
errorTest[index_errorTest[0]] = 1
error_probability_MDTest = errorTest.mean(0)
print('Probability error MD test data:',error_probability_MDTest)

tru_pred=np.where(est_class_MDTest == Target_test)
test_accuracy= len(tru_pred[0])/ len(est_class_MDTest)
#to calculate the true/false positive and the true/false negative I had used the confusion matrix 
# In case of multiclass classification, the confusion matrix
# is measured: the element in position i,j of the matrix is the probability
# that the estimated class is j given that the true class is i (the sum of the
# elements in a row must be 1).
# confusion matrix for the test data 

confusion_Matrix_test = np.zeros([16,16])
for i in range(1,17):
    for j in range(1, 17):
        if i != 11 and i != 12 and i != 13:
            indextrue_classTest = np.where(Target_test == i)
            indexest_classTest = np.where(est_class_MDTest[indextrue_classTest] == j)
            confusion_Matrix_test[j-1,i-1] = len(indexest_classTest[0])/len(indextrue_classTest[0])

ax2=[1, 2,3,  4, 5,  6, 7, 8, 9, 10, 11,  12, 13, 14, 15, 16]
plt.pcolormesh(ax2, ax2, confusion_Matrix_test)
plt.ylabel('Predicted')
plt.xlabel('Actual')
plt.title('Test Data Confusion Matrix')
plt.colorbar()
plt.show()

somm_conftest=  np.sum(confusion_Matrix_test, axis=1)     


# Note that the sensitivity and specificity can only be used in binary ==> so no true/false positive or
# true/false negative can be calculated
