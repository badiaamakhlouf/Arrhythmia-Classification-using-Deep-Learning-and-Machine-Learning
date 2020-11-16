# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 10:36:53 2017

@author: Badiaa
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%% - Exercise 1 %%%#
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
Target=clean_data_frame[:,-1] # the target is the last column of the clean_data_frame matrix
class_id=1*(Target==1)+2*(Target>1)# class_id 
Y=clean_data_frame[:,0:-1]
# normalizing the data 
m=np.mean(Y, axis=0) # mean (one valye for each column)
s=np.std(Y, axis=0) #standard deviation (one value for each column)
norm_dataset=(Y-m)/s # normalized data 
# divide the dataset onto two subsets: test and train 
data_train_norm=norm_dataset[0:226,:] # train data
data_test_norm=norm_dataset[226:452,:] # test data

# In this script we are going to implement binary classification on the last column
Target_train=Target[0:226] # the target is the last column of the clean_data_frame matrix
Target_test=Target[226:452]
Y_train=data_train_norm[:,:]
Y_test=data_test_norm[:,:]
Nnumber_rows=Y_train.shape[0]
Number_features=Y_train.shape[1]
# Each value greater to 1 will be assigned to value 2 and value 1 still for value 1
# define the two submatrices 
class_idTrain=class_id[0:226]# class_id 
class_idTest=class_id[226:452]

# identifying the index of the healthy class (value equal to 1)
healthyTrain=np.where(class_idTrain==1) 
healthyTest=np.where(class_idTest==1) 
#identifying the indexs of the Arrhythmia class (value != 1)
Arry_existTrain=np.where(class_idTrain!=1)
Arry_existTest=np.where(class_idTest!=1)
# From last column delete each line stores value 1 (healthy) class.
# ==> We obtain Matrix with only people affected by arrhythmia
Matrix_ArrhyTrain=np.delete(data_train_norm,healthyTrain[0], axis=0)
Matrix_ArrhyTest=np.delete(data_test_norm,healthyTest[0], axis=0)
Y2_Train=Matrix_ArrhyTrain
Y2_Test=Matrix_ArrhyTest
# From last column delete each line stores value different than 1 (Arrhythmic) class. 
#==> WE obtain Matrix with only healthy people ( no arrhythmia)
Matrix_healthyTrain=np.delete(data_train_norm,Arry_existTrain[0], axis=0)
Matrix_healthyTest=np.delete(data_test_norm,Arry_existTest[0], axis=0) 
Y1_Train=Matrix_healthyTrain
Y1_Test=Matrix_healthyTest

x1= Y1_Train.mean(0)
x2= Y2_Train.mean(0)
# the threshold is the mean the minimum distance between classes 
#%%  Apply the minimum distance criterion to associate each row of y with
#either est class id=1 or est class id=2.
print('minimum distance criterion')
# decision function is ||x-mean||^2
class1 = (((data_train_norm-x1))**2).sum(1) # sum the lines 
class2 = (((data_train_norm-x2))**2).sum(1) # sum the lines 

# Linear Decision Boundaries
indexMD_class1 = np.where(class1 < class2)
indexMD_class2 = np.where(class1 > class2)
# Method two : signle decision 
Decision_function= class1-class2
indexMD_class1Method2=np.where(Decision_function < 0)
indexMD_class2Method2=np.where(Decision_function > 0)
# if Decision_function==0 no decision

est_class_MD=np.zeros(len(Decision_function))
est_class_MD[indexMD_class1Method2[0]] = 1
est_class_MD[indexMD_class2Method2[0]] = 2

error_probability_MD = np.mean(np.abs(est_class_MD - class_idTest))

print(error_probability_MD)
# true positive : The case where a person is actually having arrhythmia (2) and the model classifying
# his case as arrhythmia (2) comes under True positive.
predicted_True_index= np.where(est_class_MD == 2)
true_idex=np.where(class_idTest==2)
true_positive = len(np.where(est_class_MD[true_idex[0]] == 2)[0])/len(true_idex[0])

# true negative : The case where a person NOT having arrhythmia and the model classifying 
# his case as not arrhythmia comes under True Negatives.

predicted_True_Neg_index= np.where(est_class_MD == 1)
true_idex_neg=np.where(class_idTest==1)
true_negative = len(np.where(est_class_MD[true_idex_neg[0]] == 1)[0])/len(true_idex_neg[0])

# False positive  : A person NOT having arrhythmia and the model classifying 
#his case as arrhythmia comes under False Positives.
predicted_False_pos_index= np.where(est_class_MD == 2)
false_idex_pos=np.where(class_idTest==1)

False_pos = len(np.where(est_class_MD[false_idex_pos[0]] == 2)[0])/len(false_idex_pos[0])
# false negative : A person having arrhythmia and the model classifying his case as No-arrhythmia 
# comes under False Negatives.

predicted_False_neg_index= np.where(est_class_MD == 1)
false_idex_neg=np.where(class_idTest==2)

False_neg = len(np.where(est_class_MD[false_idex_neg[0]] == 1)[0])/len(false_idex_neg[0])

print ("True positive Minimum distance:",true_positive)
print ("True negative Minimum distance:",true_negative)
print ("False positive Minimum distance:",False_pos)
print ("False negative Minimum distance:",False_neg)


# Sensitivity = TP/(TP+FN) for test dataset, it is also the true positive 
Sensitivity_MD=true_positive/(true_positive+False_neg )
print('sensitivity MD  Test:',Sensitivity_MD )
# Specificity = TN/(TN+FP) for test dataset, it is also the true negative 
Specificity_MD=true_negative/(true_negative+False_pos)
print('specificity MD  Test:',Specificity_MD)





#%%Bayes criterion
print('Bayes criterion')
pi1 = len(Y1_Train)/(len(Y1_Train)+len(Y2_Train))
pi2 = len(Y2_Train)/(len(Y1_Train)+len(Y2_Train))

N1 = len(Y1_Train)
R1 = np.dot(np.transpose(Y1_Train),Y1_Train)/N1;

eignenval1, U1= np.linalg.eig(R1) # eig is the eigendecomposition to compute 
# U1 is the matrix of eigen vectors (257 X 257)
# eignenval1 is the the vector column vector of eigenvalues
N2 = len(Y2_Train)
R2 = np.dot(np.transpose(Y2_Train),Y2_Train)/N2;
eignenval2, U2=np.linalg.eig(R2) # eig is the eigendecomposition to compute 
# U2 is the matrix of eigen vectors (257 X 257)
# eignenval2 is the the vector column of eigenvalues

# Considering only F1 features 
# ******* Feature F1 extraction******#
print('healthy')
Parameter1 = sum(np.real(eignenval1)) # the sum of the eigenvalues
percentage1 = 0.99 #  % retain at least 90% of P
Update_Parameter1 =percentage1 * Parameter1 # amount of "information" we want to keep
#we assume that lambda 1> lambda 2>  lambda 3 >...  >lambda F 
# checking my software and picking the largest eigenvalues that allow me
# to get 90% of P, so that L is as small as possible 
eignenval1_sorted=sorted(np.real(eignenval1), reverse=True)
eignenval1_sorted=np.array(eignenval1_sorted)
commulative_sum1= np.cumsum(eignenval1_sorted)
L1=0
for i in range(np.size(commulative_sum1)):
      h1=commulative_sum1[i]
      if h1 < Update_Parameter1 :
         F1=i+1
    
print(F1)
UF1=np.real(U1[0:,0:F1])
##### Repeating the same steps for UF2 ########
# Considering only F2 features 
# ******* Feature F2 extraction******#
print('Arrhythmia')
Parameter2 = sum(np.real(eignenval2)) # the sum of the eigenvalues
percentage2 = 0.99 #  % retain at least 90% of P
Update_Parameter2 =percentage2 * Parameter2 # amount of "information" we want to keep
#we assume that lambda 1> lambda 2>  lambda 3 >...  >lambda F 
# checking my software and picking the largest eigenvalues that allow me
# to get 99.9% of P, so that L is as small as possible 
eignenval2_sorted=sorted(np.real(eignenval2), reverse=True)
eignenval2_sorted=np.array(eignenval2_sorted)
commulative_sum2= np.cumsum(eignenval2_sorted)

for j in range(np.size(commulative_sum2)):
      h2=commulative_sum2[j]
      if h2 < Update_Parameter2 :
         F2=j+1
    
print(F2)

UF2=np.real(U2[0:,0:F2])
# extract Z1_train
Z1_train=np.dot(Y1_Train, UF1)
R_Z1= np.dot(np.transpose(Z1_train),Z1_train)/len(Z1_train)

#extract Z2_train
Z2_train=np.dot(Y2_Train, UF2)
R_Z2 = np.dot(np.transpose(Z2_train),Z2_train)/(len(Z2_train))

# find the means of Z1_train (those rows of z corresponding to class id=1) and
# Z2_train (those rows of z corresponding to class id=2) and call them w1 and
#w2 (each of these vectors has only F0 elements)
w1 = np.real(Z1_train.mean(0))
w2 = np.real(Z2_train.mean(0))



# project matrix y onto UF1 to get matrix s1 and onto UF2 to get matrix s2
s1 = np.dot(data_test_norm,UF1)
s2 = np.dot(data_test_norm,UF2)

# write the pdf’s of z1 fz|H1 (u) and z2 fz|H2 (u)
Pdfz1s1 = np.exp(np.diag(-0.5*np.dot(np.dot((s1-w1),np.linalg.inv(R_Z1)),(np.transpose(s1-w1)))))/np.sqrt(2*(np.pi**F1)*np.linalg.det(R_Z1))
Pdfz2s2 = np.exp(np.diag(-0.5*np.dot(np.dot((s2-w2),np.linalg.inv(R_Z2)),(np.transpose(s2-w2)))))/np.sqrt(2*(np.pi**F2)*np.linalg.det(R_Z2))

Prob1= np.pi*Pdfz1s1
Prob2=np.pi*Pdfz2s2

est_class_id_Bayes= np.where(Prob1<Prob2)
est_class_Bayes = np.ones(len(Prob2))
est_class_Bayes[est_class_id_Bayes[0]] = 2

error_probability_Bayes = np.mean(np.abs(est_class_Bayes - class_idTest))

print(error_probability_Bayes)
# true positive : The case where a person is actually having arrhythmia (2) and the model classifying
# his case as arrhythmia (2) comes under True positive.
predicted_True_indexB= np.where(est_class_Bayes == 2)
true_idex_B=np.where(class_idTest==2)
true_positiveBayes = len(np.where(est_class_Bayes[true_idex_B[0]] == 2)[0])/len(true_idex_B[0])

# true negative : The case where a person NOT having arrhythmia and the model classifying 
# his case as not arrhythmia comes under True Negatives.

predicted_True_Neg_index_Bayes= np.where(est_class_Bayes == 1)
true_idex_negBayes=np.where(class_idTest==1)
true_negative_Bayes = len(np.where(est_class_Bayes[true_idex_negBayes[0]] == 1)[0])/len(true_idex_negBayes[0])




# False positive  : A person NOT having arrhythmia and the model classifying 
#his case as arrhythmia comes under False Positives.
predicted_False_pos_indexBayes= np.where(est_class_Bayes == 2)
false_idex_pos_Bayes=np.where(class_idTest==1)

False_pos_Bayes = len(np.where(est_class_Bayes[false_idex_pos_Bayes[0]] == 2)[0])/len(false_idex_pos_Bayes[0])
# false negative : A person having arrhythmia and the model classifying his case as No-arrhythmia 
# comes under False Negatives.

predicted_False_neg_indexBayes= np.where(est_class_Bayes == 1)
false_idex_negBayes=np.where(class_idTest==2)

False_negBayes = len(np.where(est_class_Bayes[false_idex_negBayes[0]] == 1)[0])/len(false_idex_negBayes[0])

print ("True positive Bayes approach:",true_positiveBayes)
print ("True negative Bayes approach :",true_negative_Bayes)
print ("False positive Bayes approach :",False_pos_Bayes)
print ("False negative Bayes approach :",False_negBayes)

plt.plot(eignenval1,'go', label="Healthy")
plt.plot(eignenval2,'ro', label="Arrhythmic")

plt.xlabel('Feature')
plt.ylabel('Value')
plt.title('Eigenvalues')
#lt.ylim((-0.5,10))
plt.legend()
plt.show()


# Sensitivity = TP/(TP+FN)
Sensitivity_Bayes=true_positiveBayes/(true_positiveBayes+False_negBayes )
print('sensitivity Bayes Test:', Sensitivity_Bayes )
# Specificity = TN/(TN+FP)
Specificity_Bayes=true_negative_Bayes/(true_negative_Bayes+False_pos_Bayes)
print('specificity Bayes Test:', Specificity_Bayes)


ii=class_id.argsort() #sort Y just to get better plot 
class_id=class_id[ii]
Y=Y[ii,:]