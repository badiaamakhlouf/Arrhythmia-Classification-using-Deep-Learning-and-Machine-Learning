# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 09:04:19 2017

@author: badiaa
"""
import tensorflow as tf
import numpy as np
import pandas as pd

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
# Each value greater to 1 will be assigned to value 1 and value 1 still for value 0
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

# number of rows and columns
Nnumber_rows=data_train.shape[0]
Number_features=data_train.shape[1]


# define the two submatrices 
#identifying the indexs of the Healthy patient from Target class (value = 1)
healthyTrain=np.where(Target_train==0) 
healthyTest=np.where(Target_test==0) 
class_healthy_train=Target_train[healthyTrain[0]]
class_healthy_test=Target_test[healthyTest[0]]

#identifying the indexs of the patients with Arrhythmia from Target class (value != 1)
Arry_existTrain=np.where(Target_train==1)
Arry_existTest=np.where(Target_test==1)
class_Arry_train=Target_train[Arry_existTrain[0]]
class_Arry_test=Target_test[Arry_existTest[0]]

# From last column delete each line stores value 0 (healthy) class.
# ==> We obtain Matrix with only people affected by arrhythmia
Matrix_ArrhyTrain=np.delete(data_train,healthyTrain[0], axis=0)
Matrix_ArrhyTest=np.delete(data_test,healthyTest[0], axis=0)

# From last column delete each line stores value different than 1 (Arrhythmic) class. 
#==> WE obtain Matrix with only healthy people ( no arrhythmia)
Matrix_healthyTrain=np.delete(data_train,Arry_existTrain[0], axis=0)
Matrix_healthyTest=np.delete(data_test,Arry_existTest[0], axis=0) 
Y1_Train=Matrix_healthyTrain
Y1_Test=Matrix_healthyTest


#%%%%%%%
#--- initial settings
# to generate always the same results
tf.reset_default_graph()
np.random.seed(1234)
tf.set_random_seed(1234)
Num_hiddenNodes= int(np.round(Number_features/2))
n_hidden_1 = Number_features # 1st layer number of features
n_hidden_2 = Num_hiddenNodes# 2nd layer number of features
n_input = Number_features # Number of feature
n_classes = 1 # Number of classes to predict ==> output

x = tf.placeholder(tf.float32, [None, n_input]) #inputs
t = tf.placeholder(tf.float32, [None, n_classes])#desired outputs


# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['w1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['w2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer= tf.matmul(layer_2, weights['W_out']) + biases['bias_out']
    out= tf.sigmoid(out_layer)
    return out

# Store layers weight & bias
weights = {
    'w1': tf.Variable(tf.truncated_normal(shape=[n_input,n_hidden_1 ], mean=0.0, stddev=0.1,dtype=tf.float32)),
    'w2': tf.Variable(tf.truncated_normal(shape=[n_hidden_1,n_hidden_2], mean=0.0, stddev=0.1, dtype=tf.float32)),
    'W_out': tf.Variable(tf.truncated_normal([n_hidden_2, n_classes], mean=0.0, stddev=0.1, dtype=tf.float32))
}

biases = {
    'b1': tf.Variable(tf.constant(0.1, shape=(n_hidden_1,),dtype=tf.float32)),
    'b2': tf.Variable(tf.constant(0.1, shape=(n_hidden_2,), dtype=tf.float32)),
    'bias_out': tf.Variable(tf.constant(0.1, shape=(n_classes,), dtype=tf.float32))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)
# Define loss and optimizer
cost = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=t,logits= pred))
learning_rate = 0.000001
# in the training phase
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)# gradient descent was used
# Initializing the variables
init = tf.global_variables_initializer()
# Launch the graph
# generate the data
# reshape training data set 
x1 = data_train
reshaped_TargetTrain= np.reshape(Target_train,[np.shape(Target_train)[0],1])
t1 = reshaped_TargetTrain
train_data = {x: x1, t: t1}

#test the model 
# reshape test data set 
x2 = data_test
reshaped_TargetTrain= np.reshape(Target_test,[np.shape(Target_test)[0],1])
t2 = reshaped_TargetTrain
test_data = {x: x2, t: t2}


correct_prediction = tf.equal(tf.round(t), tf.round(pred))
#correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(t, 1))
# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#--- define the session
session = tf.Session()

#--- initialize
tf.global_variables_initializer().run(session=session)
# Parameters

#%% this section was used to select the best number of iteration for the optimum learning rate 
# I need to comment it otherwise I will get wrong results 
'''
Num_iterationsSet = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000,
                     1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]


#--- run the learning machine to choose the best number of iteration 
print('train_accuracy') 
for j, Num_iterations  in enumerate (Num_iterationsSet):
    for i in range( Num_iterations ):
        session.run(optimizer, feed_dict=train_data)
        #print(cost.eval(feed_dict=train_data,session=sess))
        train_accuracy = session.run(accuracy, feed_dict=train_data)
    print(train_accuracy)

#--- run the learning machine to choose the best number of iteration
print('test_accuracy') 
for kk, Num_iterations1  in enumerate (Num_iterationsSet):
    for bb in range( Num_iterations1 ):
        session.run(optimizer, feed_dict=test_data)
        test_accuracy = session.run(accuracy, feed_dict=test_data)
    print(test_accuracy)'''
#%%
# specificity and sensitivity for training data
x3= Matrix_ArrhyTrain
reshaped_class_Arry_train= np.reshape(class_Arry_train,[np.shape(class_Arry_train)[0],1])
t3 = reshaped_class_Arry_train
train_dataArry = {x: x3, t: t3}

x4= Matrix_healthyTrain
reshaped_class_healthy_train= np.reshape(class_healthy_train,[np.shape(class_healthy_train)[0],1])
t4 = reshaped_class_healthy_train
train_datahealthy = {x: x4, t: t4}
# from previous section the best number of iterations was 700 and the optimum learning rate was 0.000001
for gg  in range (700):
    session.run(optimizer, feed_dict=train_data)
    train_accuracy = session.run(accuracy, feed_dict=train_data)
    session.run(optimizer, feed_dict=test_data)
    test_accuracy = session.run(accuracy, feed_dict=test_data)
print(train_accuracy)
print(test_accuracy)
    
    
# senssitivity measures the proportion of actual positives that are correctly identified by the model
# specificity measures the proportion of actual negatives that are correctly identified by the model
sensitivitytrain = session.run(accuracy, feed_dict=train_dataArry)
print('sensitivity for train data:',sensitivitytrain)

specificitytrain  = session.run(accuracy, feed_dict=train_datahealthy)
print('specificity for train data:', specificitytrain)


# specificity and sensitivity for test data
x5= Matrix_ArrhyTest 
reshaped_class_Arry_test= np.reshape(class_Arry_test,[np.shape(class_Arry_test)[0],1])
t5 = reshaped_class_Arry_test
test_dataArry = {x: x5, t: t5}

x6= Matrix_healthyTest 
reshaped_class_healthy_test= np.reshape(class_healthy_test,[np.shape(class_healthy_test)[0],1])
t6 = reshaped_class_healthy_test
test_datahealthy = {x: x6, t: t6}


sensitivitytest = session.run(accuracy, feed_dict=test_dataArry)
print('sensitivity for test data:', sensitivitytest)

specificitytest  = session.run(accuracy, feed_dict=test_datahealthy)
print('specificity for test data:', specificitytest)









