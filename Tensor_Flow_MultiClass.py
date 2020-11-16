# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 17:02:37 2018

@author: Badiaa
"""
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#%% - Exercise 2 %%%#
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

# In this script we are going to implement Multiclass classification on the last column
Target=clean_data_frame[:,-1] # the target is the last column of the clean_data_frame matrix
Y=clean_data_frame[:,0:-1]

# parameters
n_classes = 16
n_features = Y.shape[1]
n_samples = Y.shape[0]


Target_class = np.zeros([n_samples, n_classes])
index = clean_data_frame[:,-1]-1
Target_class[np.arange(n_samples), index.astype(int)] = 1

# normalizing the data 
mu=np.mean(Y, axis=0) # mean (one valye for each column)
sigma=np.std(Y, axis=0) #standard deviation (one value for each column)
norm_dataset=(Y-mu)/sigma # normalized training data 

# Spliting the data to train/test sets
data_train=norm_dataset[0:226,:] # train data
data_test=norm_dataset[226:452,:] # test data
Target_train=Target_class[0:226] # the target is the last column of the clean_data_frame matrix
Target_test=Target_class[226:452]# 
#%%%%%%%
#--- initial settings
# to generate always the same results
tf.reset_default_graph()
np.random.seed(1234)
tf.set_random_seed(1234)
Num_hiddenNodes= int(np.round(n_features/2))
n_hidden_1 = n_features # 1st layer number of features
n_hidden_2 = Num_hiddenNodes# 2nd layer number of features
n_input = n_features # Number of feature

# Building the computations graph
x = tf.placeholder(tf.float32, [None, n_input])
t = tf.placeholder(tf.float32, [None, n_classes])

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
    out= tf.nn.softmax(out_layer)
    return out

# Store layers weight & bias
weights = {
    'w1': tf.Variable(tf.truncated_normal(shape=[n_input,n_hidden_1 ], mean=0.0, stddev=0.1)),
    'w2': tf.Variable(tf.truncated_normal(shape=[n_hidden_1,n_hidden_2], mean=0.0, stddev=0.1)),
    'W_out': tf.Variable(tf.truncated_normal([n_hidden_2, n_classes], mean=0.0, stddev=0.1))
}

biases = {
    'b1':  tf.Variable(tf.zeros([n_hidden_1])),
    'b2': tf.Variable(tf.zeros([n_hidden_2])), 
    'bias_out': tf.Variable(tf.zeros([n_classes]))
}

pred = multilayer_perceptron(x, weights, biases)

# define cost function
learningRate=0.00001
loss = tf.reduce_sum(-tf.reduce_sum(t*tf.log(pred+ learningRate))) # Cross entropy. +1e-4 ensures that log(0) is never the case

#--- optimizer structure
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learningRate).minimize(loss) # Gradient Descent Optimizer
# Basic accuracy metric
n_correct = tf.equal(tf.argmax(t, 1), tf.argmax(pred, 1))
accuracy = tf.reduce_mean(tf.cast(n_correct, tf.float32))

#--- define the session
session = tf.Session()
#--- initialize
session.run(tf.global_variables_initializer())

# reshape the dataset
# Train 
x1 = data_train
t1 = Target_train
train_data = {x: x1, t: t1}

# Test
x2 = data_test
t2 = Target_test
test_data = {x: x2, t: t2}

#%% This section was used to select the optimum learning rate and the best number of iterations

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

print('test_accuracy') 
for kk, Num_iterations1  in enumerate (Num_iterationsSet):
    for bb in range( Num_iterations1 ):
        session.run(optimizer, feed_dict=test_data)
        test_accuracy = session.run(accuracy, feed_dict=test_data)
    print(test_accuracy)'''
    
#%%    
#--- run the learning machine
# Training model
    
for gg  in range (900):
    session.run(optimizer, feed_dict=train_data)
    train_accuracy = session.run(accuracy, feed_dict=train_data)
    session.run(optimizer, feed_dict=test_data)
    test_accuracy = session.run(accuracy, feed_dict=test_data)
print(train_accuracy)
print(test_accuracy)  


# define the predicted class for the train part 
est_classTrain = np.zeros([226])
est_classTest = np.zeros([226])
#iterate over each row
for row in range(0,226):
    x_train = np.reshape(data_train[row,:],[1,257])
    t_train = np.reshape(Target_train[row,:],[1,16])
    train= {x: x_train, t: t_train}
    evaluation= pred.eval(feed_dict=train,session=session)
    est_classTrain[row] = np.argmax(evaluation[0])
    x_test=np.reshape(data_test[row,:],[1,257])
    t_test = np.reshape(Target_test[row,:],[1,16])
    test= {x: x_test, t: t_test}
    evalua= pred.eval(feed_dict=test,session=session)
    est_classTest[row] = np.argmax(evalua[0])

Mylist=[11,12,13]
# confusion matrix for the train data 
confusion_Matrix_train = np.zeros([16,16])
for tt in range(1,17):
    for jj in range(1, 17):
        if tt not in Mylist:
            indextrue_classTrain = np.where(Target[0:226] == tt)
            indexest_classTrain = np.where(est_classTrain[indextrue_classTrain] == jj)
            confusion_Matrix_train[jj-1,tt-1] = len(indexest_classTrain[0])/len(indextrue_classTrain[0])

plt.pcolormesh(range (1,17), range(1,17), confusion_Matrix_train)
#plt.matshow( confusion_Matrix_train)
plt.ylabel('Predicted')
plt.xlabel('Actual')
plt.title('Training Data Confusion Matrix')
plt.colorbar()
plt.show()

# confusion matrix for the test data 
confusion_Matrix_test = np.zeros([16,16])
for dd in range(1,17):
    for pp in range(1, 17):
        if dd not in Mylist:
            indextrue_classTest = np.where(Target[226:452] == dd)
            indexest_classTest = np.where(est_classTest[indextrue_classTest] == pp)
            confusion_Matrix_test[pp-1,dd-1] = len(indexest_classTest[0])/len(indextrue_classTest[0])

plt.pcolormesh(range(1,17), range(1,17), confusion_Matrix_test)
#plt.matshow( confusion_Matrix_train)
plt.ylabel('Predicted')
plt.xlabel('Actual')
plt.title('Test Data Confusion Matrix')
plt.colorbar()
plt.show()

targtest=Target[226:452]
targtrain=Target[0:226]