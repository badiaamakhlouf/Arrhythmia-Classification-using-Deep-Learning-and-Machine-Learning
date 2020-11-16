# Classification_with_ML-DL
## 1-Arrhythmia:
Arrhythmia is a disease that corresponds to an irregular heartbeat because heart rates can also be irregular. It is also called dysrhythmia. A normal heart rate is between 50 to 100 beats per minute. Arrhythmia and abnormal heart rates do not necessarily occur together. Arrhythmia can occur with a normal heart rate, or with heart rates that are slow (called bradyarrhythmias -- less than 50 beats per minute). Arrhythmias can also occur with rapid heart rates (called tachyarrhythmias -- faster than 100 beats per minute). 

In the United States, more than 850,000 people are hospitalized for an arrhythmia each year. 
## 2- Objectives:
This lab aims to perform two types of classifications: first is a binary classification (using only two classes) then a multiclass classification (using 16 classes) based on the following approaches: 

    1-Minimum distance criterion
    2-Bayesian approach 
    3-Neural networks (TensorFlow) approach. 
    4-Support Vector Machine

To Startwith, the provided dataset was analyzed to define each decision region and to meas-ure errors made by using the found decision region through measuring the probability of false and true positives (sensitivity) and the probability of false and true negative (specifici-ty). 
## 3- Dataset description:
The provided dataset contains 452 rows; each row represents a patient and is composed by 280 features. Moreover, the last column stores an integer number from 1 to 16 that specifies the patient’s level of cardiac arrhythmia: class 1 corresponds to absence of arrhythmia and class 16 to severe arrhythmia.
As usual, before executing the analysis process need to clear and to prepare the provided dataset via excluding columns shown as “?” and columns storing only zeros because they carry no information. The remained features were 258 instead of 280 including the last col-umn that is the target class to be estimated.
 In addition, a normalization step was performed using the mean and the standard deviation then as usual the dataset was divided into two sets: training and testing sets: the half of the total dataset was dedicated to the train phase and the other half was dedicated to the test phase. 
## 4 -For two-class (binary) classification:
This approach consists of converting the last column that contains values from 1 to 16 to a column with only two sub-classes: the disease is present or the disease is absent. The last column is the target that was deleted from the other features to perform arrhythmia classifi-cation using the remained 257 features.
### 4.1- Minimum distance criteria
The two sub-classes are the following: Class1 corresponds to absence of Arrhythmia (or healthy class) and all the values different than 1 were assigned to value 2 or class 2 (Ar-rhythmia class). 

The classification decision was based on evaluating the distance between each vector of 257 features from the original dataset and the class mean. The class mean is a vector of 257 ele-ments each element is the mean of one column from the train matrix, the train matrix can be either matrix containing only healthy people or matrix containing only people affected by the disease.

The result giving the minimum distance between the row and the class’s mean allow the assignment of this patient to that class.
The confusion matrix was used as a performance metric to evaluate this approach and to measure the probabilities of true/false positives and the probabilities of true/false negatives as follow: 

![alt text](https://github.com/BaddyMAK/Classification_with_ML-DL/blob/main/results/cm.png)

###### Figure 1: Confusion matrix representation

The Figure 1 represents the following terms: 
    a. True Positives (TP): True positives are the cases when the actual class of the data point was 2 and the predicted is also 2.
    
Ex: The case where a person is actually having arrhythmia (2) and the model classifying his case as arrhythmia (2) comes under True positive.
    b. True Negatives (TN): True negatives are the cases when the actual class of the data point was 1 and the predicted is also 1.
    
Ex: The case where a person NOT having arrhythmia and the model classifying his case as not arrhythmia comes under True Negatives.
    c. False Positives (FP): False positives are the cases when the actual class of the data point was 1 and the predicted is 2. False is because the model has predicted incorrectly and posi-tive because the class predicted was a positive one (2). 

Ex: A person NOT having arrhythmia and the model classifying his case as arrhythmia comes under False Positives.
    d. False Negatives (FN): False negatives are the cases when the actual class of the data point was 2 and the predicted is 1. False is because the model has predicted incorrectly and negative because the class predicted was a negative 1. 

Ex: A person having arrhythmia and the model classifying his case as No-arrhythmia comes under False Negatives.

The ideal wanted scenario is that the model should give 0 False Positives and 0 False Nega-tives. However, that is not the case in real life as any model will NOT be 100% accurate most of the times. The most important thing now is how to minimize those two values: False Positives and False Negatives.

### 4.2- Bayesian Approach
Equally, in this case the last column was divided onto the same classes of minimum distance criterion approach: Class1 that corresponds to healthy class and all the values different than 1 were assigned, to Arrhythmia class or value 2. 

Before deploying the Bayes criteria, need to reduce features number through performing the Karhunen-Loève decomposition and selecting only a subset of the features, this subset must corresponds to the largest eigenvalues. The developed python program automatically pick the largest eigenvalues that allow getting the target percentage and satisfying the following conditions:  

![alt text](https://github.com/BaddyMAK/Classification_with_ML-DL/blob/main/results/equation.PNG)

L is the number of extracted features and it must be as small as possible, knowing that L is less than F: the total number of features.

The confusion matrix has been changed depending on the target percentage as follow:

![alt text](https://github.com/BaddyMAK/Classification_with_ML-DL/blob/main/results/Percentage%20comparaison.PNG)

In the coming analysis and interpretation, the percentage was fixed to 0.99 because it has gave the lowest probability error.

As shown in Figure 2, the number of used eigenvec-tors was 76 for Healthy class and 67 for arrhythmic class. 

Healthy class requires more eigenvectors because the largest eigenvalues to satisfy the target condition are lower with respect to arrhythmic class, which is characterized by a larger eigenvalues. 

![alt text](https://github.com/BaddyMAK/Classification_with_ML-DL/blob/main/results/eigenval.png)

###### Figure 2: Eigenvalue representation for both healthy and Arrhythmic classes

In fact, the classification decision in Bayes criterion was based on comparing the probability that the sample is from a predefined class. The sample will be assigned to the class giving the highest probability.

### 4.3- Neural networks (TensorFlow) approach
Neural networks can be used to solve both regression problems and classification problems. The designer chooses the structure and the activation functions. Furthermore, the weights and the biases are the variables to be optimized according to the inputs and the desired out-puts. 
In this section, the last column was divided into two sub-classes class 0 for healthy patients and class 1 for patients with arrhythmia.
The gradient algorithm was used in this section because it is the typical way to find the op-timized weights and in order to deploy the gradient algorithm three values of learning coef-ficient were tried for both train and test dataset: 0.000001, 0.00001 and 0.0001.

The Figure 3 and Figure 4 were obtained and the coming selection was based on them: 

![alt text](https://github.com/BaddyMAK/Classification_with_ML-DL/blob/main/results/Figure%2025.png)

###### Figure 3: Neural network binary classification training accuracy 

![alt text](https://github.com/BaddyMAK/Classification_with_ML-DL/blob/main/results/Figure%2026.png)

###### Figure 4: Neural network binary classification testing accuracy 

From Figures 3 and 4 it is noticeable that as the learning coefficient increases the accura-cy increases for both train and test phase. 

Choosing an optimum learning rate is crucial because the network may fail in either train, or take much longer to converge to the global minima or not. With a small learning rate, the time required to train the model is lower therefore in the coming analysis the 0.000001 have been chosen as learning rate.
Fixing the number of iterations to 700 and the learning rate to 0.000001 the following table was obtained:

![alt text](https://github.com/BaddyMAK/Classification_with_ML-DL/blob/main/results/table.png)

###### Table 1: Sensitivity and Specificity for both train data and test data

Neural Network tend to perform overfitting: they adapt more to the training data than the testing data. Overfitting increases if the number of parameters, i.e. number of hidden layers and number of nodes, increases because there is more degrees of freedom and the neural network learns too well the training data.

### 4.4- Support Vector Machine approach: SVM
In this report, the SVM approach was used to perform binary classification. The same as the neural network approach the last column was divided into two sub-classes class 0 for healthy people and class 1 for patients with arrhythmia.

Before applying SVM, the half of the provided data was dedicated to the train phase and the other half was assigned to the test phase then the data was pre-processed using the PCA.

The data stored in matrix X (226 rows corresponding to 226 patients and 257 columns corre-sponding to the features) while y is the vector of 226 classes (contains either value 0 or val-ue 1). The hyperplane that separates the two classes satisfies the following equation:

![alt text](https://github.com/BaddyMAK/Classification_with_ML-DL/blob/main/results/equation%202.PNG)     

Where W is column vector with 257 elements and b is an offset equal to 0.33.

In the SVM classification report, 118 samples were assigned to class 0 and 108 samples were assigned to class 1 the same thing as the original test target vector (118 for class 0 and 108 for class 1). 

A box constraint C higher than 10 (chosen in this report) will decrease the TP and TN and will increase FP and FN that we aim to minimize in each model.
### 4.5- Result analysis

![alt text](https://github.com/BaddyMAK/Classification_with_ML-DL/blob/main/results/Table%206.png)
###### Table 2: Sensitivity and Specificity for the used approaches

![alt text](https://github.com/BaddyMAK/Classification_with_ML-DL/blob/main/results/table%207.png)
###### Table 3: TP, FP, TN, FN and Probability error
For the test dataset:
    •	Bayesian approach gave a probability error (0.265) less than the one given by the Mini-mum distance criterion (0.509).
    •	Among the four methods and with the chosen parameters in this work, Bayes Criterion offers the best results for sensitivity followed by SVM. The neural network offers the best behavior for the specificity and the worst for the sensitivity while SVM presents the worst result for specificity. 





