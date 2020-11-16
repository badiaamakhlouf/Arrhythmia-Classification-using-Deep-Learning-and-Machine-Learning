# Classification_with_ML-DL
## 1-Arrhythmia:
Arrhythmia is a disease that corresponds to an irregular heartbeat because heart rates can also be irregular. It is also called dysrhythmia. A normal heart rate is between 50 to 100 beats per minute. Arrhythmia and abnormal heart rates do not necessarily occur together. Arrhythmia can occur with a normal heart rate, or with heart rates that are slow (called bradyarrhythmias -- less than 50 beats per minute). Arrhythmias can also occur with rapid heart rates (called tachyarrhythmias -- faster than 100 beats per minute). 

In the United States, more than 850,000 people are hospitalized for an arrhythmia each year. 
## 2- Lab’s objectives:
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


