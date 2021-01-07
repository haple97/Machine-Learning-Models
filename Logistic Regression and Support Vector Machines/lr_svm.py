#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import math


# In[2]:


import sys
trainingDataFilename = str(sys.argv[1])
testDataFilename = str(sys.argv[2])
modelIdx = int(sys.argv[3])

trainingSet = pd.read_csv('./' + trainingDataFilename)
testSet = pd.read_csv('./' + testDataFilename)

# In[3]:


# trainingSet = pd.read_csv('./trainingSet.csv')
# testSet = pd.read_csv('./testSet.csv')


# In[4]:


def split_data(dataset, model_id):   # Add intercept + split into label and attribute matrix
    true_label = dataset.iloc[:,-1].to_numpy()           # 1D array of length N
    if model_id == 2: 
        true_label[true_label == 0] = -1                 # Convert true labels to [-1, 1] for SVM
    
    x = dataset.iloc[:,:-1].to_numpy()                                     # (N x A) array 
    intercept = np.ones((len(dataset.index),1), dtype = 'float')           # Intercept column of ones
    attribute_matrix = np.concatenate((x, intercept), 1)                   # (N x A+1) array
    
    return true_label, attribute_matrix


# In[5]:


# This function calculates y_hat (the predicted y, i.e. predicted P(y=1|x_i)) using a set of weights
def predict(attribute_matrix, weights):
    power_array = np.dot(weights, np.transpose(attribute_matrix))      # (1xA).(AxN) = (1xN)
    
    yhat_list = []
    for power in power_array:     # Two different ways to compute yhat to avoid overflow
        if power < 0: 
            yhat = math.exp(power)/(1+ math.exp(power))
        else: 
            yhat = 1/(1+math.exp(-power))
        yhat_list.append(yhat)
    yhat_array = np.asarray(yhat_list)
    
    return yhat_array


# In[6]:


def grad_descent(dataset, init_weights, reg_lambda, step):
    new_weights = init_weights
    iterations = 0
    iterations_max = 500
    tol = 1e-6
    old_new = 1
    
    # Split dataset into labels and matrix of attributes, with intercept column added
    true_label, attribute_matrix = split_data(dataset, 1)
    
    while (iterations <= iterations_max and old_new >= tol):
        if iterations == 0: 
            old_new = 1
        else: 
            old_new = np.linalg.norm(weights-new_weights)
        weights = new_weights        
        iterations += 1

        # Predict y given current weights
        y_pred = predict(attribute_matrix, weights)
        
        # Compute matrix of differences between predicted and true values
        diff_matrix = -true_label + y_pred                   # 1xN array
        sum_term = np.dot(diff_matrix, attribute_matrix)     # (1xN).(NxA) = (1xA) array
        
        # Compute regularization term
        reg_term = reg_lambda*weights

        # Calculate gradient vector and new weights vector
        grad_descent = sum_term + reg_term
        new_weights = weights - step*grad_descent 
        
    return weights


# In[7]:


def get_accuracy(dataset, optim_weights):
    # Split dataset into labels and matrix of attributes
    true_label, attribute_matrix = split_data(dataset, 1)
    
    # Obtain predicted y and round to obtain labels of 0 or 1
    y_pred = predict(attribute_matrix, optim_weights)
    pred_label = np.round(y_pred)
    
    # Calculate the difference in 2 label vectors; accuracy = no. of zeros in difference/no. of examples
    diff = pred_label - true_label
    accuracy = 1-(np.count_nonzero(diff)/len(dataset.index))
    return accuracy


# In[8]:


def lr(trainingSet, testSet):
    reg_lambda = 0.01
    init_weights = np.asarray([0.0 for i in range(len(trainingSet.iloc[0]))])
    step = 0.01
    
    # LR learning with gradient descent
    optim_weights = grad_descent(trainingSet, init_weights, reg_lambda, step)
    
    # Test on training set
    train_accuracy = get_accuracy(trainingSet, optim_weights)
    
    # Test on test set
    test_accuracy = get_accuracy(testSet, optim_weights)
    
    return train_accuracy, test_accuracy


# In[9]:


def subgrad_descent(dataset, init_weights, reg_lambda, step):
    new_weights = init_weights
    iterations = 0
    iterations_max = 500
    tol = 1e-6
    old_new = 1
    
    # Split dataset into labels and matrix of attributes, with intercept column added
    true_label, attribute_matrix = split_data(dataset, 2)
    
    while (iterations <= iterations_max and old_new >= tol):
        if iterations == 0: 
            old_new = 1
        else: 
            old_new = np.linalg.norm(weights-new_weights)
        weights = new_weights 
        iterations += 1
        
        # Predict y given current weights
        y_pred = np.dot(weights, np.transpose(attribute_matrix)) 
        
        # Compute regularization term
        reg_term = reg_lambda*weights
        
        # Compute (true_label*weights) for hinge loss
        hinge = true_label*y_pred           # Element-wise multiplication for every example 
        y_pseudo = true_label.copy()        # Make a copy of list - changing the copy would not change original
        for i in range(len(hinge)): 
            if hinge[i] >= 1: 
                y_pseudo[i] = 0

        # Calculate gradient vector and new weights vector
        grad_descent = reg_term - (1/len(dataset.index))*(np.dot(y_pseudo, attribute_matrix))
        
        new_weights = weights - step*grad_descent 
    
    return weights


# In[10]:


def get_accuracy_svm(dataset, optim_weights):
    # Split dataset into labels and matrix of attributes
    true_label, attribute_matrix = split_data(dataset, 2)
    
    # Obtain predicted y and squeeze to range [-1, 1]
    y_pred = np.dot(optim_weights, np.transpose(attribute_matrix))
    
    # Assign label of either +1 or -1
    y_pred[y_pred < 0] = -1
    y_pred[y_pred >= 0] = 1
    
    # Calculate the difference in 2 label vectors; accuracy = no. of zeros in difference/no. of examples
    diff = y_pred - true_label
    accuracy = 1-(np.count_nonzero(diff)/len(dataset.index))
    return accuracy


# In[11]:


def svm(trainingSet, testSet):
    init_weights = np.asarray([0.0 for i in range(len(trainingSet.iloc[0]))])
    reg_lambda = 0.01
    step = 0.5
    
    optim_weights = subgrad_descent(trainingSet, init_weights, reg_lambda, step)
    
    # Test on training set
    train_accuracy = get_accuracy_svm(trainingSet, optim_weights)
    
    # Test on test set
    test_accuracy = get_accuracy_svm(testSet, optim_weights)
    
    return train_accuracy, test_accuracy


# In[12]:


if modelIdx == 1: 
    train_LR, test_LR = lr(trainingSet, testSet)
    print('Training Accuracy LR: ', round(train_LR, 2))
    print('Test Accuracy LR: ', round(test_LR, 2))
    
if modelIdx == 2:
	train_SVM, test_SVM = svm(trainingSet, testSet)
	print('Training Accuracy SVM: ', round(train_SVM, 2))
	print('Test Accuracy SVM: ', round(test_SVM, 2))





