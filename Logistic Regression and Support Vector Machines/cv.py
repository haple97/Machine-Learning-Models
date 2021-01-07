#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt


# In[2]:


'''For LR & SVM: Add intercept + split into label and attribute matrix'''

def split_data(dataset, model_id):   
    true_label = dataset.iloc[:,-1].to_numpy()           # 1D array of length N
    if model_id == 2: 
        true_label[true_label == 0] = -1                 # Convert true labels to [-1, 1] for SVM
    
    x = dataset.iloc[:,:-1].to_numpy()                                     # (N x A) array 
    intercept = np.ones((len(dataset.index),1), dtype = 'float')           # Intercept column of ones
    
    attribute_matrix = np.concatenate((x, intercept), 1)                   # (N x A+1) array
#     print(true_label.shape, attribute_matrix.shape)
    
    return true_label, attribute_matrix


# In[3]:


'''LOGISTIC REGRESSION'''

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

# Optimize weights with gradient descent
def grad_descent(dataset, init_weights, reg_lambda, step):
    new_weights = init_weights
    iterations = 0
    iterations_max = 500
    tol = 1e-6
    old_new = 1
    
    # Split dataset into labels and matrix of attributes
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


# In[4]:


'''SUPPORT VECTOR MACHINE'''

def subgrad_descent(dataset, init_weights, reg_lambda, step):
    new_weights = init_weights
    iterations = 0
    iterations_max = 500
    tol = 1e-6
    old_new = 1
    
    # Split dataset into labels and matrix of attributes
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
        y_pseudo = true_label.copy()
        for i in range(len(hinge)): 
            if hinge[i] >= 1: 
                y_pseudo[i] = 0
        
        # Calculate gradient vector and new weights vector
        grad_descent = reg_term - (1/len(dataset.index))*(np.dot(y_pseudo, attribute_matrix))
        new_weights = weights - step*grad_descent 

    return weights

def get_accuracy_svm(dataset, optim_weights):
    # Split dataset into labels and matrix of attributes
    true_label, attribute_matrix = split_data(dataset, 2)
    
    # Obtain predicted y and squeeze to range [-1, 1]
    y_pred = np.dot(optim_weights, np.transpose(attribute_matrix)) 
    
    # Assign label of either +1 or -1
    y_pred[y_pred <= 0] = -1
    y_pred[y_pred > 0] = 1
    
    # Calculate the difference in 2 label vectors; accuracy = no. of zeros in difference/no. of examples
    diff = y_pred - true_label
    accuracy = 1-(np.count_nonzero(diff)/len(dataset.index))
    return accuracy

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


# In[5]:


'''NAIVE BAYES CLASSIFIER'''

def nbc(data):
    # Calculate priors
    all_yes = data[data['decision'] == 1]
    all_yes_ratio = (len(all_yes.index)+1)/(len(data.index)+2)    # Laplace smoothing applied on priors
    all_no = data[data['decision'] == 0]
    all_no_ratio = (len(all_no.index)+1)/(len(data.index)+2)
    
    # Calculate conditional probabilities each value of each attribute given decision
    all_columns = list(data.columns.values)
    attribute_columns = [col for col in all_columns if col not in ('decision')]
    discrete_valued_columns = ('gender', 'race', 'race_o', 'samerace', 'field') # Diff. ranges
    continuous_valued_columns = [col for col in attribute_columns 
                                 if col not in discrete_valued_columns]         # Range 1-5
    
    conditional_prob = {}  # initialize dictionary to store and access probabilities
    
    for column in attribute_columns: 
        if column in ('gender', 'samerace'): # Range: 0, 1
            min_val = 0
            max_val = 1
            num_vals = 2
        elif column in ('race', 'race_o'):   # Range: 0, 1, 2, 3, 4
            min_val = 0
            max_val = 4
            num_vals = 5
        elif column == 'field':              # Range: 0, 1, 2,...209
            min_val = 0
            max_val = 209
            num_val = 210
        else: 
            min_val = 1
            max_val = 5
            num_vals = 5
        
        for i in range(min_val, max_val+1, 1):
            key = column + str(i)   # key to access dictionary is an attribute and one of its value
            
            attribute_yes = data[(data[column] == i) & (data['decision'] == 1)]
            yes_prob = (len(attribute_yes.index)+1)/(len(all_yes.index)+num_vals) # Laplace smoothing on con. prob
            attribute_no = data[(data[column] == i) & (data['decision'] == 0)]
            no_prob = (len(attribute_no.index)+1)/(len(all_no.index)+num_vals)
            
            conditional_prob[key] = [yes_prob, no_prob]  # assign pair of values to key
    
    conditional_prob['priors'] = [all_yes_ratio, all_no_ratio]  # add prior prob to dictionary
        
    return (conditional_prob)

def Prediction(data, model):
    all_columns = list(data.columns.values)
    attribute_columns = [col for col in all_columns if col not in ('decision')]

    pred_decision = []
    p_yes_list = []
    p_no_list = []
    
    count = 0
    for i in range (len(data.index)): #len(data.index) 
        p_yes = 1  # Initialize here, not above, to reset probabilities to 1 for each entry
        p_no = 1
        for column in attribute_columns: 
            value = data.loc[i, column]               # Get value of an attribute
            p_yes *= model[column + str(value)][0]    # Obtain conditional prob from dict. & 
            p_no *= model[column + str(value)][1]     # multiply them together
            
        p_yes = p_yes*model['priors'][0]              # Multiply with the prior
        p_no = p_no*model['priors'][1]

        if  p_yes >= p_no: # Compare the two calculated probabilities of decision options
            decision = 1
        else: 
            decision = 0
    
        if decision == data.loc[i, 'decision']:  # Count number of correct prediction 
            count += 1
    
    accuracy = count/len(data.index)   # Accuracy is percentage of correct prediction
    return accuracy

def nbc_run(train_set, test_set): 
    model = nbc(train_set)
    
    training_accuracy = Prediction(train_set, model)
    test_accuracy = Prediction(test_set, model)

    return training_accuracy, test_accuracy


# In[6]:


'''OBTAIN DATA'''

# Data for LR and SVM (one-hot encoded)
trainingSet = pd.read_csv('./trainingSet.csv')

# Full data for NBC (label-encoded)
data_nbc = pd.read_csv('./dating-binned.csv', nrows = 6500)

# Obtain test-train sets for NBC, using the same frac and random state
testSet2 = data_nbc.sample(frac=0.2, random_state = 25)
trainingSet2 = data_nbc.drop(testSet2.index)
trainingSet2.reset_index(inplace = True, drop = True)


# In[7]:


'''K-FOLD VALIDATION: Shuffle and Split Data'''

# Split data into folds
def split_df(df, chunk_size): 
    chunks = list()
    num_chunks = len(df) // chunk_size + 1
    for i in range(num_chunks):
        chunks.append(df[i*chunk_size:(i+1)*chunk_size])
    return chunks

# Shuffle data and split into folds
def shuffle_split(dataset):
    # Shuffle data
    data = dataset.sample(frac=1, random_state = 18)
    data.reset_index(inplace = True, drop = True)

    # Split into 10 folds, each of 520 examples
    folds = split_df(data, 520)
    del folds[-1]

    for df in folds: 
        df.reset_index(inplace=True, drop = True)
    
    return folds


# In[8]:


# Using list of folds to obtain test and training sets
def train_test_split (i, S):
    test_set = S[i]
    Sc = S[:i] + S[(i+1):]
    Sc = pd.concat(S, ignore_index = True)
    train_set = Sc.sample(frac=t_frac, random_state = 32)
    train_set.reset_index(inplace = True, drop = True)
    return test_set, train_set


# In[9]:


# Calculate the mean, std, and standard error of any accuracy list 
def calc_stat(acc_list): 
    acc_array = np.array(acc_list)
    mean = np.mean(acc_array)
    std = np.std(acc_array, ddof = 1)
    se = std/math.sqrt(len(acc_list))
    return mean, se


# In[10]:


frac_list = [0.025, 0.05, 0.075, 0.1, 0.15, 0.2]

results_all_frac = {}
for t_frac in frac_list:
    # Create folds 
    folds = shuffle_split(trainingSet)
    folds_NBC = shuffle_split(trainingSet2)
    
    test_LR_list = []
    test_SVM_list = []
    test_NBC_list = []
    
    S = folds
    S_NBC = folds_NBC
    
    for i in range(len(S)):
        # Select 1 fold as test set and combine the rest as train set
        test_set, train_set = train_test_split(i, S)
        test_set_NBC, train_set_NBC = train_test_split(i, S_NBC)
                
        # Get test accuracy for a train-test pair (1 fold)
        _, test_LR = lr(train_set, test_set)
        _, test_SVM = svm(train_set, test_set)
        _, test_NBC = nbc_run(train_set_NBC, test_set_NBC)
        
        # Append to list to obtain list of 10 accuracy values
        test_LR_list.append(test_LR)
        test_SVM_list.append(test_SVM)
        test_NBC_list.append(test_NBC)
    
    test_acc_lists = [test_LR_list, test_SVM_list, test_NBC_list] 
    methods = ['LR', 'SVM', 'NBC']
    
    # Calculate + store average and standard error for each method across 10 folds 
    results_one_frac = {}
    for i in range(len(test_acc_lists)):
        mean, se = calc_stat(test_acc_lists[i])
        results_one_frac[methods[i]] = [mean, se]
    
    # Store in dict for all t_frac values
    results_all_frac[t_frac] = results_one_frac


# In[14]:


results_df = pd.DataFrame.from_dict(results_all_frac, orient='index')

Sc_size = 520*9
results_df['train_size'] = Sc_size*results_df.index

results_df[['LR_mean','LR_se']] = pd.DataFrame(results_df.LR.tolist(), index= results_df.index)
results_df[['SVM_mean','SVM_se']] = pd.DataFrame(results_df.SVM.tolist(), index= results_df.index)
results_df[['NBC_mean','NBC_se']] = pd.DataFrame(results_df.NBC.tolist(), index= results_df.index)

results_df


# In[15]:


fig = plt.figure(figsize = (10,7))

x = results_df['train_size']
y1 = results_df['LR_mean']
yerr1 = results_df['LR_se']
y2 = results_df['SVM_mean']
yerr2 = results_df['SVM_se']
y3 = results_df['NBC_mean']
yerr3 = results_df['NBC_se']

plt.errorbar(x, y1, yerr=yerr1, label='Logistic Regression')
plt.errorbar(x, y2, yerr=yerr2, label='Support Vector Machine')
plt.errorbar(x, y3, yerr=yerr3, label='Naive Bayes Classifier')


plt.ylim(0.4,0.8)
plt.legend(loc='lower right')
plt.xlabel('Size of Training Data (samples)')
plt.ylabel('Test Accuracy')

plt.show()


# In[16]:


'''Hypothesis Test'''
df = results_df[['train_size', 'LR_mean', 'LR_se', 'NBC_mean', 'NBC_se']]


# In[24]:


df['t_value'] = abs((df['LR_mean']-df['NBC_mean'])/(df['LR_se']-df['NBC_se']))


# In[25]:


df['df'] =  (9*(df['LR_se'].pow(2)+df['NBC_se'].pow(2)).pow(2))/(df['LR_se'].pow(4)+df['NBC_se'].pow(4))


# In[26]:


df


# In[ ]:




