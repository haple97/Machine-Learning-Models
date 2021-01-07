#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import math
import random


# In[2]:


import sys
trainingDataFilename = str(sys.argv[1])
testDataFilename = str(sys.argv[2])
modelIdx = int(sys.argv[3])

trainingSet = pd.read_csv('./' + trainingDataFilename)
testSet = pd.read_csv('./' + testDataFilename)


# In[2]:


# trainingSet = pd.read_csv('./trainingSet.csv')
# testSet = pd.read_csv('./testSet.csv')


# **Decision Tree**

# In[3]:


def calc_gini(data):
    p1 = np.count_nonzero(data['decision'])/len(data['decision'])  # P(decision = 1)
    p0 = 1-p1  # P(decision = 0)
    gini = 1 - (p1**2 + p0**2)
    return gini


# In[4]:


def split_attribute(modelID, data, used_attr):
    total_gini = calc_gini(data)
    
    attr = (data.columns.values[:-1]).tolist() # All attributes in dataset, except 'decision' (last col)
    
    if modelID == 1 or modelID == 2:   # For decision tree and bagged trees
        for item in used_attr:         # Remove all used attributes
            attr.remove(item)
        columns = attr
        
    elif modelID == 3:                  # For random forests
        k = round(math.sqrt(len(attr)))  
        k_attr = random.sample(attr, k) # Choose k attributes randomly from list of ALL attributes
        for item in used_attr:          # Remove any used attributes
            if item in k_attr: 
                k_attr.remove(item)
        columns = k_attr
    
    gini_gain_list = []    
    for column in columns:
        
        # For subgroup where value of attribute is 0 (left branch)
        subset0 = data[data[column] == 0]
        if len(subset0) == 0:
            gini0 = 0
        else: 
            gini0 = calc_gini(subset0)
        
        # For subgroup where value of attribute is 1 (right branch)
        subset1 = data[data[column] == 1]
        if len(subset1) == 0:
            gini1 = 0
        else:
            gini1 = calc_gini(subset1)
        
        gini_gain = total_gini - (len(subset0)/len(data) * gini0 + len(subset1)/len(data) * gini1)  
        gini_gain_list.append(gini_gain)
    
    if len(set(gini_gain_list)) == 1:  # If all gini gains are equal to each other
        split_attr = None
        column_id = None
    else: 
        max_index = gini_gain_list.index(max(gini_gain_list))   # Get index of the max gini gain in list
        split_attr = columns[max_index]                         # Get corresponding attribute with max gain         
        column_id =  data.columns.get_loc(split_attr)           # Get id of corresponding column in dataset
    return split_attr, column_id
        


# In[9]:


def BuildTree(modelID, data, depth, used_attr, node = {}):
    max_depth = 8
    min_examples = 50
    
    num_examples = len(data)
    
    if node is None:
        return
    
    elif num_examples == 0: 
        return
    
    elif len(set(data['decision'])) == 1:   # If all have the same label (decision)
        return {'label': data['decision'].iloc[0], 'depth': depth} # Return a leaf node of that label
    
    elif depth >= max_depth or num_examples < min_examples: # If depth reaches 8 or number of examples becomes <50
        if np.mean(data['decision']) == 0.5: # If equal number of 0 and 1
            label = random.randint(0, 1)     # Return randomly 0 or 1
        else:
            label = np.round(np.mean(data['decision'])) # Majority of 0 would lead to mean=0; similarly for majority 1
        return {'label': label, 'depth': depth}         # Return majority label as a leaf node 
    
    else:        
        split_attr, attr_id = split_attribute(modelID, data, used_attr)  
        node = {'attr': split_attr, 'attr_id': attr_id, 'depth': depth}
        
        for i in [0, 1]:
            subset = data[data[split_attr] == i]
            node[i] = BuildTree(modelID, subset, depth+1, used_attr + [split_attr], {}) 
        used_attr = used_attr + [split_attr]
        depth += 1
        tree = node
        return node


# In[10]:


def predict_decision(obs, tree):
    cur_layer = tree
    
    # If there's still attribute to split (i.e. NOT a leaf node), find that split attribute of the observation
    while cur_layer.get('attr') is not None: 
        
        # If attribute is 0, step down to left (0) branch lower layer
        if obs[cur_layer['attr_id']] == 0: 
            cur_layer = cur_layer[0]
            
        # Otherwise (attribute is 1), step down to right (1) branch lower layer    
        else: 
            cur_layer = cur_layer[1]
    # When there's no more attribute to split, this is leaf node, return label 
    else:
        return cur_layer['label']


# In[11]:


def EvalTree(data, tree):
    pred_dec_list = []
    for index, obs in data.iterrows():
        pred_dec = predict_decision(obs, tree)
        pred_dec_list.append(pred_dec)
    pred_dec_array = np.asarray(pred_dec_list)
    true_dec = np.asarray(data.iloc[:,-1])
    
    diff = pred_dec_array - true_dec
    accuracy = 1 - (np.count_nonzero(diff)/len(diff))
    
    return accuracy, pred_dec_array


# In[12]:


def decisionTree(trainingSet, testSet):
    tree = {}
    tree.clear()
    
    # Obtain tree model 
    tree = BuildTree(modelID = 1, data = trainingSet, depth = 0, used_attr = [], node = {})
    
    # Test on training set
    train_acc,train_pred_array = EvalTree(trainingSet, tree)
    print('Training Accuracy DT:', round(train_acc, 2))
    
    # Test on test set
    test_acc,test_pred_array = EvalTree(testSet, tree)
    print('Test Accuracy DT:', round(test_acc, 2))
   



# **Bagged Trees**

# In[14]:


def EvalBagging(list_pred_arrays, true_dec):
    df_pred = pd.DataFrame(list_pred_arrays).T        # df of all predictions, each column is for each tree
    pred_avg = np.round(df_pred.mean(axis = 1))       # mean of predictions from all trees
    diff = pred_avg - true_dec                        # difference between predicted and true labels
    accuracy = 1 - (np.count_nonzero(diff)/len(diff)) # accuracy = ratio of zero differences to number of all obs
    return accuracy


# In[15]:


def bagging(trainingSet, testSet): 
    num_trees = 30
    tree = {}
    
    list_train_pred_arrays = []
    list_test_pred_arrays = []
    
    for i in range(num_trees):
        random_index = np.random.choice(len(trainingSet), len(trainingSet)) 
        sample = trainingSet.iloc[random_index]
        sample.reset_index(inplace = True, drop = True)
        
        tree.clear()
        tree = BuildTree(modelID = 2, data = sample, depth = 0, used_attr = [], node = {})
        
        # Evaluate tree on training set
        _, train_pred_array = EvalTree(trainingSet, tree)
        list_train_pred_arrays.append(train_pred_array)
        
        # Evaluate tree on test set
        _, test_pred_array = EvalTree(testSet, tree)
        list_test_pred_arrays.append(test_pred_array)
    
    # Evaluate bagged trees on training and test set
    train_acc = EvalBagging(list_train_pred_arrays, trainingSet.iloc[:,-1])
    test_acc = EvalBagging(list_test_pred_arrays, testSet.iloc[:,-1])
        
    print('Training Accuracy BT:', round(train_acc, 2))
    print('Test Accuracy BT:', round(test_acc, 2))
        

# **Random Forests**

# In[17]:


def randomForests(trainingSet, testSet):
    num_trees = 30
    tree = {}
    
    list_train_pred_arrays = []
    list_test_pred_arrays = []
    
    for i in range(num_trees):
        random_index = np.random.choice(len(trainingSet), len(trainingSet)) 
        sample = trainingSet.iloc[random_index]
        sample.reset_index(inplace = True, drop = True)
        
        tree.clear()
        tree = BuildTree(modelID = 3, data = sample, depth = 0, used_attr = [], node = {})
        
        # Evaluate tree on training set
        _, train_pred_array = EvalTree(trainingSet, tree)
        list_train_pred_arrays.append(train_pred_array)
        
        # Evaluate tree on test set
        _, test_pred_array = EvalTree(testSet, tree)
        list_test_pred_arrays.append(test_pred_array)
    
    # Evaluate bagged trees on training and test set
    train_acc = EvalBagging(list_train_pred_arrays, trainingSet.iloc[:,-1])
    test_acc = EvalBagging(list_test_pred_arrays, testSet.iloc[:,-1])
        
    print('Training Accuracy RF:', round(train_acc, 2))
    print('Test Accuracy RF:', round(test_acc, 2))



# In[ ]:


if modelIdx == 1: 
    decisionTree(trainingSet, testSet)
    
elif modelIdx == 2:
    bagging(trainingSet, testSet)

elif modelIdx == 3: 
    randomForests(trainingSet, testSet)
    
else: 
	print('Model ID not valid')



