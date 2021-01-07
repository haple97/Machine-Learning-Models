#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import math
import random
import matplotlib.pyplot as plt


# **Data Splitting for Cross Validation**

# In[2]:


trainingSet = pd.read_csv('./trainingSet.csv')


# In[3]:


data = trainingSet.sample(frac=1, random_state = 18)
data.reset_index(inplace = True, drop = True)


# In[4]:


'''K-FOLD VALIDATION: Split into folds'''

# Split data into folds
def split_df(df, chunk_size): 
    chunks = list()
    num_chunks = len(df) // chunk_size + 1
    for i in range(num_chunks):
        chunks.append(df[i*chunk_size:(i+1)*chunk_size])
    return chunks

# Shuffle data and split into folds
def shuffle_split(data):
    fold_size = round(0.1*len(data))
    folds = split_df(data, fold_size) # Split into folds, each has 10% of total number examples in data 
    folds = folds[:10] # Obtain the first 10 folds only

    for df in folds: 
        df.reset_index(inplace=True, drop = True)
    
    return folds


# In[5]:


# Using list of folds to obtain test and training sets
def train_test_split (i, S, t_frac):
    test_set = S[i]
    Sc = S[:i] + S[(i+1):] # list
    Sc = pd.concat(Sc, ignore_index = True) # dataframe
    train_set = Sc.sample(frac=t_frac, random_state = 32) # sample training data with t_frac
    train_set.reset_index(inplace = True, drop = True)
    return train_set, test_set


# **Decision Tree**

# In[6]:


def calc_gini(data):
    p1 = np.count_nonzero(data['decision'])/len(data['decision'])  # P(decision = 1)
    p0 = 1-p1  # P(decision = 0)
    gini = 1 - (p1**2 + p0**2)
    return gini


# In[7]:


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


# In[8]:


def BuildTree(modelID, max_depth, data, depth, used_attr, node = {}):
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
            node[i] = BuildTree(modelID, max_depth, subset, depth+1, used_attr + [split_attr], {}) 
        used_attr = used_attr + [split_attr]
        depth += 1
        tree = node
        return node


# In[9]:


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


# In[10]:


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


# In[11]:


def decisionTree(trainingSet, testSet, d):
    tree = {}
    tree.clear()
    
    # Obtain tree model 
    tree = BuildTree(modelID = 1, max_depth = d, data = trainingSet, depth = 0, used_attr = [], node = {})
    
    # Test on training set
    train_acc,train_pred_array = EvalTree(trainingSet, tree)
    
    # Test on test set
    test_acc,test_pred_array = EvalTree(testSet, tree)
    
    return train_acc, test_acc


# **Bagging**

# In[12]:


def EvalBagging(list_pred_arrays, true_dec):
    df_pred = pd.DataFrame(list_pred_arrays).T        # df of all predictions, each column is for each tree
    pred_avg = np.round(df_pred.mean(axis = 1))       # mean of predictions from all trees
    diff = pred_avg - true_dec                        # difference between predicted and true labels
    accuracy = 1 - (np.count_nonzero(diff)/len(diff)) # accuracy = ratio of zero differences to number of all obs
    return accuracy


# In[13]:


def bagging(trainingSet, testSet, d): 
    num_trees = 30
    tree = {}
    
    list_train_pred_arrays = []
    list_test_pred_arrays = []
    
    for i in range(num_trees):
        random_index = np.random.choice(len(trainingSet), len(trainingSet)) 
        sample = trainingSet.iloc[random_index]
        sample.reset_index(inplace = True, drop = True)
        
        tree.clear()
        tree = BuildTree(modelID = 2, max_depth = d, data = sample, depth = 0, used_attr = [], node = {})
        
        # Evaluate tree on training set
        _,train_pred_array = EvalTree(trainingSet, tree) # train_single_acc
        list_train_pred_arrays.append(train_pred_array)
        
        # Evaluate tree on test set
        _, test_pred_array = EvalTree(testSet, tree) # test_single_acc
        list_test_pred_arrays.append(test_pred_array)
    
    # Evaluate bagged trees on training and test set
    train_acc = EvalBagging(list_train_pred_arrays, trainingSet.iloc[:,-1])
    test_acc = EvalBagging(list_test_pred_arrays, testSet.iloc[:,-1])
        
    return train_acc, test_acc


# **Random Forests**

# In[14]:


def randomForests(trainingSet, testSet, d):
    num_trees = 30
    tree = {}
    
    list_train_pred_arrays = []
    list_test_pred_arrays = []
    
    for i in range(num_trees):
        random_index = np.random.choice(len(trainingSet), len(trainingSet)) 
        sample = trainingSet.iloc[random_index]
        sample.reset_index(inplace = True, drop = True)
        
        tree.clear()
        tree = BuildTree(modelID = 3, max_depth = d, data = sample, depth = 0, used_attr = [], node = {})
        
        # Evaluate tree on training set
        _, train_pred_array = EvalTree(trainingSet, tree)
        list_train_pred_arrays.append(train_pred_array)
        
        # Evaluate tree on test set
        _, test_pred_array = EvalTree(testSet, tree)
        list_test_pred_arrays.append(test_pred_array)
    
    # Evaluate bagged trees on training and test set
    train_acc = EvalBagging(list_train_pred_arrays, trainingSet.iloc[:,-1])
    test_acc = EvalBagging(list_test_pred_arrays, testSet.iloc[:,-1])
        
    return train_acc, test_acc


# **Main Code for CV**

# In[15]:


# Calculate the mean, std, and standard error of any accuracy list 
def calc_stat(acc_list): 
    acc_array = np.array(acc_list)
    mean = np.mean(acc_array)
    std = np.std(acc_array, ddof = 1)
    se = std/math.sqrt(len(acc_list))
    return mean, se


# In[17]:


frac_list = [0.05, 0.075, 0.1, 0.15, 0.2]
d = 8

results_all_frac = {}
for t_frac in frac_list:
    print('t_frac:', t_frac)
    # Create folds 
    folds = shuffle_split(data)
    
    test_DT_list = []
    test_BT_list = []
    test_RF_list = []
    
    S = folds 
    for i in range(len(S)):
        print('fold:', i+1)
        # Select 1 fold as test set and combine the rest as train set
        train_set, test_set = train_test_split(i, S, t_frac)
                
        # Get test accuracy for current fold split
        _, test_DT = decisionTree(train_set, test_set, d)
        _, test_BT = bagging(train_set, test_set, d)
        _, test_RF = randomForests(train_set, test_set, d)
        
        # Append to list to obtain list of 10 accuracy values
        test_DT_list.append(test_DT)
        test_BT_list.append(test_BT)
        test_RF_list.append(test_RF)
    
    test_acc_lists = [test_DT_list, test_BT_list, test_RF_list] 
    methods = ['DT', 'BT', 'RF']
    
    # Calculate + store average and standard error for each method across 10 folds 
    results_one_frac = {}
    for i in range(len(test_acc_lists)):
        mean, se = calc_stat(test_acc_lists[i])
        results_one_frac[methods[i]] = [mean, se, test_acc_lists[i]]
    
    # Store in dict for all t_frac values
    results_all_frac[t_frac] = results_one_frac


# In[18]:


results_df = pd.DataFrame.from_dict(results_all_frac, orient='index')


# In[20]:


results_df[['DT_mean','DT_se', 'DT_acc']] = pd.DataFrame(results_df.DT.tolist(), index= results_df.index)
results_df[['BT_mean','BT_se', 'BT_acc']] = pd.DataFrame(results_df.BT.tolist(), index= results_df.index)
results_df[['RF_mean','RF_se', 'RF_acc']] = pd.DataFrame(results_df.RF.tolist(), index= results_df.index)

results_df


# In[21]:


fold_list = np.arange(1,11).tolist()

DT_acc = pd.DataFrame(results_df.DT_acc.tolist(), index= results_df.index, 
                      columns = ['DT_' + str(i) for i in fold_list])
BT_acc = pd.DataFrame(results_df.BT_acc.tolist(), index= results_df.index,
                     columns = ['BT_' + str(i) for i in fold_list])
RF_acc = pd.DataFrame(results_df.RF_acc.tolist(), index= results_df.index,
                     columns = ['RF_' + str(i) for i in fold_list])

results_df = results_df.join(DT_acc)
results_df = results_df.join(BT_acc)
results_df = results_df.join(RF_acc)
results_df = results_df.drop(columns = ['DT', 'BT', 'RF', 'DT_acc', 'BT_acc', 'RF_acc'])
results_df


# In[22]:


results_df.to_csv('./results_df_frac.csv', index = False)
# results_df = pd.read_csv('./results_df_frac.csv')


# **Visualization**

# In[23]:


fig = plt.figure(figsize = (10,7))

x = frac_list
y1 = results_df['DT_mean']
yerr1 = results_df['DT_se']
y2 = results_df['BT_mean']
yerr2 = results_df['BT_se']
y3 = results_df['RF_mean']
yerr3 = results_df['RF_se']

plt.errorbar(x, y1, yerr=yerr1, label='Decision Tree')
plt.errorbar(x, y2, yerr=yerr2, label='Bagged Trees')
plt.errorbar(x, y3, yerr=yerr3, label='Random Forests')


# plt.ylim(0.4,0.8)
plt.legend(loc='upper left')
plt.xlabel('Training Fraction')
plt.ylabel('Test Accuracy (10-fold CV)')

plt.show()


# In[24]:


fig.savefig('cv_frac.png')


# In[ ]:




