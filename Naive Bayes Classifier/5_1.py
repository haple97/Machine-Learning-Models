#!/usr/bin/env python
# coding: utf-8

# In[65]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[67]:


def nbc(t_frac):
    training_set = pd.read_csv('./trainingSet.csv')
    data = training_set.sample(frac=t_frac, random_state = 47)
    
    # Calculate priors
    all_yes = data[data['decision'] == 1]
    all_yes_ratio = (len(all_yes.index)+1)/(len(data.index)+2)    # Laplace smoothing applied on priors
    all_no = data[data['decision'] == 0]
    all_no_ratio = (len(all_no.index)+1)/(len(data.index)+2)
    
    # Calculate conditional probabilities each value of each attribute given decision
    all_columns = list(data.columns.values)
    attribute_columns = [col for col in all_columns if col not in ('decision')]
    discrete_valued_columns = ('gender', 'race', 'race_o', 'samerace', 'field')
    continuous_valued_columns = [col for col in attribute_columns if col not in discrete_valued_columns]
    
    conditional_prob = {}  # initialize dictionary
    
    for column in attribute_columns: 
        if column in ('gender', 'samerace'): 
            min_val = 0
            max_val = 1
            num_vals = 2
        elif column in ('race', 'race_o'):
            min_val = 0
            max_val = 4
            num_vals = 5
        elif column == 'field':
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
    
    conditional_prob['priors'] = [all_yes_ratio, all_no_ratio]
        
    return (conditional_prob)


# In[68]:


def Prediction(data, model):
    all_columns = list(data.columns.values)
    attribute_columns = [col for col in all_columns if col not in ('decision')]

    pred_decision = []
    p_yes_list = []
    p_no_list = []
    
    count = 0
    for i in range (len(data.index)): #len(data.index) 
        p_yes = 1
        p_no = 1
        for column in attribute_columns: 
            value = data.loc[i, column]
            p_yes *= model[column + str(value)][0]
            p_no *= model[column + str(value)][1]
            
        p_yes = p_yes*model['priors'][0]
        p_no = p_no*model['priors'][1]

        if  p_yes >= p_no: 
            decision = 1
        else: 
            decision = 0
    
        if decision == data.loc[i, 'decision']:
            count += 1
    
    accuracy = count/len(data.index)

    return accuracy


# In[69]:


model = nbc(0.5)

training_set = pd.read_csv('./trainingSet.csv')
training_accuracy = Prediction(training_set, model)
print ('Training Accuracy: ', round(training_accuracy,2))

test_set = pd.read_csv('./testSet.csv')
test_accuracy = Prediction(test_set, model)
print ('Test Accuracy: ', round(test_accuracy,2))

