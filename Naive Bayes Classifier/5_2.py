#!/usr/bin/env python
# coding: utf-8

# In[17]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[18]:


'''Discretize continuous values into bins'''

def Discretize (b): 
    dating = pd.read_csv('./dating.csv')
    all_columns = list(dating.columns.values)
    continuous_valued_columns = [col for col in all_columns 
                                 if col not in ('gender', 'race', 'race_o', 'samerace', 'field', 'decision')]

    preference_scores_of_participant = ['attractive_important', 'sincere_important', 'intelligence_important',
                                        'funny_important', 'ambition_important', 'shared_interests_important']
    preference_scores_of_partner = ['pref_o_attractive', 'pref_o_sincere', 'pref_o_intelligence', 'pref_o_funny',
                                    'pref_o_ambitious', 'pref_o_shared_interests']

    # Assign bin based on value of item; replace old column with new column of binned values

    # Provide min and max values of each column to divide bins
    for column in continuous_valued_columns:
        if (column in preference_scores_of_participant) or (column in preference_scores_of_partner): 
            min_val = 0    # normalized columns 
            max_val = 1
        elif column in ('age', 'age_o'): 
            min_val = 18
            max_val = 58
        elif column == 'interests_correlate':
            min_val = -1
            max_val = 1
        else: 
            min_val = 0
            max_val = 10

        bin_num = b
        bins = np.arange(min_val, max_val + 0.000000000000001, (max_val-min_val)/bin_num)
    
        # Wrap outliers inside the defined range (min, max)
        dating.loc[dating[column] < min_val, column] = min_val
        dating.loc[dating[column] > max_val, column] = max_val

        # Bin, including the right value (a <= x < b)
        dating[column] = np.digitize(dating[column], bins)

        # Assign x = b as belonging to the last bin 
        dating.loc[dating[column] > bin_num, column] = bin_num

        # Count number of entries for each bin
        num_items_by_bin = []
        for i in range(bin_num):
            count = len(dating[dating[column] == i+1].index)
            num_items_by_bin.append(count)

    #     print(column+':', num_items_by_bin)

    # dating.to_csv('./dating-binned.csv', index = False)
    return dating


# In[19]:


'''Split'''

def Split(b):
    dating_binned = pd.read_csv('./dating-binned-'+str(b)+'.csv')
#     dating_binned = dating

    test_set = dating_binned.sample(frac=0.2, random_state = 47)
    test_set.to_csv('./testSet-'+str(b)+'.csv', index = False)

    training_set = dating_binned.drop(test_set.index)
    training_set.to_csv('./trainingSet-'+str(b)+'.csv', index = False)


# In[20]:


'''Train model'''

def nbc(t_frac, b):
    training_set = pd.read_csv('./trainingSet-'+str(b)+'.csv')
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
            max_val = b
            num_vals = b
        
        for i in range(min_val, max_val+1, 1):
            key = column + str(i)   # key to access dictionary is an attribute and one of its value
            
            attribute_yes = data[(data[column] == i) & (data['decision'] == 1)]
            yes_prob = (len(attribute_yes.index)+1)/(len(all_yes.index)+num_vals) # Laplace smoothing on con. prob
            attribute_no = data[(data[column] == i) & (data['decision'] == 0)]
            no_prob = (len(attribute_no.index)+1)/(len(all_no.index)+num_vals)
            
            conditional_prob[key] = [yes_prob, no_prob]  # assign pair of values to key
    
    conditional_prob['priors'] = [all_yes_ratio, all_no_ratio]
        
    return (conditional_prob)


# In[21]:


'''Predict decision with model & Return accuracy of model on a dataset'''

def Prediction(data, model):
    all_columns = list(data.columns.values)
    attribute_columns = [col for col in all_columns if col not in ('decision')]

    pred_decision = []
    p_yes_list = []
    p_no_list = []
    
    count = 0
    for i in range (len(data.index)): 
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


# In[22]:


B = [2, 5, 10, 50, 100, 200]
training_B = []
test_B = []
    
for b in B: 
    # Discretize
    data = Discretize(b)
    data.to_csv('./dating-binned-'+str(b)+'.csv', index = False)
    
    # Split train-test
    Split(b)
    
    # Obtain model
    model = nbc(1, b)
    
    # Obtain dataset to test model on
    training_set = pd.read_csv('./trainingSet-'+str(b)+'.csv')
    test_set = pd.read_csv('./testSet-'+str(b)+'.csv')
    
    print('Bin size: ', b)
    
    # Predict model on training set
    training_accuracy = Prediction(training_set, model)
    print ('Training Accuracy: ', round(training_accuracy,2))
    training_B.append(training_accuracy)

    # Predict model on test set
    test_accuracy = Prediction(test_set, model)
    print ('Test Accuracy: ', round(test_accuracy,2))
    test_B.append(test_accuracy)

plt.plot(B, training_B, label = "Training Accuracy")
plt.plot(B, test_B, label = "Test Accuracy")
plt.xlabel('Number of Bins')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

