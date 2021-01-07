#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


dating = pd.read_csv('./dating.csv')


# In[3]:


all_columns = list(dating.columns.values)


# In[4]:


continuous_valued_columns = [col for col in all_columns 
                             if col not in ('gender', 'race', 'race_o', 'samerace', 'field', 'decision')]


# In[5]:


preference_scores_of_participant = ['attractive_important', 'sincere_important', 'intelligence_important',
                                    'funny_important', 'ambition_important', 'shared_interests_important']
preference_scores_of_partner = ['pref_o_attractive', 'pref_o_sincere', 'pref_o_intelligence', 'pref_o_funny',
                                'pref_o_ambitious', 'pref_o_shared_interests']


# In[6]:


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
        
    bin_num = 5
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
    
    print(column+':', num_items_by_bin)


# In[7]:


dating.to_csv('./dating-binned.csv', index = False)

