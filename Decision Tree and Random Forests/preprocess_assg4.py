#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
quotes_changed = 0
lowered_case = 0
label_encoding = {}


# In[2]:


def perform_label_encoding(column):
    column = column.astype('category')
    codes_for_column = {}
    for i, category in enumerate(column.cat.categories):
        codes_for_column[category] = i
    label_encoding[column.name] = codes_for_column
    return column.cat.codes


# In[3]:


# Read first 6500 rows dataset and drop 'race', 'race_o', 'field'
data = pd.read_csv('./dating-full.csv', nrows = 6500)
decision_col = data['decision']    # Save this column to insert back at the end later on
data = data.drop(columns = ['race', 'race_o', 'field'], axis = 1)


# In[4]:


# Label encoding for gender column
data[['gender']] = data[['gender']].apply(perform_label_encoding)

# Drop old categorical column (gender)
data = data.drop(columns = ['gender'], axis = 1)


# In[5]:


# Normalize preference scores of the participant
columns1  = ['attractive_important', 'sincere_important', 'intelligence_important',
             'funny_important', 'ambition_important', 'shared_interests_important']
columns2  = ['pref_o_attractive', 'pref_o_sincere', 'pref_o_intelligence',
             'pref_o_funny', 'pref_o_ambitious', 'pref_o_shared_interests']
data[columns1] = data[columns1].div(data[columns1].sum(axis=1), axis=0)
data[columns2] = data[columns2].div(data[columns2].sum(axis=1), axis=0)


# In[6]:


# Move the target class to the end
data = data.drop(['decision'], axis = 1)
data['decision'] = decision_col


# In[7]:


# Discretize to binary values
all_columns = list(data.columns.values)
continuous_valued_columns = [col for col in all_columns 
                             if col not in ('gender', 'samerace', 'decision')]

for column in continuous_valued_columns: 
    data[column] = pd.cut(data[column], bins = 2, labels = [0, 1], include_lowest = True)


# In[8]:


# Split data into training and test set
test_set = data.sample(frac=0.2, random_state = 47)
test_set.reset_index(inplace=True, drop = True)
test_set.to_csv('./testSet.csv', index = False)

training_set = data.drop(test_set.index)
training_set.reset_index(inplace=True, drop = True)
training_set.to_csv('./trainingSet.csv', index = False)

print('Training and test sets split and saved as csv files')

