#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pandas as pd
import numpy as np
quotes_changed = 0
lowered_case = 0
onehot_encoding = {} # initiate dictionary to store column_name:encoded_label


# In[17]:


def perform_onehot_encoding(column, column_name):
    coded_column_df = pd.get_dummies(column, prefix = column_name).iloc[:,:-1]
    
    coded_unique_df = coded_column_df.drop_duplicates()    # Keep only unique rows
    
    coded_list = coded_unique_df.values.tolist()              # Create list from df rows
    coded_list.sort(reverse=True)
    
    column = column.astype('category')
    
    codes_for_column = {}
    for i, category in enumerate(column.cat.categories):
        codes_for_column[category] = coded_list[i]
    onehot_encoding[column.name] = codes_for_column
    return coded_column_df


# In[18]:


def remove_quotes(x):
    global quotes_changed
    if "'" in x:
        quotes_changed = quotes_changed + 1
        return x.replace("'", "")
    else:
        return x


# In[19]:


def to_lower(x):
    global lowered_case
    if x.islower():
        return x
    else:
        lowered_case = lowered_case + 1
        return x.lower()


# In[20]:


#Read the dataset
data = pd.read_csv('dating-full.csv', nrows = 6500)
decision_col = data['decision']    # Save this column to insert back at the end later on


# In[21]:


#Remove quotes
data['race'] = data['race'].apply(remove_quotes)
data['race_o'] = data['race_o'].apply(remove_quotes)
data['field'] = data['field'].apply(remove_quotes)


# In[22]:


#Convert to lowercase
data['field'] = data['field'].apply(to_lower)


# In[23]:


#Normalize preference scores of the participant
columns1  = ['attractive_important', 'sincere_important', 'intelligence_important','funny_important', 'ambition_important', 'shared_interests_important']
columns2  = ['pref_o_attractive', 'pref_o_sincere', 'pref_o_intelligence',
             'pref_o_funny', 'pref_o_ambitious', 'pref_o_shared_interests']
data[columns1] = data[columns1].div(data[columns1].sum(axis=1), axis=0)
data[columns2] = data[columns2].div(data[columns2].sum(axis=1), axis=0)


# In[24]:


gender_encoded = perform_onehot_encoding(data['gender'], 'gender')
race_encoded = perform_onehot_encoding(data['race'], 'race')
race_o_encoded = perform_onehot_encoding(data['race_o'], 'race_o')
field_encoded = perform_onehot_encoding(data['field'], 'field')

data_new = pd.concat([data, race_encoded, race_o_encoded, gender_encoded, field_encoded], axis=1)

print('Mapped vector for female in column gender:', onehot_encoding['gender']['female'])
print('Mapped vector for Black/African American in column race:', onehot_encoding['race']['Black/African American'])
print('Mapped vector for Other in column race_o:', onehot_encoding['race_o']['Other'])
print('Mapped vector for economics in column field:', onehot_encoding['field']['economics'])


# In[25]:


#Drop old categorical columns
data_new = data_new.drop(columns = ['gender', 'race', 'race_o', 'field'], axis = 1)


# In[26]:


#Move the target class to the end
data_new = data_new.drop(['decision'], axis = 1)
data_new['decision'] = decision_col


# In[27]:


#Save the csv file
data_new.to_csv('dating.csv', index = False)


# In[28]:


# Split data into training and test set
test_set = data_new.sample(frac=0.2, random_state = 25)
test_set.to_csv('./testSet.csv', index = False)

training_set = data_new.drop(test_set.index)
training_set.to_csv('./trainingSet.csv', index = False)


# In[ ]:




