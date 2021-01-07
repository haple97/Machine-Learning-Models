#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[9]:


dating = pd.read_csv('./dating.csv')


# In[10]:


rating_of_partner_from_participant = ['attractive_partner', 'sincere_partner', 'intelligence_partner',
                                     'funny_partner', 'ambition_partner', 'shared_interests_partner']


# In[11]:


attribute_values = []
num_values = []
for column in rating_of_partner_from_participant: 
    values = dating[column].unique()        # Get list of unique values for each attribute column  
    values.sort()                           # Sort in increasing order
    attribute_values.append(values)         # Create list of list of unique values 
    num_values.append(len(values))


# In[12]:


attribute_values_percentages = []
for column in rating_of_partner_from_participant: 
    index_column = rating_of_partner_from_participant.index(column)
    each_attr_percentage = []
    for i in range(num_values[index_column]):  
    # For each unique value of each attribute column, 
        # filter for a dataframe of a single attribute value 
        participant_attr_val = dating[(dating[column] == attribute_values[index_column][i])]
        # filter for a dataframe with decision of 1 (i.e. yes to 2nd date) from the above dataframe
        participant_2date = participant_attr_val[participant_attr_val['decision'] == 1]
        # ratio of no. of items between the above 2 dataframes
        success_rate = participant_2date.shape[0]/participant_attr_val.shape[0]
        each_attr_percentage.append(success_rate)
    attribute_values_percentages.append(each_attr_percentage)


# In[13]:


for i in range(len(attribute_values)):
    x = attribute_values[i]
    y = attribute_values_percentages[i]

    fig, ax = plt.subplots()

    ax.set_ylabel('Success rate')
    ax.set_xlabel(rating_of_partner_from_participant[i] +' values')

    ax.set_title('Success rate by different values of ' + rating_of_partner_from_participant[i])

    plt.scatter(x, y, alpha=0.5)
    
plt.show()

