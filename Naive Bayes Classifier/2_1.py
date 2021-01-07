#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


dating = pd.read_csv('./dating.csv')


# In[3]:


## (i) 

# (a) Divide into subsets by gender
dating_male = dating[dating['gender'] == 1]
dating_female = dating[dating['gender'] == 0]


# In[4]:


preference_scores_of_participant = ['attractive_important', 'sincere_important', 'intelligence_important',
                                    'funny_important', 'ambition_important', 'shared_interests_important']
preference_scores_of_partner = ['pref_o_attractive', 'pref_o_sincere', 'pref_o_intelligence', 'pref_o_funny',
                                'pref_o_ambitious', 'pref_o_shared_interests']


# In[5]:


# (b) Mean values for each column of each subset
means_male = []
for column in preference_scores_of_participant: 
    mean_column_male = dating_male[column].mean()
    means_male.append(mean_column_male)


# In[6]:


means_female = []
for column in preference_scores_of_participant: 
    mean_column_female = dating_female[column].mean()
    means_female.append(mean_column_female)


# In[7]:


# (c) Plot for comparison

ind = np.arange(len(means_male))  # the x locations for the groups
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind - width/2, means_male, width, label='Male')
rects2 = ax.bar(ind + width/2, means_female, width, label='Female')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Importance')
ax.set_title('Importance of attributes by gender')
ax.set_xticks(ind)
ax.set_xticklabels(('attractive', 'sincere', 'intelligence',
                    'funny', 'ambition', 'shared_interests'), rotation = 45)
ax.legend()

fig.tight_layout()

plt.show()


# In[8]:


# Observations: To male, attractiveness (apperance) plays an important role in considering a partner, 
# while for female, sincerity, intelligence, ambitiousness, and shared interests (internal factors) 
# are slightly more important. 
# To both genders, how funny the partner is is equally important. 

