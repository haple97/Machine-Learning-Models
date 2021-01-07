#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


dating_full = pd.read_csv('./dating-full.csv')
dating_full


# In[3]:


## (i)
# Strip quotes and save as new columns
dating_full['new_race'] = dating_full['race'].str.strip('\'')
dating_full['new_race_o'] = dating_full['race_o'].str.strip('\'')
dating_full['new_field'] = dating_full['field'].str.strip('\'')

# Compare every pair of columns to count number of cells changed
C1 = np.where((dating_full['new_race'] == dating_full['race']), 'Same', 'Different')
diff_race = C1.tolist().count('Different')

C2 = np.where((dating_full['new_race_o'] == dating_full['race_o']), 'Same', 'Different')
diff_race_o = C2.tolist().count('Different')

C3 = np.where ((dating_full['new_field'] == dating_full['field']), 'Same', 'Different')
diff_field = C3.tolist().count('Different')

cells_changed = diff_race + diff_race_o + diff_field
print ('Quotes removed from ' + str(cells_changed) + ' cells.')


# In[4]:


## (ii)
# Lowercase field column 
dating_full['low_field'] = dating_full['new_field'].str.lower()

# Compare old and new column to count number of cells changed 
C4 = np.where((dating_full['new_field'] == dating_full['low_field']), 'Same', 'Different')
diff_lfield = C4.tolist().count('Different')
print('Standardized ' + str(diff_lfield) + ' cells to lower case')


# In[5]:


# Remove old columns and change new columns' names
dating_full = dating_full.drop(columns = ['race', 'race_o', 'field', 'new_field'])
dating_full = dating_full.rename(columns = {'new_race':'race', 'new_race_o':'race_o', 'low_field':'field'})


# In[6]:


## (iii)
# Return unique values of each column into an array and sort alphabetically
gender_val = dating_full['gender'].unique()
gender_val.sort()
print('Value assigned for male in column gender: ', gender_val.tolist().index('male'))

# Encode to a new column using index number of a gender type in the unique list above
dating_full['gender'] = dating_full.apply(lambda x: gender_val.tolist().index(x.gender), axis = 1)


# In[7]:


race_val = dating_full['race'].unique()
race_val.sort()
print('Value assigned for European/Caucasian-American in column race: ', 
      race_val.tolist().index('European/Caucasian-American'))
dating_full['race'] = dating_full.apply(lambda x: race_val.tolist().index(x.race), axis = 1)


# In[8]:


race_o_val = dating_full['race_o'].unique()
race_o_val.sort()
print('Value assigned for Latino/Hispanic American in column race o: ', 
      race_o_val.tolist().index('Latino/Hispanic American'))
dating_full['race_o'] = dating_full.apply(lambda x: race_o_val.tolist().index(x.race_o), axis = 1)


# In[9]:


field_val = dating_full['field'].unique()
field_val.sort()
print('Value assigned for law in column field: ', field_val.tolist().index('law'))
dating_full['field'] = dating_full.apply(lambda x: field_val.tolist().index(x.field), axis = 1)


# In[10]:


## (iv)
# Normalize for 6 columns of preference_scores_of_participant
dating_full['total'] = dating_full.apply(lambda x: x.attractive_important + x.sincere_important +
                                         x.intelligence_important + x.funny_important + 
                                         x.ambition_important + x.shared_interests_important, axis = 1)
dating_full['attractive_important'] = dating_full.apply(lambda x: x.attractive_important/x.total, axis = 1)
dating_full['sincere_important'] = dating_full.apply(lambda x: x.sincere_important/x.total, axis = 1)
dating_full['intelligence_important'] = dating_full.apply(lambda x: x.intelligence_important/x.total, axis = 1)
dating_full['funny_important'] = dating_full.apply(lambda x: x.funny_important/x.total, axis = 1)
dating_full['ambition_important'] = dating_full.apply(lambda x: x.ambition_important/x.total, axis = 1)
dating_full['shared_interests_important'] = dating_full.apply(lambda x: x.shared_interests_important/x.total, axis = 1)


# In[11]:


# Normalize for 6 columns of preference_scores_of_partner
dating_full['total_partner'] = dating_full.apply(lambda x: x.pref_o_attractive + x.pref_o_sincere +
                                         x.pref_o_intelligence + x.pref_o_funny + 
                                         x.pref_o_ambitious + x.pref_o_shared_interests, axis = 1)
dating_full['pref_o_attractive'] = dating_full.apply(lambda x: x.pref_o_attractive/x.total_partner, axis = 1)
dating_full['pref_o_sincere'] = dating_full.apply(lambda x: x.pref_o_sincere/x.total_partner, axis = 1)
dating_full['pref_o_intelligence'] = dating_full.apply(lambda x: x.pref_o_intelligence/x.total_partner, axis = 1)
dating_full['pref_o_funny'] = dating_full.apply(lambda x: x.pref_o_funny/x.total_partner, axis = 1)
dating_full['pref_o_ambitious'] = dating_full.apply(lambda x: x.pref_o_ambitious/x.total_partner, axis = 1)
dating_full['pref_o_shared_interests'] = dating_full.apply(lambda x: x.pref_o_shared_interests/x.total_partner, axis = 1)


# In[12]:


# Print mean values of each column

print('Mean of attractive_important: ', str(round(dating_full['attractive_important'].mean(), 2)))
print('Mean of sincere_important: ', str(round(dating_full['sincere_important'].mean(), 2)))
print('Mean of intelligence_important: ', str(round(dating_full['intelligence_important'].mean(), 2)))
print('Mean of funny_important: ', str(round(dating_full['funny_important'].mean(), 2)))
print('Mean of ambition_important: ', str(round(dating_full['ambition_important'].mean(), 2)))
print('Mean of shared_interests_important: ', str(round(dating_full['shared_interests_important'].mean(), 2)))
print('Mean of pref_o_attractive: ', str(round(dating_full['pref_o_attractive'].mean(), 2)))
print('Mean of pref_o_sincere: ', str(round(dating_full['pref_o_sincere'].mean(), 2)))
print('Mean of pref_o_intelligence: ', str(round(dating_full['pref_o_intelligence'].mean(), 2)))
print('Mean of pref_o_funny: ', str(round(dating_full['pref_o_funny'].mean(), 2)))
print('Mean of pref_o_ambitious: ', str(round(dating_full['pref_o_ambitious'].mean(), 2)))
print('Mean of pref_o_shared_interests: ', str(round(dating_full['pref_o_shared_interests'].mean(), 2)))


# In[13]:


# Remove unnecessary columns
dating_full = dating_full.drop(columns = ['total', 'total_partner'])


# In[14]:


# Save to new file 
dating_full.to_csv('./dating.csv', index = False)

