#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


dating_binned = pd.read_csv('./dating-binned.csv')


# In[3]:


test_set = dating_binned.sample(frac=0.2, random_state = 47)
test_set.to_csv('./testSet.csv', index = False)


# In[4]:


training_set = dating_binned.drop(test_set.index)
training_set.to_csv('./trainingSet.csv', index = False)

