#!/usr/bin/env python
# coding: utf-8

# In[75]:


import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt 
import random


# In[76]:


pixel = ['pixel' + str(i) for i in range(1, 785)]
col_names = ['id', 'label'] + pixel

data = pd.read_csv('./cs573_hw5/digits-raw.csv', names = col_names, header = None)
data


# In[77]:


digits = list(set(data['label']))
np.random.seed(0)

img_list = {}
for i in digits: 
    cluster = data[data['label'] == i]                          # each digit as one cluster - 10 total (0 to 9)
    obs = cluster.sample(random_state = random.randint(0,50))   # one random observation
    data_obs = obs.values[0][2:]                                # get values of observation as numpy array
                                                                # [0] to get inside tuple, [2:] to omit id & label
    img = np.reshape(data_obs, (28,28))                         # reshape (784,) to (28,28)
    img_list[i] = img                                           # record to dict


# In[78]:


fig,ax = plt.subplots(2,5)
fig.set_size_inches(15,6)

for i in range(5):
    ax[0,i].imshow(img_list[i], cmap='gray')
    ax[1,i].imshow(img_list[i+5], cmap='gray')

fig.show()


# In[79]:


col_names_emb = ['id', 'label', 'feature_1', 'feature_2']
data_emb = pd.read_csv('./cs573_hw5/digits-embedding.csv', names = col_names_emb, header = None)
data_emb


# In[80]:


sample_index = np.random.randint(0, len(data_emb), size=1000)
sample = data_emb.iloc[sample_index]


# In[81]:


fig, ax = plt.subplots(figsize = (10,10))

# for digit in digits:
x = sample['feature_1']
y = sample['feature_2']
c = sample['label']

scatter = ax.scatter(x, y, c=c, cmap = 'Set3', edgecolors='none')

legend1 = ax.legend(*scatter.legend_elements(), loc="upper left", title="Class Label")

ax.add_artist(legend1)

plt.show()

