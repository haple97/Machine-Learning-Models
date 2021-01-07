#!/usr/bin/env python
# coding: utf-8

# In[15]:


import pandas as pd
import numpy as np
import math
import random
import scipy
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
from scipy import cluster


# In[2]:


np.random.seed(0)


# In[3]:


col_names_emb = ['id', 'label', 'feature_1', 'feature_2']
data_emb = pd.read_csv('./cs573_hw5/digits-embedding.csv', names = col_names_emb, header = None)


# In[63]:


# Get dataset of 100 images
data = pd.DataFrame(columns = col_names_emb)
for i in range(10):
    subset = data_emb[data_emb['label'] == i]
    points_index = np.random.randint(0, len(subset), size=10)
    points = subset.iloc[points_index]
    data = data.append(points)
    
data.reset_index(inplace = True, drop = True)


# **Dendrograms with different methods**

# In[12]:


Z = linkage(np.array(data.iloc[:,2:]), 'single')
fig = plt.figure(figsize=(25, 10))
dn = dendrogram(Z)
plt.show()


# In[13]:


Z2 = linkage(np.array(data.iloc[:,2:]), 'complete')
fig = plt.figure(figsize=(25, 10))
dn = dendrogram(Z2)
plt.show()


# In[14]:


Z3 = linkage(np.array(data.iloc[:,2:]), 'average')
fig = plt.figure(figsize=(25, 10))
dn = dendrogram(Z3)
plt.show()


# **Calculate WC-SSD and SC for various K (by cutting tree) and 3 above trees**

# In[68]:


def get_centroids(K, cluster_data, data):
    centroids = []
    for i in range(K):
        index = np.where(cluster_data == i)[0]
#         print(index)
        subset = data.iloc[index.tolist(), :]
        centroids.append(np.average(subset, axis=0))
    return np.array(centroids)


# In[69]:


def get_data_groups(data, pred_cluster, K):
    '''
    Parameters: 
    data: data without cluster label, type: array
    pred_cluster: predicted cluster labels for all observations in data, type: array
    '''
    K_groups = []
    for k in range(K):
        index = np.where(pred_cluster == k)[0]  # array
        subset = data[index.tolist()]           # array
        K_groups.append(subset)                 # list of arrays
    return K_groups


# In[70]:


def calc_wc_ssd(K_groups, centroids, K):
    ssd_all_k = 0
    for k in range(K):
        subset = K_groups[k] 
        ssd_same_k = 0
        for i in range(len(subset)):
            ssd_same_k += (np.linalg.norm(subset[i]-centroids[k]))**2  # sum of squares of differences for each K    
        ssd_all_k += ssd_same_k # sum of squares of differences for all values of K
    return ssd_all_k


# In[74]:


# Using list of clusters to split one cluster from the rest of the clusters, return them as 2 arrays
def split_clusters(i, S):
    cur_group = S[i]
    other_groups = S[:i] + S[(i+1):] 
    other_groups = np.vstack(other_groups)  # stack other arrays in list into one array
    return cur_group, other_groups

def calc_sc(K_groups, K):
    
    Si_all_k = 0
    for k in range(K):
        cur_group, other_groups = split_clusters(k, K_groups)  # 2 np arrays
        len_wc = len(cur_group)
        len_oc = len(other_groups)
        
        Si_same_k = 0
        for i in range(len_wc):
            
            # Distance to points within group
            x = cur_group[i]*np.ones((len_wc-1,2))    # [x1, x1, x1, x1...x1] - array of length (len_wc-1)
            others = np.delete(cur_group, i, axis=0)  # the rest of the cluster [x2, x3, x4....xn] (same length)
            norm_array = np.linalg.norm(x-others, axis=1)  # array of norms [|x1-x2|, |x1-x3|..., |x1-xn|]
            A = np.sum(norm_array.tolist())/(len_wc-1) # within-group average distance
            
            # Distance to points of other groups
            x2 = cur_group[i]*np.ones((len_oc,2))
            norm_array2 = np.linalg.norm(x2-other_groups, axis = 1)
            B = np.sum(norm_array2.tolist())/len_oc
            
            # Add up Si for all points in the same cluster
            Si_same_k += (B-A)/max(A,B)
        
        # Add up Si for all points in all clusters (whole dataset)
        Si_all_k += Si_same_k
    
    # SC is average of Si values in whole dataset 
    SC = Si_all_k/(len_wc + len_oc)
    
    return SC


# In[72]:


def calc_entropy(label):
    unique, counts = np.unique(label, return_counts=True)
    p_array = counts/len(label)
    entropy = -np.sum(p_array*np.log(p_array))
    label_dict = dict(zip(unique, p_array))
    return entropy, label_dict

def calc_nmi(pred_cluster, true_label):
    # Calc H(C) and H(G)
    entropy_pred, pred_dict = calc_entropy(pred_cluster)
    entropy_true, true_dict = calc_entropy(true_label)
    
    # Calc I(C,G)
    unique_true = np.unique(true_label)
    unique_pred = np.unique(pred_cluster)
    df = pd.DataFrame(np.vstack([pred_cluster, true_label]).T, columns = ['predict', 'true'])
    
    I_cg = 0
    for c in unique_true:
        I_g = 0
        for g in unique_pred:
            subset_cg = df[(df['true'] == c) & (df['predict'] == g)]
            p_cg = len(subset_cg)/len(df)
            if p_cg != 0:
                I_g += p_cg*np.log(p_cg/(pred_dict[g]*true_dict[c]))
        I_cg += I_g
            
    NMI = I_cg/(entropy_pred + entropy_true)
    return NMI


# In[77]:


K_list = [2, 4, 8, 16, 32]
method_list = ['single', 'complete', 'average']
Z_list = [Z, Z2, Z3]

coord_data = data.iloc[:,2:]

results = {}
i = 0
for Z in Z_list: 
    ssd_list = []
    sc_list = []
    
    for K in K_list: 
        pred_cluster = cluster.hierarchy.cut_tree(Z, n_clusters=K)
        pred_cluster = pred_cluster.flatten()
        centroids = get_centroids(K, pred_cluster, coord_data)
        # Get clusters/groups of data as list of arrays, based on pred_cluster returned
        K_groups = get_data_groups(np.array(coord_data), pred_cluster, K)
        wc_ssd = calc_wc_ssd(K_groups, centroids, K)
        sc = calc_sc(K_groups, K)
        
        ssd_list.append(wc_ssd)
        sc_list.append(sc)
    
    results[method_list[i]] = {'wc_ssd': ssd_list, 'sc': sc_list}
    i += 1
    


# In[78]:


fig, axs = plt.subplots(3, 2, figsize = (10, 10))

for i in range(3):
    y_ssd = results[method_list[i]]['wc_ssd']
    y_sc = results[method_list[i]]['sc']
    
    axs[i, 0].plot(K_list, y_ssd)
    axs[i, 0].set_title(method_list[i] + ' WC-SSD')
    axs[i, 0].set(xlabel='K', ylabel='WC-SSD')
    
    axs[i, 1].plot(K_list, y_sc)
    axs[i, 1].set_title(method_list[i] + ' SC')
    axs[i, 1].set(xlabel='K', ylabel='SC')

fig.tight_layout(pad=3.0)
plt.show()


# **NMI for final selection - K = 8**

# In[79]:


pred_cluster = cluster.hierarchy.cut_tree(Z2, n_clusters=K)
pred_cluster = pred_cluster.flatten()
true_label = data.iloc[:,1].values
nmi = calc_nmi(pred_cluster, true_label)
print('NMI:', round(nmi, 2))


# In[ ]:




