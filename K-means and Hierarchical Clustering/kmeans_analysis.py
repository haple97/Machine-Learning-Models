#!/usr/bin/env python
# coding: utf-8

# In[103]:


import pandas as pd
import numpy as np
import math
import random


# In[105]:


# seed_list = [0, 40, 100, 150, 410, 73, 1945, 3004, 2101, 6873]
seed = 6873
np.random.seed(seed)


# In[106]:


col_names_emb = ['id', 'label', 'feature_1', 'feature_2']
data1 = pd.read_csv('./cs573_hw5/digits-embedding.csv', names = col_names_emb, header = None)


# In[107]:


data2 = data1[data1['label'].isin([2, 4, 6, 7])]
data3 = data1[data1['label'].isin([6, 7])]


# In[108]:


data2.reset_index(inplace = True, drop = True)
data3.reset_index(inplace = True, drop = True)


# In[109]:


def get_cluster(x, centroids, K):
    dist_list = []
    for i in range(K):
        dist = np.linalg.norm(x - centroids[i])
        dist_list.append(dist)
    return dist_list.index(min(dist_list))  # return the index of the smallest distance


# In[110]:


def train_kmeans(all_coord_data, K, max_iter, tol):  
    # Initiate centroids 
    K_index = np.random.randint(0, len(all_coord_data), size=K)
    init_centroids = all_coord_data.iloc[K_index]
    init_coord_data = all_coord_data.drop(init_centroids.index)    
    
    # Turn all df into np array
    init_centroids = init_centroids.values   
    all_coord_data = all_coord_data.values
    init_coord_data = init_coord_data.values
    
    # Initiate values for stopping criteria
    cur_iter = 0
    eps_list = 1000 * np.ones(K)
    
    # Create dict to record centroids
    all_centroids = {}
    all_centroids[cur_iter] = init_centroids
    
    while cur_iter <= max_iter:  # and all(eps_list) >= tol
        if cur_iter == 0: 
            centroids = init_centroids
            coord_data = init_coord_data
        else: 
            centroids = centroids
            coord_data = all_coord_data
        
        prev_centroids = all_centroids[cur_iter] 
        
        # Find cluster labels for all data using current centroids 
        cluster_data = []
        for i in range(len(coord_data)):
            cluster_id = get_cluster(coord_data[i], centroids, K)
            cluster_data.append(cluster_id)
        cluster_data = np.array(cluster_data)
        
        # Get new centroids 
        for i in range(K):
            index = np.where(cluster_data == i)[0]
            subset = coord_data[index.tolist()]
            centroids[i] = np.average(subset, axis=0)
        
        # Get stopping criteria            
        cur_iter += 1
        all_centroids[cur_iter] = centroids    # save new centroids to dict
#         print(cur_iter)
        
    return cluster_data, all_centroids[max_iter]  # return assigned cluster labels using centroids of 50th iteration


# In[111]:


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


# In[112]:


def calc_wc_ssd(K_groups, centroids, K):
    ssd_all_k = 0
    for k in range(K):
        subset = K_groups[k] 
        ssd_same_k = 0
        for i in range(len(subset)):
            ssd_same_k += (np.linalg.norm(subset[i]-centroids[k]))**2  # sum of squares of differences for each K    
        ssd_all_k += ssd_same_k # sum of squares of differences for all values of K
    return ssd_all_k


# In[113]:


# Using list of clusters to split one cluster from the rest of the clusters, return them as 2 arrays
def split_clusters(i, S):
    cur_group = S[i]
    other_groups = S[:i] + S[(i+1):] 
    other_groups = np.vstack(other_groups)  # stack other arrays in list into one array
    return cur_group, other_groups


# In[114]:


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


# In[115]:


def calc_entropy(label):
    unique, counts = np.unique(label, return_counts=True)
    p_array = counts/len(label)
    entropy = -np.sum(p_array*np.log(p_array))
    label_dict = dict(zip(unique, p_array))
    return entropy, label_dict


# In[116]:


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


# In[117]:


def kmeans(data, K):    
    # Set stopping criteria
    max_iter = 50
    tol = 1e-7
    all_coord_data = data.iloc[:,2:]   # Only use coordinates data (df)
    true_label = data.iloc[:,1].values # np.array of true class labels
    
    # Train model
    pred_cluster, centroids = train_kmeans(all_coord_data, K, max_iter, tol)
    
    # Get clusters/groups of data as list of arrays, based on pred_cluster returned
    K_groups = get_data_groups(np.array(all_coord_data), pred_cluster, K)
    
    # Evaluate model
    wc_ssd = calc_wc_ssd(K_groups, centroids, K)
    sc = calc_sc(K_groups, K)
#     nmi = calc_nmi(pred_cluster, true_label)
    
    return wc_ssd, sc


# **Selection of K for 3 datasets**

# In[118]:


data_list = [data1, data2, data3]
K_list = [2, 4, 8, 16, 32]

results = {}
for i in range(len(data_list)):
    print('data:', i)
    ssd_list = []
    sc_list = []
    for K in K_list:
        print('K:', K)
        wc_ssd, sc = kmeans(data_list[i], K)
        ssd_list.append(wc_ssd)
        sc_list.append(sc)
    results['data'+str(i+1)] = {'wc_ssd': ssd_list, 'sc': sc_list}


# In[119]:


import matplotlib.pyplot as plt


# In[120]:


fig, axs = plt.subplots(3, 2, figsize = (10, 10))

for i in range(3):
    y_ssd = results['data'+str(i+1)]['wc_ssd']
    y_sc = results['data'+str(i+1)]['sc']
    
    axs[i, 0].plot(K_list, y_ssd)
    axs[i, 0].set_title('data'+str(i+1) + ' WC-SSD')
    axs[i, 0].set(xlabel='K', ylabel='WC-SSD')
    
    axs[i, 1].plot(K_list, y_sc)
    axs[i, 1].set_title('data'+str(i+1) + ' SC')
    axs[i, 1].set(xlabel='K', ylabel='SC')

fig.tight_layout(pad=3.0)
plt.show()


# In[121]:


results_df1 = pd.DataFrame.from_dict(results['data1'], orient='index')
results_df2 = pd.DataFrame.from_dict(results['data2'], orient='index')
results_df3 = pd.DataFrame.from_dict(results['data3'], orient='index')
results_df = pd.concat([results_df1, results_df2, results_df3], keys = ['data1', 'data2', 'data3'])


# In[122]:


results_df.to_csv('./results_rand'+str(seed)+'.csv')


# **Analysis of Sensitivity to Initial Centroids (Varied Seed)**
# 
# seed_list = [0, 40, 100, 150, 410, 73, 1945, 3004, 2101, 6873]

# In[123]:


def calc_stat(acc_list): 
    acc_array = np.array(acc_list)
    mean = np.mean(acc_array)
    std = np.std(acc_array, ddof = 1)
    se = std/math.sqrt(len(acc_list))
    return mean, se


# In[124]:


seed_list = [0, 40, 100, 150, 410, 73, 1945, 3004, 2101, 6873]
columns = ['dataset', 'score', 'K2', 'K4', 'K8', 'K16', 'K32']

all_results = []
for seed in seed_list: 
    results_seed = pd.read_csv('./results_rand'+str(seed)+'.csv', names = columns)
    all_results.append(results_seed)
all_results = pd.concat(all_results, keys = seed_list)


# In[133]:


summary = {}
for data in ['data1', 'data2', 'data3']:
    summary_score = {}
    for score in ['wc_ssd', 'sc']:
        subset = all_results[(all_results['dataset'] == data) & (all_results['score'] == score)]
        mean_list = []
        se_list = []
        for i in range(2,7):
            score_list = subset.iloc[:, i].tolist()
            mean, se = calc_stat(score_list)
            mean_list.append(mean)
            se_list.append(se)
        summary_score[score] = {'mean': mean_list, 'se': se_list}
    summary[data] = summary_score


# In[140]:


fig, axs = plt.subplots(3, 2, figsize = (10, 10))

for i in range(3):
    ssd_mean = summary['data'+str(i+1)]['wc_ssd']['mean']
    ssd_se = summary['data'+str(i+1)]['wc_ssd']['se']
    sc_mean = summary['data'+str(i+1)]['sc']['mean']
    sc_se = summary['data'+str(i+1)]['sc']['se']
    
    axs[i, 0].errorbar(K_list, ssd_mean, ssd_se, ecolor = 'r', capsize = 4)
    axs[i, 0].set_title('data'+str(i+1) + ' WC-SSD')
    axs[i, 0].set(xlabel='K', ylabel='WC-SSD')
    
    axs[i, 1].errorbar(K_list, sc_mean, sc_se, ecolor = 'r', capsize = 4)
    axs[i, 1].set_title('data'+str(i+1) + ' SC')
    axs[i, 1].set(xlabel='K', ylabel='SC')

fig.tight_layout(pad=3.0)
plt.show()


# **Re-run with chosen K, calculate NMI, and plot clusters**

# In[27]:


def kmeans_nmi(data, K):    
    # Set stopping criteria
    max_iter = 50
    tol = 1e-7
    all_coord_data = data.iloc[:,2:]   # Only use coordinates data (df)
    true_label = data.iloc[:,1].values # np.array of true class labels
    
    # Train model
    pred_cluster, centroids = train_kmeans(all_coord_data, K, max_iter, tol)

    nmi = calc_nmi(pred_cluster, true_label)
    
    return nmi, pred_cluster


# In[28]:


data_list = [data1, data2, data3]
K_list = [8, 4, 8]

nmi_list = []
pred_cluster_list = []
for (data, K) in zip(data_list, K_list):
    nmi, pred_cluster = kmeans_nmi(data, K)
    nmi_list.append(nmi)
    pred_cluster_list.append(pred_cluster)


# In[29]:


print('NMI:', np.round(nmi_list, 2))


# In[40]:


sample_index = np.random.randint(0, len(data3), size=1000)
sample = data3.iloc[sample_index]


# In[41]:


cluster = pred_cluster_list[2][sample_index]


# In[42]:


fig, ax = plt.subplots(figsize = (10,10))

# for digit in digits:
x = sample['feature_1']
y = sample['feature_2']
c = cluster

scatter = ax.scatter(x, y, c=c, cmap = 'Set3', edgecolors='none')

legend1 = ax.legend(*scatter.legend_elements(), loc="upper left", title="Cluster Label")

ax.add_artist(legend1)

plt.show()


# In[ ]:




