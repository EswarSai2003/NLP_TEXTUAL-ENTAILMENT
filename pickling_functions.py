#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle


# In[2]:


def dump_pickle(data,file_path):
  #dumping dataframes to pickle files to extract them easily
  with open(file_path + '.' + 'pkl', 'wb') as f:
    pickle.dump(data, f)
    
    


# In[3]:


def load_pickle(file_path):

# loading preprocessed data from pickle file
 with open(file_path + '.'+ 'pkl', 'rb') as f:
    return pickle.load(f)
    

