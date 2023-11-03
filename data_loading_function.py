#!/usr/bin/env python
# coding: utf-8

# In[ ]:
#importing Essential libraries for  loading the data
import pandas as pd

#function to load the data from file if file path is given
def load_data(base_path,file_name,file_type):
        file_path = base_path + file_name + '.' + file_type
        if file_type == 'jsonl':
            dataframe =  pd.read_json(file_path,lines = True)
        elif file_type == 'csv':
            dataframe =  pd.read_csv(file_path)
        elif file_type == 'xlsx':
            dataframe =  pd.read_excel(file_path)
        return dataframe

