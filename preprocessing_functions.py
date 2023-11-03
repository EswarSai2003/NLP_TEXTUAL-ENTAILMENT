#!/usr/bin/env python
# coding: utf-8

# In[1]:


#IMPORTING LIBRARIES ESSENTIAL FOR PRE PROCESSING DATA:
#Importing libraries essential for sentence cleaning in each of the datasets:
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
#using slearn library to encode the gold_labels to integers as a part of preprocessing for further analysis
import sklearn
from sklearn.preprocessing import LabelEncoder





# In[2]:


#function to select only useful columns in train,dev,test data frames
def useful_columns(dataframe,useful_attributes):
    result = dataframe[useful_attributes]
    return result


# In[3]:


#function to drop rows containing null values:
def remove_null(dataframe):
    result = dataframe.dropna()
    return result


# In[4]:


#function to drop rows where gold label is not given:
def remove_goldlabel_notassigned(dataframe,column_name):
    return dataframe[dataframe[column_name] != '-']


# In[5]:


#function to clean the pair of sentences[premise and hypothesis]:
def clean_text(sentence):
  sentence = sentence.lower()#converting all letters to lower case
  word_tokens = word_tokenize(sentence)#tokenising(splitting) the sentence to words in vocabulary
  stopwords_english = stopwords.words('english')#list of stop words in english
  clean_words = []
  for word in word_tokens:
    if (word not in stopwords_english) and (word not in string.punctuation):
      clean_words.append(word)#appending clean words to list
  lemmatized_words = []
  lemma = WordNetLemmatizer()#instantiating class to perform lemmatisation
  for item in clean_words:
    lemmatized_word = lemma.lemmatize(item)
    lemmatized_words.append(lemmatized_word)#lemmatizing all words and appending them to a list
  processed_text = ' '.join(lemmatized_words)#joining all lemmatized words 
  return processed_text


# In[6]:


#function to remove rows with sentences having empty spaces 
def remove_empty_sentence(dataframe):
    dataframe = dataframe[dataframe['sentence1'].str.strip().astype(bool)]
    dataframe = dataframe[dataframe['sentence2'].str.strip().astype(bool)]
    return dataframe



# In[7]:


#assigning integer values to each of the three gold labels(encoding gold labels) using below function:
def goldlabel_encoding(dataframe,column_name):
    label_encoder = LabelEncoder()
    result = label_encoder.fit_transform(dataframe[column_name])
    return result


# In[8]:

#using above functions to define a function to preprocess entire dataframe at once
def pre_process(dataset) :
    dataset = useful_columns(dataset,['sentence1','sentence2','gold_label'])
    dataset = remove_goldlabel_notassigned(dataset,'gold_label')
    dataset = remove_null(dataset)
    dataset['sentence1'] = dataset['sentence1'].apply(clean_text)
    dataset['sentence2'] = dataset['sentence2'].apply(clean_text) 
    dataset = remove_empty_sentence(dataset)
    dataset['integer_label'] = goldlabel_encoding(dataset,'gold_label')
    return dataset 

