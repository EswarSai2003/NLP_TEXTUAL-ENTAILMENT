#!/usr/bin/env python
# coding: utf-8
#This file contains all functions necessary for training,testing the data and for plotting graphs 
# In[1]:


# pip install transformers
# pip install sentencepiece
# !pip install cloud-tpu-client==0.10 https://storage.googleapis.com/tpu-pytorch/wheels/torch_xla-1.7-cp36-cp36m-linux_x86_64.whl


# In[2]:


import pandas as pd
import re
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader, SequentialSampler, RandomSampler
from torch.nn.utils.rnn import pad_sequence
from transformers import BertForSequenceClassification, AdamW
from transformers import BertTokenizer
import time
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[4]:


def loading_data(df):
    MAX_LEN = 512
    token_ids = []
    mask_ids = []
    seg_ids = []
    y = df['integer_label'].to_list()
    premise_list = df['sentence1'].to_list()
    hypothesis_list = df['sentence2'].to_list()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    for (premise, hypothesis) in zip(premise_list, hypothesis_list):
      premise_id = tokenizer.encode(premise, add_special_tokens = False)
      hypothesis_id = tokenizer.encode(hypothesis, add_special_tokens = False)
      pair_token_ids = [tokenizer.cls_token_id] + premise_id + [tokenizer.sep_token_id] + hypothesis_id + [tokenizer.sep_token_id]
      premise_len = len(premise_id)
      hypothesis_len = len(hypothesis_id)

      segment_ids = torch.tensor([0] * (premise_len + 2) + [1] * (hypothesis_len + 1))  # sentence 0 and sentence 1
      attention_mask_ids = torch.tensor([1] * (premise_len + hypothesis_len + 3))  # mask padded values

      token_ids.append(torch.tensor(pair_token_ids))
      seg_ids.append(segment_ids)
      mask_ids.append(attention_mask_ids)
    token_ids = pad_sequence(token_ids, batch_first=True)
    mask_ids = pad_sequence(mask_ids, batch_first=True)
    seg_ids = pad_sequence(seg_ids, batch_first=True)
    y = torch.tensor(y)
    dataset = TensorDataset(token_ids, mask_ids, seg_ids, y)
    return dataset


# In[5]:


def get_data_loaders(dataset,batch_size=16, shuffle=True):
    data_loader = DataLoader(
      dataset,
      shuffle=shuffle,
      batch_size=batch_size
    )

   
    return data_loader


# In[6]:


def multi_acc(y_pred, y_label):
  acc = (torch.log_softmax(y_pred, dim=1).argmax(dim=1) == y_label).sum().float() / float(y_label.size(0))
  return acc


# In[7]:


#model path is place where saved trained model gets stored and it gets updated for every epoch only val loss decreases over time to avoid overfitting
def training(model, train_loader, val_loader, optimizer, model_path):
    best_val_loss = float('inf')
    best_val_acc = 0
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    # check if a checkpoint file exists
    if os.path.isfile(model_path):
        print(f"Loading model checkpoint from {model_path}")
        checkpoint = torch.load(model_path,map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_val_loss = checkpoint['best_val_loss']
        best_val_acc = checkpoint['best_val_acc']
        train_losses = checkpoint['train_losses']
        val_losses = checkpoint['val_losses']
        train_accuracies = checkpoint['train_accuracies']
        val_accuracies = checkpoint['val_accuracies']
        epoch = checkpoint['epoch'] + 1
    else:
        epoch = 1
    print(f"Epoch {epoch} is going on now")
    
    start = time.time()
    model.train()
    total_train_loss = 0
    total_train_acc  = 0
    for batch_idx, (pair_token_ids, mask_ids, seg_ids, y) in enumerate(train_loader):
            optimizer.zero_grad()
            pair_token_ids = pair_token_ids.to(device)
            mask_ids = mask_ids.to(device)
            seg_ids = seg_ids.to(device)
            labels = y.to(device)

            loss, prediction = model(pair_token_ids, 
                                     token_type_ids=seg_ids, 
                                     attention_mask=mask_ids, 
                                     labels=labels).values()

            acc = multi_acc(prediction, labels)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            total_train_acc  += acc.item()

    train_acc  = total_train_acc/len(train_loader)
    train_accuracies.append(train_acc)
    train_loss = total_train_loss/len(train_loader)
    train_losses.append(train_loss)

    # validation with development dataset to decide to save the model or not
    model.eval()
    total_val_loss = 0
    total_val_acc = 0
    with torch.no_grad():
     for batch_idx, (pair_token_ids, mask_ids, seg_ids, y) in enumerate(val_loader):
        pair_token_ids = pair_token_ids.to(device)
        mask_ids = mask_ids.to(device)
        seg_ids = seg_ids.to(device)
        labels = y.to(device)

        loss, prediction = model(pair_token_ids, 
                                token_type_ids=seg_ids, 
                                attention_mask=mask_ids, 
                                labels=labels).values()

        acc = multi_acc(prediction, labels)

        total_val_loss += loss.item()
        total_val_acc += acc.item()

    val_loss = total_val_loss/len(val_loader)
    val_losses.append(val_loss)
    val_acc = total_val_acc/len(val_loader)
    val_accuracies.append(val_acc)

    if val_loss < best_val_loss:
            print(f"Validation loss decreased from {best_val_loss:.4f} to {val_loss:.4f} in {epoch} epoch . Saving the model...")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': val_loss,
                'best_val_acc': val_acc,
                'train_losses': train_losses,
                'val_losses' : val_losses,
                'train_accuracies' : train_accuracies,
                'val_accuracies': val_accuracies},model_path)
            best_val_loss = val_loss
            best_val_acc = val_acc
    print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Time: {time.time()-start:.2f}s")

  
       



# In[8]:

def Loss_plots(train_losses,val_losses):

    # create a figure and axes object
    fig, ax = plt.subplots()

    # plot the two lists as dots connected by lines on the same plot with different colors and labels
    ax.plot(range(1, len(train_losses)+1), train_losses, color='blue', label='Training_Loss', marker='o')
    ax.plot(range(1, len(val_losses)+1), val_losses, color='red', label='Validation_Loss', marker='o')

    # set the axis labels and legend
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()

    # show the plot
    plt.show()


# In[9]:


def Accuracy_plots(train_acc,val_acc):

    # create a figure and axes object
    fig, ax = plt.subplots()

    # plot the two lists as dots connected by lines on the same plot with different colors and labels
    ax.plot(range(1, len(train_acc)+1), train_acc, color='blue', label='Training_Accuracy', marker='o')
    ax.plot(range(1, len(val_acc)+1), val_acc, color='red', label='Validation_Accuracy', marker='o')

    # set the axis labels and legend
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend()

    # show the plot
    plt.show()


def testing(model_path, test_loader):
    # Load the model on CPU device
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
 # Replace with your actual model class
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    test_loss = 0
    test_acc = 0
    with torch.no_grad():
        for batch_idx, (pair_token_ids, mask_ids, seg_ids, y) in enumerate(test_loader):
            pair_token_ids = pair_token_ids.to(device)
            mask_ids = mask_ids.to(device)
            seg_ids = seg_ids.to(device)
            labels = y.to(device)

            loss, prediction = model(pair_token_ids, 
                                     token_type_ids=seg_ids, 
                                     attention_mask=mask_ids, 
                                     labels=labels).values()

            acc = multi_acc(prediction, labels)

            test_loss += loss.item()
            test_acc += acc.item()

    test_loss /= len(test_loader)
    test_acc /= len(test_loader)

    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")








