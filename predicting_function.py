#!/usr/bin/env python
# coding: utf-8

# # Defining function to predict gold label if sentence and premise are given

# In[1]:


# !pip install transformers
# !pip install sentencepiece


# In[2]:


import torch
from transformers import BertForSequenceClassification
from transformers import BertTokenizer


# In[3]:


import preprocessing_functions as pf


# In[4]:


#defining function to predict gold_label when premise and hypothesis are given along with model_path where it is stored
def predict(model_path, premise, hypothesis):
    
    premise = pf.clean_text(premise)
    hypothesis = pf.clean_text(hypothesis)
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    premise_id = tokenizer.encode(premise, add_special_tokens=False)
    hypothesis_id = tokenizer.encode(hypothesis, add_special_tokens=False)
    pair_token_ids = [tokenizer.cls_token_id] + premise_id + [tokenizer.sep_token_id] + hypothesis_id + [tokenizer.sep_token_id]
    segment_ids = torch.tensor([0] * (len(premise_id) + 2) + [1] * (len(hypothesis_id) + 1)).unsqueeze(0)  # sentence 0 and sentence 1
    attention_mask_ids = torch.tensor([1] * len(pair_token_ids)).unsqueeze(0)  # mask padded values

    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    with torch.no_grad():
        logits = model(torch.tensor(pair_token_ids).unsqueeze(0), token_type_ids=segment_ids, attention_mask=attention_mask_ids)[0]
        prediction = torch.argmax(logits).item()
    class_names = ['contradiction','Entailment','Neutral']
    output = class_names[prediction]
    return output


# In[5]:


#Taking pair of sentences and predicting the relationship between them
predict('/Users/vkadava/Desktop/PROJECT3/saved_model.pth',"A woman with a green headscarf, blue shirt and a very big grin.","The woman is young.") #gives 'Neutral' as output
        
        
        

