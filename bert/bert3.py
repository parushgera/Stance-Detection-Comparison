#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[267]:


import string
import re
import os
import nltk
import pandas as pd
import numpy as np
import json
import tensorflow as tf
from tensorflow import keras

SEED = 1013
np.random.seed(SEED)

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras import Model
from sklearn.model_selection import StratifiedKFold


# In[268]:


import pandas as pd 
import numpy as np 
import torch.nn as nn
from pytorch_pretrained_bert import BertTokenizer, BertModel
import torch
from torchnlp.datasets import imdb_dataset
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report


# In[269]:


def train_and_test(t): # feed raw text in BERT so we will not preprocess
    
    sentence_maxlen = 0
    x_train = []
    y_train = []
    all_favor_tweets = []
    all_against_tweets = []
    
    with open(train_data_file, 'r') as trainfile:
        for line in trainfile:
            
            line = line.replace('#SemST', '').strip()
            line = line.split('\t')
            
            if line[0].strip() != 'ID' and line[3].strip() == 'FAVOR' and line[1] == t:
                tweet = line[2]
                #tweet = process_tweet(tweet)
                if len(tweet) > sentence_maxlen:
                    sentence_maxlen = len(tweet)
                all_favor_tweets.append(tweet)
            elif line[0].strip() != 'ID' and line[3].strip() == 'AGAINST' and line[1] == t:
                tweet = line[2]
                #tweet = process_tweet(tweet)
                if len(tweet) > sentence_maxlen:
                    sentence_maxlen = len(tweet)
                all_against_tweets.append(tweet)
            
    x_train = all_favor_tweets + all_against_tweets
    y_train = np.append(np.ones(len(all_favor_tweets)), np.zeros(len(all_against_tweets))) 
    
    pp = len(x_train)
    x_test = []
    y_test = []
    all_favor_tweets_test = []
    all_against_tweets_test = []
    with open(test_data_file, 'r') as testfile:
        for line in testfile:
            line = line.replace('#SemST', '').strip()
            line = line.split('\t')
        

            if line[0] != 'ID' and line[3] == 'FAVOR' and line[1] == t:
                tweet = line[2]
                #tweet = process_tweet(tweet)
                if len(tweet) > sentence_maxlen:
                    sentence_maxlen = len(tweet)
                all_favor_tweets_test.append(tweet)
            elif line[0].strip() != 'ID' and line[3].strip() == 'AGAINST' and line[1] == t:
                tweet = line[2]
                #tweet = process_tweet(tweet)
                if len(tweet) > sentence_maxlen:
                    sentence_maxlen = len(tweet)
                all_against_tweets_test.append(tweet)

    x_test = all_favor_tweets_test + all_against_tweets_test
    y_test = np.append(np.ones(len(all_favor_tweets_test)), np.zeros(len(all_against_tweets_test)))

    
    return x_train, y_train, x_test, y_test, sentence_maxlen, pp


# In[270]:


train_data_file = '/home/parush/stance/Experiments/stance_mohammed/train.txt'
test_data_file = '/home/parush/stance/Experiments/stance_mohammed/test.txt'
TARGETS = [ 'Atheism','Climate Change is a Real Concern', 'Feminist Movement','Hillary Clinton', 'Legalization of Abortion' ]

df = pd.read_csv(train_data_file, sep='\t')
df1 = pd.read_csv(test_data_file, sep='\t')
t = TARGETS[3]


# In[271]:


#train_data_file = '/home/parush/stance/Experiments/SomasundaranWiebe-politicalDebates/train.txt'
#test_data_file = '/home/parush/stance/Experiments/SomasundaranWiebe-politicalDebates/test.txt'
#TARGETS = ['god','healthcare','guns','gayRights','abortion', 'creation']
#df = pd.read_csv(train_data_file, sep='\t')

#df1 = pd.read_csv(test_data_file, sep='\t')
#t = TARGETS[3]

# In[272]:


# train_data_file = '/home/parush/stance/Experiments/Data_MPCHI/train.txt'
# test_data_file = '/home/parush/stance/Experiments/Data_MPCHI/test.txt'
# TARGETS = ['Are E-Cigarettes safe?','Does MMR Vaccine lead to autism in children?',
#       'Does Sunlight exposure lead to skin cancer?','Does Vitamin C prevent common cold?',
#       'Should women take HRT post-menopause?']
# df = pd.read_csv(train_data_file, sep='\t')
# df1 = pd.read_csv(test_data_file, sep='\t')
# 


# In[273]:


train_texts, train_labels, test_texts, test_labels, sen_maxlen, pp = train_and_test(t)


# In[274]:


# Bert Classifier
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
train_tokens = list(map(lambda t: ['[CLS]'] + tokenizer.tokenize(t)[:511], train_texts)) #cannot be more than 512
test_tokens = list(map(lambda t: ['[CLS]'] + tokenizer.tokenize(t)[:511], test_texts))


# In[275]:


train_tokens_ids = list(map(tokenizer.convert_tokens_to_ids, train_tokens))
test_tokens_ids = list(map(tokenizer.convert_tokens_to_ids, test_tokens))

train_tokens_ids = pad_sequences(train_tokens_ids, maxlen=128, truncating="post", padding="post", dtype="int")
test_tokens_ids = pad_sequences(test_tokens_ids, maxlen=128, truncating="post", padding="post", dtype="int")

train_y = np.array(train_labels) == 1
test_y = np.array(test_labels) == 1


# In[276]:


class BertBinaryClassifier(nn.Module):
    def __init__(self, dropout=0.1):
        super(BertBinaryClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, tokens, masks=None):
        _, pooled_output = self.bert(tokens, attention_mask=masks, output_all_encoded_layers=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        proba = self.sigmoid(linear_output)
        return proba


# In[277]:


train_masks = [[float(i > 0) for i in ii] for ii in train_tokens_ids]
test_masks = [[float(i > 0) for i in ii] for ii in test_tokens_ids]
train_masks_tensor = torch.tensor(train_masks)
test_masks_tensor = torch.tensor(test_masks)

train_tokens_tensor = torch.tensor(train_tokens_ids)
train_y_tensor = torch.tensor(train_y.reshape(-1, 1)).float()
test_tokens_tensor = torch.tensor(test_tokens_ids)
test_y_tensor = torch.tensor(test_y.reshape(-1, 1)).float()


# In[278]:


BATCH_SIZE = 32 # you can change it
EPOCHS = 25 # do not train for too many epochs as a large model like this will surely overfit and overflow the ram and disk
learning_rate = 2e-5 # you can decrease it more like 3e-5 but do not increase it more than 1e-5


# In[279]:


'''
This is how you read data into Pytorch

'''
train_dataset =  torch.utils.data.TensorDataset(train_tokens_tensor, train_masks_tensor, train_y_tensor)
train_sampler =  torch.utils.data.RandomSampler(train_dataset)
train_dataloader =  torch.utils.data.DataLoader(train_dataset, sampler=train_sampler, batch_size=BATCH_SIZE)

test_dataset =  torch.utils.data.TensorDataset(test_tokens_tensor, test_masks_tensor, test_y_tensor)
test_sampler =  torch.utils.data.SequentialSampler(test_dataset)
test_dataloader =  torch.utils.data.DataLoader(test_dataset, sampler=test_sampler, batch_size=BATCH_SIZE)


# In[280]:


bert_clf = BertBinaryClassifier()
optimizer = torch.optim.Adam(bert_clf.parameters(), lr=learning_rate)


# In[281]:


for epoch_num in range(EPOCHS):
    bert_clf.train()
    train_loss = 0
    for step_num, batch_data in enumerate(train_dataloader):
        token_ids, masks, labels = tuple(t for t in batch_data)
        probas = bert_clf(token_ids, masks)
        loss_func = nn.BCELoss()
        batch_loss = loss_func(probas, labels)
        train_loss += batch_loss.item()
        bert_clf.zero_grad()
        batch_loss.backward()
        optimizer.step()
        print('Epoch: ', epoch_num + 1)
        print("\r" + "{0}/{1} loss: {2} ".format(step_num, len(train_texts) / BATCH_SIZE, train_loss / (step_num + 1)))


# In[282]:


# Evaluation
bert_clf.eval()
bert_predicted = []
all_logits = []
with torch.no_grad():
    for step_num, batch_data in enumerate(test_dataloader):
      
        token_ids, masks, labels = tuple(t for t in batch_data)
        logits = bert_clf(token_ids, masks)
        loss_func = nn.BCELoss()
        loss = loss_func(logits, labels)
        numpy_logits = logits.cpu().detach().numpy()

        bert_predicted += list(numpy_logits[:, 0] > 0.5)
        all_logits += list(numpy_logits[:, 0])


# In[283]:


print("For: ",t,classification_report(test_y, bert_predicted,digits=4))


# In[226]:


# from sklearn.metrics import roc_curve
# import sklearn
# from sklearn.metrics import f1_score

# fpr, tpr, threshold = roc_curve(y_test, y_predict)
# roc_auc = sklearn.metrics.auc(_fpr, _tpr)
# f1_score = f1_score(y_test, y_predict)
# print('AUROC: {}'.format(roc_auc), file=open("output.txt","a"))
# print('f1-score: {}'.format(f1_score), file=open("output.txt","a"))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




