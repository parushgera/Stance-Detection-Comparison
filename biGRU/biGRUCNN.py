#!/usr/bin/env python
# coding: utf-8

# In[14]:



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
#nltk.download('stopwords')
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords, twitter_samples 
#from utils import *
#from parameters import *
from nltk.stem import PorterStemmer
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dropout,Concatenate,Dense, Embedding, LSTM, SpatialDropout1D, Flatten, GRU, Bidirectional, Conv1D, Input,MaxPooling1D
from sklearn.model_selection import train_test_split
from tensorflow.keras import Model
from sklearn.model_selection import StratifiedKFold
stemmer = PorterStemmer()
tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
stopwords_english = stopwords.words('english')


#train_data_file = '/home/parush/stance/Experiments/stance_mohammed/train.txt'
#test_data_file = '/home/parush/stance/Experiments/stance_mohammed/test.txt'
#TARGETS = [ 'Atheism','Climate Change is a Real Concern', 'Feminist Movement','Hillary Clinton', 'Legalization of Abortion' ]


train_data_file = '/home/parush/stance/Experiments/SomasundaranWiebe-politicalDebates/train.txt'
test_data_file = '/home/parush/stance/Experiments/SomasundaranWiebe-politicalDebates/test.txt'
TARGETS = ['god','healthcare','guns','gayRights','abortion', 'creation']


# train_data_file = '/home/parush/stance/Experiments/Data_MPCHI/train.txt'
# test_data_file = '/home/parush/stance/Experiments/Data_MPCHI/test.txt'
# TARGETS = ['Are E-Cigarettes safe?','Does MMR Vaccine lead to autism in children?',
#       'Does Sunlight exposure lead to skin cancer?','Does Vitamin C prevent common cold?',
#       'Should women take HRT post-menopause?']




def process_tweet(tweet):
    '''
    Input: 
        tweet: a string containing a tweet
    Output:
        tweets_clean: a list of words containing the processed tweet
    
    '''
    # remove stock market tickers like $GE
    tweet = re.sub(r'\$\w*', '', tweet)
    # remove old style retweet text "RT"
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    # remove hyperlinks
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
    # remove hashtags
    # only removing the hash # sign from the word
    tweet = re.sub(r'#', '', tweet)
    # tokenize tweets
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)
    ### START CODE HERE ###
    tweets_clean = []
    for word in tweet_tokens:
        if (word not in stopwords_english and # remove stopwords
            word not in string.punctuation): # remove punctuation
            #tweets_clean.append(word)
            stem_word = stemmer.stem(word) # stemming word
            tweets_clean.append(stem_word)
    ### END CODE HERE ###
    return tweets_clean



def train_and_test(t):
    
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
                tweet = process_tweet(tweet)
                if len(tweet) > sentence_maxlen:
                    sentence_maxlen = len(tweet)
                all_favor_tweets.append(tweet)
            elif line[0].strip() != 'ID' and line[3].strip() == 'AGAINST' and line[1] == t:
                tweet = line[2]
                tweet = process_tweet(tweet)
                if len(tweet) > sentence_maxlen:
                    sentence_maxlen = len(tweet)
                all_against_tweets.append(tweet)
            
    x_train = all_favor_tweets + all_against_tweets
    y_train = np.append(np.ones(len(all_favor_tweets)), np.zeros(len(all_against_tweets))) 
    
    
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
                tweet = process_tweet(tweet)
                if len(tweet) > sentence_maxlen:
                    sentence_maxlen = len(tweet)
                all_favor_tweets_test.append(tweet)
            elif line[0].strip() != 'ID' and line[3].strip() == 'AGAINST' and line[1] == t:
                tweet = line[2]
                tweet = process_tweet(tweet)
                if len(tweet) > sentence_maxlen:
                    sentence_maxlen = len(tweet)
                all_against_tweets_test.append(tweet)

    x_test = all_favor_tweets_test + all_against_tweets_test
    y_test = np.append(np.ones(len(all_favor_tweets_test)), np.zeros(len(all_against_tweets_test)))
    
    return x_train, y_train, x_test, y_test, sentence_maxlen
                
def tweet_to_tensor(processed_tweet, vocab_dict, unk_token="__UNK__"):
    tensor = []
    unk_ID = vocab_dict[unk_token]
    for word in processed_tweet:
        word_ID = vocab_dict[word] if word in vocab_dict else unk_ID
        tensor.append(word_ID)
    return tensor
           
                
def load_embeddings(embedding,dim):
    if embedding == 'twitter':
        path = '/home/parush/stance/Experiments/embeddings/twitter/glove.twitter.27B.'+str(dim)+'d.txt'
    else:
        path ='/home/parush/stance/Experiments/embeddings/wikipedia/glove.6B.'+str(dim)+'d.txt'
    word_embeddings = {}
    with open(path, 'r') as f:
        for each_emb in f:
            emb = each_emb.split(' ')
            word_embeddings[emb[0]] = np.asarray(emb[1:], dtype='float32')
    return word_embeddings
               
            
def get_embeddings(embedding,dim):
    if embedding == 'twitter':
        embedding_matrix_twitter = np.zeros((vocab_size, dim))
        word_embeddings_twitter = load_embeddings(embedding, dim)
        print(embedding_matrix_twitter[0])
        for each_word,index in Vocab.items():
            if each_word in word_embeddings_twitter:
                embedding_matrix_twitter[index] = word_embeddings_twitter[each_word]
        return embedding_matrix_twitter
    else:
        embedding_matrix_wikipedia = np.zeros((vocab_size, dim))
        word_embeddings_wikipedia = load_embeddings(embedding, dim)
        for each_word,index in Vocab.items():
            if each_word in word_embeddings_wikipedia:
                embedding_matrix_wikipedia[index] = word_embeddings_wikipedia[each_word]
        return embedding_matrix_wikipedia
        


def biLSTM():
    model = Sequential()
    model.add(Embedding(embeddings_weights.shape[0], embeddings_weights.shape[1], weights=[embeddings_weights]))
    model.add(Dropout(0.2))
    model.add(LSTM(64,return_sequences=True,dropout=0.3))
    model.add(Bidirectional(LSTM(64,dropout=0.3)))
    #model.add(Flatten())
    #add a dropout here
    model.add(Dropout(0.5))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(loss = 'binary_crossentropy', optimizer='adam',metrics = ['accuracy'])
    return model

def biLSTMCNN():
    inputs = Input(shape=(sentence_maxlen,))
    embedded_inputs = Embedding(embeddings_weights.shape[0], embeddings_weights.shape[1], weights=[embeddings_weights])(inputs)
    embedded_inputs = Dropout(0.2)(embedded_inputs)
    lstm = Bidirectional(LSTM(64,return_sequences=True,dropout=0.3))(embedded_inputs)
    convs = []
    for each_filter_size in [3,4,5]:
        #print(rnn.shape)
        each_conv = Conv1D(100, each_filter_size, activation='relu')(lstm)
        each_conv = MaxPooling1D(sentence_maxlen-each_filter_size+1)(each_conv)
        each_conv = Flatten()(each_conv)
        #print(each_conv.shape)
        convs.append(each_conv)
        
    output = Concatenate()(convs)
    output = Dropout(0.5)(output)
    output = (Dense(1,activation='sigmoid'))(output)
    model = Model(inputs=inputs, outputs=output)
    model.compile(loss = 'binary_crossentropy', optimizer='adam',metrics = ['accuracy']) 
    return model

def biGRU():
    model = Sequential()
    model.add(Embedding(embeddings_weights.shape[0], embeddings_weights.shape[1], weights=[embeddings_weights]))
    model.add(Dropout(0.2))
    model.add(Bidirectional(GRU(64,return_sequences=True,dropout=0.3)))
    model.add(Bidirectional(GRU(64,dropout=0.3)))
    model.add(Dropout(0.5))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(loss = 'binary_crossentropy', optimizer='adam',metrics = ['accuracy'])
    return model

def biGRUCNN():
    inputs = Input(shape=(sentence_maxlen,))
    embedded_inputs = Embedding(embeddings_weights.shape[0], embeddings_weights.shape[1], weights=[embeddings_weights])(inputs)
    embedded_inputs = Dropout(0.2)(embedded_inputs)
    rnn = Bidirectional(GRU(64,return_sequences=True,dropout=0.3))(embedded_inputs)
    convs = []
    for each_filter_size in [3,4,5]:
        #print(rnn.shape)
        each_conv = Conv1D(100, each_filter_size, activation='relu')(rnn)
        each_conv = MaxPooling1D(sentence_maxlen-each_filter_size+1)(each_conv)
        each_conv = Flatten()(each_conv)
        #print(each_conv.shape)
        convs.append(each_conv)
        
    output = Concatenate()(convs)
    output = Dropout(0.5)(output)
    output = (Dense(1,activation='sigmoid'))(output)
    model = Model(inputs=inputs, outputs=output)
    model.compile(loss = 'binary_crossentropy', optimizer='adam',metrics = ['accuracy'])    
    
    return model
    
        


for target in TARGETS:
    x_train, y_train, x_test, y_test, sentence_maxlen = train_and_test(target)
    Vocab = {'__PAD__': 0, '__</e>__': 1, '__UNK__': 2} 
    for processed_tweet in x_train: 
        for word in processed_tweet:
            if word not in Vocab: 
                Vocab[word] = len(Vocab)
    print("Total words in vocab are",target,len(Vocab))
    vocab_size = len(Vocab)
    embeddings_weights = get_embeddings('wikipedia',100) 
    for i in range(len(x_train)):
        tweet_tensor = tweet_to_tensor(x_train[i], Vocab)
        if len(tweet_tensor) < sentence_maxlen:
            diff = sentence_maxlen - len(tweet_tensor)
            n_pad = [0]*diff
            tweet_tensor = tweet_tensor + n_pad
        x_train[i] = tweet_tensor
    for i in range(len(x_test)):
        tweet_tensor = tweet_to_tensor(x_test[i], Vocab)
        if len(tweet_tensor) < sentence_maxlen:
            diff = sentence_maxlen - len(tweet_tensor)
            n_pad = [0]*diff
            tweet_tensor = tweet_tensor + n_pad
        x_test[i] = tweet_tensor





    x_train = np.array(x_train)

    x_test = np.array(x_test)

    y_train = np.asarray(y_train).astype('float32').reshape((-1,1))
    y_test = np.asarray(y_test).astype('float32').reshape((-1,1))





    print("Sentence maxlength {}, vocab size : {}, in target: {}".format(sentence_maxlen,vocab_size,target))





    model = biGRUCNN()
    print(model.summary())


    # In[21]:


    x_train.shape
    seed = 7


    # In[22]:


    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    cvscores = []

    for train, val in kfold.split(x_train, y_train):
        
        
        model.fit(x_train[train], y_train[train], epochs = 50, batch_size = 16, verbose=1)
        scores = model.evaluate(x_train[val], y_train[val], verbose=0)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        cvscores.append(scores[1] * 100)
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

    y_pred = np.round(model.predict(x_test))

    print("For: ",target,classification_report(y_test, y_pred, digits=4))
# model.save('/home/parush/stance/Experiments/saved_model/debates/biGRU')
# model.save('/home/parush/stance/Experiments/saved_model/debates/biGRU.h5')

