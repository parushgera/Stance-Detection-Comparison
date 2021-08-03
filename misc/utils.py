

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

def train_and_test():
    
    sentence_maxlen = 0
    x_train = []
    y_train = []
    all_favor_tweets = []
    all_against_tweets = []
    
    with open(train_data_file, 'r') as trainfile:
        for line in trainfile:
            
            line = line.replace('#SemST', '').strip()
            line = line.split('\t')
            
            if line[0].strip() != 'ID' and line[3].strip() == 'FAVOR':
                tweet = line[2]
                tweet = process_tweet(tweet)
                if len(tweet) > sentence_maxlen:
                    sentence_maxlen = len(tweet)
                all_favor_tweets.append(tweet)
            elif line[0].strip() != 'ID' and line[3].strip() == 'AGAINST':
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
        

            if line[0] != 'ID' and line[3] == 'FAVOR':
                tweet = line[2]
                tweet = process_tweet(tweet)
                if len(tweet) > sentence_maxlen:
                    sentence_maxlen = len(tweet)
                all_favor_tweets_test.append(tweet)
            elif line[0].strip() != 'ID' and line[3].strip() == 'AGAINST':
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
        
                        