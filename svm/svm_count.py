#!/usr/bin/env python
# coding: utf-8




import pandas as pd
import re
import string
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
#nltk.download('stopwords')
#nltk.download('punkt')
from nltk.util import ngrams
#from google.colab import drive
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from scipy import sparse
import os
import pickle


# In[ ]:





# In[8]:


# Loading Data
df = pd.read_csv('/home/parush/stance/Experiments/SomasundaranWiebe-politicalDebates/train.txt', sep='\t')
df1 = pd.read_csv('/home/parush/stance/Experiments/SomasundaranWiebe-politicalDebates/test.txt', sep='\t')



#Use below lines only when training different classifiers for different targerts and testing on their corrosponding data.
t = ['god','healthcare','guns','gayRights','abortion', 'creation']
# df = df[(df["Target"]==t[4]) ]
# df1 = df1[(df1["Target"]==t[4]) ]

print("The length of train data is {}".format(len(df)))
print("The length of test data is {}".format(len(df1)))


# In[18]:


df


# In[283]:


vectorizer = 'count'   # set 'count' or 'tfidf'
analyzer = 'word'  # set 'word' or 'both' ( word and char)


# In[284]:


if vectorizer == 'count':
    if analyzer == 'word':
        vectorizer = CountVectorizer(analyzer='word',ngram_range=(1,1))
    else:
        vectorizer = CountVectorizer(analyzer='word',ngram_range=(1,3))
        char_vectorizer = CountVectorizer(analyzer='char',ngram_range=(2,5))
else:
    if analyzer == 'word':
        vectorizer = TfidfVectorizer(analyzer='word',ngram_range=(1,1))
    else:
        vectorizer = TfidfVectorizer(analyzer='word',ngram_range=(1,3))
        char_vectorizer = TfidfVectorizer(analyzer='char',ngram_range=(2,5))
        
        
        
        


# In[ ]:





# In[285]:


#List of FAVOR Tweets
def get_training_data_and_labels(df):
    df_train_favor = df.loc[df['Stance'] == 'FAVOR']
    df_train_favor = df_train_favor.reset_index(drop=True)
    train_favor_tweets = df_train_favor['Tweet'].tolist()
    
    # List of AGAINST Tweets
    df_train_against = df.loc[df['Stance'] == 'AGAINST']
    df_train_against = df_train_against.reset_index(drop=True)
    train_against_tweets = df_train_against['Tweet'].tolist()
    
    #Favor + Against Tweets and Labels
    train_corpus = train_favor_tweets + train_against_tweets
    train_labels = np.append(np.ones((len(train_favor_tweets))) , np.zeros((len(train_against_tweets))))
    
    
    if analyzer == 'word':
        ngram_vectorized_data = vectorizer.fit_transform(train_corpus)
        
        return ngram_vectorized_data, train_labels
    else:
        ngram_vectorized_data = vectorizer.fit_transform(train_corpus)
        char_vectorized_data = char_vectorizer.fit_transform(train_corpus)
        l = np.hstack((ngram_vectorized_data.toarray(), char_vectorized_data.toarray()))
        train_vectorized_data = sparse.csr_matrix(l)
        
        return train_vectorized_data, train_labels 


# In[ ]:





# In[286]:


#preparing test_data
def get_test_data_and_labels(df1):
    df_test_favor = df1.loc[df1['Stance']=='FAVOR']
    df_test_favor = df_test_favor.reset_index(drop=True)
    test_favor_tweets = df_test_favor['Tweet'].tolist()
    print(len(test_favor_tweets))
    
    
    df_test_against = df1.loc[df1['Stance'] == 'AGAINST']
    df_test_against = df_test_against.reset_index(drop=True)
    test_against_tweets = df_test_against['Tweet'].tolist()
    print(len(test_against_tweets))
    
    
    test_corpus = test_favor_tweets + test_against_tweets
    test_labels = np.append(np.ones((len(test_favor_tweets))) , np.zeros((len(test_against_tweets))))
    
    if analyzer == 'word':
        test_ngram_vectorized_data = vectorizer.transform(test_corpus)
        
        return test_ngram_vectorized_data, test_labels
    else:
        test_ngram_vectorized_data = vectorizer.transform(test_corpus)
        test_char_vectorized_data = char_vectorizer.transform(test_corpus)
        l2 = np.hstack((test_ngram_vectorized_data.toarray(), test_char_vectorized_data.toarray()))
        test_vectorized_data = sparse.csr_matrix(l2)
        
        return test_vectorized_data,test_labels
    
    
    
    


# In[ ]:





# In[287]:


X_train, y_train =  get_training_data_and_labels(df)
X_test, y_test = get_test_data_and_labels(df1)


# In[288]:


# Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]}, {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
scores = ['precision', 'recall']


for score in scores:
    
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(
        SVC(), tuned_parameters, scoring='%s_macro' % score
    )
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()


# In[ ]:

# save the model to disk
filename = 'count_word.sav'
pickle.dump(clf, open(filename, 'wb'))



# In[273]:


# df_all = pd.read_csv('test.csv')
# X_test, y_test = get_test_data_and_labels(df_all)
# y_true, y_pred = y_test, clf.predict(X_test)
# print('Report for ', classification_report(y_true, y_pred))


# In[289]:


# t = ['Atheism', 'Climate Change is a Real Concern', 'Feminist Movement', 'Hillary Clinton', 'Legalization of Abortion']

# for target in t:
#     df_test = df1[(df1["Target"]== target) ]
#     X_test, y_test = get_test_data_and_labels(df_test)
#     y_true, y_pred = y_test, clf.predict(X_test)
#     print('Report for ',target, classification_report(y_true, y_pred))


# In[ ]:





# In[ ]:


#print(len(test_count_vectorizer.get_feature_names()))

#test_indexed_data = hstack((np.array(range(0,test_vectorized_data.shape[0]))[:,None], test_vectorized_data)) #adding a column for index and stacking data 3614 X 100285
#test_indexed_data.shape

#X_train, X_test, y_train, y_test = train_test_split(indexed_data, labels, test_size=0.4, random_state=0, shuffle = True)
#X_train,, y_train = indexed_data , labels
#data_train_index = X_train[:,0]
#print(X_test.shape)
#print(data_train_index)
#X_train = X_train[:,1:]
#data_test_index = X_test[:,0]
#print(data_test_index)
#X_test = X_test[:,1:]


# In[ ]:





# In[ ]:





# In[ ]:




