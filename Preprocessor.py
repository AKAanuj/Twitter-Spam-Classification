#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd
import numpy as np
import nltk
from sklearn.base import BaseEstimator,TransformerMixin
from nltk import PorterStemmer,WordNetLemmatizer,TweetTokenizer,pos_tag,FreqDist
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as sw
import string
import re
import sklearn
import random


# In[12]:


class DataPreprocessor(BaseEstimator,TransformerMixin):
    def __init__(self,EMAILREGEX="[a-zA-Z][a-zA-Z0-9]*@[a-zA-Z]+(.com)",WEBREGEX="(http://|https://)?(www\.|wap\.)?([a-zA-Z ]+)\.([a-zA-Z]+)(/.*)?",PHONEREGEX="(\+)?([0-9]{2})?[0-9]{10,}",MONEYREGEX="(\$)([0-9]*([\.][0-9]+)?)",number_of_features=0):
        self.MONEYREGEX=MONEYREGEX
        self.PHONEREGEX=PHONEREGEX
        self.EMAILREGEX=EMAILREGEX
        self.WEBREGEX=WEBREGEX
        self.STOPWORDS=list(sw.union(stopwords.words('english')))
        self.number_of_features=number_of_features
        
    def fit(self,X,y=None,**fit_params):
        #print("fit called")
        return self
    
    def transform(self,X,y=None,**fit_params):
        #print("transform called")
        self.EMAILREP_="EMAIL_ADDRESS_HERE"
        self.WEBREP_="WEBSITE_ADDRESS_HERE"
        self.PHONEREP_="PHONE_NUMBER_HERE"
        self.MONEYREP_="MONEY_HERE"
        self.exclude_=(self.EMAILREP_,self.WEBREP_,self.PHONEREP_,self.MONEYREP_)
        
        Xcolumns=X.columns
        X=X.dropna(how="any")
        X=X.values.squeeze()
        
        tokenizer=TweetTokenizer()
        lemmatizer=WordNetLemmatizer()
        stemmer=PorterStemmer()

        for i in range(len(X)):
            print("{} out of {}".format(i,len(X)))
            X[i]=re.sub("(?P<chars>\w)(?P=chars){2,}","\g<chars>",X[i])#remove repeating characters
            X[i]=re.sub(self.EMAILREGEX,self.EMAILREP_,X[i])
            X[i]=re.sub(self.WEBREGEX,self.WEBREP_,X[i])
            X[i]=re.sub(self.MONEYREGEX,self.MONEYREP_,X[i])
            X[i]=re.sub(self.PHONEREGEX,self.PHONEREP_,X[i])
            X[i]=re.sub("\s+"," ",X[i])#replace whitespace spaces with one space
            X[i]=re.sub("^\s+|\s+$","",X[i])#strip leading and trailing white spaces
            X[i]=tokenizer.tokenize(X[i])
            X[i]=[i for i in X[i] if i not in list(string.punctuation)]#remove punctuations
            #X[i]=[i for i in X[i] if len(i)==1 and i in ["a","i","I","A"]]
            X[i]=[lemmatizer.lemmatize(x) if x not in self.exclude_ and dict(pos_tag(X[i]))[x]!="NNP" else x for x in X[i]] 
            X[i]=[stemmer.stem(x) if x not in self.exclude_ and dict(pos_tag(X[i]))[x]!="NNP" else x for x in X[i]]
            X[i]=[x for x in X[i] if x not in self.STOPWORDS]
        
        self.word_features_=[]
        for i in X:
            self.word_features_.extend(i)
        self.word_features_=list(dict(FreqDist(self.word_features_).most_common(self.number_of_features)).keys())
        self.output_=pd.DataFrame(np.zeros([len(X),self.number_of_features]))
        self.output_.columns=self.word_features_
              
        for i in range(X.shape[0]):
            for j in X[i]:
                if j in self.word_features_:
                    self.output_.iloc[i,self.word_features_.index(j)]+=1
        return self.output_
        


# In[ ]:




