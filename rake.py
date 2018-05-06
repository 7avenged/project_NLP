# Adapted from: github.com/aneesha/RAKE/rake.py

from __future__ import division
import operator
import nltk
import string

#import sys
#reload(sys)  
#sys.setdefaultencoding('utf8')
#str = unicode(str, errors='replace')

import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
import numpy as np
import re
from urllib import urlopen
import string
from bs4 import BeautifulSoup
import string
import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from nltk.stem.wordnet import WordNetLemmatizer
import nltk
from nltk.corpus import stopwords
from nltk import wordpunct_tokenize
import sys
from nltk import wordpunct_tokenize
from nltk.corpus import stopwords
from sklearn.decomposition import NMF, LatentDirichletAllocation
#from gensim import corpora

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from progressbar import ProgressBar
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer


n_samples = 2000
n_features = 1000
n_topics = 10
n_top_words = 20

#df = pd.read_csv("sampledata.csv", )
#df["prediction"] = df["prediction"].astype(str)  #convert column to string s
#X=df["Information"]

df = pd.read_csv("sampledata.csv", )
df["Information"] = df["Information"].astype(str)  #convert column to string s
X=df["Information"]
X1 = X

##X1 = X[10]
#print(X).head()
#print(type(X)) 
#print(type(X1))
#for i in len(range(X)):
 # print(type(X[i]) )


def isPunct(word):
  return len(word) == 1 and word in string.punctuation

def isNumeric(word):
  try:
    float(word) if '.' in word else int(word)
    return True
  except ValueError:
    return False

class RakeKeywordExtractor:

  def __init__(self):
    self.stopwords = set(nltk.corpus.stopwords.words())
    self.top_fraction = 1 # consider top third candidate keywords by score

  def _generate_candidate_keywords(self, sentences):
    phrase_list = []
    for sentence in sentences:
      words = map(lambda x: "|" if x in self.stopwords else x,
        nltk.word_tokenize(sentence.lower()))
      phrase = []
      for word in words:
        if word == "|" or isPunct(word):
          if len(phrase) > 0:
            phrase_list.append(phrase)
            phrase = []
        else:
          phrase.append(word)
    return phrase_list

  def _calculate_word_scores(self, phrase_list):
    word_freq = nltk.FreqDist()
    word_degree = nltk.FreqDist()
    for phrase in phrase_list:
      degree = len(filter(lambda x: not isNumeric(x), phrase)) - 1
      for word in phrase:
        ##word_freq.inc(word)
        word_freq[word] += 1
        #word_degree.inc(word, degree) # other words
        word_degree[word]+= 1# other words
    for word in word_freq.keys():
      word_degree[word] = word_degree[word] + word_freq[word] # itself
    # word score = deg(w) / freq(w)
    word_scores = {}
    for word in word_freq.keys():
      word_scores[word] = word_degree[word] / word_freq[word]
    return word_scores

  def _calculate_phrase_scores(self, phrase_list, word_scores):
    phrase_scores = {}
    for phrase in phrase_list:
      phrase_score = 0
      for word in phrase:
        phrase_score += word_scores[word]
      phrase_scores[" ".join(phrase)] = phrase_score
    return phrase_scores
    
  def extract(self, text, incl_scores=False):
    sentences = nltk.sent_tokenize(text)
    phrase_list = self._generate_candidate_keywords(sentences)
    word_scores = self._calculate_word_scores(phrase_list)
    phrase_scores = self._calculate_phrase_scores(
      phrase_list, word_scores)
    sorted_phrase_scores = sorted(phrase_scores.iteritems(),
      key=operator.itemgetter(1), reverse=True)
    n_phrases = len(sorted_phrase_scores)
    if incl_scores:
      return sorted_phrase_scores[0:int(n_phrases/self.top_fraction)]
    else:
      return map(lambda x: x[0],
        sorted_phrase_scores[0:int(n_phrases/self.top_fraction)])

#def test():
 # rake = RakeKeywordExtractor()
  #for i in len(range(X)):
   # print(X[i])
    #keywords = rake.extract(X[i])
    #print keywords

def test(text):
  rake = RakeKeywordExtractor()
  keywords = rake.extract(X[i], incl_scores=True)
  
  print keywords
  return keywords

for i in range(len(X)):
  #X[i] = X[i].encode('utf8')
  X[i] = str(X[i]) #Basically it removes encoding
  X[i] = X[i].decode('utf-8')
  print(type(X[i]))
  X1[i] = test(X[i])
  #test(X[i])
df8 = pd.DataFrame({"zarooriinfo" : X1 })
df8.to_csv("zorooriinfo.csv", index=False)

  

    
  

#test()