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



from sklearn.datasets import fetch_20newsgroups
from progressbar import ProgressBar
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.svm import SVC

noise_list = ["is", "a", "this", "...","ourselves", "hers", "between", "yourself", "but", "again", "there", "about", "once", "during", "out", "very", "having", "with", "they", "own", "an", "be", "some", "for", "do", "its", "yours", "such", "into", "of", "most", "itself", "other", "off", "is", "s", "am", "or", "who", "as", "from", "him", "each", "the", "themselves", "until", "below", "are", "we", "these", "your", "his", "through", "don", "nor", "me", "were", "her", "more", "himself", "this", "down", "should", "our", "their", "while", "above", "both", "up", "to", "ours", "had", "she", "all", "no", "when", "at", "any", "before", "them", "same", "and", "been", "have", "in", "will", "on", "does", "yourselves", "then", "that", "because", "what", "over", "why", "so", "can", "did", "not", "now", "under", "he", "you", "herself", "has", "just", "where", "too", "only", "myself", "which", "those", "i", "after", "few", "whom", "t", "being", "if", "theirs", "my", "against", "a", "by", "doing", "it", "how", "further", "was", "here", "than"] 

def remove_noise(input_text):
    words = input_text.split() 
    noise_free_words = [word for word in words if word not in noise_list] 
    noise_free_text = " ".join(noise_free_words) 
    return noise_free_text


def cleanhtml(raw_html):
  cleanr = re.compile('<.*?>')
  cleantext = re.sub(cleanr, '', raw_html)
  return cleantext    


def stem(text):
    porter = nltk.PorterStemmer()
    stemmed = porter.stem(text) 
    return stemmed


def for_eng(text):   
    for i in range(len(X)):        
	    X[i] = remove_noise(X[i])        #remove stopwords

    for i in range(len(X)):
        X[i] = cleanhtml(X[i])   	           #remove html chars with regex

    print(type(i))

    for i in range(len(X)):             #remove puncutation marks
        X[i] = X[i].translate(None, string.punctuation)   
    

    for i in range(len(X)):	                                            
        X[i] = X[i].lower()    #set all to lowercase


    for i in range(len(X)):
    #print(type(X[i]))		
	    X[i] = X[i].decode('utf-8', errors='replace')        #????decode to utf-8 for no issues with other languages(chinesse etc)

    for i in range(len(X)):	  ##stemming text -> grows-grow, leaves-leaf USED PORTER STEMMER AS SNOBALL WANTED LANGUAGE AND THIS IS MULTI LANGUAGE
        X[i] = stem(X[i])
                               
    for i in range(len(X)):	  ##lemmatize text after stemming so that any word left can be stemmed for the algorithm to understand better
        lem = WordNetLemmatizer()
        X[i] = lem.lemmatize(X[i], "v")
    
    for i in range(len(X)):
	    X[i] = wordpunct_tokenize(X[i])

    return X        



##################

import sys
#----------------------------------------------------------------------
def _calculate_languages_ratios(text):

    languages_ratios = {}

    tokens = wordpunct_tokenize(text)
    words = [word.lower() for word in tokens]

    # Compute per language included in nltk number of unique stopwords appearing in analyzed text
    for language in stopwords.fileids():
        stopwords_set = set(stopwords.words(language))
        words_set = set(words)
        common_elements = words_set.intersection(stopwords_set)

        languages_ratios[language] = len(common_elements) # language "score"

    return languages_ratios


#----------------------------------------------------------------------
def detect_language(text):

    ratios = _calculate_languages_ratios(text)

    most_rated_language = max(ratios, key=ratios.get)

    return most_rated_language

    language = detect_language(text)

    print language
#####################


df = pd.read_csv("sampledata.csv", )
df["Information"] = df["Information"].astype(str)  #convert column to string s
X=df["Information"]


for i in range(len(X)):
    text = X[i]
    language = detect_language(text)
    
    print language

for i in range(len(X)):
    text = for_eng(X[i])
#X = X.encode('utf-8')
#df8 = pd.DataFrame({"prediction" :X})
#df8.to_csv("cleaned.csv", index=False)