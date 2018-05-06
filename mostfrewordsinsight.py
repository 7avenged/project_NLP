import nltk
import pandas as pd
from nltk.corpus import sentiwordnet as swn



#df = pd.read_csv("sampledata.csv", )
##df["Information"] = df["Information"].astype(str)  #convert column to string s
##X=df["Information"]
#X1 = X


df = pd.read_csv("cleaned.csv", )
df["prediction"] = df["prediction"].astype(str)  #convert column to string s
X=df["prediction"]
X1 = X

#print(X[4])

Xnew = ' '.join(X)  #(df['text'])

Xnew = Xnew.replace("u'","")
#for i in Xnew:
#    Xnew = i.replace("u'","")
  #  s = str.cat
    
    #s = s.str.cat(X[i])

#    print type(X)
x=0
y=0

allWords = nltk.tokenize.word_tokenize(Xnew)
allWordDist = nltk.FreqDist(w.lower() for w in allWords)
stopwords = nltk.corpus.stopwords.words('english')
allWordExceptStopDist = nltk.FreqDist(w.lower() for w in allWords if w not in stopwords) 
for word, frequency in allWordDist.most_common(100):
    word = word.encode('utf-8')  
    #print(type(word)) #string      
    #print('%s;%d' % (word, frequency) )
    #if word == "'" or word == " " or word == "," or word == "i" or word == "[" or word == "]" :
     #   word =word.replace(word,"random")





    sentiword = word
    #sentiword1 = sentiword.replace("u'","")
    #sentiword1 = sentiword.replace("'","")
    #sentiword1 = sentiword.replace(",","")

    #print(type(sentiword)) #str
    #arg = sentiword.join(sentiword + '.n.03')
    arg = '{}.n.01'.format(sentiword)
    #print(arg)

    #print(arg)
    #print type(arg)
    #arg = arg.astype(str)
    #print(swn.senti_synset(arg) )
    #temp = 'breakdown'      #nltk.corpus.reader.wordnet.WordNetError: no lemma u"'" with part of speech u'n'
    #arg1 = '{}.n.01'.format(temp)
    #print(temp)
    #print(type(temp))


    try:
        analysis = swn.senti_synset(arg)
        pos = analysis.pos_score()
        x=x+pos
        neg = analysis.neg_score()
        y=y+neg
            
               
    except Exception:
        pass


print x
print y    
if x>y:
    print("customers got satisfying replies")
else:
    print("customers did not get satisfying replies")
print("The percentage amount agent gave positive replies: %f " % ((x-y)/x) *100)        
                
        
    #print('%s;%d' % (word, frequency).encode('utf-8') )
    #mostCommon= allWordDist.most_common(10).keys()
    #rint(X[i])
    #for w in X[4]:
    #    print w
    #allWords = nltk.tokenize.word_tokenize(X[i])
    



    