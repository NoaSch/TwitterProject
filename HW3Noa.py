import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils.extmath import density
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
import re
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from nltk.stem.snowball import SnowballStemmer
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
# have to use latin1 even though it results in a lot of dead characters
import re
from collections import Counter
import itertools

twitterDataSet = pd.read_csv("c:\corpus\gender-classifier-DFE-791531.csv", encoding='latin1')
#twitterDataSet = pd.read_csv("c:\gender-classifier-DFE-791531.csv")
print(twitterDataSet.head())



'''
def normalize_text(text):
    # just in case
    text = text.lower()

    # remove punctuation that is not word-internal (e.g., hyphens, apostrophes)
    text = re.sub('\s\W', ' ', text)
    text = re.sub('\W\s', ' ', text)


    # make sure we didn't introduce any double spaces
    text = re.sub('\s+', ' ', text)

    return text
'''

#clean a tweet

def cleanData(text):
    #print(text)
    #print("******************************")
    text = text.lower()

    # remove punctuation that is not word-internal (e.g., hyphens, apostrophes)
    text = re.sub('\s\W', ' ', text)
    text = re.sub('\W\s', ' ', text)

    # make sure we didn't introduce any double spaces
    text = re.sub('\s+', ' ', text)

    stop_words = stopwords.words('english') + list(string.punctuation)
    stemmer = SnowballStemmer("english")

    #for doc in data:
    docAfterTokenize = word_tokenize(text) #text after tokenize (from full sentences to separated words)
    filteredTwit = [word for word in docAfterTokenize if word not in stop_words]
    twitAfterStemming = [stemmer.stem(word) for word in filteredTwit] #stem each word in the text
    twitAfterStemmingStr = ' '.join(twitAfterStemming)
    return twitAfterStemmingStr


twitterDataSet['text_norm'] = [cleanData(s) for s in twitterDataSet['text']]
#twitterDataSet['description_norm'] = [cleanData(s) for s in twitterDataSet['description']]



termsPerCat = {} #dictionary of all categories and for each one- list of all terms (after cleaning) in category

#get all the common terms and their frequency in a category
def getCommonTermsPerCat (gender):
    allTermsInCat = list(itertools.chain(*termsPerCat[gender]))#make list of all terms (transfer list of lists to one list)
    termsCounter = Counter(allTermsInCat)
    return (termsCounter.most_common(10))


#get all the common terms in the corpus
def getCommonMedTerms (cleanTextsPerTerms):
    global commonTermsInCorpus
    allTerms = list(itertools.chain(*cleanTextsPerTerms))#make list of all terms (transfer list of lists to one list)
    commonTermsInCorpus = [term for term, word_count in Counter(allTerms).most_common(20)]


    #get the tweet foreach gender
twitsPerGender ={};
twitsPerGender['male'] = [];
twitsPerGender['female'] = [];
twitsPerGender['unknown'] = [];
twitsPerGender['brand'] = [];
for index, row in twitterDataSet.iterrows():
    twitsPerGender[row['gender']].append(row['text_norm'])
#count the twits in each category
print("Class Distribution:")
print("***************************************************")
for key, value in twitsPerGender.iteritems():
    print('{}: {}'.format(key,len(value)))
    #print (len(value))


