'''import os
os.environ["KERAS_BACKEND"] = "theano"
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
# fix random seed for reproducibility\n
#numpy.random.seed(7)'''



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
from sets import Set

import tweepy
from tweepy import OAuthHandler
from tweepy import Stream
from tweepy.streaming import StreamListener
# function to split the data for cross-validation
from sklearn.model_selection import train_test_split
from time import time

twitterDataSet = pd.read_csv("c:\corpus\gender-classifier-DFE-791531.csv", encoding='latin1')
#twitterDataSet = pd.read_csv("c:\gender-classifier-DFE-791531.csv")
#print(twitterDataSet.head())


#clean a tweet

def cleanData(text):
    #print(text)
    #print("******************************")
    text = text.lower()

    # remove punctuation, spaces digits and "https"
    text = re.sub('\s\W',' ',text)
    text = re.sub('\W,\s',' ',text)
    text = re.sub(r'[^\w]', ' ', text)
    text = re.sub("\d+", "", text)
    text = re.sub('\s+',' ',text)
    text = re.sub('[!@#$_]', '', text)
    text = text.replace("https","")
    text = text.replace(",","")
    text = text.replace("[\w*"," ")

    stop_words = stopwords.words('english') + list(string.punctuation)
    stemmer = SnowballStemmer("english")

    twitAfterTokenize = word_tokenize(text) #text after tokenize (from full sentences to separated words)
    filteredTwit = [word for word in twitAfterTokenize if word not in stop_words]
    twitAfterStemming = [stemmer.stem(word) for word in filteredTwit] #stem each word in the text
    twitAfterStemmingStr = ' '.join(twitAfterStemming)
    return twitAfterStemmingStr

#update the corpus with the clean twits
twitterDataSet['text_norm'] = [cleanData(s) for s in twitterDataSet['text']]
#twitterDataSet['description_norm'] = [cleanData(s) for s in twitterDataSet['description']]

print("Class Distribution:")
print("***************************************************")
print(twitterDataSet.gender.value_counts())
print("")

#termsPerCat = {} #dictionary of all categories and for each one- list of all terms (after cleaning) in category

#get all the common terms and their frequency in a category
def getCommonTermsPerCat (gender):
    allTermsInCat = list(itertools.chain(*twitsPerGender[gender]))#make list of all terms (transfer list of lists to one list)
    termsCounter = Counter(allTermsInCat)
    return (termsCounter.most_common(30))


#get all the common terms in the corpus
#def getCommonMedTerms (cleanTextsPerTerms):
#    global commonTermsInCorpus
#    allTerms = list(itertools.chain(*cleanTextsPerTerms))#make list of all terms (transfer list of lists to one list)
#    commonTermsInCorpus = [term for term, word_count in Counter(allTerms).most_common(20)]


#get the tweet foreach gender
twitsPerGender ={};
twitsPerGender['male'] = [];
twitsPerGender['female'] = [];
for index, row in twitterDataSet.iterrows(): # iterate each row
    #twitsPerGender[row['gender']].append(row['text_norm']) #if we want that twit will save as string not list
    twitsPerGender[row['gender']].append(row['text_norm'].split())
#count the twits in each category
#print("Class Distribution:")
#print("***************************************************")
#for key, value in twitsPerGender.iteritems():
    #print('{}: {} twits'.format(key,len(value)))
commonWords = [];
##get the terms distribution
table_of_most_frequent = {};# array of tuples - term and requenchy
most_freq_byCat = {};
temp = {};
commTermsInAll = []
commSet = Set(commTermsInAll);
for gen in twitsPerGender:
    #print (gen)
    lst = getCommonTermsPerCat(gen);
    table_of_most_frequent[gen] = lst;
    most_freq_byCat[gen] = [];
    temp[gen] = [];

for gen in twitsPerGender:
  for commTerm, word_count in table_of_most_frequent[gen]:
      most_freq_byCat[gen].append(commTerm)

for gen in twitsPerGender:
    ##remove terms that apear in all categories as common
    for commTerm in most_freq_byCat[gen]:
        if(commTerm not in commSet):
            counter = 1
            for otherGen in twitsPerGender:
                if(otherGen != gen and commTerm in most_freq_byCat[otherGen]):
                    counter = counter +1
            if(counter == len(twitsPerGender)):
                commSet.add(commTerm)
                #print(commTerm)

#remove the terms that common in all categories
for gen in twitsPerGender:
    for i in table_of_most_frequent[gen]:
        if i[0] not in commSet:
            temp[gen].append(i)
    table_of_most_frequent[gen] = temp[gen];

#plot the results
for gen in twitsPerGender:
    df3 = pd.DataFrame(table_of_most_frequent[gen], columns = ['Word', 'Count'])
    ax = df3.plot.bar(title ="most frequent terms in category: " + gen, x='Word',y='Count')
    x_offset = -0.4
    y_offset = 2
    for p in ax.patches:
        b = p.get_bbox()
        val = b.y1 + b.y0
        ax.annotate(val, ((b.x0 + b.x1)/2 + x_offset, b.y1 + y_offset))
    #print(gen)
    #print("**************************")
    #print(df3)
df2 = pd.DataFrame(table_of_most_frequent)
df2.transpose()
print("")
print("Terms Frequency:") #without terms that common apear in alll category
print("***************************************************")
print (df2)
plt.show()

print("")
print("The terms that are common in all categories:")
print("***************************************************")
print (commSet)

('# Q2\n'
 '\n'
 '#remove all rows in dataset that have Nan value in "gender:confidence" column\n'
 'gender_confidence = twitterDataSet[\'gender:confidence\'][np.where(np.invert(np.isnan(twitterDataSet[\'gender:confidence\'])))[0]]\n'
 'twitterDataSet_confident = twitterDataSet[twitterDataSet[\'gender:confidence\']==1]\n'
 '\n'
 'textsAfterClean = twitterDataSet_confident[\'text_norm\']\n'
 'genders = twitterDataSet_confident[\'gender\']\n'
 '\n'
 '# split into train and test sets using cross validation\n'
 'x_train, x_test, y_train, y_test = train_test_split(textsAfterClean, genders, test_size=0.2)\n'
 '\n'
 '\n'
 '#this function recieve: clf-the chosen classifier, x_train and x_test- BOW/TF-IDF matrix (from feature extraction)\n'
 '#then train the model clf and return the results\n'
 'def benchmark(clf, x_train, x_test):\n'
 '\n'
 '    print(\'_\' * 80)\n'
 '    print("Training: ")\n'
 '    print(clf)\n'
 '    t0 = time()\n'
 '    clf.fit(x_train, y_train)\n'
 '    train_time = time() - t0\n'
 '    print("train time: %0.3fs" % train_time)\n'
 '\n'
 '    t0 = time()\n'
 '    pred = clf.predict(x_test)\n'
 '    test_time = time() - t0\n'
 '    print("test time:  %0.3fs" % test_time)\n'
 '\n'
 '    score = metrics.accuracy_score(y_test, pred)\n'
 '    print("accuracy:   %0.3f" % score)\n'
 '\n'
 '    clf_descr = str(clf).split(\'(\')[0]\n'
 '    return clf_descr, score, train_time, test_time\n'
 '\n'
 '#function that receive the results of the training models and displays the comparison between them\n'
 'def makePlot (results):\n'
 '    indices = np.arange(len(results))\n'
 '\n'
 '    results = [[x[i] for x in results] for i in range(4)]\n'
 '\n'
 '    clf_names, score, training_time, test_time = results\n'
 '    training_time = np.array(training_time) / np.max(training_time)\n'
 '    test_time = np.array(test_time) / np.max(test_time)\n'
 '\n'
 '    plt.figure(figsize=(12, 8))\n'
 '    plt.title("Score")\n'
 '    plt.barh(indices, score, .2, label="score", color=\'navy\')\n'
 '    plt.barh(indices + .3, training_time, .2, label="training time",\n'
 '             color=\'c\')\n'
 '    plt.barh(indices + .6, test_time, .2, label="test time", color=\'darkorange\')\n'
 '    plt.yticks(())\n'
 '    plt.legend(loc=\'best\')\n'
 '    plt.subplots_adjust(left=.25)\n'
 '    plt.subplots_adjust(top=.95)\n'
 '    plt.subplots_adjust(bottom=.05)\n'
 '\n'
 '    for i, c in zip(indices, clf_names):\n'
 '        plt.text(-.3, i, c)\n'
 '    plt.show()\n'
 '\n'
 '\n'
 '#feature extraction\n'
 'vectorizer = CountVectorizer()\n'
 'BOW_train = vectorizer.fit_transform(x_train)\n'
 'BOW_test = vectorizer.transform(x_test)\n'
 '\n'
 'Tfidf_vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words=\'english\')\n'
 'Tfidf_train = Tfidf_vectorizer.fit_transform(x_train)\n'
 'Tfidf_test = Tfidf_vectorizer.transform(x_test)\n'
 '\n'
 '### Results of the classify using Bag of words feature extraction:\n'
 'results_BOW = []\n'
 'for clf, name in (\n'
 '        (SGDClassifier(),"SVM"),\n'
 '        (MultinomialNB(),"Naive Bayes")):\n'
 '    print(\'=\' * 80)\n'
 '    print(name)\n'
 '    results_BOW.append(benchmark(clf, BOW_train,BOW_test))\n'
 'makePlot(results_BOW)\n'
 '\n'
 '### Results of the classify using TF-IDF feature extraction:\n'
 'results_Tfidf = []\n'
 'for clf, name in (\n'
 '        (SGDClassifier(),"SVM"),\n'
 '        (MultinomialNB(),"Naive Bayes")):\n'
 '    print(\'=\' * 80)\n'
 '    print(name)\n'
 '    results_Tfidf.append(benchmark(clf, Tfidf_train,Tfidf_test))\n'
 'makePlot(results_Tfidf)\n'
 '\n'
 '\n'
 '### Tune each model parameters to optimize the results using pipeline\n'
 'svmResults  = []\n'
 'NBResults = []\n'
 '\n'
 '#### find the best parameters for both the feature extraction-BOW and the classifier-Naive Bayes:\n'
 '\n'
 'nb_clf1 = Pipeline([(\'vect\', CountVectorizer()),(\'clf\', MultinomialNB())])\n'
 'parameters_clf1 =  {\n'
 '    \'vect__max_df\': (0.3,0.5,1.0),\n'
 '    \'clf__alpha\': (0.01,1.0),\n'
 '    \'clf__fit_prior\':(False,True)}\n'
 'gs_clf1 = GridSearchCV(nb_clf1, parameters_clf1, n_jobs=1)\n'
 'gs_clf1 = gs_clf1.fit(x_train,y_train)\n'
 'print(\'Best score: \',gs_clf1.best_score_)\n'
 'NBResults.append(gs_clf1.best_score_)\n'
 'print(\'Best params: \',gs_clf1.best_params_)\n'
 '\n'
 '#### find the best parameters for both the feature extraction-Tf idf and the classifier-Naive Bayes:\n'
 'nb_clf2 = Pipeline([(\'vect\', TfidfVectorizer()),(\'clf\', MultinomialNB())])\n'
 'parameters_clf2 =  {\n'
 '    \'vect__max_df\': (0.3,0.5,1.0),\n'
 '    \'vect__sublinear_tf\':(True, False),\n'
 '    \'clf__alpha\': (0.01,1.0),\n'
 '    \'clf__fit_prior\':(False,True)}\n'
 'gs_clf2 = GridSearchCV(nb_clf2, parameters_clf2, n_jobs=1)\n'
 'gs_clf2 = gs_clf2.fit(x_train,y_train)\n'
 'NBResults.append(gs_clf2.best_score_)\n'
 'print(\'Best score: \',gs_clf2.best_score_)\n'
 'print(\'Best params: \',gs_clf2.best_params_)\n'
 '\n'
 '#### find the best parameters for both the feature extraction-BOW and the classifier-SVM:\n'
 'svm_clf3 = Pipeline([\n'
 '    (\'vect\', CountVectorizer()),\n'
 '    (\'clf\', SGDClassifier()),\n'
 '])\n'
 'parameters_clf3 = {\n'
 '    \'vect__max_df\': (0.3,0.5,1.0),\n'
 '    \'clf__alpha\': (0.001,0.0001,0.00001,0.000001),\n'
 '    \'clf__penalty\': (\'elasticnet\',\'none\', \'l2\',\'l1\'),\n'
 '    \'clf__epsilon\':(0.1,0.2)}\n'
 'gs_clf3 = GridSearchCV(svm_clf3, parameters_clf3, n_jobs=1)\n'
 'gs_clf3 = gs_clf3.fit(x_train,y_train)\n'
 'svmResults.append(gs_clf3.best_score_);\n'
 'print(\'Best score: \',gs_clf3.best_score_)\n'
 'print(\'Best params: \',gs_clf3.best_params_)\n'
 '\n'
 '#### find the best parameters for both the feature extraction-Tf idf and the classifier-SVM:\n'
 'svm_clf4 = Pipeline([\n'
 '    (\'vect\', TfidfVectorizer()),\n'
 '    (\'clf\', SGDClassifier()),\n'
 '])\n'
 'parameters_clf4 = {\n'
 '    \'vect__max_df\': (0.3,0.5,1.0),\n'
 '    \'vect__sublinear_tf\':(True, False),\n'
 '    \'clf__alpha\': (0.001,0.0001,0.00001,0.000001),\n'
 '    \'clf__penalty\': (\'elasticnet\',\'none\', \'l2\',\'l1\'),\n'
 '    \'clf__epsilon\':(0.1,0.2)}\n'
 'gs_clf4 = GridSearchCV(svm_clf4, parameters_clf4, n_jobs=1)\n'
 'gs_clf4 = gs_clf4.fit(x_train,y_train)\n'
 'svmResults.append(gs_clf4.best_score_);\n'
 'print(\'Best score: \',gs_clf4.best_score_)\n'
 'print(\'Best params: \',gs_clf4.best_params_)\n'
 '\n'
 '### Use the best models selected in the previous steps for prediction on the test set and present the accuracy for each machine learning model\n'
 '\n'
 '#### Results of  the model Naive Bayes (with the receive pipeline\'s parameters):\n'
 '#Naive Bayes best with Bag of word\n'
 '#Best params:  {\'vect__max_df\': 0.5, \'clf__fit_prior\': True, \'clf__alpha\': 1.0}\n'
 'count_vectorizer1 = CountVectorizer(max_df=0.5, stop_words=\'english\')\n'
 'count_train1 = count_vectorizer1.fit_transform(x_train)\n'
 'count_test1 = count_vectorizer1.transform(x_test)\n'
 'results1 = []\n'
 'clf_nb=MultinomialNB(fit_prior=True,alpha=1.0)\n'
 'print("Naive Bayes BOW")\n'
 'results1.append(benchmark(clf_nb, count_train1,count_test1))\n'
 '\n'
 '#### Results of  the model SVM (with the receive pipeline\'s parameters):\n'
 '#SVM best with Bag of word\n'
 '#Best params: {\'vect__max_df\': 0.5, \'clf__penalty\': \'l2\', \'clf__epsilon\': 0.1, \'clf__alpha\': 0.001}\n'
 'count_vectorizer1 = CountVectorizer(max_df=0.5, stop_words=\'english\')\n'
 'count_train2 = count_vectorizer1.fit_transform(x_train)\n'
 'count_test2 = count_vectorizer1.transform(x_test)\n'
 'results2 = []\n'
 'clf_svm=SGDClassifier(penalty=\'l2\', alpha=0.001, epsilon=0.1)\n'
 'print("SVM BOW")\n'
 'results2.append(benchmark(clf_svm, count_train2,count_test2))\n'
 '\n'
 '\n'
 '\n'
 'import os\n'
 'os.environ["KERAS_BACKEND"] = "theano"\n'
 'import numpy\n'
 'from keras.models import Sequential\n'
 'from keras.layers import Dense\n'
 'from keras.layers import LSTM\n'
 'from keras.layers.embeddings import Embedding\n'
 'from keras.preprocessing import sequence\n'
 '# fix random seed for reproducibility\n'
 '#numpy.random.seed(7)\n')



'''

#Q3 - Tweeter

#Set the credential of our twitter App
consumer_key = 'Sctv4QyeP6y4GkxzRMGlfkqtj'
consumer_secret = 'xpi8fDuUieBL17FddrH81RClqSvRR8PnkRfT3NaenGm8Kq5inW'
access_token = '950991759663882240-9AWQESLQpmCoRRkroF7GqOrJRlT6TK8'
access_secret = 'IevZtm0wWTnnMUWjr5WNXGi1awEc4EtfajOncbDX1tts8'

auth = OAuthHandler(consumer_key,consumer_secret)
auth.set_access_token(access_token,access_secret)

api = tweepy.API(auth)

import json

def process_or_store(tweet):
    print(json.dumps(tweet))

#class that listen to the tweeter
class MyListener(StreamListener):

    def on_data(self, data):
        try:
            with open('tweets.json', 'a') as f:
                f.write(data)
                return True
        except BaseException as e:
            print("Error on_data: %s" % str(e))
        return True

    def on_error(self, status):
        print(status)
        return True


twitter_stream = Stream(auth, MyListener())
twitter_stream.filter(track=['#SHOOPING'])
'''


#PreProcecing the Tweets by the example preprocecing (Before we send them to Q1)
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

punctuation = list(string.punctuation)
stop = stopwords.words('english') + punctuation + ['rt', 'via', 'the', u'\u2019', u'\u2026', 'The', u'de', u'\xe9']

import re

emoticons_str = r"""
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )"""

regex_str = [
    emoticons_str,
    r'<[^>]+>',  # HTML tags
    r'(?:@[\w_]+)',  # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)",  # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+',  # URLs

    r'(?:(?:\d+,?)+(?:\.?\d+)?)',  # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])",  # words with - and '
    r'(?:[\w_]+)',  # other words
    r'(?:\S)'  # anything else
]

tokens_re = re.compile(r'(' + '|'.join(regex_str) + ')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^' + emoticons_str + '$', re.VERBOSE | re.IGNORECASE)


def tokenize(s):
    return tokens_re.findall(s.encode('ascii', 'ignore'))


def preprocess(s, lowercase=False):
    tokens = tokenize(s)
    if lowercase:
        tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
    return tokens


#Get the information from the JsonFile
import json

twitterStreamingDataSet = {}
#counter =0
twitterStreamingDataSet['text'] = []
twitterStreamingDataSet['description'] = []
with open('shoppingCarTweets1000.json', 'r') as f: #read this file
    while True:
        line = f.readline()  # read the next line
        #counter = counter +1
        #print (counter)
        if not line: break
        tweet = json.loads(line)  # load it as Python dict
        #get the text of the tweet
        terms_only = [term for term in preprocess(tweet['text'])
            if term not in stop]
                  #and term.startswith(('#', '@'))]
    #same for description
        if(tweet['user']['description']):
            terms_only_description = [term for term in preprocess(tweet['user']['description'])
                  if term not in stop]
        else:
            terms_only_description = []
        #convert the terms list to strinng
        terms_only_strings  = ""
        terms_only_strings_desc = ""
        for tweet_terms in terms_only:
            terms_only_strings += " " +tweet_terms
        twitterStreamingDataSet['text'].append(terms_only_strings) #add the new tweet to he tweets List
        # convert eacj term array to list - description field
        terms_only_strings_description = []
        for tweet_terms_desc in terms_only_description:
            terms_only_strings_desc += " " + tweet_terms_desc
        twitterStreamingDataSet['description'].append(terms_only_strings_desc)

#clean the tweets in the same cleaner of Q!
twitterStreamingDataSet['text_norm'] = [cleanData(s) for s in twitterStreamingDataSet['text']]
twitterStreamingDataSet['description_norm'] = [cleanData(s) for s in twitterStreamingDataSet['description']]


print ("done")