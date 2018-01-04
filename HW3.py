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
from sets import Set
# function to split the data for cross-validation
from sklearn.model_selection import train_test_split
from time import time

twitterDataSet = pd.read_csv("c:\gender-classifier-DFE-791531.csv", encoding='latin1')
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
twitsPerGender['unknown'] = [];
twitsPerGender['brand'] = [];
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
print("Tthe terms that are common in all categories:")
print("***************************************************")
print (commSet)




# Q2

#remove all rows in dataset that have Nan value in "gender:confidence" column
gender_confidence = twitterDataSet['gender:confidence'][np.where(np.invert(np.isnan(twitterDataSet['gender:confidence'])))[0]]
twitterDataSet_confident = twitterDataSet[twitterDataSet['gender:confidence']==1]

textsAfterClean = twitterDataSet_confident['text_norm']
genders = twitterDataSet_confident['gender']

# split into train and test sets using cross validation
x_train, x_test, y_train, y_test = train_test_split(textsAfterClean, genders, test_size=0.2)


#this function recieve: clf-the chosen classifier, x_train and x_test- BOW/TF-IDF matrix (from feature extraction)
#then train the model clf and return the results
def benchmark(clf, x_train, x_test):

    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(x_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(x_test)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    score = metrics.accuracy_score(y_test, pred)
    print("accuracy:   %0.3f" % score)

    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, train_time, test_time

#function that receive the results of the training models and displays the comparison between them
def makePlot (results):
    indices = np.arange(len(results))

    results = [[x[i] for x in results] for i in range(4)]

    clf_names, score, training_time, test_time = results
    training_time = np.array(training_time) / np.max(training_time)
    test_time = np.array(test_time) / np.max(test_time)

    plt.figure(figsize=(12, 8))
    plt.title("Score")
    plt.barh(indices, score, .2, label="score", color='navy')
    plt.barh(indices + .3, training_time, .2, label="training time",
             color='c')
    plt.barh(indices + .6, test_time, .2, label="test time", color='darkorange')
    plt.yticks(())
    plt.legend(loc='best')
    plt.subplots_adjust(left=.25)
    plt.subplots_adjust(top=.95)
    plt.subplots_adjust(bottom=.05)

    for i, c in zip(indices, clf_names):
        plt.text(-.3, i, c)
    plt.show()


#feature extraction
vectorizer = CountVectorizer()
BOW_train = vectorizer.fit_transform(x_train)
BOW_test = vectorizer.transform(x_test)

Tfidf_vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
Tfidf_train = Tfidf_vectorizer.fit_transform(x_train)
Tfidf_test = Tfidf_vectorizer.transform(x_test)

### Results of the classify using Bag of words feature extraction:
results_BOW = []
for clf, name in (
        (SGDClassifier(),"SVM"),
        (MultinomialNB(),"Naive Bayes")):
    print('=' * 80)
    print(name)
    results_BOW.append(benchmark(clf, BOW_train,BOW_test))
makePlot(results_BOW)

### Results of the classify using TF-IDF feature extraction:
results_Tfidf = []
for clf, name in (
        (SGDClassifier(),"SVM"),
        (MultinomialNB(),"Naive Bayes")):
    print('=' * 80)
    print(name)
    results_Tfidf.append(benchmark(clf, Tfidf_train,Tfidf_test))
makePlot(results_Tfidf)

### Tune each model parameters to optimize the results using pipeline
svmResults  = []
NBResults = []

#### find the best parameters for both the feature extraction-BOW and the classifier-Naive Bayes:

nb_clf1 = Pipeline([('vect', CountVectorizer()),('clf', MultinomialNB())])
parameters_clf1 =  {
    'vect__max_df': (0.3,0.5,1.0),
    'clf__alpha': (0.01,1.0),
    'clf__fit_prior':(False,True)}
gs_clf1 = GridSearchCV(nb_clf1, parameters_clf1, n_jobs=1)
gs_clf1 = gs_clf1.fit(x_train,y_train)
print('Best score: ',gs_clf1.best_score_)
NBResults.append(gs_clf1.best_score_)
print('Best params: ',gs_clf1.best_params_)

#### find the best parameters for both the feature extraction-Tf idf and the classifier-Naive Bayes:
nb_clf2 = Pipeline([('vect', TfidfVectorizer()),('clf', MultinomialNB())])
parameters_clf2 =  {
    'vect__max_df': (0.3,0.5,1.0),
    'vect__sublinear_tf':(True, False),
    'clf__alpha': (0.01,1.0),
    'clf__fit_prior':(False,True)}
gs_clf2 = GridSearchCV(nb_clf2, parameters_clf2, n_jobs=1)
gs_clf2 = gs_clf2.fit(x_train,y_train)
NBResults.append(gs_clf2.best_score_)
print('Best score: ',gs_clf2.best_score_)
print('Best params: ',gs_clf2.best_params_)

#### find the best parameters for both the feature extraction-BOW and the classifier-SVM:
svm_clf3 = Pipeline([
    ('vect', CountVectorizer()),
    ('clf', SGDClassifier()),
])
parameters_clf3 = {
    'vect__max_df': (0.3,0.5,1.0),
    'clf__alpha': (0.001,0.0001,0.00001,0.000001),
    'clf__penalty': ('elasticnet','none', 'l2','l1'),
    'clf__epsilon':(0.1,0.2)}
gs_clf3 = GridSearchCV(svm_clf3, parameters_clf3, n_jobs=1)
gs_clf3 = gs_clf3.fit(x_train,y_train)
svmResults.append(gs_clf3.best_score_);
print('Best score: ',gs_clf3.best_score_)
print('Best params: ',gs_clf3.best_params_)

#### find the best parameters for both the feature extraction-Tf idf and the classifier-SVM:
svm_clf4 = Pipeline([
    ('vect', TfidfVectorizer()),
    ('clf', SGDClassifier()),
])
parameters_clf4 = {
    'vect__max_df': (0.3,0.5,1.0),
    'vect__sublinear_tf':(True, False),
    'clf__alpha': (0.001,0.0001,0.00001,0.000001),
    'clf__penalty': ('elasticnet','none', 'l2','l1'),
    'clf__epsilon':(0.1,0.2)}
gs_clf4 = GridSearchCV(svm_clf4, parameters_clf4, n_jobs=1)
gs_clf4 = gs_clf4.fit(x_train,y_train)
svmResults.append(gs_clf4.best_score_);
print('Best score: ',gs_clf4.best_score_)
print('Best params: ',gs_clf4.best_params_)

### Use the best models selected in the previous steps for prediction on the test set and present the accuracy for each machine learning model

#### Results of  the model Naive Bayes (with the receive pipeline's parameters):
#Naive Bayes best with Bag of word
#Best params:  {'vect__max_df': 0.5, 'clf__fit_prior': True, 'clf__alpha': 1.0}
count_vectorizer1 = CountVectorizer(max_df=0.5, stop_words='english')
count_train1 = count_vectorizer1.fit_transform(x_train)
count_test1 = count_vectorizer1.transform(x_test)
results1 = []
clf_nb=MultinomialNB(fit_prior=True,alpha=1.0)
print("Naive Bayes BOW")
results1.append(benchmark(clf_nb, count_train1,count_test1))

#### Results of  the model SVM (with the receive pipeline's parameters):
#SVM best with Bag of word
#Best params: {'vect__max_df': 0.5, 'clf__penalty': 'l2', 'clf__epsilon': 0.1, 'clf__alpha': 0.001}
count_vectorizer1 = CountVectorizer(max_df=0.5, stop_words='english')
count_train2 = count_vectorizer1.fit_transform(x_train)
count_test2 = count_vectorizer1.transform(x_test)
results2 = []
clf_svm=SGDClassifier(penalty='l2', alpha=0.001, epsilon=0.1)
print("SVM BOW")
results2.append(benchmark(clf_svm, count_train2,count_test2))

