################################################################
# This is DS_10_19 NLP final project - Itai & Pnina  07.2020
################################################################

# Add the Required Libraries
############################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

import spacy
import en_core_web_sm
from spacy.lang.en import English
from spacy.tokenizer import Tokenizer
from spacy import displacy
from spacy.util import minibatch, compounding
from spacy.lemmatizer import Lemmatizer
from spacy.lookups import Lookups
# from spacy.lang.en import LEMMA_INDEX,LEMMA_EXP,LEMMA_RULES

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import model_selection, naive_bayes, svm
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

# Importing Libraries and Loading Datasets
from sklearn.datasets import load_files
from sklearn.preprocessing import LabelEncoder
from sklearn import datasets

# for Word2vec method
import gensim     # The main module, contains Word2Vec functions

# for RNN model
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Masking, Embedding, Activation

from collections import Counter


# Adds support to use `...` as the delimiter for sentence detection 
####################################################################
def set_custom_boundaries(doc):
    for token in doc[:-1]:
        if token.text == '...':
           doc[token.i+1].is_sent_start = True
    return doc    

# Function we will use in the token process
####################################################################
def customize_tokenizer(nlp):
    # Adds support to use - as the delimiter for tokenization
    return Tokenizer(custom_nlp.vocab, prefix_search=prefix_re.search,
                     suffix_search=suffix_re.search,
                     infix_finditer=infix_re.finditer,
                     token_match=None
                     )

# read input file
####################################################################    
# apple_data = pd.read_csv('C://Users//pnina///Google Drive//Personal//Data Science//dataset_apple_twitter_sample.csv',encoding='utf8')
apple_data = pd.read_csv('C://Users//pnina///Google Drive//Personal//Data Science//dataset_apple_twitter.csv',encoding='utf8')
X =  apple_data
# X.head()      # X.head().T

# Dealing imbalanced data using oversampling - by increasing the size of rare samples (sentiment = 1)
##################################################################################################### 
plt.figure(figsize=(4 ,6 ))
sns.countplot(X['sentiment'])
plt.show()

X.sentiment.value_counts()

X_class_1 = X[X['sentiment'] == 1]
X_class_1_new = X_class_1.sample(700, replace=True)
X_class_not_1 = X[X['sentiment'] != 1]
X_new = pd.concat([X_class_1_new,X_class_not_1])
X=X_new

plt.figure(figsize=(4 ,6 ))
sns.countplot(X['sentiment'])
plt.show()

X.sentiment.value_counts()

# Data pre-processing - Using spacy
###################################################################
custom_nlp  = en_core_web_sm.load()
custom_nlp.add_pipe(set_custom_boundaries, before='parser')

# declare params for customize_tokenizer function
prefix_re = spacy.util.compile_prefix_regex(custom_nlp.Defaults.prefixes)
suffix_re = spacy.util.compile_suffix_regex(custom_nlp.Defaults.suffixes)
infix_re = re.compile(r'''[-~]''')

# Create a dictionary of abbreviations, and replace them within the data text with words themselves
with open("C:/Users/pnina/Google Drive/Personal/Data Science/abbreviations_words.txt",'r') as i:
    abbreviations = {}        
    for line in i:
        line = line.split()
        abbreviations[line[0]] = line[1:] 
X['text'] = X['text'].replace(abbreviations, regex=True)        

# Change all the text to lower case  
X['text'] = [entry.lower() for entry in X['text']]
# X.head()
      
# https://github.com/RobertJGabriel/Google-profanity-words/blob/master/list.txt  
# Tag curse words                                                              
curse_words = pd.read_csv("C:/Users/pnina/Google Drive/Personal/Data Science/curse_words.csv",encoding='utf8')
curse_words_list = curse_words['curse_words'].tolist() 
X['text'] = X['text'].replace(curse_words_list, 'curse', regex=True) 

# Tag zero 
zero_variations_list = [' 0 ',' 0.0 ',' 0.00 ']
X['text'] = X['text'].replace(zero_variations_list, 'zero', regex=True)


# Remove hashtags                                                                                           
X['text'] = X['text'].replace(r'@', '', regex=True).replace(r'Note', '',regex=True)
# Remove #       
X['text'] = X['text'].replace(r'#', '', regex=True).replace(r'Note', '',regex=True)
# Remove urls common prefixes                                                                                
X['text'] = X['text'].replace(r'http\S', '', regex=True).replace(r'www\S','', regex=True) 
# Apple variations 
apple_variations_list = ['APPL','aapl']
X['text'] = X['text'].replace(apple_variations_list, 'apple', regex=True) 


# Create stopwords set 
spacy_stopwords = custom_nlp.Defaults.stop_words 
inc_words_list = ['Inc','Inc.'] 
all_stopwords = list(spacy_stopwords) + inc_words_list 

                               
##################################################################    
# the main loop - reading the test column from the dataframe X
##################################################################
X_post = X.copy()
my_documents = []

custom_nlp.tokenizer = customize_tokenizer(custom_nlp)
    
for i in range(0, len(X)):    
    
    # Create Doc object of sapcy.tokens.doc module 
    custom_tokenizer_doc = custom_nlp(X['text'].iloc[i])
  
    token_list1 = [] 
    # Tokenization & lemmatizing   
    for token in custom_tokenizer_doc:
        # print (i, token.text , token.lemma_, 
        #        token.tag_, token.pos_, spacy.explain(token.tag_), token.shape_,
        #        token.is_alpha, token.is_punct, token.is_space, token.is_stop)
        if (token.is_punct == False and token.is_space == False
            and token.text != token.shape_):
            token_list1.append(token.lemma_) 
 
    token_list2 = []   
    # Stopwords  
    token_list2 = [word for word in token_list1  
                   if word.lower() not in all_stopwords]
 
    my_documents.append(token_list2)
    X_post['text'].iloc[i] = token_list2 

###################################################################   
# tf-idf  
###################################################################
tf = TfidfVectorizer(ngram_range=(1,1),      
                     max_features=2000,
                     stop_words='english', 
                     sublinear_tf=True, 
                     min_df=5, 
                     max_df=0.7,
                     lowercase=False
                     )

# tfidf_matrix = tf.fit_transform(my_documents).toarray()
tfidf_matrix = tf.fit_transform(list(X_post['text'].apply(lambda x: ' '.join (x)).values)).toarray()
print(tfidf_matrix)

feature_names = tf.get_feature_names() 
print(len(feature_names))
print(feature_names[1000:1100])

# Spliting the data
X_train, X_test, y_train, y_test = train_test_split(tfidf_matrix, X_post['sentiment'],
                                                    test_size=0.2, random_state=0)
   
# Creating Naive Bayes Model Using Sckit-learn  
###################################################################    
clf = MultinomialNB(alpha=0.1, fit_prior=True, class_prior=None)
y_pred = clf.fit(X_train, y_train).predict(X_test)
print('MultinomialNB:', confusion_matrix(y_test,y_pred))
print('MultinomialNB:',classification_report(y_test,y_pred))
print('MultinomialNB:',accuracy_score(y_test, y_pred))

# Using GridSearchCV
naive_bayes_model = MultinomialNB(alpha=0.1, fit_prior=True, class_prior=None)
my_param_grid = [{ "alpha": [0.2, 0.3, 0.4, 0.5],"fit_prior": [True, False]}]
k = 5
naive_bayes_gs = GridSearchCV(naive_bayes_model, my_param_grid, cv=k)
naive_bayes_gs.fit(X_train, y_train)
y_pred = naive_bayes_gs.predict(X_test)
print("Naive_bayes best parameters:", naive_bayes_gs.best_params_)
print('MultinomialNB - confusion_matrix:', confusion_matrix(y_test,y_pred))
print('MultinomialNB - classification_report:',classification_report(y_test,y_pred))
print('MultinomialNB - accuracy:',accuracy_score(y_test, y_pred))


# Creating  svm.SVC Model Using Sckit-learn  
###################################################################    
clf= svm.SVC(kernel='linear', C=1, gamma=1)
y_pred = clf.fit(X_train, y_train).predict(X_test)
print('svm.SVC:', confusion_matrix(y_test,y_pred))
print('svm.SVC:',classification_report(y_test,y_pred))
print('svm.SVC:',accuracy_score(y_test, y_pred))



# Creating Logistic-Regression Model Using Sckit-learn  
###################################################################  
# Grid search (cross validation is part of it)
Logistic_Regression = LogisticRegression(C=0.1)
my_param_grid = [{ "C": [0.8, 1, 1.2, 1.4]}]
k = 5
logistic_regression_gs = GridSearchCV(Logistic_Regression, my_param_grid, cv=k)
logistic_regression_gs.fit(X_train, y_train)

## Assessment
cm = confusion_matrix(y_true=y_test,
                      y_pred=logistic_regression_gs.predict(X_test))
pd.DataFrame(cm, 
             index=logistic_regression_gs.classes_, 
             columns=logistic_regression_gs.classes_)

accuracy_score(y_true=y_test,
               y_pred=logistic_regression_gs.predict(X_test))

print("Logistic_regression best parameters:", logistic_regression_gs.best_params_)
print("Logistic_regression train score:",logistic_regression_gs.score(X_train, y_train))
print("Logistic_regression test score:",logistic_regression_gs.score(X_test, y_test))


######################################################################################################################################################################
# Training the Word2Vec model  
######################################################################################################################################################################
word2vec_model = gensim.models.Word2Vec(my_documents, size=100, min_count=1, 
                                    window=5, iter=100)
pretrained_weights = word2vec_model.wv.syn0
vocab_size, emdedding_size = pretrained_weights.shape
print('Result embedding shape:', pretrained_weights.shape)
print('Checking similar words:')
for word in ['model', 'network', 'train', 'learn']:
  most_similar = ', '.join('%s (%.2f)' % (similar, dist) 
                           for similar, dist in word2vec_model.most_similar(word)[:8])
  print('  %s -> %s' % (word, most_similar))

def word2idx(word):
  return word2vec_model.wv.vocab[word].index
def idx2word(idx):
  return word2vec_model.wv.index2word[idx]


######################################################################################################################################################################
# create network (RNN)
######################################################################################################################################################################
model_rnn = Sequential()
model_rnn.add(Embedding(input_dim=vocab_size, output_dim=emdedding_size, 
                    weights=[pretrained_weights],trainable=False,mask_zero=True))
model_rnn.add(Masking(mask_value=0.0))
model_rnn.add(LSTM(units=emdedding_size))
model_rnn.add(Dense(units=vocab_size,activation='relu'))
model_rnn.add(Dropout(0.5))
model_rnn.add(Dense(units=vocab_size,activation='softmax'))
model_rnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])  
 
## Run model
# Split
X_train, X_test, y_train, y_test = train_test_split(pretrained_weights, X_post['sentiment'],
                                                    test_size=0.3, random_state=0)

# fit
model_rnn.fit(X_train, y_train, 
          batch_size= 128, epochs=30, verbose=2, validation_split=0.1)

# score 
score = model_rnn.evaluate(X_test, y_test, verbose=0)
print('\nScore: ', score)

print (model_rnn.summary())