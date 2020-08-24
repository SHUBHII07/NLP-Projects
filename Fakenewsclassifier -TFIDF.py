# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 10:31:26 2020

@author: Shubhangi sakarkar
"""

import pandas as pd
import numpy as np
df=pd.read_csv('train.csv',)


## data preprocessing

## checking for null values
df.isnull().sum()
## droping the missing values
df.dropna(inplace=True)

## reseting the index
messages=df.copy()
messages.reset_index(inplace=True)


##removing the characters that are not required
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
ps=PorterStemmer()
corpus=[]
for i in range(0, len(messages)):
     review = re.sub('[^a-zA-Z]', ' ', messages['text'][i])
     review = review.lower()
     review = review.split()
    
     review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
     review = ' '.join(review)
     corpus.append(review)

## creating bag of words model to extract features from text
     
from sklearn.feature_extraction.text import TfidfVectorizer
cv=TfidfVectorizer(max_features=5000,ngram_range=(1,3))
X=cv.fit_transform(corpus).toarray()
y=messages.label


## training testing
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)


## get the parameters and feature name
cv.get_feature_names()
cv.get_params()

##final dataframwe
final_df=pd.DataFrame(X_train,columns=cv.get_feature_names())


## Creating machine learning model
from sklearn.naive_bayes import MultinomialNB
clf=MultinomialNB()
clf.fit(X_train,y_train)
y_predict=clf.predict(X_test)


## measuring the performance of classifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
acc=accuracy_score(y_test,y_predict)
cr=classification_report(y_test, y_predict)
cm=confusion_matrix(y_test, y_predict)


## Hypertuning the parameters
classifier=MultinomialNB(alpha=0.1)
previous_score=0
for alpha in np.arange(0,1,.1):
    sub_classifier=MultinomialNB(alpha=alpha)
    sub_classifier.fit(X_train,y_train)
    y_prd=sub_classifier.predict(X_test)
    score = accuracy_score(y_test, y_prd)
    if score>previous_score:
        classifier=sub_classifier
    print("Alpha: {}, Score : {}".format(alpha,score))

    


#### Passive Aggressive classifier
from sklearn.linear_model import PassiveAggressiveClassifier
clf2=PassiveAggressiveClassifier()
clf2.fit(X_train,y_train)
y_pred=clf2.predict(X_test)

## performance measurement
ac2=accuracy_score(y_test,y_pred)
cm2=confusion_matrix(y_test,y_pred)
cr2=classification_report(y_test,y_pred)
















