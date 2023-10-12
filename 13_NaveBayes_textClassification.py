# -*- coding: utf-8 -*-
"""
@author: Pachimatla Rajesh

-Naive Bayes algorithm application

"""
### import the libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from sklearn.datasets import fetch_20newsgroups
#it will fetch 20 news groups from the library, very ones

from sklearn.feature_extraction.text import TfidfVectorizer
"""
The TfidfVectorizer is text data preprocessing and feature extraction, 
particularly in NLP and machine learning tasks.
It is a numerical statistic that reflects the importance of a word in a document 
relative to a collection of documents (corpus).
"""
from sklearn.naive_bayes import MultinomialNB

from sklearn.pipeline import make_pipeline

"""
We create a machine learning pipeline using make_pipeline. 
The pipeline consists of two main steps:

TfidfVectorizer(): This step converts text data into TF-IDF (term frequency - 
inverse document frequency features. MultinomialNB(): This step is a 
Multinomial Naive Bayes classifier, often used for text classification tasks.
"""
from sklearn.metrics import confusion_matrix

data = fetch_20newsgroups()

data.target_names
#gives different catagories of the news

categories = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 
              'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 
              'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 
              'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 
              'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 
              'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 
              'talk.religion.misc']

#lets create train and test subsets from the database
train = fetch_20newsgroups(subset='train', categories=categories)

test = fetch_20newsgroups(subset='test', categories=categories)

print(train.data[5])

#Creating a model on multinomial naive bayes 
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
#A pipe line is created here

model.fit(train.data, train.target)

labels = model.predict(test.data)

mat = confusion_matrix(test.target, labels)

sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
                xticklabels=train.target_names, yticklabels=train.target_names)
plt.xlabel('true label')
plt.ylabel('predicted label');
    
def predict_category(s, train=train, model=model):
    pred = model.predict([s])
    return train.target_names[pred[0]]

predict_category('Jesus Christ')
#soc.religion.christian
predict_category('Sending load to International Space Station ISS')
#sci.space
predict_category('Suzuki Hayabusa is a very fast motorcycle')
#rec.motorcycles
predict_category('Audi is better than BMW')
#rec.autos
########### end of Script ############33
    