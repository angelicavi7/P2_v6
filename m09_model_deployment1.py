#!/usr/bin/python

import pandas as pd
from sklearn.externals import joblib
import sys
import os
import numpy as np
import nltk
from nltk.stem.snowball import SnowballStemmer
import re, string

clf = joblib.load(os.path.dirname(__file__) + '/movie_genre_clf.pkl') 
vect1 = joblib.load(os.path.dirname(__file__) + '/movie_genre_vect1.pkl') 
stemmer = SnowballStemmer('english')
stopwords_per=['the']
cols = ['Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family',
        'Fntasy', 'Film-Noir', 'History', 'Horror', 'Music', 'Musical', 'Mystery', 'News', 'Romance',
        'Sci-Fi', 'Short', 'Sport', 'Thriller', 'War', 'Western']

def puntuacion(texto):
    texto1="".join([char for char in texto if char not in string.punctuation])
    return texto1

def numeracion(texto):
    texto1="".join([char for char in texto if not char.isdigit()])
    return texto1

def tokenize(text):
    tokens=re.split('\W+',text)
    return tokens

def remove_stopwords(texto):
    texto1=[word for word in texto if word not in stopwords_per]
    return texto1

def stemming_snowball(texto):
    texto1=[stemmer.stem(word) for word in texto]
    return texto1
    
def predict_genre(plot,title,year):

    # Create features
    plot= title+' '+plot
    plot=puntuacion(plot)
    plot=plot.replace(" N ", '')
    plot=plot.lower()
    plot=numeracion(plot)
    largo= len(plot)- plot.count(" ")
    plot_tokens=tokenize(plot)
    plot_tokens_ww=remove_stopwords(plot_tokens)
    plot_stemmed_snow=stemming_snowball(plot_tokens_ww)
    plot_stemmed_snow1= vect1.transform(plot_stemmed_snow)
        
    # Make prediction
    y_pred_test_genres = clf.predict_proba(plot_stemmed_snow1)
    index_=["Prob"]
    res = pd.DataFrame(y_pred_test_genres,columns=cols)[:1]
    res.index=index_
    res=res.transpose()
    res=res.sort_values(by="Prob", ascending=False)
    res=res[:3]
    res=res.drop(["Prob"],axis=1)
    res=res.transpose()
    return res