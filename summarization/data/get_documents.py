"""
Created on Mon Dec 25 20:29:34 2017

@author: Aleksa Kocić

Data fetching module. Contains methods for gathering data for test summarization.
"""

from sklearn.datasets import fetch_20newsgroups
from nltk.corpus import movie_reviews
from nltk.corpus import reuters
from random import shuffle

def get_documents():
    """
    Get documents from 20 News Groups, Movie Reviews and Reuters corpora.
    
    Returns:
        list of str: Small subset of documents from News Groups, Movie Reviews 
            and Reuters corpora
    """
    dataset = fetch_20newsgroups(subset='all', shuffle=True, 
                              remove=('headers', 'footers', 'quotes'))
    corpus_20newsgroups = dataset.data[:5]
    
    tuples = [(movie_reviews.raw(fileid), category)
              for category in movie_reviews.categories()
              for fileid in movie_reviews.fileids(category)]
    
    corpus_movies = [tuple_[0] for tuple_ in tuples]
    shuffle(corpus_movies)
    corpus_movies = corpus_movies[:5]
    
    tuples = [(reuters.raw(fileid), reuters.categories(fileid)) for fileid in reuters.fileids()]
    corpus_reuters = [tuple_[0] for tuple_ in tuples]
    shuffle(corpus_reuters)
    corpus_reuters = corpus_reuters[:5]
    
    corpus = list()
    corpus.extend(corpus_20newsgroups)
    corpus.extend(corpus_movies)
    corpus.extend(corpus_reuters)
    
    return corpus