"""
Created on Fri Dec  1 17:40:29 2017

@author: Aleksa Kocić

Feature extraction module. Contains methods for feature extraction from 
scikit-learn library.
"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from feature_extraction import get_ngrams
from nltk.stem import PorterStemmer

def scikit_bag_of_words_frequencies(document, ngram_range=(1, 1), 
                                    binary=False):
    """
    Get bag of words with frequencies based on a raw document of text.
    
    Args:
        document (list of str): Normalized text to be transformed
                                into bag of words
        ngram_range (tuple of int, int): Start and end range for ngrams
        binary (bool): True if only indicator of presence of word in document
                       is needed, else False
    
    Returns:
        sklearn.feature_extraction.text.CountVectorizer: Contains information
            about bag of words, such as names of features
        scipy.sparse.csr.csr_matrix: Feature matrix where every word is
            represented by a row in a matrix and every column is a word
            of a document
    """
    stemmer = PorterStemmer()
    analyzer = CountVectorizer(stop_words='english',
                               ngram_range=ngram_range).build_analyzer()
    
    def stemmed_words(document):
        return (stemmer.stem(word) for word in analyzer(document))
    
    count_vectorizer = CountVectorizer(analyzer=stemmed_words, binary=binary)
    
    bag_of_words = count_vectorizer.fit_transform(document)
    return count_vectorizer, bag_of_words

def scikit_bag_of_words_tfidf(document, ngram_range=(1, 1)):
    """
    Get bag of words with tfidf based on a raw document of text.
    
    Args:
        document (list of str): Normalized text to be transformed
                                into bag of words
        ngram_range (tuple of int, int): Start and end range for ngrams
    
    Returns:
        sklearn.feature_extraction.text.CountVectorizer: Contains information
            about bag of words, such as names of features
        scipy.sparse.csr.csr_matrix: Feature matrix where every word is
            represented by a row in a matrix and every column is a word
            of a document
    """
    stemmer = PorterStemmer()
    analyzer = TfidfVectorizer(stop_words='english',
                               ngram_range=ngram_range).build_analyzer()
    
    def stemmed_words(document):
        return (stemmer.stem(word) for word in analyzer(document))
    
    tfidf_vectorizer = TfidfVectorizer(analyzer=stemmed_words)
    
    bag_of_words = tfidf_vectorizer.fit_transform(document)
    return tfidf_vectorizer, bag_of_words