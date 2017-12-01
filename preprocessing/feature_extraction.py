"""
Created on Mon Nov 27 19:01:47 2017

@author: Aleksa Kocić

Feature extraction module. Contains methods for feature extraction from
normalized text.
"""

import math
from nltk import ngrams
from collections import Counter

def bag_of_words_existence(document, words):
    """
    Creates a bag of words from normalized text that are present in a
    predefined set of words.
    
    Args:
        document (list of str): Normalized text to be transformed
                                into bag of words
        words (set of str): Predefined set of words
    
    Returns:
        set of str: Set of words from normalized text which are present
                    in predefined set of words
    """
    return set([word for word in document if word in words])

def bag_of_words_frequencies(document):
    """
    Creates a bag of words from normalized text which represents
    word:number of word appearances in text.
    
    Args:
        document (list of str): Normalized text to be transformed
                                into bag of words
    Returns:
        dict of str:int pairs: Dictionary of word:number of word appearances 
                               in text
    """
    counted_words = Counter(document)
    return dict(counted_words)

def term_frequency(term, document):
    """
    Calculate frequency of a term in a document, normalized by number of terms
    in a document.
    
    Args:
        term (str): A term for which frequency is calculated
        document (list of str): Document containing normalized text
        
    Returns:
        float: Normalized frequency of a term in a document
    """    
    return sum([1 for word in document if word == term]) / len(document)

def _inverse_document_frequency(term, corpus):
    """
    Calculate inverse document frequency for a term in a corpus of documents.
    
    Args:
        term (str): A term for which frequency is calculated
        corpus (list of list of str): List of documents containing normalized
                                      text
    
    Returns:
        float: Inverse document frequency of a term in a corpus    
    """
    # Add 1 to both nominator and denominator to avoid zero division:
    # assume existence of a document in corpus to which every term belongs to
    return math.log((len(corpus) + 1)/(1 + sum([1 for document in corpus
                                                  if term in set(document)])))

def tfidf(term, document, corpus):
    """
    Calculate tfidf value for a term for specific document in corpus.
    
    Args:
        term (str): A term for which frequency is calculated
        document (list of str): Document containing normalized text
        corpus (list of list of str): List of documents containing normalized
                                      text
    
    Returns:
        float: Tfidf measure of term in a document which belongs to specific
               corpus
    """
    tf = term_frequency(term, document)
    idf = _inverse_document_frequency(term, corpus)
    return tf * idf

def bag_of_words_tfidf(document, corpus):
    """
    Creates a bag of words from normalized text which represents
    word:tfidf of word in a document.
    
    Args:
        document (list of str): Document containing normalized text
        corpus (list of list of str): List of documents containing normalized
                                      text
    Returns:
        dict of str:float pairs: Dictionary of word:tfidf measure of a word 
                                 in text
    """
    document = set(document)
    bag_of_words = dict()
    
    for word in document:
        bag_of_words[word] = tfidf(word, document, corpus)
    
    return bag_of_words

def get_ngrams(document, n):
    """
    Get ngram from a document of normalized text.
    
    Args:
        document (list of str): Document containing normalized text
        n (int): Number of words in a ngram
    
    Returns:
        list of str: List of ngrams from a document
    """
    return list(ngrams(document, n))

def bag_of_ngrams_simple(document, ngrams, n):
    """
    Creates a bag of ngrams from normalized text that are present in predefined
    set of ngrams.
    
    Args:
        document (list of str): Document containing normalized text
        ngrams (set of tuple of str): Predefined set of ngrams
        n (int): Length of ngram
    Returns:
        set of tuple of str: Set of ngrams present in predefined set of ngrams
    """
    document = get_ngrams(document, n)
    return {ngram for ngram in document if ngram in ngrams}

def bag_of_ngrams_frequencies(document, n):
    """
    Creates a bag of ngrams from normalized text which represents
    ngram:number of ngram appearances in text.
    
    Args:
        document (list of str): Normalized text to be transformed
                                into bag of ngrams
        n (int): Length of ngram
    Returns:
        dict of tuple of str:int pairs: Dictionary of ngram:number of ngram 
                                        appearances in text
    """
    counted_freqs = Counter(get_ngrams(document, n))
    return dict(counted_freqs)

def _ngram_term_frequency(ngram_term, document, n):
    """
    Calculate frequency of a ngram in a document, normalized by number of 
    ngrams in a document.
    
    Args:
        ngram (list str): A ngram for which frequency is calculated
        document (list of str): Document containing normalized text
        n (int): Length of ngram
        
    Returns:
        float: Normalized frequency of a ngram in a document
    """
    return sum([1 for ngram in get_ngrams(document, n)
               if ngram == ngram_term])
    
def _ngram_inverse_document_frequency(ngram, corpus, n):
    """
    Calculate inverse document frequency for a ngram in a corpus of documents.
    
    Args:
        ngram (list ofstr): A ngram for which frequency is calculated
        corpus (list of list of str): List of documents containing normalized
                                      text
        n (int): Length of ngram
    
    Returns:
        float: Inverse document frequency of a ngram in a corpus    
    """
    # Add 1 to both nominator and denominator to avoid zero division:
    # assume existence of a document in corpus to which every ngram belongs to
    return math.log((len(corpus) + 1)/(1 + sum([1 for document in corpus
                                                  if ngram in
                                                  set(
                                                    get_ngrams(document, n)
                                                    )])))

def _ngram_tfidf(ngram, document, corpus, n):
    """
     Calculate tfidf value for a ngram for specific document in corpus.
    
    Args:
        ngram (list of str): A ngram for which frequency is calculated
        document (list of str): Document containing normalized text
        corpus (list of list of str): List of documents containing normalized
                                      text
        n (int): Length of ngram
    
    Returns:
        float: Tfidf measure of ngram in a document which belongs to specific
               corpus
    """
    return _ngram_term_frequency(ngram, document, n) * \
           _ngram_inverse_document_frequency(ngram, corpus, n)

def bag_of_ngrams_tfidf(document, corpus, n):
    """
    Creates a bag of ngrams from normalized text which represents
    ngram:tfidf of a ngram in a document.
    
    Args:
        document (list of str): Document containing normalized text
        corpus (list of list of str): List of documents containing normalized
                                      text
        n (int): Length of ngram
        
    Returns:
        dict of tuple of str:float pairs: Dictionary of ngram:tfidf measure of 
                                         a word in text
    """
    document_ngrams = set(get_ngrams(document, n))
    bag_of_words = dict()
    
    for ngram in document_ngrams:
        bag_of_words[ngram] = _ngram_tfidf(ngram, document, corpus, n)
    
    return bag_of_words