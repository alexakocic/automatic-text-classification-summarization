"""
Created on Mon Nov 27 19:01:47 2017

@author: Aleksa Kocić

Feature extraction module. Contains methods for feature extraction from
normalized text.
"""

import math
from collections import Counter

def bag_of_words_simple(normalized_text, words):
    """
    Creates a bag of words from normalized text that are present in a
    predefined set of words.
    
    Args:
        normalized_text (list of str): Normalized text to be transformed
                                       into bag of words
        words (set of str): Predefined set of words
    
    Returns:
        set of str: Set of words from normalized text which are present
                    in predefined set of words
    """
    return set([word for word in normalized_text if word in words])

def bag_of_words_frequencies(normalized_text):
    """
    Creates a bag of words from normalized text which represents
    word:number of word appearances in text.
    
    Args:
        normalized_text (list of str): Normalized text to be transformed
                                       into bag of words
    Returns:
        dict of str:int pairs: Dictionary of word:number of word appearances 
                               in text
    """
    counted_words = Counter(normalized_text)
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