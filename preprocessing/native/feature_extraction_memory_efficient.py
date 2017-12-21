"""
Created on Wed Dec 13 18:33:03 2017

@author: Aleksa Kocić

Feature extraction module. Contains methods for feature extraction
from normalized text.
"""

import numpy as np
from nltk import ngrams
from collections import Counter
from scipy.sparse import csr_matrix
import math

class FeatureExtractor:
    """
    Contains methods for extracting features from document based on a training set.
    """
    def __init__(self, vocabulary, ngram_range):
        self.vocabulary = vocabulary
        self.ngram_range = ngram_range
    
    def extract_features(document):
        """
        Extract features from document based on a training set.
        
        Args:
            document (list of str): Normalized document
        
        Returns:
            scipy.sparse.csr_matrix of int/float/bool values: Sparse list (1D matrix) of features
        """
        raise NotImplementedError()

class BagOfWordsExtractor(FeatureExtractor):
    """
    Contains methods for extracting bag of words from document based on a training set.
    """
    def __init__(self, vocabulary, binary, ngram_range):
        super().__init__(vocabulary, ngram_range)
        self.binary = binary
        
    def extract_features(self, document):
        """
        Extract features from document based on a training set.
        
        Args:
            document (list of str): Normalized document
        
        Returns:
            scipy.sparse.csr_matrix of int/bool values: Sparse list (1D matrix) of features
        """
        if self.ngram_range != (1, 1):
            document = _get_ngram_range(document, self.ngram_range)
            
        if not self.binary:
            word_count = Counter(document)
            row = [0] * len(self.vocabulary)
        else:
            word_count = dict()
            for word in set(document):
                word_count[word] = word in self.vocabulary.keys()
            row = [False] * len(self.vocabulary)
            
        for word in word_count:
            row[self.vocabulary[word]] = word_count[word]
        
        return csr_matrix(row)

class TfidfExtractor(FeatureExtractor):
    """
    Contains methods for extracting tfidf from document based on a training set.
    """
    def __init__(self, vocabulary, idf_dict, ngram_range):
        super().__init__(vocabulary, ngram_range)
        self.idf_dict = idf_dict
    
    def extract_features(self, document):
        """
        Extract features from document based on a training set.
        
        Args:
            document (list of str): Normalized document
        
        Returns:
            scipy.sparse.csr_matrix of float values: Sparse list (1D matrix) of features
        """
        if self.ngram_range != (1, 1):
            document = _get_ngram_range(document, self.ngram_range)
        
        row = [0.0] * len(self.vocabulary)
        
        for term in document:
            if term in self.vocabulary.keys():
                tf = _term_frequency(term, document)
                idf = self.idf_dict[term]
                row[self.vocabulary[term]] = tf * idf
        
        return csr_matrix(row)

def _csr_vappend(a, b):
    """ Takes in 2 csr_matrices and appends the second one to the bottom of the first one. 
    Much faster than scipy.sparse.vstack but assumes the type to be csr and overwrites
    the first matrix instead of copying it. The data, indices, and indptr still get copied.
        
    Args:
        a (scipy.sparse.csr_matrix): Sparse matrix
        b (scipy.sparse.csr_matrix): Sparse matrix
            
    Returns:
        scipy.sparse.csr_matrix: Stacked sparce matrix
    """
    if a == None:
        return b
    elif b == None:
        return a
    
    a.data = np.hstack((a.data,b.data))
    a.indices = np.hstack((a.indices,b.indices))
    a.indptr = np.hstack((a.indptr,(b.indptr + a.nnz)[1:]))
    a._shape = (a.shape[0]+b.shape[0],b.shape[1])
    return a

def _get_ngrams(document, n):
    """
    Get ngram from a document of normalized text.
    
    Args:
        document (list of str): Document containing normalized text
        n (int): Number of words in a ngram
    
    Returns:
        list of tuple of str: List of ngrams from a document
    """
    return list(ngrams(document, n))

def _get_ngram_range(document, ngram_range):
    """
    Get ngrams from a document of normalized text.
    
    Args:
        document (list of str): Document containing normalized text
        ngram_range (tuple of int, int): Ngram range
    
    Returns:
        list of tuple of str: List of ngrams from a document
    """
    min_ = ngram_range[0]
    max_ = ngram_range[1]
    ngrams_list = list()
    n = min_
    
    while n <= max_:
        ngrams_list.extend(_get_ngrams(document, n))
        n += 1
    
    return ngrams_list

def _create_vocabulary(corpus, ngram_range=(1, 1)):
    """
    Create vocabulary with word - id pairs.
    
    Args:
        corpus (list of list of str): List of normalized documents
        ngram_range (tuple of int, int): Ngram range
    
    Returns:
        dict of str:int pairs: Vocabulary with word - id pairs
    """
    id_ = 0
    vocabulary = dict()
    
    # Generate the vocabulary for matrix
    for document in corpus:
        if ngram_range != (1, 1):
            document = _get_ngram_range(document, ngram_range)
        for word in set(document):
            if word not in vocabulary.keys():
                vocabulary[word] = id_
                id_ += 1
                
    return vocabulary

def bag_of_words(corpus, binary=False, ngram_range=(1, 1)):
    """
    Create bag of words presentation of a corpus of normalized text.
    
    Args:
        corpus (list of list of str): List of normalized documents
        binary (bool): False means counting frequencies, True means only existence
        ngram_range (tuple of int, int): Ngram range
    
    Returns:
        scipy.sparse.csr_matrix of int/float/bool values: Sparse matrix of features
    """
    vocabulary = _create_vocabulary(corpus, ngram_range)
    feature_extractor = BagOfWordsExtractor(vocabulary, binary, ngram_range)
    features = None
    
    for document in corpus:
        if ngram_range != (1, 1):
            document = _get_ngram_range(document, ngram_range)
            
        if binary:
            word_count = dict()
            for word in set(document):
                word_count[word] = word in vocabulary.keys()
            row = [False] * len(vocabulary)
        else:
            word_count = Counter(document)
            row = [0] * len(vocabulary)
            
        for word in word_count:
            row[vocabulary[word]] = word_count[word]
        sparse_row = csr_matrix(row)
        features = _csr_vappend(features, sparse_row)
        
    return features, feature_extractor

def _term_frequency(term, document):
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

def _tfidf(term, document, idf_dict):
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
    tf = _term_frequency(term, document)
    #idf = _inverse_document_frequency(term, corpus)
    return tf * idf_dict[term]

def _tfidf_document(document, corpus, idf_dict):
    """
    Calculate tfidf dictionary for every term in a document in a given corpus.
    
    Args:
        document (list of str): Normalized document
        corpus (list of list of str): List of normalized documents
    
    Returns:
        dict of str/tuple of str: float: Tfidf dictionaries
    """
    tfidfs = dict()
    
    for term in document:
        tfidfs[term] = _tfidf(term, document, idf_dict)
    
    return tfidfs

def _idf_dict(corpus):
    words = set([word for document in corpus for word in document])
    idf_dict = dict()
    n = len(corpus)
    
    for word in words:
        idf_dict[word] = math.log((n + 1)/(1 + sum([1 for document in corpus
                                                  if word in set(document)])))
    
    return idf_dict

def tfidf(corpus, ngram_range=(1, 1)):
    """
    Calculate tfidf for every document in a corpus.
    
    Args:
        corpus (list of list of str): List of normalized documents
        ngram_range (tuple of int, int): Range of ngrams to be considered in text
    
    Returns:
        dict of str/tuple of str: float: Tfidf dictionaries
    """
    
    #save_corpus = bag_of_words(corpus, binary=True, ngram_range=ngram_range)[0]
    vocabulary = _create_vocabulary(corpus, ngram_range)
    if ngram_range != (1, 1):
        corpus = [_get_ngram_range(document, ngram_range) for document in corpus]
        
    idf_dict = _idf_dict(corpus)
    
    feature_extractor = TfidfExtractor(vocabulary, idf_dict, ngram_range)
    features = None
    
    for document in corpus:
        tfidf_ = _tfidf_document(document, corpus, idf_dict)
        row = [0.0] * len(vocabulary)
        for word, value in tfidf_.items():
            row[vocabulary[word]] = tfidf_[word]
        sparse_row = csr_matrix(row)
        features = _csr_vappend(features, sparse_row)
    
    return features, feature_extractor