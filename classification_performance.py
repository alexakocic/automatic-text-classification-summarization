"""
Created on Fri Dec 22 00:35:03 2017

@author: Aleksa Kocić
"""

from preprocessing.normalization import Normalizer
from preprocessing import feature_extraction as fe
from classification.data import train_test_data as ttd

def prepare_data(train_data, test_data, type_='bow', binary=False, ngram_range=(1, 1)):
    """
    Transform train and test corpuses into vectors of features ready for classification.
    
    Args:
        train_data (list of str): Corpus of documents
        test_data (list of str): Corpus of documents
        type_ (str): Type of features: Bag of Words or Tfidf
        binary (bool): Bag of Words has binary values if True, or else values are
            frequencies. Use only if type_ if 'bow'.
        ngram_range (tuple of int): Calculate features based on ngrams
        
    Returns:
        tuple of scipy.sparse.csr.csr_matrix: Train and test documents features
    """
    normalizer = Normalizer()
    
    train_data = [normalizer.normalize_text(document) for document in train_data]
    train_data = [' '.join(document) for document in train_data]
    
    if type_ == 'bow':
        vectorizer, train_vectors = fe.scikit_bag_of_words_frequencies(corpus=train_data, 
                                                                          binary=binary,
                                                                          ngram_range=ngram_range,
                                                                          normalize=False)
    elif type_ == 'tfidf':
        vectorizer, train_vectors = fe.scikit_bag_of_words_tfidf(corpus=train_data,
                                                                    ngram_range=ngram_range,
                                                                    normalize=False)
    else:
        raise ValueError("Wrong value for type_ parameter. Type help(prepare_data) to see the list of possible values.")
    
    test_vectors = vectorizer.transform(test_data)
    return train_vectors, test_vectors