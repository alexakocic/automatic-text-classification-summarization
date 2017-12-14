"""
Created on Sun Dec  3 21:19:06 2017

@author: Aleksa Kocić

Feature extraction module. Contains class for converting corpus of text 
documents into feature matrix (list of feature vectors).
"""

from normalization import Normalizer
import feature_extraction
from feature_extraction_thirdparty import scikit_bag_of_words_frequencies
from feature_extraction_thirdparty import scikit_bag_of_words_tfidf

class FeatureExtractor:
    """
    Contains methods for corpus normalization.
    """
    def __init__(self):
        # Text normalizer
        self.normalizer = Normalizer()
    
    def bag_of_words(self, corpus, ngram_range=(1, 1), type_="binary"):
        """
        Generate bag of words for each document of a corpus.
        
        Args:
            corpus (list of str): List of documents
            ngram_range (tuple of int): Minimum and maximum size of ngrams in text
                                        used only if type is *-ngram
            ngram_size (int): Size of a ngram
            tupe (int): Type of bag of words:
                            - binary
                            - frequency
                            - tfidf
                            - binary-ngram
                            - frequency-ngram
                            - tfidf-ngram
            
            Returns:
                list of str/tuple of str:int pairs: Bag of words/ngrams
        """
        corpus = [self.normalizer.normalize_text(document) for document in corpus]
        
        if type_ == "binary":
            bag_of_words = feature_extraction.bag_of_words_binary_corpus(corpus)
        elif type_ == "frequency":
            bag_of_words = feature_extraction.bag_of_words_frequencies_corpus(corpus)
        elif type_ == "tfidf":
            bag_of_words = feature_extraction.bag_of_words_tfidf_corpus(corpus)
        elif type_ == "binary-ngram":
            bag_of_words = feature_extraction.bag_of_ngrams_binary_corpus(corpus, ngram_range[0])
        elif type_ == "frequency-ngram":
            bag_of_words = feature_extraction.bag_of_ngrams_frequencies_range_corpus(corpus, ngram_range)
        elif type_ == "tfidf-ngram":
            bag_of_words = feature_extraction.bag_of_ngrams_tfidf_range_corpus(corpus, ngram_range)
        else:
            raise ValueError("""Wrong type_ input. Type help(bag_of_words) to see supported types.""")
        
        return bag_of_words
    
    def feature_matrix(self, corpus, ngram_range=(1, 1), type_="binary"):
        """
        Generate feature matrix for each document of a corpus.
        
        Args:
            corpus (list of str): List of documents
            ngram_range (tuple of int): Minimum and maximum size of ngrams in text
                                        used only if type is *-ngram
            ngram_size (int): Size of a ngram
            tupe (int): Type of bag of words:
                            - binary
                            - frequency
                            - tfidf
                            - binary-ngram
                            - frequency-ngram
                            - tfidf-ngram
            
            Returns:
                ###
        """
        bag_of_words = self.bag_of_words(corpus, ngram_range, type_)
        vocabulary = dict()
        
        id_ = 0
        
        for document in bag_of_words:
            for word in document:
                if not word in vocabulary.keys():
                    vocabulary[word] = id_
                    id_ += 1
        
        sorted_vocabulary = sorted(vocabulary.items(), key = lambda x:x[1])
        feature_matrix = list()
        
        for document in bag_of_words:
            vector = list()
            for word in sorted_vocabulary:
                try:
                    vector.append(document[word[0]])
                except KeyError:
                    # If word is not present in bag of words, fill respective
                    # column with default value
                    if type_.startswith("binary"):
                        vector.append(False)
                    elif type_.startswith("frequency"):
                        vector.append(0)
                    elif type_.startswith("tfidf"):
                        vector.append(0.0)
            feature_matrix.append(vector) 
        
        return vocabulary, feature_matrix
    
    def feature_matrix_sklearn(self, corpus, ngram_range=(1, 1), binary=False,
                               type_=0):
        """
        Generate feature matrix for each document of a corpus.
        
        Args:
            corpus (list of str): Raw documents to be transformed into matrix of
                                  bags of words
            ngram_range (tuple of int, int): Start and end range for ngrams
            binary (bool): True if only indicator of presence of word in document
                           is needed, else False
            type_ (int): 0 - frequencies, 1 - tfidf
        
        Returns:
            list of dict of str/tuple of str:int pairs: Matrix of 
                word/ngram:frequency or tfidf measure of a word in text
        """
        if type_ == 0:
            count_vectorizer, feature_matrix = scikit_bag_of_words_frequencies(corpus,
                                                                         ngram_range,
                                                                         binary
                                                                         )
        elif type_ == 1:
            count_vectorizer, feature_matrix = scikit_bag_of_words_tfidf(corpus,
                                                                       ngram_range
                                                                       )
        return count_vectorizer.vocabulary_, feature_matrix.toarray(), feature_matrix.toarray().tolist()