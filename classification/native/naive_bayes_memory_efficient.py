"""
Created on Sat Dec 16 14:29:10 2017

@author: Aleksa Kocić

Naive Bayes classifier.
"""

from scipy.sparse import csr_matrix
from math import log
import sys

class NaiveBayes:
    """
    Naive Bayes classifier.
    Based on: https://web.stanford.edu/class/cs124/lec/naivebayes.pdf
    """
    
    def __class_probability(self, class_, classes):
        """
        Calculate probability P(c), or frequency of documents that belong to 
        class c based on training corpus.
        
        Args:
            class_ (str): Name of a class
            classes (list of str): List of classes
        
        Returns:
            float: Probability of class
        """
        return sum([1 for class__ in classes if class__ == class_]) / len(classes)
    
    def __word_if_class_probability(self, word, class_, training_vectors, classes):
        """
        Calculate probability P(w|c), or frequency of word w appearing among
        all words in all documents classified with class c.
        
        Args:
            word (str): A word whose frequency is calculated
            class_ (str): Class of documents based on which word frequency
                          is calculated
            training_vectors (scipy.sparse.csr_matrix): Training vectors
            vocabulary (dict of str:int pairs): All words from training set
                with their id-s
        
        Returns:
            float: Probability of word based on class
        """
        index = self.vocabulary[word]
        size = training_vectors.shape[0]
        
        count = 0
        sum_of_words = 0
        for i in range(size):
            if classes[i] == class_:
                count += training_vectors[i, index]
                sum_of_words += training_vectors[i, :].sum(axis=1)[0, 0]
        
        # Include add-1 smoothing if word is not in any document to avoid
        # zeroing out Cmap calculation
        return (count + 1)/(sum_of_words + len(self.vocabulary))
    
    def __class_probabilities(self, classes):
        """
        Calculate probabilities P(c[i]), or frequency of documents that belong to 
        each class c[i] based on training corpus.
        
        Args:
            classes (list of str): Names of classes
        
        Returns:
            dict of str:float values: Dictionary of classes with their respective
                                      probabilities
        """
        probabilities = dict()
        
        for class_, id_ in self.classes.items():
            probabilities[class_] = self.__class_probability(class_, classes)
        
        return probabilities
    
    def __word_probabilities_for_classes(self, training_vectors, classes):
        """
        Calculate probabilities P(w[i]|c[i]), or frequency of words w[i] appearing among
        all words in all documents classified with classes c[i].
        
        Args:
            training_vectors (scipy.sparse.csr_matrix): Training vectors
            classes (list of str): Names of classes

        Returns:
            scipy.sparse.csr_matrix: Matrix of probabilities where row represents
                word and column represents class
        """
        probabilities = list()
        ordered_words = [word for id_, word in sorted([(id_, word) for word, id_ in self.vocabulary.items()])]
        
        for word in ordered_words:
            row = [0.0] * len(self.classes)
            for class_, class_id in self.classes.items():
                row[class_id] = self.__word_if_class_probability(word, class_, training_vectors, classes)
            probabilities.append(row)
        
        return csr_matrix(probabilities)
    
    def train(self, training_vectors, classes, vocabulary):
        """
        Train classification model.
        
        Args:
            training_vectors (scipy.sparse.csr_matrix): Vectors for training
                Naive Bayes classifier
        """
        self.classes = dict()
        id_ = 0
        for class_ in set(classes): 
            self.classes[class_] = id_
            id_ += 1
            
        self.vocabulary = vocabulary
        
        self.word_probabilities = self.__word_probabilities_for_classes(training_vectors, 
                                                                   classes)
        self.class_probabilities = self.__class_probabilities(classes)
        
    def classify(self, vectors):
        """
        Use trained model to classify vectors
        
        Args:
            vectors (scipy.sparse.csr_matrix): Vectors to be classified
            
        Returns:
            str: Class label of document
        """
        classes = list()
        n = vectors.shape[0]
        m = vectors.shape[1]
        
        for i in range(n):
            max_probability = -sys.float_info.max 
            cmap = ""
            print(i)
            
            for class_, class_id in self.classes.items():
                class_probability = self.class_probabilities[class_]
                sum_ = log(class_probability)
                
                for j in range(m):
                    if vectors[i, j] > 0:
                        sum_ += log(self.word_probabilities[j, class_id])
                
                if sum_ > max_probability:
                    max_probability = sum_
                    cmap = class_
                    
            classes.append(cmap)
            
        return classes