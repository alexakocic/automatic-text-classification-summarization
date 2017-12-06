"""
Created on Mon Dec  4 20:52:10 2017

@author: Aleksa Kocić
"""

from collections import Counter

class NaiveBayes:
    """
    Naive Bayes classifier.
    Based on: https://web.stanford.edu/class/cs124/lec/naivebayes.pdf
    """
    
    def __class_probability(self, class_, training_vectors):
        """
        Calculate probability P(c), or frequency of documents that belong to 
        class c based on training corpus.
        
        Args:
            class_ (str): Name of a class
            training_vectors (list of tuple of dict, str): List of classified 
                                                           bag of words
        
        Returns:
            float: Probability of class
        """
        doccount = sum([1 for vector in training_vectors 
                        if vector[1] == class_])
        return doccount / len(training_vectors)
    
    def __word_if_class_probability(self, word, class_, training_vectors, vocabulary):
        """
        Calculate probability P(w|c), or frequency of word w appearing among
        all words in all documents classified with class c.
        
        Args:
            word (str): A word whose frequency is calculated
            class_ (str): Class of documents based on which word frequency
                          is calculated
            training_vectors (list of tuple of dict, str): List of classified 
                                                           bag of words
            vocabulary (set of str): Set of all words from training set
        
        Returns:
            float: Probability of word based on class
        """
        vectors = [vector[0] for vector in training_vectors
                     if vector[1] == class_]
        
        count = 0
        for vector in vectors:
            if word in vector.keys():
                count += vector[word]
        
        sum_of_words = sum([value for vector in vectors
                            for word, value in vector])
        
        # Include add-1 smoothing if word is not in any document to avoid
        # zeroing out Cmap calculation
        return (count + 1)/(sum_of_words + len(vocabulary))
    
    def __class_probabilities(self, classes, training_vectors):
        """
        Calculate probabilities P(c[i]), or frequency of documents that belong to 
        each class c[i] based on training corpus.
        
        Args:
            classes (list of str): Names of classes
            training_vectors (list of tuple of dict, str): List of classified 
                                                           bag of words
        
        Returns:
            dict of str:float values: Dictionary of classes with their respective
                                      probabilities
        """
        probabilities = dict()
        
        for class_ in classes:
            probabilities[class_] = self.__class_probability(class_, training_vectors)
        
        return probabilities
    
    def __word_probabilities_for_classes(self, vocabulary, classes, training_vectors):
        """
        Calculate probabilities P(w[i]|c[i]), or frequency of words w[i] appearing among
        all words in all documents classified with classes c[i].
        
        Args:
            vocabulary (set of str): Set of all words from training set
            classes (list of str): Names of classes
            training_vectors (list of tuple of dict, str): List of classified 
                                                           bag of words

        Returns:
            dict of str:dict of str:float values: Dictionary of every word's
                from vocabulary probabilities based on every class
        """
        probabilities = dict()
        
        for word in vocabulary:
            class_probabilities = dict()
            for class_ in classes:
                class_probabilities[class_] = self.__word_if_class_probability(word, 
                                                                              class_,
                                                                              training_vectors,
                                                                              vocabulary)
            probabilities[word] = class_probabilities
        
        return probabilities
    
    def train(self, training_vectors):
        """
        Train classification model.
        
        Args:
            training_vectors (list of tuple of dict, str): List of classified 
                                                           bag of words
        """
        vectors = [vector[0] for vector in training_vectors]
        self.classes = set([vector[1] for vector in training_vectors])
        self.vocabulary = set([word for vector in vectors for word in vector.keys()])
        
        self.word_probabilities = self.__word_probabilities_for_classes(self.vocabulary, 
                                                                   self.classes, 
                                                                   training_vectors)
        self.class_probabilities = self.__class_probabilities(self.classes, training_vectors)        

    def classify(self, document):
        """
        Use trained model to classify a document.
        
        Args:
            document (dict of str:bool pairs): Bag of words containing information
                about which word from vocabulary is present in a document
                
        Returns:
            str: Class label of document
        """
        words = [word for word in document if document[word]]
        
        max_probability = 0.0
        
        for class_ in self.classes:
            class_probability = self.class_probabilities[class_]
            word_probabilities_product = 1.0
            
            for word in words:
                word_probabilities_product *= self.word_probabilities[word][class_]
                
            class_probability *= word_probabilities_product
            
            if class_probability > max_probability:
                max_probability = class_probability
                cmap = class_
        
        return cmap