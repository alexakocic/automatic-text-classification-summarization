"""
Created on Fri Dec 22 00:35:03 2017

@author: Aleksa Kocić

Perform classification methods evaluation with different datasets and parameters.
"""

from preprocessing.normalization import Normalizer
from preprocessing import feature_extraction as fe
from classification.data import train_test_data as data
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MultiLabelBinarizer
from classification.classification import create_classification_model_and_evaluate
import numpy as np

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

def convert_multiclass_list_to_labels(predictions, mlb):
    pass

def generate_data_and_test_classifiers(classifier, test_set_percentage, data_getters, messages, multilabel=None):
    """
    Test classifier performances based on different datasets and parameters.
    
    Args:
        classifier: Classification model
        test_set_percentage (float): Percentage of test set from whole corpus
        data_getters (list of functions): List of function to call for fetching data
        messages (list of str): List of messages to print for each data getter
        multilabel (list of bool): List of boolean values indicating whether 
    """
    ngram_ranges = [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)]
    
    print(np.round(test_set_percentage * 100, 2), "test set size")
    
    i = 0
    for data_getter in data_getters:
        print(messages[i])
        train_data, test_data, train_labels, test_labels = data_getter(test_set_percentage)
        
        is_multilabel = False
        
        if multilabel and multilabel[i]:
            mlb = MultiLabelBinarizer()
            mlb.fit(train_labels + test_labels)
            train_labels = mlb.transform(train_labels)
            test_labels = mlb.transform(test_labels)
            is_multilabel = True
        
        normalizer = Normalizer()
    
        train_data = [normalizer.normalize_text(document) for document in train_data]
        train_data = [' '.join(document) for document in train_data]
    
        test_data = [normalizer.normalize_text(document) for document in test_data]
        test_data = [' '.join(document) for document in test_data]
        
        print("Bag of Words frequencies:")
        for ngram_range in ngram_ranges:
            print("Ngram range", ngram_range)
            train_vectors, test_vectors = prepare_data(train_data, test_data, type_='bow', 
                                                       binary=False, ngram_range=ngram_range)
            if not is_multilabel:
                create_classification_model_and_evaluate(classifier, train_vectors, train_labels, 
                                                         test_vectors, test_labels)
            else:
                create_classification_model_and_evaluate(classifier, train_vectors, train_labels, 
                                                         test_vectors, test_labels, multilabel=True,
                                                         mlb=mlb)   
                
            print()
        
        print("---------------------------------------------------\n")
        
        print("Bag of Words binary:")
        for ngram_range in ngram_ranges:
            print("Ngram range", ngram_range)
            train_vectors, test_vectors = prepare_data(train_data, test_data, type_='bow', 
                                                       binary=True, ngram_range=ngram_range)
            if not is_multilabel:
                create_classification_model_and_evaluate(classifier, train_vectors, train_labels, 
                                                         test_vectors, test_labels)
            else:
                create_classification_model_and_evaluate(classifier, train_vectors, train_labels, 
                                                         test_vectors, test_labels, multilabel=True,
                                                         mlb=mlb)   
            print()
            
        print("---------------------------------------------------\n")
        
        print("Tfidf:")
        for ngram_range in ngram_ranges:
            print("Ngram range", ngram_range)
            train_vectors, test_vectors = prepare_data(train_data, test_data, type_='tfidf', 
                                                       ngram_range=ngram_range)
            if not is_multilabel:
                create_classification_model_and_evaluate(classifier, train_vectors, train_labels, 
                                                         test_vectors, test_labels)
            else:
                create_classification_model_and_evaluate(classifier, train_vectors, train_labels, 
                                                         test_vectors, test_labels, multilabel=True,
                                                         mlb=mlb)
            print()
        i += 1


print("Multinomial Naive Bayes - 70% train 30% test set:")
generate_data_and_test_classifiers(MultinomialNB(), 0.3, [data.get_movie_reviews, data.get_20newsgroups], ["Movie reviews dataset", "20 News Groups dataset"])
print("Multinomial Naive Bayes - 80% train 20% test set:")
generate_data_and_test_classifiers(MultinomialNB(), 0.2, [data.get_movie_reviews, data.get_20newsgroups], ["Movie reviews dataset", "20 News Groups dataset"])
print("Multinomial Naive Bayes - 90% train 10% test set:")
generate_data_and_test_classifiers(MultinomialNB(), 0.1, [data.get_movie_reviews, data.get_20newsgroups], ["Movie reviews dataset", "20 News Groups dataset"])

print("K Nearest Neighbors with default settings - 70% train 30% test set:")
generate_data_and_test_classifiers(KNeighborsClassifier(), 0.3, [data.get_movie_reviews, data.get_20newsgroups, data.get_reuters], ["Movie reviews dataset", "20 News Groups dataset", "Reuters dataset"], [False, False, True])
print("K Nearest Neighbors with default settings - 80% train 20% test set:")
generate_data_and_test_classifiers(KNeighborsClassifier(), 0.2, [data.get_movie_reviews, data.get_20newsgroups, data.get_reuters], ["Movie reviews dataset", "20 News Groups dataset", "Reuters dataset"], [False, False, True])
print("K Nearest Neighbors with default settings - 90% train 10% test set:")
generate_data_and_test_classifiers(KNeighborsClassifier(), 0.1, [data.get_movie_reviews, data.get_20newsgroups, data.get_reuters], ["Movie reviews dataset", "20 News Groups dataset", "Reuters dataset"], [False, False, True])

print("K Nearest Neighbors with k=3 and weights - 70% train 30% test set:")
generate_data_and_test_classifiers(KNeighborsClassifier(n_neighbors=3, weights='distance'), 0.3, [data.get_movie_reviews, data.get_20newsgroups, data.get_reuters], ["Movie reviews dataset", "20 News Groups dataset", "Reuters dataset"], [False, False, True])
print("K Nearest Neighbors with k=3 and weights - 80% train 20% test set:")
generate_data_and_test_classifiers(KNeighborsClassifier(n_neighbors=3, weights='distance'), 0.2, [data.get_movie_reviews, data.get_20newsgroups, data.get_reuters], ["Movie reviews dataset", "20 News Groups dataset", "Reuters dataset"], [False, False, True])
print("K Nearest Neighbors with k=3 and weights - 90% train 10% test set:")
generate_data_and_test_classifiers(KNeighborsClassifier(n_neighbors=3, weights='distance'), 0.1, [data.get_movie_reviews, data.get_20newsgroups, data.get_reuters], ["Movie reviews dataset", "20 News Groups dataset", "Reuters dataset"], [False, False, True])

print("Decision Tree with gini impurity - 70% train 30% test set:")
generate_data_and_test_classifiers(DecisionTreeClassifier(), 0.3, [data.get_movie_reviews, data.get_20newsgroups, data.get_reuters], ["Movie reviews dataset", "20 News Groups dataset", "Reuters dataset"], [False, False, True])
print("Decision Tree with gini impurity - 80% train 20% test set:")
generate_data_and_test_classifiers(DecisionTreeClassifier(), 0.2, [data.get_movie_reviews, data.get_20newsgroups, data.get_reuters], ["Movie reviews dataset", "20 News Groups dataset", "Reuters dataset"], [False, False, True])
print("Decision Tree with gini impurity - 90% train 10% test set:")
generate_data_and_test_classifiers(DecisionTreeClassifier(), 0.1, [data.get_movie_reviews, data.get_20newsgroups, data.get_reuters], ["Movie reviews dataset", "20 News Groups dataset", "Reuters dataset"], [False, False, True])

print("Decision Tree with entropy - 70% train 30% test set:")
generate_data_and_test_classifiers(DecisionTreeClassifier(criterion='entropy'), 0.3, [data.get_movie_reviews, data.get_20newsgroups, data.get_reuters], ["Movie reviews dataset", "20 News Groups dataset", "Reuters dataset"], [False, False, True])
print("Decision Tree with entropy - 80% train 20% test set:")
generate_data_and_test_classifiers(DecisionTreeClassifier(criterion='entropy'), 0.2, [data.get_movie_reviews, data.get_20newsgroups, data.get_reuters], ["Movie reviews dataset", "20 News Groups dataset", "Reuters dataset"], [False, False, True])
print("Decision Tree with entropy - 90% train 10% test set:")
generate_data_and_test_classifiers(DecisionTreeClassifier(criterion='entropy'), 0.1, [data.get_movie_reviews, data.get_20newsgroups, data.get_reuters], ["Movie reviews dataset", "20 News Groups dataset", "Reuters dataset"], [False, False, True])

print("Support Vector Machine - 70% train 30% test set:")
generate_data_and_test_classifiers(SVC(criterion='entropy'), 0.3, [data.get_movie_reviews, data.get_20newsgroups], ["Movie reviews dataset", "20 News Groups dataset"])
print("Support Vector Machine - 80% train 20% test set:")
generate_data_and_test_classifiers(SVC(criterion='entropy'), 0.2, [data.get_movie_reviews, data.get_20newsgroups], ["Movie reviews dataset", "20 News Groups dataset"])
print("Support Vector Machine - 90% train 10% test set:")
generate_data_and_test_classifiers(SVC(criterion='entropy'), 0.1, [data.get_movie_reviews, data.get_20newsgroups], ["Movie reviews dataset", "20 News Groups dataset"])