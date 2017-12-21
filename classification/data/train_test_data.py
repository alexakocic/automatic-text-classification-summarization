"""
Created on Thu Dec 21 17:02:42 2017

@author: Aleksa Kocić

Contains methods for getting data from different corpora.
"""

from sklearn.datasets import fetch_20newsgroups
from sklearn.cross_validation import train_test_split
from nltk.corpus import movie_reviews
from nltk.corpus import reuters
from sklearn.preprocessing import MultiLabelBinarizer

def get_20newsgroups(test_data_proportion=0.3):
    """
    Get 20 Newsgroups corpus dataset
    """
    dataset = fetch_20newsgroups(subset='all', shuffle=True, 
                              remove=('headers', 'footers', 'quotes'))
    corpus, labels = dataset.data, dataset.target
    corpus, labels = remove_empty_documents(corpus, labels)
    labels = [dataset.target_names[label] for label in labels]
    
    train_data, test_data, train_labels, test_labels = prepare_datasets(corpus,
                                                                        labels,
                                                                        test_data_proportion)
    return train_data, test_data, train_labels, test_labels


def get_movie_reviews(test_data_proportion=0.3):
    """
    Get Movie Reviews corpus dataset
    """
    tuples = [(movie_reviews.raw(fileid), category)
              for category in movie_reviews.categories()
              for fileid in movie_reviews.fileids(category)]
    
    corpus = [tuple_[0] for tuple_ in tuples]
    labels = [tuple_[1] for tuple_ in tuples]
    
    corpus, labels = remove_empty_documents(corpus, labels)
    
    train_data, test_data, train_labels, test_labels = prepare_datasets(corpus,
                                                                        labels,
                                                                        test_data_proportion)
    return train_data, test_data, train_labels, test_labels

def get_reuters(test_data_proportion=0.3):
    """
    Get Reuters corpus dataset
    """
    tuples = [(reuters.raw(fileid), reuters.categories(fileid)) for fileid in reuters.fileids()]
    corpus = [tuple_[0] for tuple_ in tuples]
    labels = [tuple_[1] for tuple_ in tuples]
    
    corpus, labels = remove_empty_documents(corpus, labels)
    
    train_data, test_data, train_labels, test_labels = prepare_datasets(corpus,
                                                                        labels,
                                                                        test_data_proportion)
    return train_data, test_data, train_labels, test_labels

def prepare_datasets(corpus, labels, test_data_proportion=0.3):
    """
    Split documents and their labels into training and test sets.
    
    Args:
        corpus (list of str): List of documents
        labels (list of str): List of document labels
        test_data_proportion (float): A percentage of corpus which will be test data
    """
    train_data, test_data, train_labels, test_labels = train_test_split(corpus,
                                                                        labels,
                                                                        test_size=test_data_proportion,
                                                                        random_state=42)
    return train_data, test_data, train_labels, test_labels

def remove_empty_documents(corpus, labels):
    """
    Remove empty documents from corpus.
    
    Args:
        corpus (list of str): List of documents
        labels (list of str): List of document labels
    """
    filtered_corpus = list()
    filtered_labels = list()
    
    for doc, label in zip(corpus, labels):
        if doc.strip():
            filtered_corpus.append(doc)
            filtered_labels.append(label)
    
    return filtered_corpus, filtered_labels