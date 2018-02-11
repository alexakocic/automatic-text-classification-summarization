"""
Created on Thu Dec 21 23:51:44 2017

@author: Aleksa Kocić

Contains generic methods for testing classification models performance.
"""
from classification.evaluation import classification_evaluation as ce

def create_classification_model_and_evaluate(classifier, train_vectors,
                                             train_labels, test_vectors,
                                             test_labels, multilabel=False, mlb=None):
    """
    Generic method for testing classification models performance.
    
    Args:
        classifier (sklearn classifier): A classification model
        train_vectors (scipy.sparse.csr.csr_matrix): Training set
        train_labels (list of str): Labels for training set
        test_vectors (scipy.sparse.csr.csr_matrix): Test set
        test_labels (list of str): Labels for test set
        multilabel (bool): Is train data multilabeled
        mlb (sklearn.preprocessing.MultiLabelBinarizer): Multilabel binarizer
    """
    classifier.fit(train_vectors, train_labels)
    predictions = classifier.predict(test_vectors)
    
    ce.evaluate_print(test_labels, predictions)
    
    if multilabel:
        test_labels = [list(label) for label in mlb.inverse_transform(test_labels)]
        predictions = [list(label) for label in mlb.inverse_transform(predictions)]
    
    return classifier, test_labels, predictions