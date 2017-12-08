# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 21:05:44 2017

@author: Aleksa Kocić
"""

from sklearn.datasets import fetch_20newsgroups

data = fetch_20newsgroups(subset='all', shuffle=True, remove=('headers',
                                                              'footers',
                                                              'quotes'))

def remove_empty_docs(corpus, labels):
    filtered_corpus = []
    filtered_labels = []
    for doc, label in zip(corpus, labels):
        if doc.strip():
            filtered_corpus.append(doc)
            filtered_labels.append(label)
    
    return filtered_corpus, filtered_labels

print(data)