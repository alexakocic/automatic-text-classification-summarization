# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 21:05:44 2017

@author: Aleksa Kocić
"""

from sklearn.datasets import fetch_20newsgroups

dataset = fetch_20newsgroups(subset='all', shuffle=True, remove=('headers',
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

corpus, labels = dataset.data, dataset.target
labels = [dataset.target_names[label] for label in labels]

corpus, labels = remove_empty_docs(corpus, labels)

normalizer = Normalizer()

print("Normalizing")
normalized_corpus = [normalizer.normalize_text(document) for document in corpus]

# normalized_corpus = [normalizer.normalize_text(document) for document in corpus]
print("Extracting features")
feature_vectors = bag_of_words_frequencies_corpus(normalized_corpus)
training_set = feature_vectors[:int(0.8 * len(feature_vectors))]
test_set = feature_vectors[int(0.8 * len(feature_vectors)):]
training_set = list(zip(training_set, labels))
naive_bayes = NaiveBayes()
print("Training")
naive_bayes.train(training_set)