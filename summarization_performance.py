"""
Created on Mon Dec 25 21:19:37 2017

@author: Aleksa Kocić

Test performance of summarization algorithms.
"""

from summarization.summarization_algorithms import summarization_gensim
from summarization.summarization_algorithms import lsa
from summarization.summarization_algorithms import textrank
from summarization.data.get_documents import get_documents
from preprocessing.normalization import Normalizer
from preprocessing import feature_extraction as fe

normalizer = Normalizer()

corpus = get_documents()

corpus_for_summarizing = list()

for document in corpus:
    document = normalizer.tokenize_sentences(document)
    normalized_document = [normalizer.normalize_text(sent) for sent in document]
    normalized_document = [' '.join(sent) for sent in normalized_document]
    vectorizer, features =  fe.scikit_bag_of_words_tfidf(normalized_document, 
                                                         normalize=False)
    
    corpus_for_summarizing.append((features, document))

with open("summarization_results_gensim.txt", "w") as f:
    f.write("Gensim:\n\n")
    
    for item in corpus:
        # Summary should contain third of original document
        summary = summarization_gensim(item, 0.33)
        summary = '\n'.join(summary)
        f.write("Original:\n")
        f.write(item)
        f.write("\n\n")
        f.write("Summary:\n")
        f.write(summary)
        f.write("\n------------------------------------------------\n\n")
    

with open("summarization_results_lsa.txt", "w") as f:
    f.write("LSA:\n\n")
    
    for item in corpus_for_summarizing:
        # Summary should contain third of original document
        summary = lsa(item[0], item[1], num_sentences=int(0.33 * len(item[1])))
        summary = '\n'.join(summary)
        f.write("Original:\n")
        f.write(document)
        f.write("\n\n")
        f.write("Summary:\n")
        f.write(summary)
        f.write("\n------------------------------------------------\n\n")

with open("summarization_results_textrank.txt", "w") as f:
    f.write("TextRank:\n\n")
    
    for item in corpus_for_summarizing:
        # Summary should contain third of original document
        summary = textrank(item[0], item[1], num_sentences=int(0.33 * len(item[1])))
        summary = '\n'.join(summary)
        f.write("Original:\n")
        f.write(document)
        f.write("\n\n")
        f.write("Summary:\n")
        f.write(summary)
        f.write("\n------------------------------------------------\n\n")