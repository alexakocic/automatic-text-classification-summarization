"""
Created on Sun Dec 24 17:07:27 2017

@author: Aleksa Kocić

Text summarization module. Contains methods for text summarization.
"""
from scipy.sparse.linalg import svds
import numpy as np
import networkx
from gensim.summarization import summarize

def summarization_gensim(text, ratio):
    """
    Summarize text using gensim's text summarizer. Gensim text summarizer
    uses TextRank algorithm.
    
    Args:
        text (str): Text to be summarized
        ratio (float): Percentege of original text to be present in summary
    
    Returns:
        str: Summarized document
    """
    summary = summarize(text, ratio=ratio)
    return summary

def low_rank_svd(matrix, k):
    """
    Create SVD matrix.
    
    Args:
        matrix (scipy.sparse.csr_matrix): Matrix to be transformed
        k (int): Number of largest singular values
    
    Returns:
        (tuple of scipy.sparse.csr_matrix): U, S and V transponed matrices
            respectively
    """
    
    u, s, vt = svds(matrix, k=k)
    return u, s, vt

def lsa(feature_matrix, sentences, num_sentences=3, num_topics=3, treshold=0.5):
    """
    Latent Semantic Analysis unsupervised summarizaton.
    
    Args:
        feature_matrix (scipy.sparse.csr_matrix): Matrix of features
        sentences (list of str): List of sentences in document
        num_sentences (int): Which number of sentences should summarized
            text contain
        num_topic (int): Number of topics the algorithms supposes there are
            in text
        treshold (float): Minimal percentage of maximum singular value a
            singular value should have not to be zeroed
    
    Returns:
        str: Summarized document
    """
    feature_matrix = feature_matrix.transpose()
    feature_matrix = feature_matrix.multiply(feature_matrix > 0)
    
    u, s, vt = low_rank_svd(feature_matrix, num_topics)
    min_sigma_value = max(s) * treshold
    s[s < min_sigma_value] = 0
    
    ss = np.sqrt(np.dot(np.square(s), np.square(vt)))
    top_sentence_indices = ss.argsort()[-num_sentences:][::-1]
    top_sentence_indices.sort()
    
    summarized_text = ""
    
    for index in top_sentence_indices:
        summarized_text += sentences[index]
    
    return summarized_text

def textrank(features, sentences, num_sentences=3):
    """
    TextRank unsupervised summarization.
    
    Args:
        feature_matrix (scipy.sparse.csr_matrix): Matrix of features
        sentences (list of str): List of sentences in document
        num_sentences (int): Which number of sentences should summarized
            text contain
    
    Returns:
        str: Summarized document
    """
    similarity_matrix = (features * features.T)
    similarity_graph = networkx.from_scipy_sparse_matrix(similarity_matrix)
    scores = networkx.pagerank(similarity_graph)
    
    ranked_sents = sorted(((score, index) for index, score in scores.items()),
                          reverse=True)
    
    top_sentence_indices = [ranked_sents[index][1] for index in range(num_sentences)]
    top_sentence_indices.sort()
    
    summarized_text = ""
    
    for index in top_sentence_indices:
        summarized_text += sentences[index]
    
    return summarized_text