"""
Created on Sat Nov 18 17:18:51 2017

@author: Aleksa Kocić

Text normalization module. Contains methods for transforming raw text into 
normalized version of it.
"""

import nltk

def tokenize_sentences(text):
    """
    For a given text, retrieve list of sentences it is made of.
    
    Args:
        text (str): Input text
    
    Returns:
        list: List of text sentences that input text is made of
    """
    return nltk.tokenize.sent_tokenize(text)

# TODO: not implemented
def tokenize_words(sentence):
    return False 