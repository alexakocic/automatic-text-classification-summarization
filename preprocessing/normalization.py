"""
Created on Sat Nov 18 17:18:51 2017

@author: Aleksa Kocić

Text normalization module. Contains methods for transforming raw text into 
normalized version of it.
"""

import nltk
import spell_check as sc

def tokenize_sentences(text):
    """
    For a given text, retrieve list of sentences it is made of.
    
    Args:
        text (str): Input text
    
    Returns:
        list: List of sentences that input text is made of
    """
    return nltk.tokenize.sent_tokenize(text)

def tokenize_words(sentence):
    """
    For a given sentence, retrieve list of words it is made of.
    
    Args:
        sentence (str): Input sentence
    
    Returns:
        list: List of words that input sentence is made of
    """
    return nltk.tokenize.word_tokenize(sentence)

def tokenize(text):
    """
    For a given text, retrieve list of all words it consists of.
    
    Args:
        text (str): Input text
    
    Returns:
        list: List of words that input text is made of
    """
    return [word for sentence in tokenize_sentences(text) 
            for word in tokenize_words(sentence)]
    