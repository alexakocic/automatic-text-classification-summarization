"""
Created on Sat Nov 18 17:18:51 2017

@author: Aleksa Kocić

Text normalization module. Contains methods for transforming raw text into 
normalized version of it.
"""

import nltk
from spell_check import SpellChecker
import re

def _tokenize_sentences(text):
    """
    For a given text, retrieve list of sentences it is made of.
    
    Args:
        text (str): Input text
    
    Returns:
        list: List of sentences that input text is made of
    """
    return nltk.tokenize.sent_tokenize(text)

def _tokenize_words(sentence):
    """
    For a given sentence, retrieve list of words it is made of.
    
    Args:
        sentence (str): Input sentence
    
    Returns:
        list of str: List of words that input sentence is made of
    """
    return nltk.tokenize.word_tokenize(sentence)

def _tokenize(text):
    """
    For a given text, retrieve list of all words it consists of.
    
    Args:
        text (str): Input text
    
    Returns:
        list of str: List of words that input text is made of
    """
    return [word for sentence in _tokenize_sentences(text) 
            for word in _tokenize_words(sentence)]

def _remove_non_word_tokens(word_tokens):
    """
    From a list of word tokens, remove all tokens that made only of special
    characters.
    
    Args:
        word_tokens (list of str): Input tokens
    
    Returns:
        list of str: List of all input word tokens that are not made
                     exclusively from special characters
    """
    return [word_token for word_token in word_tokens 
            if not re.match(r'^[^A-Za-z0-9]+$', word_token)]

def _cleanse_word_tokens(word_tokens):
    """
    Remove all special characters from word tokens and leave only letters
    and numbers.
    
    Args:
        word_tokens (list of str): Input tokens
    
    Returns:
        list of str: Input word tokens cleansed from special characters 
    """
    cleansed_word_tokens = [re.sub('[^A-Za-z0-9]+', '', word_token)
            for word_token in word_tokens]
    return list(filter(None, cleansed_word_tokens))

def _correct_word_spelling(word_tokens):
    """
    Detect wrongly spelled words and try to correct them. If correction fails,
    it is considered that the word is wrong and wields no meaning so it is
    removed from list of tokens.
    
    Args:
        word_tokens (list of str): Input tokens
    
    Returns:
        list of str: Input word tokens with corrected spelling, or deleted
                     if wrongly spelled and not possible to correct
    """
    spell_checker = SpellChecker()
    corrected_words = [spell_checker.correct_word(word) 
                       for word in word_tokens]
    return [corrected_word for corrected_word in corrected_words
            if spell_checker.is_known_word(corrected_word)]