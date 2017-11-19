# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 17:01:12 2017

@author: Aleksa Kocić

Spell checking module. Contains methods for spell checking and correcting
misspelled words.
"""

from nltk.corpus import wordnet
# only temporary!
import random

def word_edit1(word):
    """
    Get all strings that are one edit away from the input word.
    
    Args:
        word (str): Word for finding edits
    
    Returns:
        set: All words that are one edit away from input word
    """
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    
    # Get all possible pairs of prefixes and suffixes of word such that
    # prefix + suffix = word
    pairs = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    
    # Get all words that are generated when one letter is omitted
    # from the original word
    deletes = [a + b[1:] for (a, b) in pairs if b]
    
    # Get all words that are generated when two letters in the word 
    # are transposed
    transposes = [a + b[1] + b[0] + b[2:] for (a, b) in pairs if len(b) > 1]
    
    # Get all words that are generated when one letter is replaced with
    # any other letter in word
    replaces = [a + c + b[1:] for (a, b) in pairs for c in alphabet if b]
    
    # Get all words that are generated when one letter is inserted at
    # any place in word
    inserts = [a + c + b[1:] for (a, b) in pairs for c in alphabet]
    
    return set(deletes + transposes + replaces + inserts)

def word_edit2(word):
    """
    Get all strings that are two edits away from the input word.
    
    Args:
        word (str): Word for finding edits
    
    Returns:
        set: All words that are two edits away from input word
    """
    return {edit2 for edit1 in word_edit1(word) for edit2 in word_edit1(edit1)}

def is_known_word(word):
    """
    Checks if word exists in English language.
    
    Args:
        word (str): A word which existence is checked
    
    Returns:
        bool: True if word is a part of English language, False if it isn't
    """
    # Presence in English language is determined by using WordNet:
    # if a word exists in WordNet database, then it is considered it is
    # a regular English word
    return True if wordnet.synsets(word, lang='eng') else False

def known_words(words):
    """
    Given the set of words, retrieve the subset of words which exist in
    English language.
    
    Args:
        words (list): List of words
    
    Returns:
        list: Subset of input list which consists only of words which exist
              in english language
    """
    return {word for word in words if is_known_word(word)}

# TODO: not implemented; should calculate word frequency from some corpora
def word_probability(word):
    return random.uniform(0, 1)

def correct_word(word):
    """
    Get most probable correct form of misspelled word.
    
    Args:
        word (str): Misspelled word
    
    Returns:
        str: Most probable correct form of misspelled word
    """
    # 1 edit distance has more importance than 2 edit distance
    possible_words = known_words(word_edit1(word)) or \
                     known_words(word_edit2(word))
                     
    possible_words_probabilities = [(word_probability(word), word) for 
                                    word in possible_words]
    
    # Calculate the word with highest frequency in English language
    # among possible words
    most_probable_word = max(possible_words_probabilities)[1]
    
    return most_probable_word