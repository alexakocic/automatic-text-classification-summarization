"""
Created on Sun Nov 19 17:01:12 2017

@author: Aleksa Kocić

Spell checking module. Contains methods for spell checking and correcting
misspelled words.
"""

from nltk import FreqDist
from nltk.corpus import wordnet
from nltk.corpus import gutenberg
from nltk.corpus import webtext
from nltk.corpus import brown
from nltk.corpus import reuters

def _english_word_frequencies():
    """
    Get frequencies of english words based on four corpora:
    Gutenberg Corpus, Web and Chat Text, Brown Corpus, Reuters Corpus
    
    Returns:
        tuple: Frequencies of words based on Gutenberg, Web and Chat Text, 
               Brown and Reuters corpora, respectively.
    """
    gutenberg_freqs = FreqDist(gutenberg.words())

    webtext_freqs = FreqDist(webtext.words())
    brown_freqs = FreqDist(brown.words())
    reuters_freqs = FreqDist(reuters.words())
    
    return gutenberg_freqs, webtext_freqs, brown_freqs, reuters_freqs

class SpellChecker:
    """
    Contains methods for correcting misspelled words and checking if a word
    exists in English language. Words edited by maximum number of 2 edits can 
    be corrected.
    """
    def __init__(self):
        self.word_frequencies = _english_word_frequencies()

    def __word_edit1(self, word):
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
        transposes = [a + b[1] + b[0] + b[2:] for (a, b) in pairs 
                      if len(b) > 1]
        
        # Get all words that are generated when one letter is replaced with
        # any other letter in word
        replaces = [a + c + b[1:] for (a, b) in pairs for c in alphabet if b]
        
        # Get all words that are generated when one letter is inserted at
        # any place in word
        inserts = [a + c + b[1:] for (a, b) in pairs for c in alphabet]
        
        return set(deletes + transposes + replaces + inserts)
    
    def __word_edit2(self, word):
        """
        Get all strings that are two edits away from the input word.
        
        Args:
            word (str): Word for finding edits
        
        Returns:
            set: All words that are two edits away from input word
        """
        return {edit2 for edit1 in self.__word_edit1(word) 
                for edit2 in self.__word_edit1(edit1)}
    
    def is_known_word(self, word):
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
    
    def __known_words(self, words):
        """
        Given the set of words, retrieve the subset of words which exist in
        English language.
        
        Args:
            words (list): List of words
        
        Returns:
            list: Subset of input list which consists only of words which exist
                  in english language
        """
        return {word for word in words if self.is_known_word(word)}
    
    def __word_frequency(self, word):
        """
        Get total frequency of a given word based on Gutenberg, Web and Chat 
        Text, Brown and Reuters corpora
        
        Args:
            word (str): A word which frequency is to be calculated
        
        Returns:
            int: Frequency of a word based on Gutenberg, Web and Chat Text, 
            Brown and Reuters corpora
        """
        sum_freq = 0
        for freq in self.word_frequencies:
            sum_freq += freq[word]
        
        return sum_freq
    
    # TODO: DELETE!
    def word_frequency(self, word):
        """
        Get total frequency of a given word based on Gutenberg, Web and Chat 
        Text, Brown and Reuters corpora
        
        Args:
            word (str): A word which frequency is to be calculated
        
        Returns:
            int: Frequency of a word based on Gutenberg, Web and Chat Text, 
            Brown and Reuters corpora
        """
        sum_freq = 0
        for freq in self.word_frequencies:
            sum_freq += freq[word]
        
        return sum_freq
    
    def correct_word(self, word):
        """
        Get most probable correct form of misspelled word.
        
        Args:
            word (str): Misspelled word
        
        Returns:
            str: Most probable correct form of misspelled word
        """
        # If word is an existing word in English, it is considered
        # correctly spelled
        if self.is_known_word(word):
            return word
        
        # One edit distance has more importance than two edit distance
        possible_words = self.__known_words(self.__word_edit1(word)) or \
                         self.__known_words(self.__word_edit2(word))
        
        # Word cannot be corrected, no one edit or two edits away words found 
        if not possible_words:
            return word
        
        # Probability goes first because when using max function,
        # it sorts elements based on first element                 
        possible_words_frequencies = [(self.__word_frequency(word), word) for 
                                        word in possible_words]
        
        # Calculate the word with highest frequency in English language
        # among possible words
        most_probable_word = max(possible_words_frequencies)[1]
        
        return most_probable_word