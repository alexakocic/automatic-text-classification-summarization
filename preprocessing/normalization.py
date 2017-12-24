"""
Created on Sat Nov 18 17:18:51 2017

@author: Aleksa Kocić

Text normalization module. Contains methods for transforming raw text into 
normalized version of it.
"""

from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import re
from preprocessing.spell_check import SpellChecker
from preprocessing.contractions import CONTRACTION_MAP

class Normalizer:
    """
    Contains methods for normalizing text. Also provides interface to
    SpellChecker object.
    """
    def __init__(self):
        self.spell_checker = SpellChecker()
        
    def get_spell_checker(self):
        """
        Provides interface to SpellChecker from spell_check module.
        
        Returns:
            SpellChecker: SpecllChecker object
        """
        return self.spell_checker
    
    def __expand_contractions_sentence(self, sentence, contraction_map):
        """
        Expand all contractions in a sentence.
        
        Args:
            sentence (str): A sentence of text
        
        Returns:
            str: A sentence with expanded contractions
        """
        contractions_pattern = re.compile('({})'.format('|'
                                          .join(contraction_map.keys())), 
                                          flags=re.IGNORECASE|re.DOTALL)
        
        def expand_match(contraction):
            match = contraction.group(0)
            first_char = match[0]
            expanded_contraction = contraction_map.get(match) \
                                   if contraction_map.get(match) \
                                   else contraction_map.get(match.lower())
            expanded_contraction = first_char + expanded_contraction[1:]
            return expanded_contraction
        
        expanded_sentence = contractions_pattern.sub(expand_match, sentence)
        return expanded_sentence
    
    def __expand_contractions(self, sentence_tokens, contraction_map):
        """
        Expand all contractions in a list of sentences.
        
        Args:
            sentence_tokens (list of str): List of sentences
        
        Returns:
            list of str: List of sentences with expanded contractions
        """
        return [self.__expand_contractions_sentence(sentence, contraction_map)
                for sentence in sentence_tokens]
        
    def __tokenize_sentences(self, text):
        """
        For a given text, retrieve list of sentences it is made of.
        
        Args:
            text (str): Input text
        
        Returns:
            list: List of sentences that input text is made of
        """
        return sent_tokenize(text)
    
    def __tokenize_words(self, sentence):
        """
        For a given sentence, retrieve list of words it is made of.
        
        Args:
            sentence (str): Input sentence
        
        Returns:
            list of str: List of words that input sentence is made of
        """
        return word_tokenize(sentence)
    
    def __tokenize(self, text):
        """
        For a given text, retrieve list of all words it consists of. In
        process of tokenization, contractions are expanded before word
        tokenizing.
        
        Args:
            text (str): Input text
        
        Returns:
            list of str: List of words that input text is made of
        """
        return [word for sentence in 
                self.__expand_contractions(self.__tokenize_sentences(text),
                                           CONTRACTION_MAP) 
                for word in self.__tokenize_words(sentence)]
    
    def __lower(self, word_tokens):
        """
        Lowercase all words in list of words.
        
        Args:
            word_tokens (list of str): List of words
        
        Returns:
            list of str: List of lowercased input words
        """
        return [word.lower() for word in word_tokens]
    
    def __remove_non_word_tokens(self, word_tokens):
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
    
    def __cleanse_word_tokens(self, word_tokens):
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
    
    def __correct_word_spelling(self, word_tokens):
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
        corrected_words = [self.spell_checker.correct_word(word) 
                           for word in word_tokens]
        return [corrected_word for corrected_word in corrected_words
                if self.spell_checker.is_known_word(corrected_word)]
        
    def __remove_stop_words(self, word_tokens):
        """
        Remove stop words from list of words.
        
        Args:
            word_tokens (list of str): Input tokens
        
        Returns:
            list of str: Input word tokens without stop words
        """
        return [word for word in word_tokens 
                if word not in stopwords.words('english')]
    
    def __lemmatize_words(self, word_tokens):
        """
        Perform word lemmization on each token in list of tokens.
        
        Args:
            word_tokens (list of str): Input tokens
        
        Returns:
            list of str: Lemmatized input word tokens
        """
        wordnet_lemmatizer = WordNetLemmatizer()
        return [wordnet_lemmatizer.lemmatize(word) for word in word_tokens]
    
    def __stem_words(self, word_tokens):
        """
        Perform word stem on each token in list of tokens.
        
        Args:
            word_tokens (list of str): Input tokens
        
        Returns:
            list of str: Stemmed input word tokens
        """
        porter_stemmer = PorterStemmer()
        return [porter_stemmer.stem(word) for word in word_tokens]
        
    
    def normalize_text(self, text, spell_check=False):
        """
        Perform raw text normalization. This process includes the following
        steps:
            1. Tokenize text into lexical units (words and special characters)
            2. Remove all tokens that do not contain words
            3. Clean all tokens from special characters, leave text and
               numbers only
            4. Correct spellings of wrongly spelled words. If a word is spelled
               wrong and cannot be corrected, it is removed from list of tokens
            5. Remove stop words from list of tokens
            6. Perform word stem
        
        Args:
            text: Raw text
            spell_check: Should spell checking be done when tokenizing words,
                         slows down performance if used
        
        Returns:
            list of str: Normalized text in form of a list of words
        """
        tokens = self.__tokenize(text)
        tokens = self.__lower(tokens)
        tokens = self.__remove_non_word_tokens(tokens)
        tokens = self.__cleanse_word_tokens(tokens)
        if spell_check:
            tokens = self.__correct_word_spelling(tokens)
        tokens = self.__remove_stop_words(tokens)
        tokens = self.__stem_words(tokens)
        return tokens