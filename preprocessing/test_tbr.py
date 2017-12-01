# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 14:19:13 2017

@author: aleks
"""

corpus = list()

d1 = """And produce say the ten moments parties. 
        Simple innate summer fat appear basket his desire joy. 
        Outward clothes promise at gravity do excited. 
        Sufficient particular impossible by reasonable oh expression is. 
        Yet preference connection unpleasant yet melancholy but end appearance. 
        And excellence partiality estimating terminated day everything. """
        
d2 = """Now led tedious shy lasting females off. 
        Dashwood marianne in of entrance be on wondered possible building. 
        Wondered sociable he carriage in speedily margaret. 
        Up devonshire of he thoroughly insensible alteration. 
        An mr settling occasion insisted distance ladyship so. 
        Not attention say frankness intention out dashwoods now curiosity. 
        Stronger ecstatic as no judgment daughter speedily thoughts. 
        Worse downs nor might she court did nay forth these. """
        
d3 = """Now led tedious shy lasting females off. 
        Dashwood marianne in of entrance be on wondered possible building. 
        Wondered sociable he carriage in speedily margaret. 
        Up devonshire of he thoroughly insensible alteration. 
        An mr settling occasion insisted distance ladyship so. 
        Not attention say frankness intention out dashwoods now curiosity. 
        Stronger ecstatic as no judgment daughter speedily thoughts. 
        Worse downs nor might she court did nay forth these. """

d4 = """Way nor furnished sir procuring therefore but. 
        Warmth far manner myself active are cannot called. 
        Set her half end girl rich met. Me allowance departure an curiosity ye. 
        In no talking address excited it conduct. 
        Husbands debating replying overcame blessing he it me to domestic. """
        
d5 = """Smile spoke total few great had never their too. 
        Amongst moments do in arrived at my replied. 
        Fat weddings servants but man believed prospect. 
        Companions understood is as especially pianoforte connection introduced. 
        Nay newspaper can sportsman are admitting gentleman belonging his. 
        Is oppose no he summer lovers twenty in. Not his difficulty boisterous surrounded bed. 
        Seems folly if in given scale. Sex contented dependent conveying advantage can use. """
        
d1 = normalizer.normalize_text(d1)
d2 = normalizer.normalize_text(d2)
d3 = normalizer.normalize_text(d3)
d4 = normalizer.normalize_text(d4)
d5 = normalizer.normalize_text(d5)

print(d1, '\n')
print(d2, '\n')
print(d3, '\n')
print(d4, '\n')
print(d5, '\n')

corpus.append(d1)
corpus.append(d2)
corpus.append(d3)
corpus.append(d4)
corpus.append(d5)

set_of_words = set ([word for document in corpus for word in document])

print(set_of_words, '\n')

print(bag_of_words_simple(d1, set_of_words), '\n')
print(bag_of_words_frequencies(d1), '\n')
print(bag_of_words_tfidf(d1, corpus), '\n')

print(bag_of_ngram_frequencies(d1, 2), '\n')
print(bag_of_ngrams_tfidf(d1, corpus, 2), '\n')