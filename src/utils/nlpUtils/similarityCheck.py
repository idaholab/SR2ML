# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
'''
Created on Sep 20, 2022

@author: mandd
'''

'''
Needed:
> import nltk
> nltk.download('wordnet')
> nltk.download('omw-1.4')
'''
# sources:
#   https://www.holisticseo.digital/python-seo/nltk/wordnet
#   https://stackoverflow.com/questions/42446521/check-whether-there-is-any-synonyms-between-two-word-sets


from nltk.corpus import wordnet as wn

def findNyms(word):
  '''
  This method is designed to find the following for a provided word:
  * synonyms : words having similar meaning
  * antonyms : words having opposite meaning
  * hypernyms: words that provide a generalization of word
  * hyponyms : words that "inherit" from word and have been modified for a specific purpose
  * meronyms : words that represents constituents that are "part of" word
  * holonyms : words that represent the "whole" where word is "part of"
  '''

  nymsDict = {}
  nymsDict['antonyms']  = []
  nymsDict['synonyms']  = []
  nymsDict['hypernyms'] = []
  nymsDict['hyponyms']  = []
  nymsDict['meronyms']  = []
  nymsDict['holonyms']  = []

  for synset in wn.synsets(word):
    for lemma in synset.lemmas():
      if lemma.name()!=word:
        nymsDict['synonyms'].append(lemma.name())
      if lemma.antonyms():
        for ant in lemma.antonyms():
          nymsDict['antonyms'].append(ant.name())
      if synset.hyponyms():
        for hypo in synset.hyponyms():
          nymsDict['hyponyms'].append(hypo.lemmas()[0].name())
      if synset.hypernyms():
        for hyper in synset.hypernyms():
          nymsDict['hypernyms'].append(hyper.lemmas()[0].name())
      if synset.part_holonyms():
        for hol in synset.part_holonyms():
          nymsDict['holonyms'].append(hol.lemmas()[0].name())
      if synset.part_meronyms():
        for mer in synset.part_meronyms():
          nymsDict['meronyms'].append(mer.lemmas()[0].name())

  for key in nymsDict:
    nymsDict[key] = set(nymsDict[key])
  return nymsDict

def findSimilarity(word1, word2):
  '''
  This method is designed to find similarity between two words.
  The returned value is in [0,1] interval; if the returned value
  is 1 then two words are direct synonyms. If no connecting path
  between the two words can be determined, None is returned,.
  '''
  synset1 = wn.synsets(word1)
  synset2 = wn.synsets(word2)

  distance = wn.path_similarity(synset1[0],synset2[0])

  return distance
