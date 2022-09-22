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


from nltk.corpus import wordnet

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

  for synset in wordnet.synsets(word):
    for lemma in synset.lemmas():
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
