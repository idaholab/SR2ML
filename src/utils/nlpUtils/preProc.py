'''
Created on May 3, 2021

@author: mandd
'''

# External Imports
import spacy  # MIT License
import textacy # Apache Software License (Apache)
import nltk as nltk # Apache Software License (Apache License, Version 2.0)
#import neuralcoref # MIT License (not available yet for spacy >= 3.0)

# Internal Imports


def syntaticAnalysis(text):
  # 1- initialize NLP engine
  # this dictionary needs to be installed separately: python3 -m spacy download en
  nlp = spacy.load("en_core_web_sm")
  
  # 2- sentence tokenization
  # this requires an additional installation step:  nltk.download('punkt')
  sentencesSet = nltk.sent_tokenize(text)
  
  # 3- syntactic analysis
  processedSentences = []
  for sentence in sentencesSet:
    # 3.1- identify syntactic elements
    doc = nlp(sentence)
    syntaticTranslation={}
    for tok in doc:
      syntaticTranslation[tok] = [tok.dep_ , tok.pos_]  
  
    processedSentences.append(syntaticTranslation)
  return processedSentences

def semanticAnalysis(syntaticTranslation):
  pass

def coreferenceResolution(triples):
  # use Stanford CoreNLP: https://stanfordnlp.github.io/CoreNLP/
  pass

text = "London is the capital of England. Westminster is located in London."
syntaticTranslation = syntaticAnalysis(text)
print(syntaticTranslation)
