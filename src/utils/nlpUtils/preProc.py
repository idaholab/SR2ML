'''
Created on May 3, 2021

@author: mandd
'''

# External Imports
import spacy  # MIT License
from spacy import displacy
from pathlib import Path
import textacy # Apache Software License (Apache)
import nltk as nltk # Apache Software License (Apache License, Version 2.0)
#import neuralcoref # MIT License (not available yet for spacy >= 3.0)
import stanza # pip install stanza # Apache License, Version 2.0
# Internal Imports


def syntaticAnalysis(text, plot=False):
  # 1- initialize NLP engine
  # this dictionary needs to be installed separately: python3 -m spacy download en
  # installation of en_core_web_sm: python -m spacy download en_core_web_sm
  nlp = spacy.load("en_core_web_sm")

  # 2- sentence tokenization
  # this requires an additional installation step:  nltk.download('punkt')
  sentencesSet = nltk.sent_tokenize(text)

  # 3- syntactic analysis
  processedSentences = []
  for index,sentence in enumerate(sentencesSet):
    # 3.1- identify syntactic elements
    doc = nlp(sentence)

    # this line is for visualization on file
    if plot:
      svg = displacy.render(doc, style='dep', jupyter=False)
      filename = 'sentence_'+ str(index) +'.svg'
      output_path = Path(filename)
      output_path.open('w', encoding='utf-8').write(svg)

    syntaticTranslation={}
    for tok in doc:
      syntaticTranslation[tok] = [tok.dep_ , tok.pos_]

    processedSentences.append(syntaticTranslation)
  return processedSentences


text = "London is the capital of England. Westmister is located in London."
syntaticTranslation = syntaticAnalysis(text, plot=True)
#print(syntaticTranslation)
