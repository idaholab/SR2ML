# modified from https://github.com/nihitsaxena95/sentence-similarity-wordnet-sementic/blob/master/SentenceSimilarity.py
import nltk
from pywsd.lesk import simple_lesk
import numpy as np
from nltk.corpus import wordnet, wordnet_ic

import logging
log_format = "%(asctime)s %(message)s"
logging.basicConfig(
    filename="sentence_similarity_computing.log",
    filemode="a",
    level=logging.INFO,
    format=log_format,
    datefmt="%m/%d %I:%M:%S %p",
)
logger = logging.getLogger("SentenceSimilarity")


class SentenceSimilarity:

  def __init__(self, disambiguationMethod='simple_lesk', similarityMethod='wup_similarity', wordOrder=False):
    """
    """
    self.validDisambiguation = ['simple_lesk']
    self.wordnetSimMethod = ["path_similarity", "wup_similarity", "lch_similarity", "res_similarity", "jcn_similarity", "lin_similarity"]
    self.validSimilarity = self.wordnetSimMethod
    self.wordOrder = wordOrder
    if disambiguationMethod.lower() not in self.validDisambiguation:
      raise ValueError(f'Inappropriate argument value for "disambiguationMethod", valid values are {self.validDisambiguation}')
    if similarityMethod.lower() not in self.validSimilarity:
      raise ValueError(f'Inappropriate argument value for "similarityMethod", valid values are {self.validSimilarity}')
    self.disambiguationMethod = disambiguationMethod.lower()
    self.similarityMethod = similarityMethod.lower()

    self.brownIc = wordnet_ic.ic('ic-brown.dat')

  def identifyNounAndVerbForComparison(self, sentence):
    """
      Taking out Noun and Verb for comparison word based
    """
    tokens = nltk.word_tokenize(sentence)
    pos = nltk.pos_tag(tokens)
    pos = [p for p in pos if p[1].startswith('N') or p[1].startswith('V')]
    return pos

  def wordSenseDisambiguation(self, sentence):
    """
      removing the disambiguity by getting the context
    """
    pos = self.identifyNounAndVerbForComparison(sentence)
    sense = []
    for p in pos:
      if self.disambiguationMethod == 'simple_lesk':
        sense.append(simple_lesk(sentence, p[0], pos=p[1][0].lower()))
      else:
        raise NotImplementedError(f"Mehtod {self.disambiguationMethod} not implemented yet!")
    return set(sense)

  def getSimilarity(self, arr1, arr2, vectorLen):
    """
    """
    vector = [0.0] * vectorLen
    count = 0
    for i,a1 in enumerate(arr1):
      allSimilarityIndex=[]
      for a2 in arr2:
        a1Edit = wordnet.synset(a1.name())
        a2Edit = wordnet.synset(a2.name())
        if self.similarityMethod in self.wordnetSimMethod:
          if self.similarityMethod == "path_similarity" or self.similarityMethod == "wup_similarity" or self.similarityMethod == "lch_similarity":
            similarity = getattr(wordnet, self.similarityMethod)(a1Edit, a2Edit)
          else:
            similarity = getattr(wordnet, self.similarityMethod)(a1Edit, a2Edit, self.brownIc)
        else:
          raise NotImplementedError(f"Similarity method {self.similarityMethod} is not Implemented yet!")
        if similarity != None:
          allSimilarityIndex.append(similarity)
        else:
          allSimilarityIndex.append(0.0)
      allSimilarityIndex = sorted(allSimilarityIndex, reverse = True)
      vector[i]=allSimilarityIndex[0]
      if vector[i] >= 0.804:
        count +=1
    return vector, count


  def shortestPathDistance(self, sense1, sense2):
    """
    """
    #getting the shortest path to get the similarity
    if len(sense1) >= len(sense2):
      grtSense = len(sense1)
    else:
      grtSense = len(sense2)
    v1, c1 = self.getSimilarity(sense1, sense2, grtSense)
    v2, c2 = self.getSimilarity(sense2, sense1, grtSense)
    return np.array(v1),np.array(v2),c1,c2

  def loopWordnet(self, method, sysnet1, sysnet2, ic=None):
    """
    """
    method = method.lower()
    if ic is None:
      ic = self.brownIc
    if method not in self.wordnetSimMethod:
      raise ValueError(f'Method {method} is not a valid wordnet method! try {self.wordnetSimMethod}')
    scoreList = []
    for item1 in sysnet1:
      for item2 in sysnet2:
        if method in ["res_similarity", "jcn_similarity", "lin_similarity"]:
          if item1.name().split(".")[1] == item2.name().split(".")[1]:
            try:
              score = getattr(wordnet, method)(item1, item2, ic)
              scoreList.append(score)
            except:
              continue
        else:
          score = getattr(wordnet, method)(item1, item2)
          scoreList.append(score)
    return scoreList

  def main(self, sentence1, sentence2):
    """
    """
    sense1 = self.wordSenseDisambiguation(sentence1)
    sense2 = self.wordSenseDisambiguation(sentence2)
    v1,v2,c1,c2 = self.shortestPathDistance(sense1,sense2)
    # print(v1, v2, c1, c2)
    dot = np.dot(v1,v2)
    print("dot", dot) # getting the dot product
    tow = (c1+c2)/1.8
    final_similarity = dot/tow
    print("similarity",final_similarity)
