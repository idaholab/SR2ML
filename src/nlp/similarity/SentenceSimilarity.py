# modified from https://github.com/nihitsaxena95/sentence-similarity-wordnet-sementic/blob/master/SentenceSimilarity.py
# Method proposed by: https://arxiv.org/pdf/1802.05667.pdf

import nltk
from pywsd.lesk import simple_lesk
import numpy as np
from nltk.corpus import wordnet, wordnet_ic
import simUtils

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

  def __init__(self, disambiguationMethod='simple_lesk', similarityMethod='semantic_similarity_synsets', wordOrderContribution=0.0):
    """
      Options for the initiation for this class
    """
    self.validDisambiguation = ['simple_lesk', 'original_lesk', 'cosine_lesk', 'adapted_lesk', 'max_similarity']
    self.wordnetSimMethod = ["path_similarity", "wup_similarity", "lch_similarity", "res_similarity", "jcn_similarity", "lin_similarity"]
    self.validSimilarity = self.wordnetSimMethod + ["semantic_similarity_synsets"]
    self.wordOrder = wordOrderContribution
    if disambiguationMethod.lower() not in self.validDisambiguation:
      raise ValueError(f'Inappropriate argument value for "disambiguationMethod", valid values are {self.validDisambiguation}')
    if similarityMethod.lower() not in self.validSimilarity:
      raise ValueError(f'Inappropriate argument value for "similarityMethod", valid values are {self.validSimilarity}')
    self.disambiguationMethod = disambiguationMethod.lower()
    self.similarityMethod = similarityMethod.lower()
    self.brownIc = wordnet_ic.ic('ic-brown.dat')

  def setParameters(self, paramDict):
    """
      Method to set the parameters
    """
    for key, value in paramDict:
      if key in self.__dict__:
        setattr(self, key, value)

  def constructSimilarityVectorPawarMagoMethod(self, arr1, arr2):
    """
      Construct the similarity vector
      @ In, arr1, set of wordnet.Synset for one sentence
      @ In, arr2, set of wordnet.Synset for the other sentence
      @ Out, vector, list, list of similarity vector
      @ Out, count, int, the number of words that have high similarity >=0.804
    """
    # vectorLen, the large length between arr1 and arr2
    if len(arr1) >= len(arr2):
      vectorLen = len(arr1)
    else:
      vectorLen = len(arr2)
    vector = [0.0] * vectorLen
    count = 0
    for i,a1 in enumerate(arr1):
      allSimilarityIndex=[]
      for a2 in arr2:
        if a1 is None or a2 is None:
          similarity = 0.
        else:
          a1Edit = wordnet.synset(a1.name())
          a2Edit = wordnet.synset(a2.name())
          similarity = simUtils.synsetsSimilarity(a1Edit, a2Edit, method=self.similarityMethod)
        if similarity != None:
          allSimilarityIndex.append(similarity)
        else:
          allSimilarityIndex.append(0.0)
      allSimilarityIndex = sorted(allSimilarityIndex, reverse = True)
      vector[i]=allSimilarityIndex[0]
      # According to Rubinstein 1965, the benchmark synonymy value of two word is 0.8025.
      if vector[i] >= 0.8025:
        count +=1
    vector = np.asarray(vector)
    return vector, count

  def sentenceSimilarity(self, sentence1, sentence2, method='pm_disambiguation', infoContentNorm=False):
    """
      sentence similarity calculation
    """
    if method.lower() == 'pm_disambiguation':
      similarity = self.sentenceSimilarityPawarMagoMethod(sentence1, sentence2)
    elif method.lower() == 'best_sense':
      similarity = self.sentenceSimialrityBestSense(sentence1, sentence2, infoContentNorm)
    else:
      raise ValueError(f'{method} is not a valid option, please try "pm_disambiguation" or "best_sense"')
    return similarity

  def sentenceSimilarityPawarMagoMethod(self, sentence1, sentence2):
    """
      Proposed method from https://arxiv.org/pdf/1802.05667.pdf
      @ In, sentence1, str, first sentence used to compute sentence similarity
      @ In, sentence2, str, second sentence used to compute sentence similarity
      @ Out, similarity, float, [0, 1], the computed similarity for given two sentences
    """
    _, sense1 = simUtils.sentenceSenseDisambiguationPyWSD(sentence1, senseMethod=self.disambiguationMethod, simMethod='path')
    _, sense2 = simUtils.sentenceSenseDisambiguationPyWSD(sentence2, senseMethod=self.disambiguationMethod, simMethod='path')
    v1, c1 = self.constructSimilarityVectorPawarMagoMethod(sense1,sense2)
    v2, c2 = self.constructSimilarityVectorPawarMagoMethod(sense2,sense1)
    # FIXME: check the following algorithms with benchmarks
    # dot = np.dot(v1,v2)
    # from original paper
    dot = np.linalg.norm(v1)*np.linalg.norm(v2)
    # print("dot", dot) # getting the dot product
    tow = (c1+c2)/1.8
    if tow == 0.:
      tow = len(v1)/2.0
    semanticSimilarity = dot/tow
    # print("similarity",semanticSimilarity)
    similarity = (1-self.wordOrder) * semanticSimilarity + self.wordOrder * simUtils.wordOrderSimilaritySentences(sentence1, sentence2)
    return similarity


  def sentenceSimialrityBestSense(self, sentence1, sentence2, infoContentNorm=False):
    """
      Proposed method from https://github.com/anishvarsha/Sentence-Similaritity-using-corpus-statistics
      Compute sentence similarity using both semantic and word order similarity
      The semantic similarity is based on maximum word similarity between one word and another sentence
      @ In, sentence1, str, first sentence used to compute sentence similarity
      @ In, sentence2, str, second sentence used to compute sentence similarity
      @ In, infoContentNorm, bool, True if statistics corpus is used to weight similarity vectors
      @ Out, similarity, float, [0, 1], the computed similarity for given two sentences
    """
    similarity = (1-self.wordOrder) * simUtils.semanticSimilaritySentences(sentence1, sentence2, infoContentNorm) + self.wordOrder * simUtils.wordOrderSimilaritySentences(sentence1, sentence2)
    return similarity
