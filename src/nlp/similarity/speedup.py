import sys
import math
import numpy as np

import nltk
from nltk import word_tokenize as tokenizer
from nltk.corpus import wordnet as wn


def convertToSynsets(wordSet):
  wordList = list(set(wordSet))
  synsets = [list(wn.synsets(word)) for word in wordList]
  return wordList, synsets

def identifyBestSynset(word, wordList, synsets):
  if word in wordList:
    i = wordList.index(word)
    wordList.remove(word)
    synsets.remove(synsets[i])
  synsets = [item for sub in synsets for item in sub]
  wordSyns = wn.synsets(word)
  if len(wordSyns) == 0:
    return None

  max = -1
  bestSyn = wordSyns[0]
  for syn in wordSyns:
    similarity = [wn.path_similarity(syn, syn1) for syn1 in synsets]
    temp = np.max(similarity)
    if temp > max:
      bestSyn = syn
      max = temp
  return bestSyn

def synsetListSimilarity(synsetList1, synsetList2, delta=0.85):
  """

  """
  similarity = delta * semanticSimilaritySynsetList(synsetList1, synsetList2) + (1.0-delta)* wordOrderSimilaritySynsetList(synsetList1, synsetList2)
  return similarity


def wordOrderSimilaritySynsetList(synsetList1, synsetList2):
  """
  """
  synSet = list(set(synsetList1).union(set(synsetList2)))
  index = {syn[1]: syn[0] for syn in enumerate(synSet)}
  r1 = constructSynsetOrderVector(synsetList1, synSet, index)
  r2 = constructSynsetOrderVector(synsetList2, synSet, index)
  srTemp = np.linalg.norm(r1-r2)/np.linalg.norm(r1+r2)
  return 1-srTemp

def constructSynsetOrderVector(synsets, jointSynsets, index):
  """
  """
  vector = np.zeros(len(jointSynsets))
  i = 0
  synsets = set(synsets)
  for syn in jointSynsets:
    if syn in synsets:
      vector[i] = index[syn]
    else:
      synSimilar, similarity = identifyBestSimilarWordFromSynsets(syn, synsets)
      if similarity > 0.4:
        vector[i] = index[synSimilar]
      else:
        vector[i] = 0
    i +=1
  return vector

def identifyBestSimilarWordFromSynsets(syn, synsets):
  """
  """
  similarity = 0.0
  bestSyn = None
  for syn1 in synsets:
    temp = semanticSimilaritySynsets(syn, syn1)
    if temp > similarity:
      similarity = temp
      bestSyn = syn1
  return bestSyn, similarity

def semanticSimilaritySynsets(synsetA, synsetB, disambiguation=False):
  """
    Compute the similarity between two synset using semantic analysis
    e.g., using both path length and depth information in wordnet
    @ In, synsetA, wordnet.synset, the first synset
    @ In, synsetB, wordnet.synset, the second synset
    @ Out, similarity, float, [0, 1], the similarity score
  """
  shortDistance = pathLength(synsetA, synsetB, disambiguation=disambiguation)
  maxHierarchy = scalingDepthEffect(synsetA, synsetB, disambiguation=disambiguation)
  similarity = shortDistance*maxHierarchy
  return similarity

def pathLength(synsetA, synsetB, alpha=0.2, disambiguation=False):
  """
    Path length calculation using nonlinear transfer function between two Wordnet Synsets
    The two Synsets should be the best Synset Pair (e.g., disambiguation should be performed)
    @ In, synsetA, wordnet.synset, synset for first word
    @ In, synsetB, wordnet.synset, synset for second word
    @ In, alpha, float, a constant in monotonically descreasing function, exp(-alpha*wordnetpathLength),
      parameter used to scale the shortest path length. For wordnet, the optimal value is 0.2
    @ In, disambiguation, bool, True if disambiguation have been performed for the given synsets
    @ Out, shortDistance, float, [0, 1], the shortest distance between two synsets using exponential descreasing
      function.
  """
  # synsetA = wn.synset(synsetA.name())
  # synsetB = wn.synset(synsetB.name())
  maxLength = sys.maxsize
  if synsetA is None or synsetB is None:
    return 0.0
  # ? The original word is difference, but their synset are the same, we assume the path length is zero
  if synsetA == synsetB:
    maxLength = 0.0
  else:
    if not disambiguation:
      lemmaSetA = set([str(word.name()) for word in synsetA.lemmas()])
      lemmaSetB = set([str(word.name()) for word in synsetB.lemmas()])
      #this line if the word is none
      if len(lemmaSetA.intersection(lemmaSetB)) > 0:
        maxLength = 1.0
      else:
        maxLength = synsetA.shortest_path_distance(synsetB)
        if maxLength is None:
          maxLength = 0.0
    else:
      # when disamigutation is performed, we should avoid to check lemmas
      maxLength = synsetA.shortest_path_distance(synsetB)
      if maxLength is None:
        maxLength = 0.0
  shortDistance = math.exp(-alpha*maxLength)
  return shortDistance

def scalingDepthEffect(synsetA, synsetB, beta=0.45, disambiguation=False):
  """
    Words at upper layers of hierarchical semantic nets have more general concepts and less semantic similarity
    between words than words at lower layers. This method is used to scale the similarity behavior with repect
    to depth h, e.g., [exp(beta*h)-exp(-beta*g)]/[exp(beta*h)+exp(-beta*g)]
    The two Synsets should be the best Synset Pair (e.g., disambiguation should be performed)
    @ In, synsetA, wordnet.synset, synset for first word
    @ In, synsetB, wordnet.synset, synset for second word
    @ In, beta, float, parameter used to scale the shortest depth, for wordnet, the optimal value is 0.45
    @ In, disambiguation, bool, True if disambiguation have been performed for the given synsets
    @ out, treeDist, float, [0, 1], similary score based on depth effect in wordnet
  """
  maxLength = sys.maxsize
  smoothingFactor = beta
  if synsetA is None or synsetB is None:
    return 0.0

  if synsetA == synsetB:
    if disambiguation:
      return 1.0
    else:
      # The following is from original code, I think it should be return 1.0 when synset are the same
      # I think a new similarity calculations should be proposed
      # values for different lengths and their similarity score:
      # len simScore
      # 0   0.0
      # 1   0.4218990052500079
      # 2   0.7162978701990244
      # 3   0.874053287886007
      # 4   0.9468060128462682
      # 5   0.9780261147388136
      # 6   0.9910074536781175
      maxLength = max(word[1] for word in synsetA.hypernym_distances())
  else:
    hypernymWordA = {word[0]: word[1] for word in synsetA.hypernym_distances()}
    hypernymWordB = {word[0]: word[1] for word in synsetB.hypernym_distances()}
    commonValue = set(hypernymWordA.keys()).intersection(set(hypernymWordB.keys()))
    if len(commonValue) <=0:
      maxLength = 0
    else:
      distances = []
      for common in commonValue:
        commonWordADist = 0
        commonWordBDist = 0
        if common in hypernymWordA:
          commonWordADist = hypernymWordA[common]
        if common in hypernymWordB:
          commonWordBDist = hypernymWordB[common]
        maxDistance = max(commonWordADist, commonWordBDist)
        distances.append(maxDistance)
      maxLength = max(distances)
  treeDist = (math.exp(smoothingFactor*maxLength)- math.exp(-smoothingFactor*maxLength))/(math.exp(smoothingFactor*maxLength) + math.exp(-smoothingFactor*maxLength))

  return treeDist

def semanticSimilaritySynsetList(synsetList1, synsetList2):
  """
  """
  synSet = set(synsetList1).union(set(synsetList2))
  wordVectorA = constructSemanticVector(synsetList1, synSet)
  wordVectorB = constructSemanticVector(synsetList2, synSet)

  semSimilarity = np.dot(wordVectorA, wordVectorB)/(np.linalg.norm(wordVectorA)*np.linalg.norm(wordVectorB))
  return semSimilarity


def constructSemanticVector(syns, jointSyns):
  """
  """
  synSet = set(syns)
  vector = np.zeros(len(jointSyns))

  i = 0
  for jointSyn in jointSyns:
    if jointSyn in synSet:
      vector[i] = 1
    else:
      _, similarity = identifyBestSimilarWordFromSynsets(jointSyn, synSet)
      if similarity >0.2:
        vector[i] = similarity
      else:
        vector[i] = 0.0
    i+=1
  return vector
