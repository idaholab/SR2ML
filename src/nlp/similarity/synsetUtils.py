import sys
import math
import numpy as np

from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic

def synsetListSimilarity(synsetList1, synsetList2, delta=0.85):
  """
    Compute similarity for synsetList pair
    @ In, synsetList1, list, list of synset
    @ In, synsetList2, list, list of synset
    @ In, delta, float, between 0 and 1, factor for semantic similarity contribution
    @ Out, similarity, float, the similarity score
  """
  similarity = delta * semanticSimilaritySynsetList(synsetList1, synsetList2) + (1.0-delta)* wordOrderSimilaritySynsetList(synsetList1, synsetList2)
  return similarity

def wordOrderSimilaritySynsetList(synsetList1, synsetList2):
  """
    Compute word order similarity for synsetList pair
    @ In, synsetList1, list, list of synset
    @ In, synsetList2, list, list of synset
    @ Out, float, word order similarity score
  """
  # keep the order (works for python3.7+)
  synSet = list(dict.fromkeys(synsetList1+synsetList2))
  # synSet = list(set(synsetList1).union(set(synsetList2)))
  index = {syn[1]: syn[0] for syn in enumerate(synSet)}
  r1 = constructSynsetOrderVector(synsetList1, synSet, index)
  r2 = constructSynsetOrderVector(synsetList2, synSet, index)
  srTemp = np.linalg.norm(r1-r2)/np.linalg.norm(r1+r2)
  return 1-srTemp

def constructSynsetOrderVector(synsets, jointSynsets, index):
  """
    Construct synset order vector for word order similarity calculation
    @ In, synsets, list of synsets
    @ In, jointSynsets, list of joint synsets
    @ In, index, int, index for synsets
    @ Out, vector, np.array, synset order vector
  """
  vector = np.zeros(len(jointSynsets))
  i = 0
  synsets = set(synsets)
  for syn in jointSynsets:
    if syn in synsets:
      vector[i] = index[syn]
    else:
      synSimilar, similarity = identifyBestSimilarSynsetFromSynsets(syn, synsets)
      if similarity > 0.4:
        vector[i] = index[synSimilar]
      else:
        vector[i] = 0
    i +=1
  return vector

def identifyBestSimilarSynsetFromSynsets(syn, synsets):
  """
    Identify best similar synset from synsets
    @ In, syn, wn.synset, synset
    @ In, synsets, list of synsets
    @ Out, bestSyn, the best similar synset in synsets
    @ Out, similarity, the best similarity score
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
  synsetA = wn.synset(synsetA.name())
  synsetB = wn.synset(synsetB.name())
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
    Compute the similarity between two synsetList using semantic analysis
    i.e., compute the similarity using both path length and depth information in wordnet
    @ In, synsetList1, list, the list of synset
    @ In, synsetList2, list, the list of synset
    @ Out, semSimilarity, float, [0, 1], the similarity score
  """
  synSet = set(synsetList1).union(set(synsetList2))
  wordVectorA = constructSemanticVector(synsetList1, synSet)
  wordVectorB = constructSemanticVector(synsetList2, synSet)

  semSimilarity = np.dot(wordVectorA, wordVectorB)/(np.linalg.norm(wordVectorA)*np.linalg.norm(wordVectorB))
  return semSimilarity

def constructSemanticVector(syns, jointSyns):
  """
    Construct semantic vector
    @ In, syns, list of synsets
    @ In, jointSyns, list of joint synsets
    @ Out, vector, numpy.array, the semantic vector
  """
  synSet = set(syns)
  vector = np.zeros(len(jointSyns))

  i = 0
  for jointSyn in jointSyns:
    if jointSyn in synSet:
      vector[i] = 1
    else:
      _, similarity = identifyBestSimilarSynsetFromSynsets(jointSyn, synSet)
      if similarity >0.2:
        vector[i] = similarity
      else:
        vector[i] = 0.0
    i+=1
  return vector

def synsetsSimilarity(synsetA, synsetB, method='semantic_similarity_synsets', disambiguation=True):
  """
    Compute synsets similarity
    @ In, synsetA, wordnet.synset, the first synset
    @ In, synsetB, wordnet.synset, the second synset
    @ In, method, str, the method used to compute synset similarity
      one of ['semantic_similarity_synsets', 'path', 'wup', 'lch', 'res', 'jcn', 'lin']
    @ In, disambiguation, bool, True if disambiguation has been already performed
    @ Out, similarity, float, [0, 1], the similarity score
  """
  method = method.lower()
  if method != 'semantic_similarity_synsets' and not method.endswith('_similarity'):
    method = '_'.join([method, 'similarity'])
  wordnetSimMethod = ["path_similarity", "wup_similarity", "lch_similarity", "res_similarity", "jcn_similarity", "lin_similarity"]
  sematicSimMethod = ['semantic_similarity_synsets']
  synsetA = wn.synset(synsetA.name())
  synsetB = wn.synset(synsetB.name())
  if method in wordnetSimMethod:
    if method in ["path_similarity", "wup_similarity"]:
      similarity = getattr(wn, method)(synsetA, synsetB)
    elif method == "lch_similarity":
      if synsetA.name().split(".")[1] == synsetB.name().split(".")[1]:
        similarity = getattr(wn, method)(synsetA, synsetB)
      else:
        similarity = 0.0
    else:
      brownIc = wordnet_ic.ic('ic-brown.dat')
      if synsetA.name().split(".")[1] == synsetB.name().split(".")[1]:
        try:
          similarity = getattr(wn, method)(synsetA, synsetB, brownIc)
        except:
          similarity = 0.0
      else:
        similarity = 0.0
  elif method in sematicSimMethod:
    similarity = semanticSimilaritySynsets(synsetA, synsetB, disambiguation=disambiguation)
  else:
    raise ValueError(f'{method} is not valid, please use one of {wordnetSimMethod+sematicSimMethod}')

  return similarity


def constructSemanticVectorUsingDisambiguatedSynsets(wordSynsets, jointWordSynsets, simMethod='semantic_similarity_synsets'):
  """
    Construct semantic vector while disambiguation has been already performed
    @ In, wordSynsets, set/list, set of words synsets
    @ In, jointWords, set, set of joint words synsets
    @ In, simMethod, str, method for similarity analysis in the construction of semantic vectors
      one of ['semantic_similarity_synsets', 'path', 'wup', 'lch', 'res', 'jcn', 'lin']
    @ Out, vector, numpy.array, semantic vector with disambiguation
  """
  wordSynsets = set(wordSynsets)
  vector = np.zeros(len(jointWordSynsets))
  for i, jointSynset in enumerate(jointWordSynsets):
    simVector = []
    if jointSynset in wordSynsets:
      vector[i] = 1
    else:
      for synsetB in wordSynsets:
        similarity = synsetsSimilarity(jointSynset, synsetB, method=simMethod, disambiguation=True)
        simVector.append(similarity)
      maxSim = max(simVector)
      # if similarity < 0.2, treat it as noise and reset it to 0.0
      if maxSim >= 0.2:
        vector[i] = maxSim
      else:
        vector[i] = 0.0
  return vector


def semanticSimilarityUsingDisambiguatedSynsets(synsetsA, synsetsB, simMethod='semantic_similarity_synsets'):
  """
    Compute semantic similarity for given synsets while disambiguation has been already performed for given synsets
    @ In, synsetsA, set/list, list of synsets
    @ In, synsetsB, set/list, list of synsets
    @ In, simMethod, str, method for similarity analysis in the construction of semantic vectors
      one of ['semantic_similarity_synsets', 'path', 'wup', 'lch', 'res', 'jcn', 'lin']
    @ Out, semSimilarity, float, [0, 1], the similarity score
  """
  jointWordSynsets = set(synsetsA).union(set(synsetsB))
  wordVectorA = constructSemanticVectorUsingDisambiguatedSynsets(synsetsA, jointWordSynsets, simMethod=simMethod)
  wordVectorB = constructSemanticVectorUsingDisambiguatedSynsets(synsetsB, jointWordSynsets, simMethod=simMethod)
  semSimilarity = np.dot(wordVectorA, wordVectorB)/(np.linalg.norm(wordVectorA)*np.linalg.norm(wordVectorB))
  return semSimilarity
