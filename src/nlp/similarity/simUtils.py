import sys
import math
import numpy as np
# https://github.com/alvations/pywsd
from pywsd.lesk import simple_lesk, original_lesk, cosine_lesk, adapted_lesk
from pywsd import disambiguate
from pywsd.similarity import max_similarity as maxsim

import nltk
from nltk import word_tokenize as tokenizer
from nltk.corpus import brown
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic

"""
  Methods proposed by: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1644735
  Codes are modified from https://github.com/anishvarsha/Sentence-Similaritity-using-corpus-statistics
"""

def sentenceSimilarity(sentenceA, sentenceB, infoContentNorm=False, delta=0.85):
  """
    Compute sentence similarity using both semantic and word order similarity
    The semantic similarity is based on maximum word similarity between one word and another sentence
    @ In, sentenceA, str, first sentence used to compute sentence similarity
    @ In, sentenceB, str, second sentence used to compute sentence similarity
    @ In, infoContentNorm, bool, True if statistics corpus is used to weight similarity vectors
    @ In, delta, float, [0,1], similarity contribution from semantic similarity, 1-delta is the similarity
      contribution from word order similarity
    @ Out, similarity, float, [0, 1], the computed similarity for given two sentences
  """
  similarity = delta * semanticSimilaritySentences(sentenceA, sentenceB, infoContentNorm) + (1.0-delta)* wordOrderSimilaritySentences(sentenceA, sentenceB)
  return similarity


def wordOrderSimilaritySentences(sentenceA, sentenceB):
  """
    Compute sentence similarity using word order similarity
    @ In, sentenceA, str, first sentence used to compute sentence similarity
    @ In, sentenceB, str, second sentence used to compute sentence similarity
    @ Out, similarity, float, [0, 1], the computed word order similarity for given two sentences
  """
  wordsA = tokenizer(sentenceA)
  wordsB = tokenizer(sentenceB)
  wordSet = list(set(wordsA).union(set(wordsB)))
  index = {word[1]: word[0] for word in enumerate(wordSet)}
  r1 = constructWordOrderVector(wordsA, wordSet, index)
  r2 = constructWordOrderVector(wordsB, wordSet, index)
  srTemp = np.linalg.norm(r1-r2)/np.linalg.norm(r1+r2)
  return 1-srTemp

def constructWordOrderVector(words, jointWords, index):
  """
    Construct word order vector
    @ In, words, set of words, a set of words for one sentence
    @ In, jointWords, set of joint words, a set of joint words for both sentences
    @ In, index, dict, word index in the joint set of words
    @ Out, vector, numpy.array, the word order vector
  """
  vector = np.zeros(len(jointWords))
  i = 0
  wordSet = set(words)
  for jointWord in jointWords:
    if jointWord in wordSet:
      vector[i] = index[jointWord]
    else:
      wordSimilar, similarity = identifyBestSimilarWordFromWordSet(jointWord, wordSet)
      if similarity > 0.4:
        vector[i] = index[wordSimilar]
      else:
        vector[i] = 0
    i +=1
  return vector

def semanticSimilaritySentences(sentenceA, sentenceB, infoContentNorm):
  """
    Compute sentence similarity using semantic similarity
    The semantic similarity is based on maximum word similarity between one word and another sentence
    @ In, sentenceA, str, first sentence used to compute sentence similarity
    @ In, sentenceB, str, second sentence used to compute sentence similarity
    @ In, infoContentNorm, bool, True if statistics corpus is used to weight similarity vectors
    @ Out, semSimilarity, float, [0, 1], the computed similarity for given two sentences
  """
  wordsA = tokenizer(sentenceA)
  wordsB = tokenizer(sentenceB)
  wordSet = set(wordsA).union(set(wordsB))
  wordVectorA = constructSemanticVector(wordsA, wordSet, infoContentNorm)
  wordVectorB = constructSemanticVector(wordsB, wordSet, infoContentNorm)

  semSimilarity = np.dot(wordVectorA, wordVectorB)/(np.linalg.norm(wordVectorA)*np.linalg.norm(wordVectorB))
  return semSimilarity


def constructSemanticVector(words, jointWords, infoContentNorm):
  """
    Construct semantic vector
    @ In, words, set of words, a set of words for one sentence
    @ In, jointWords, set of joint words, a set of joint words for both sentences
    @ In, infoContentNorm, bool, consider word statistics in Brown  Corpus if True
    @ Out, vector, numpy.array, the semantic vector
  """
  wordSet = set(words)
  vector = np.zeros(len(jointWords))
  if infoContentNorm:
    wordCount, brownDict = brownInfo()
  i = 0
  for jointWord in jointWords:
    if jointWord in wordSet:
      vector[i] = 1
      if infoContentNorm:
        vector[i] = vector[i]*math.pow(content(jointWord, wordCount, brownDict), 2)
    else:
      similarWord, similarity =  identifyBestSimilarWordFromWordSet(jointWord, wordSet)
      if similarity >0.2:
        vector[i] = similarity
      else:
        vector[i] = 0.0
      if infoContentNorm:
        vector[i] = vector[i]*content(jointWord, wordCount, brownDict)* content(similarWord, wordCount, brownDict)
    i+=1
  return vector

def brownInfo():
  """
    Compute word dict and word numbers in NLTK brown corpus
    @ In, None
    @ Out, wordCount, int, the total number of words in brown
    @ Out, brownDict, dict, the brown word dict, {word:count}
  """
  brownDict = {}
  wordCount = 0
  for sent in brown.sents():
    for word in sent:
      key = word.lower()
      if key not in brownDict:
        brownDict[key] = 0
      brownDict[key]+=1
      wordCount+=1
  return wordCount, brownDict

def content(wordData, wordCount=0, brownDict=None):
  """
    Employ statistics from Brown Corpus to compute the information content of given word in the corpus
    ref: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1644735
    information content I(w) = 1 - log(n+1)/log(N+1)
    The significance of a word is weighted using its information content. The assumption here is
    that words occur with a higher frequency (in corpus) contain less information than those
    that occur with lower frequencies.
    @ In, wordData, string, a given word
    @ In, wordCount, int, the total number of words in brown corpus
    @ In, brownDict, dict, the brown word dict, {word:count}
    @ Out, content, float, [0, 1], the information content of a word in the corpus
  """
  if wordCount == 0:
    wordCount, brownDict = brownInfo()
  wordData = wordData.lower()
  if wordData not in brownDict:
    n = 0
  else:
    n = brownDict[wordData]
  informationContent = math.log(n+1)/math.log(wordCount+1)
  return 1.0-informationContent

def identifyBestSimilarWordFromWordSet(wordA, wordSet):
  """
    Identify the best similar word in a word set for a given word
    @ In, wordA, str, a given word that looking for the best similar word in a word set
    @ In, wordSet, set/list, a pool of words
    @ Out, word, str, the best similar word in the word set for given word
    @ Out, similarity, float, [0, 1], similarity score between the best pair of words
  """
  similarity = 0.0
  word = ""
  for wordB in wordSet:
    temp = semanticSimilarityWords(wordA, wordB)
    if temp > similarity:
      similarity = temp
      word = wordB
  return word, similarity

def semanticSimilarityWords(wordA, wordB):
  """
    Compute the similarity between two words using semantic analysis
    First identify the best similar synset pair using wordnet similarity, then compute the similarity
    using both path length and depth information in wordnet
    @ In, wordA, str, the first word
    @ In, wordB, str, the second word
    @ Out, similarity, float, [0, 1], the similarity score
  """
  if wordA.lower() == wordB.lower():
    return 1.0
  bestPair = identifyBestSimilarSynsetPair(wordA, wordB)
  if bestPair[0] is None or bestPair[1] is None:
    return 0.0
  # disambiguation is False since only two words is provided and there is no additional information content
  similarity = semanticSimilaritySynsets(bestPair[0], bestPair[1], disambiguation=False)
  return similarity

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

def identifyBestSimilarSynsetPair(wordA, wordB):
  """
    Identify the best synset pair for given two words using wordnet similarity analysis
    @ In, wordA, str, the first word
    @ In, wordB, str, the second word
    @ Out, bestPair, tuple, (first synset, second synset), identified best synset pair using wordnet similarity
  """
  similarity = -1.0
  synsetsWordA = wn.synsets(wordA)
  synsetsWordB = wn.synsets(wordB)

  if len(synsetsWordA) == 0 or len(synsetsWordB) == 0:
    return None, None
  else:
    similarity = -1.0
    bestPair = None, None
    for synsetWordA in synsetsWordA:
      for synsetWordB in synsetsWordB:
        # TODO: may change to general similarity method
        temp = wn.path_similarity(synsetWordA, synsetWordB)
        if temp > similarity:
          similarity = temp
          bestPair = synsetWordA, synsetWordB
    return bestPair

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

#################################################


"""
  Methods proposed by: https://arxiv.org/pdf/1802.05667.pdf
  Codes are modified from https://github.com/nihitsaxena95/sentence-similarity-wordnet-sementic/blob/master/SentenceSimilarity.py
"""

def identifyNounAndVerbForComparison(sentence):
  """
    Taking out Noun and Verb for comparison word based
    @ In, sentence, string, sentence string
    @ Out, pos, list, list of dict {token/word:pos_tag}
  """
  tokens = nltk.word_tokenize(sentence)
  pos = nltk.pos_tag(tokens)
  pos = [p for p in pos if p[1].startswith('N') or p[1].startswith('V')]
  return pos

def sentenceSenseDisambiguation(sentence, method='simple_lesk'):
  """
    removing the disambiguity by getting the context
    @ In, sentence, str, sentence string
    @ In, method, str, the method for disambiguation, this method only support simple_lesk method
    @ Out, sense, set, set of wordnet.Synset for the estimated best sense
  """
  pos = identifyNounAndVerbForComparison(sentence)
  sense = []
  for p in pos:
    if method.lower() == 'simple_lesk':
      sense.append(simple_lesk(sentence, p[0], pos=p[1][0].lower()))
    else:
      raise NotImplementedError(f"Mehtod {method} not implemented yet!")
  return set(sense)

###########################################################################

"""
  Extended methods
"""
def wordsSimilarity(wordA, wordB, method='semantic_similarity_synsets'):
  """
    General method for compute words similarity
    @ In, wordA, str, the first word
    @ In, wordB, str, the second word
    @ In, method, str, the method used to compute word similarity
    @ Out, similarity, float, [0, 1], the similarity score
  """
  method = method.lower()
  wordnetSimMethod = ["path_similarity", "wup_similarity", "lch_similarity", "res_similarity", "jcn_similarity", "lin_similarity"]
  sematicSimMethod = ['semantic_similarity_synsets']
  if method not in sematicSimMethod and not method.endswith('_similarity'):
    method = method + '_similarity'
  if method not in sematicSimMethod + wordnetSimMethod:
    raise ValueError(f'{method} is not valid, please use one of {wordnetSimMethod+sematicSimMethod}')
  bestPair = identifyBestSimilarSynsetPair(wordA, wordB)
  if bestPair[0] is None or bestPair[1] is None:
    return 0.0
  # when campare words only, we assume there is no disambiguation required.

  similarity = synsetsSimilarity(bestPair[0], bestPair[1], method=method, disambiguation=False)
  return similarity

# TODO: Utilize Spacy wordvector similarity to improve the similarity score when the POS for given synset are not the same
# i.e., synsetA.name().split(".")[1] != synsetB.name().split(".")[1], the similarity will be set to 0.0
# Similarity calculated by word1.similarity(word2)
# calibration calibrate 0.715878
# failure fail 0.6802347
# replacement replace 0.73397416
# leak leakage 0.652573

# Similarity calculated by wordnet:
# calibration calibrate 0.09090909090909091
# failure fail 0.125
# replacement replace 0.1111111111111111
# leak leakage 1.0

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


def wordSenseDisambiguation(word, sentence, senseMethod='simple_lesk', simMethod='path'):
  """
    removing the disambiguity by getting the context
    @ In, word, str/list/set, given word or set of words
    @ In, sentence, str, sentence that will be used to disambiguate the given word
    @ In, senseMethod, str, method for disambiguation, one of ['simple_lesk', 'original_lesk', 'cosine_lesk', 'adapted_lesk', 'max_similarity']
    @ In, simMethod, str, method for similarity analysis when 'max_similarity' is used,
      one of ['path', 'wup', 'lch', 'res', 'jcn', 'lin']
    @ Out, sense, str/list/set, the type for given word, identified best sense for given word with disambiguation performed using given sentence
  """
  method = senseMethod.lower()
  simMethod = simMethod.lower()
  validMethod = ['simple_lesk', 'original_lesk', 'cosine_lesk', 'adapted_lesk', 'max_similarity']
  validSimMethod = ['path', 'wup', 'lch', 'res', 'jcn', 'lin']
  if method not in validMethod:
    raise ValueError(f'{method} is not valid option, please try to use one of {validMethod}')
  if simMethod not in validSimMethod:
    raise ValueError(f'{simMethod} is not valid option, please try to use one of {validSimMethod}')
  if isinstance(word, str):
    tokens = nltk.word_tokenize(word)
  elif isinstance(word, list) or isinstance(word, set):
    tokens = [nltk.word_tokenize(w)[0] for w in word]
  else:
    raise ValueError(f'{word} format is not valid, please input string, set/list of string')
  pos = nltk.pos_tag(tokens)
  sense = []
  for p in pos:
    if method == 'simple_lesk':
      sense.append(simple_lesk(sentence, p[0], pos=p[1][0].lower()))
    elif method == 'original_lesk':
      sense.append(original_lesk(sentence, p[0]))
    elif method == 'adapted_lesk':
      sense.append(adapted_lesk(sentence, p[0], pos=p[1][0].lower()))
    elif method == 'cosine_lesk':
      sense.append(cosine_lesk(sentence, p[0], pos=p[1][0].lower()))
    elif method == 'max_similarity':
      sense.append(maxsim(sentence, p[0], pos=p[1][0].lower(), option=simMethod))
    else:
      raise NotImplementedError(f"Mehtod {method} not implemented yet!")
  if isinstance(word, str):
    return sense[0]
  elif isinstance(word, list):
    return sense
  else:
    return set(sense)


# use disambiguate function to disambiguate sentences
def sentenceSenseDisambiguationPyWSD(sentence, senseMethod='simple_lesk', simMethod='path'):
  """
    Wrap for sentence sense disambiguation method from pywsd
    https://github.com/alvations/pywsd
    @ In, sentence, str, given sentence
    @ In, senseMethod, str, method for disambiguation, one of ['simple_lesk', 'original_lesk', 'cosine_lesk', 'adapted_lesk', 'max_similarity']
    @ In, simMethod, str, method for similarity analysis when 'max_similarity' is used,
      one of ['path', 'wup', 'lch', 'res', 'jcn', 'lin']
    @ Out, wordList, list, list of words from sentence that has an identified synset from wordnet
    @ Out, synsetList, list, list of corresponding synset for wordList
  """
  method = senseMethod.lower()
  simMethod = simMethod.lower()
  validMethod = ['simple_lesk', 'original_lesk', 'cosine_lesk', 'adapted_lesk', 'max_similarity']
  validSimMethod = ['path', 'wup', 'lch', 'res', 'jcn', 'lin']
  if method not in validMethod:
    raise ValueError(f'{method} is not valid option, please try to use one of {validMethod}')
  if simMethod not in validSimMethod:
    raise ValueError(f'{simMethod} is not valid option, please try to use one of {validSimMethod}')
  if method == 'simple_lesk':
    sentSense = disambiguate(sentence, simple_lesk, prefersNone=True, keepLemmas=True)
  elif method == 'original_lesk':
    sentSense = disambiguate(sentence, original_lesk, prefersNone=True, keepLemmas=True)
  elif method == 'adapted_lesk':
    sentSense = disambiguate(sentence, adapted_lesk, prefersNone=True, keepLemmas=True)
  elif method == 'cosine_lesk':
    sentSense = disambiguate(sentence, cosine_lesk, prefersNone=True,  keepLemmas=True)
  elif method == 'max_similarity':
    sentSense = disambiguate(sentence, maxsim, similarity_option=simMethod, prefersNone=True,  keepLemmas=True)
  # sentSense: a list of tuples, [(word, lemma, wn.synset/None)]
  wordList = list([syn[0] for syn in sentSense if syn[-1] is not None])
  synsetList = list([syn[-1] for syn in sentSense if syn[-1] is not None])
  return wordList, synsetList

# Sentence similarity after disambiguation

def sentenceSimilarityWithDisambiguation(sentenceA, sentenceB, senseMethod='simple_lesk', simMethod='semantic_similarity_synsets', disambiguationSimMethod='path', delta=0.85):
  """
    Compute semantic similarity for given two sentences that disambiguation will be performed
    @ In, sentenceA, str, first sentence
    @ In, sentenceB, str, second sentence
    @ In, senseMethod, str, method for disambiguation, one of ['simple_lesk', 'original_lesk', 'cosine_lesk', 'adapted_lesk', 'max_similarity']
    @ In, simMethod, str, method for similarity analysis in the construction of semantic vectors
      one of ['semantic_similarity_synsets', 'path', 'wup', 'lch', 'res', 'jcn', 'lin']
    @ In, disambiguationSimMethod, str, method for similarity analysis when 'max_similarity' is used,
      one of ['path', 'wup', 'lch', 'res', 'jcn', 'lin']
    @ In, delta, float, [0,1], similarity contribution from semantic similarity, 1-delta is the similarity
      contribution from word order similarity
    @ Out, similarity, float, [0, 1], the computed similarity for given two sentences
  """
  simMethod = simMethod.lower()
  disambiguationSimMethod = disambiguationSimMethod.lower()
  if disambiguationSimMethod not in ['path', 'wup', 'lch', 'res', 'jcn', 'lin']:
    raise ValueError(f'Option for "disambiguationSimMethod={disambiguationSimMethod}" is not valid, please try one of "path, wup, lch, res, jcn, lin" ')
  _, synsetsA = sentenceSenseDisambiguationPyWSD(sentenceA, senseMethod=senseMethod, simMethod=disambiguationSimMethod)
  _, synsetsB = sentenceSenseDisambiguationPyWSD(sentenceB, senseMethod=senseMethod, simMethod=disambiguationSimMethod)
  similarity = delta * semanticSimilarityUsingDisambiguatedSynsets(synsetsA, synsetsB, simMethod=simMethod) + (1.0-delta)* wordOrderSimilaritySentences(sentenceA, sentenceB)
  return similarity

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
