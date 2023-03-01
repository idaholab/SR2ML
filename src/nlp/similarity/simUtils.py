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

def sentenceSimialrity(sentenceA, sentenceB, infoContentNorm, delta=0.85):
	similarity = delta * semanticSimilaritySentences(sentenceA, sentenceB, infoContentNorm) + (1.0-delta)* wordOrderSimilaritySentences(sentenceA, sentenceB)
	return similarity


def wordOrderSimilaritySentences(sentenceA, sentenceB):
	"""
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
		@ Out,
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
				vector[i] = 0.2
			else:
				vector[i] = 0.0
			if infoContentNorm:
				vector[i] = vector[i]*content(jointWord, wordCount, brownDict)* content(similarWord, wordCount, brownDict)
		i+=1
	return vector

def brownInfo():
	"""
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
	"""
	similarity = -1.0
	word = ""
	for wordB in wordSet:
		temp = semanticSimilarityWords(wordA, wordB)
		if temp > similarity:
			similarity = temp
			word = wordB
	return word, similarity

def semanticSimilarityWords(wordA, wordB, disambiguation=False):
	"""
	"""
	if wordA.lower() == wordB.lower():
		return 1.0
	bestPair = identifyBestSimilarSynsetPair(wordA, wordB)
	similarity = semanticSimilaritySynsets(bestPair[0], bestPair[1], disambiguation=disambiguation)
	return similarity

def semanticSimilaritySynsets(synsetA, synsetB, disambiguation=False):
	"""
	"""
	shortDistance = PathLength(synsetA, synsetB, disambiguation=disambiguation)
	maxHierarchy = ScalingDepthEffect(synsetA, synsetB, disambiguation=disambiguation)
	return shortDistance*maxHierarchy

def identifyBestSimilarSynsetPair(wordA, wordB):
	"""
		Disambiguation and identify the best synset pair for given two words
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

def PathLength(synsetA, synsetB, alpha=0.2, disambiguation=False):
	"""
		Path length calculation using nonlinear transfer function between two Wordnet Synsets
		The two Synsets should be the best Synset Pair (e.g., disambiguation should be performed)
		@ In, synsetA, wordnet.synset,
		@ In, synsetB, wordnet.synset,
		@ In, alpha, float, a constant in monotonically descreasing function, exp(-alpha*wordnetPathLength),
			parameter used to scale the shortest path length. For wordnet, the optimal value is 0.2
		@ Out,
	"""
	synsetA = wn.synset(synsetA.name())
	synsetB = wn.synset(synsetB.name())
	maxLength = sys.maxsize
	if synsetA is None or synsetB is None:
		return 0.0
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

def ScalingDepthEffect(synsetA, synsetB, beta=0.45, disambiguation=False):
	"""
		Words at upper layers of hierarchical semantic nets have more general concepts and less semantic similarity
		between words than words at lower layers. This method is used to scale the similarity behavior with repect
		to depth h, e.g., [exp(beta*h)-exp(-beta*g)]/[exp(beta*h)+exp(-beta*g)]
		The two Synsets should be the best Synset Pair (e.g., disambiguation should be performed)
		@ In, synsetA, wordnet.synset,
		@ In, synsetB, wordnet.synset,
		@ In, beta, float, parameter used to scale the shortest depth, for wordnet, the optimal value is 0.45
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
		@ In, sentence, string, sentence string
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
def wordsSimilarity(wordA, wordB, method=None):
	"""
	"""
	bestPair = identifyBestSimilarSynsetPair(wordA, wordB)
	# when campare words only, we assume there is no disambiguation required.
	similarity = synsetsSimilarity(bestPair[0], bestPair[1], method=method, disambiguation=False)
	return similarity

def synsetsSimilarity(synsetA, synsetB, method='semantic_similarity_synsets', disambiguation=True):
	"""
	"""
	method = method.lower()
	wordnetSimMethod = ["path_similarity", "wup_similarity", "lch_similarity", "res_similarity", "jcn_similarity", "lin_similarity"]
	sematicSimMethod = ['semantic_similarity_synsets']
	if method in wordnetSimMethod:
		if method in ["path_similarity", "wup_similarity", "lch_similarity"]:
			similarity = getattr(wn, method)(wn.synset(synsetA.name()), wn.synset(synsetB.name()))
		else:
			brownIc = wordnet_ic.ic('ic-brown.dat')
			similarity = getattr(wn, method)(wn.synset(synsetA.name()), wn.synset(synsetB.name()), brownIc)
	elif method in sematicSimMethod:
		similarity = semanticSimilaritySynsets(synsetA, synsetB, disambiguation=disambiguation)
	else:
		raise ValueError(f'{method} is not valid, please use one of {wordnetSimMethod+sematicSimMethod}')

	return similarity


def wordSenseDisambiguation(word, sentence, senseMethod='simple_lesk', simMethod='path'):
	"""
		removing the disambiguity by getting the context
		@ In, sentence, string, sentence string
		@ Out, sense, set, set of wordnet.Synset for the estimated best sense
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


