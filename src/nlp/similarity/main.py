import simUtils
from SentenceSimilarity import SentenceSimilarity
import pandas as pd
import os

currentPath = os.path.dirname(os.path.abspath(__file__))
wordPairsFile = os.path.join(currentPath, "benchmark", "noun_pairs.csv")
sentencePairsFile = os.path.join(currentPath, "benchmark", "Li2006_sim_sents.csv")

fromFile = True
if fromFile:
  wordPairsData = pd.read_csv(wordPairsFile,  header=0)
  sentencePairsData = pd.read_csv(sentencePairsFile, header=0)
else:
  # Results from https://arxiv.org/pdf/1802.05667.pdf
  # with wordOrder = 0.15, word order threshold: 0.4, semantic threshold: 0.2
  wordPairs = [
      ["asylum", "fruit", 0.21],
      ["autograph", "shore", 0.29],
      ["autograph", "signature", 0.55],
      ["automobile", "car", 0.64],
      ["bird", "woodland", 0.33],
      ["boy", "rooster", 0.53],
      ["boy", "lad", 0.66],
      ["boy", "sage", 0.51],
      ["cemetery", "graveyard", 0.73],
      ["coast", "forest", 0.36],
      ["coast", "shore", 0.76],
      ["cock", "rooster", 1.00],
      ["cord", "smile", 0.33],
      ["cord", "string", 0.68],
      ["cushion", "pillow", 0.66],
      ["forest", "graveyard", 0.55],
      ["forest", "woodland", 0.70],
      ["furnace", "stove", 0.72],
      ["glass", "tumbler", 0.65],
      ["grin", "smile", 0.49],
      ["gem", "jewel", 0.83],
      ["hill", "woodland", 0.59],
      ["hill", "mound", 0.74],
      ["implement", "tool", 0.75],
      ["journey", "voyage", 0.52],
      ["magician", "oracle", 0.44],
      ["magician", "wizard", 0.65],
      ["midday", "noon", 1.0],
      ["oracle", "sage", 0.43],
      ["serf", "slave", 0.39]
  ]
  wordPairsData = pd.DataFrame(wordPairs, columns=['word1', 'word2', 'similarity'])

  sentencePairs = [
      ["I like that bachelor.", "I like that unmarried man.", 0.561],
      ["Red alcoholic drink.", "A bottle of wine.", 0.585],
      ["I have a hammer.", "Take some nails.", 0.508],
      ["John is very nice.", "Is John very nice?", 0.977],
      ["It is a dog.", "It is a log.", 0.623],
      ["Red alcoholic drink.", "An English dictionary.", 0.0],
      ["Dogs are animals.", "They are common pets.", 0.738],
      ["Canis familiaris are animals.", "Dogs are common pets.", 0.362],
      ["A glass of cider.", "A full cup of apple juice.", 0.678],
      ["I have a hammer.", "Take some apples.", 0.121],
      ["It is a dog.", "It is a pig.", 0.790],
      ["I have a pen.", "Where is ink?", 0.129],
      ["I have a pen.", "Where do you live?", 0.0],
      ["Red alcoholic drink.", "Fresh orange juice.", 0.611],
      ["Red alcoholic drink.", "Fresh apple juice.", 0.420],
      ["It is a dog.", "That must be your dog.", 0.739],
  ]
  sentencePairsData = pd.DataFrame(sentencePairs, columns=['sent1', 'sent2', 'similarity'])

if __name__ == '__main__':

  wordSimSemantic = []
  print("Words Similairty using semanticSimilarityWords:")
  for index, wordPair in wordPairsData.iterrows():
    calculated = simUtils.semanticSimilarityWords(wordPair['word1'], wordPair['word2'])
    wordSimSemantic.append(calculated)
    print(" ".join([str(e)+'\t' for e in wordPair.to_numpy()]), calculated)

  print("Words Similairty using wordsSimilarity semantic_similarity_synsets method:")
  for index, wordPair in wordPairsData.iterrows():
    calculated = simUtils.wordsSimilarity(wordPair['word1'], wordPair['word2'], method='semantic_similarity_synsets')
    print(" ".join([str(e)+'\t' for e in wordPair.to_numpy()]), calculated)

  wordSimPath = []
  print("Words Similairty using wordsSimilarity wordnet path method:")
  for index, wordPair in wordPairsData.iterrows():
    calculated = simUtils.wordsSimilarity(wordPair['word1'], wordPair['word2'], method='path')
    wordSimPath.append(calculated)
    print(" ".join([str(e)+'\t' for e in wordPair.to_numpy()]), calculated)

  senSimBestSense = []
  print("Sentence Similarity Best Sense:")
  for index, sentPair in sentencePairsData.iterrows():
    calculated = simUtils.sentenceSimilarity(sentPair['sent1'], sentPair['sent2'], False)
    # calculated = sentenceSimialrity(sentPair[0], sentPair[1], True)
    senSimBestSense.append(calculated)
    print(" ".join([str(e)+'\t' for e in sentPair.to_numpy()]), calculated)

  senSimDisabiguate = []
  print("Sentence Similarity with Disambiguation:")
  for index, sentPair in sentencePairsData.iterrows():
    calculated = simUtils.sentenceSimilarityWithDisambiguation(sentPair['sent1'], sentPair['sent2'], senseMethod='simple_lesk', simMethod='semantic_similarity_synsets', delta=1.0)
    senSimDisabiguate.append(calculated)
    print(" ".join([str(e)+'\t' for e in sentPair.to_numpy()]), calculated)

  # simObj = SentenceSimilarity()
  # print("Sentence Similarity Best Sense (Class Object):")
  # for index, sentPair in sentencePairsData.iterrows():
  #   calculated = simObj.sentenceSimilarity(sentPair['sent1'], sentPair['sent2'], method='best_sense')
  #   # calculated = sentenceSimialrity(sentPair[0], sentPair[1], True)
  #   print(" ".join([str(e)+'\t' for e in sentPair.to_numpy()]), calculated)

  # print("Sentence Similarity PM_Disambiguation:")
  # for index, sentPair in sentencePairsData.iterrows():
  #   calculated = simObj.sentenceSimilarity(sentPair['sent1'], sentPair['sent2'], method='pm_disambiguation')
  #   # calculated = sentenceSimialrity(sentPair[0], sentPair[1], True)
  #   print(" ".join([str(e)+'\t' for e in sentPair.to_numpy()]), calculated)

  def computePearson(name1, name2, data1, data2, type='Word'):
    pcoef = stats.pearsonr(data1, data2).statistic
    print(f'{type} Correlation ({name1} vs. {name2}):', pcoef)

  if fromFile:
    goldMetric = 'RG_similarity'
    simMetrics = ['Lee2014', 'PV2018']
    from scipy import stats
    computePearson(name1=goldMetric, name2='semantic', data1=wordSimSemantic, data2=wordPairsData[goldMetric])
    computePearson(name1=goldMetric, name2='Path', data1=wordSimPath, data2=wordPairsData[goldMetric])
    for m in simMetrics:
      computePearson(name1=goldMetric, name2=m, data1=wordPairsData[m], data2=wordPairsData[goldMetric])

    goldMetric = 'MeanHumanSimilarity'
    simMetrics = ['PV2018']
    computePearson(name1=goldMetric, name2='semantic', data1=senSimBestSense, data2=sentencePairsData[goldMetric], type='Sentence Best Sense')
    computePearson(name1=goldMetric, name2='semantic disambiguate', data1=senSimDisabiguate, data2=sentencePairsData[goldMetric], type='Sentence Disambiguation')
    for m in simMetrics:
      computePearson(name1=goldMetric, name2=m, data1=sentencePairsData[m], data2=sentencePairsData[goldMetric], type='Sentence')

