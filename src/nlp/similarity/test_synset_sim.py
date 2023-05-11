import synsetUtils
import time

bankSents = ['I went to the bank to deposit my money',
'The river bank was full of dead fishes']

plantSents = ['The workers at the industrial plant were overworked',
'The plant was no longer bearing flowers']

print("================ Testing sentenceSimilarity ================\n")
sents = [bankSents, plantSents]
for sent in sents:
  wordList1, synsets1 = synsetUtils.convertToSynsets([e.strip() for e in sent[0].split()])
  wordList2, synsets2 = synsetUtils.convertToSynsets([e.strip() for e in sent[1].split()])
  bestSyn1 = [synsetUtils.identifyBestSynset(word, wordList1, synsets1) for word in wordList1]
  bestSyn2 = [synsetUtils.identifyBestSynset(word, wordList2, synsets2) for word in wordList2]
  bestSyn1 = list(filter(None, bestSyn1))
  bestSyn2 = list(filter(None, bestSyn2))
  print(wordList1)
  print(wordList2)
  print(bestSyn1)
  print(bestSyn2)
  st = time.time()
  similarity = synsetUtils.synsetListSimilarity(bestSyn1, bestSyn2, delta=.8)
  print(sent[0], 'vs', sent[1], 'similarity', similarity)

  print('%s second'% (time.time()-st))

# TODO: Use disambiguation to determine synsets for each sentence
