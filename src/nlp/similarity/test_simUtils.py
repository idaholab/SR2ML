import simUtils


bankSents = ['I went to the bank to deposit my money',
'The river bank was full of dead fishes']

plantSents = ['The workers at the industrial plant were overworked',
'The plant was no longer bearing flowers']

print("================ Testing sentenceSenseDisambiguationPyWSD ================\n")
for method in ['simple_lesk', 'original_lesk', 'cosine_lesk', 'adapted_lesk', 'max_similarity']:
  simMethod = 'path'
  for sent in bankSents:
    wordList, synsetList = simUtils.sentenceSenseDisambiguationPyWSD(sent, senseMethod=method, simMethod=simMethod)
    print(sent)
    print('==>', method, wordList, synsetList)


print("================ Testing wordSenseDisambiguation ================\n")
for method in ['simple_lesk', 'original_lesk', 'cosine_lesk', 'adapted_lesk', 'max_similarity']:
  simMethod = 'path'
  word = 'plant'
  for sent in plantSents:
    synsetList = simUtils.wordSenseDisambiguation(word, sent, senseMethod=method, simMethod=simMethod)
    print(sent)
    print('==>', method, synsetList)


print("================ Check two words similarity in two sentences ================\n")
word = 'plant'
method = 'simple_lesk'
simMethod = 'path'
sent = plantSents
synsetA = simUtils.wordSenseDisambiguation(word, sent[0], senseMethod=method, simMethod=simMethod)
synsetB = simUtils.wordSenseDisambiguation(word, sent[1], senseMethod=method, simMethod=simMethod)
for m in ["path_similarity", "wup_similarity", "lch_similarity", "res_similarity", "jcn_similarity", "lin_similarity"] + ['semantic_similarity_synsets']:
  similarity = simUtils.synsetsSimilarity(synsetA, synsetB, m)
  print(word, 'synsetA:', synsetA, 'synsetB:', synsetB, 'method:', m, 'similarity:', similarity)


word = 'bank'
sent = bankSents
synsetA = simUtils.wordSenseDisambiguation(word, sent[0], senseMethod=method, simMethod=simMethod)
synsetB = simUtils.wordSenseDisambiguation(word, sent[1], senseMethod=method, simMethod=simMethod)
for m in ["path_similarity", "wup_similarity", "lch_similarity", "res_similarity", "jcn_similarity", "lin_similarity"] + ['semantic_similarity_synsets']:
  similarity = simUtils.synsetsSimilarity(synsetA, synsetB, m)
  print(word, 'synsetA:', synsetA, 'synsetB:', synsetB, 'method:', m, 'similarity:', similarity)


print("================ Testing sentenceSimilarity ================\n")
sents = [bankSents, plantSents]
for sent in sents:
  similarity = simUtils.sentenceSimilarity(sent[0], sent[1], infoContentNorm=False, delta=0.85)
  print(sent[0], 'vs', sent[1], 'similarity', similarity)

print("================ Testing sentenceSimilarityWithDisambiguation ================\n")
sents = [bankSents, plantSents]
for sent in sents:
  similarity = simUtils.sentenceSimilarityWithDisambiguation(sent[0], sent[1], senseMethod='simple_lesk', simMethod='semantic_similarity_synsets', delta=0.85)
  print(sent[0], 'vs', sent[1], 'similarity', similarity)
