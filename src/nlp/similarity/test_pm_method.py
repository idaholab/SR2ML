from SentenceSimilarity import SentenceSimilarity



sents = ['A gem is a jewel or stone that is used in jewellery.', 'A jewel is a precious stone used to decorate valuable things that you wear, such as rings or necklaces.']

simObj = SentenceSimilarity()
print("Sentence Similarity PM_Disambiguation:")
calculated = simObj.sentenceSimilarity(sents[0], sents[1], method='pm_disambiguation')
print(" ".join([str(e)+'\t' for e in sents]), calculated)
