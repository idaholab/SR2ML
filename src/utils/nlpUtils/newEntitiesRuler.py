'''
Created on Sep 27, 2022

@author: mandd
'''

import tagKeywordListReader as tklr

def patternCreator(tagDict):
  """
    
    @ In, tagDict, dict, []
    @ Out, patterns, [], []
  """
  patterns = []
  for tag in tagDict:
    for elem in tagDict[tag]:
      listElem ={"label": tag, "pattern": elem}
      patterns.append(listElem)
  return patterns

def unitPatternCreator(measureDict):
  unitPatterns = []
  for measure in measureDict:
    for elem in measureDict[measure]:
      listElem ={"label": 'unit', "pattern": elem}
      unitPatterns.append(listElem)
  return unitPatterns

'''
Example of usage:

> from spacy.lang.en import English
> tagDict = tklr.keyWordListGenerator('../../nlp/data/tag_keywords_lists.xlsx')
> tagsDict, acronymsDict = tklr.cleanTagDict(tagDict)
> patterns = patternCreator(tagDict)
> measureDict = tklr.extractUnits('../../nlp/data/tag_keywords_lists.xlsx')
> unitPatterns = unitPatternCreator(measureDict)

> nlp = English()
> ruler = nlp.add_pipe("entity_ruler")
> ruler.add_patterns(patterns)
> ruler.add_patterns(unitPatterns)
> doc = nlp('Pump shaft temperature was measured at about 45 Pa')
> print([(ent.text, ent.label_) for ent in doc.ents])
  [('shaft', 'comp_mech_rot'), ('temperature', 'prop'), ('Pa', 'unit')]
  
'''
