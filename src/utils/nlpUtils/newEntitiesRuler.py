# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
'''
Created on Sep 27, 2022

@author: mandd
'''

import tagKeywordListReader as tklr

def patternCreator(tagDict):
  """
    This method is designed to create patterns from the tags and the corresponding list of
    keywords.
    @ In, tagDict, dict, dictionary containing tags and keywords. This dictionary is generated
                         by keyWordListGenerator() located in tagKeywordListReader.py from the
                         file nlp/data/tag_keywords_lists.xlsx
    @ Out, patterns, list, list of patterns: {"label": label_ID, "pattern": keyword}
  """
  patterns = []
  for tag in tagDict:
    for elem in tagDict[tag]:
      listElem ={"label": tag, "pattern": elem}
      patterns.append(listElem)
  return patterns

def unitPatternCreator(measureDict):
  """
    This method is designed to to create patterns in order to identify unit of measures
    @ In, measureDict, dict, dictionary which contains, for each quantity, a list of commonly
                       used units, e.g.,
                            {'Pressure': ['pa', ' torr', ' barr', ' atm', ' psi']}
    @ Out, patterns, list, list of patterns: {"label": "unit", "pattern": unitID}
  """
  unitPatterns = []
  for measure in measureDict:
    for elem in measureDict[measure]:
      listElem ={"label": "unit", "pattern": elem}
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
> doc = nlp('Pump shaft temperature was measured at about 45 C')
> print([(ent.text, ent.label_) for ent in doc.ents])
  [('shaft', 'comp_mech_rot'), ('temperature', 'prop'), ('C', 'unit')]
'''
