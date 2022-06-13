'''
Created on Mar 22, 2022

@author: mandd
'''
# from here space 3.2 version is needed - spacy 3.2 was installed using PIP, CONDA was setting spacy to 2.3
# NER for OPM model entities


#Import the requisite library
import spacy
from OPLparser import OPLentityParser, OPLtextParser
from spacy.pipeline import EntityRuler
from spacy import displacy
from pathlib import Path
import re

def NERentityParser(text):
  nlp = spacy.load('en_core_web_sm')

  # Print out current pipeline
  print(*nlp.pipeline, sep='\n')

  # Print out current NER entities
  print(nlp.get_pipe("ner").labels)

  doc = nlp(text)

  # Print out NER entities identified in text
  print(*[(e.text, e.label_) for e in doc.ents], sep=' ')

  # Save NER pipeline applied to text on SVG file
  svg = displacy.render(doc, style='ent', jupyter=False)
  filename = 'OPMparser.svg'
  output_path = Path(filename)
  output_path.open('w', encoding='utf-8').write(svg)

def opmFormEntityParser(text, formEntitiesList, functionEntitiesList):
  nlp = spacy.load('en_core_web_sm')

  if nlp.has_pipe('entity_ruler'):
    nlp.remove_pipe('entity_ruler')

  #patterns = [{'label':'OPM_form',
  #             'pattern': [{'TEXT': {'IN':formEntitiesList}, 'ENT_TYPE':'OPM_form'}]}]

  patterns = [{'label'  :'OPM_FORM',
               'pattern': {'TEXT': {'IN':formEntitiesList}}}]

  entity_ruler = EntityRuler(nlp, patterns=patterns, overwrite_ents=True)
  nlp.add_pipe('entity_ruler')
  doc = nlp(text)

  # Print out NER entities identified in text
  print(*[(e.text, e.label_) for e in doc.ents], sep=' ')

  # Save NER pipeline applied to text on SVG file
  svg = displacy.render(doc, style='ent', jupyter=False)
  filename = 'OPMparser.svg'
  output_path = Path(filename)
  output_path.open('w', encoding='utf-8').write(svg)


''' Testing workflow '''
formList, functionList = OPLentityParser('pump_OPL.html')
text_OPM = 'Pump inspection revealed excessive Impeller degradation.'
text_base = 'Over the last quarter Apple sold nearly 20 thousand iPods for a profit of $6 million. By contrast, my kids sold a lot of lemonade'
#NERentityParser(text_base)
print(formList)
opmFormEntityParser(text_OPM, formList, functionList)
