# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
Created on March, 2022

@author: wangc, mandd
"""

import logging
import spacy
from nlp.CreatePatterns import CreatePatterns
from nlp.ConjectureEntity import ConjectureEntity
from nlp.GeneralEntity import GeneralEntity
from nlp.TemporalAttributeEntity import TemporalAttributeEntity
from nlp.TemporalRelationEntity import TemporalRelationEntity
from nlp.LocationEntity import LocationEntity
# sr2mlPath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# sys.path.append(sr2mlPath)


# logging.basicConfig(format='%(asctime)s %(name)-20s %(levelname)-8s %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.ERROR)
logging.basicConfig(format='%(asctime)s %(name)-20s %(levelname)-8s %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.DEBUG)

# To enable the logging to both file and console, the logger for the main should be the root,
# otherwise, a function to add the file handler and stream handler need to be created and called by each module.
# logger = logging.getLogger(__name__)
logger = logging.getLogger()
# # create file handler which logs debug messages
fh = logging.FileHandler(filename='main_nlp_opm.log', mode='w')
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s %(name)-20s %(levelname)-8s %(message)s')
fh.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh)

#####################################################################

if __name__ == "__main__":
  # load nlp model and matcher
  nlp = spacy.load("en_core_web_lg", exclude=[])
  # remove the 'ner' pipeline which is used for ORG and DATE entity recognization
  if nlp.has_pipe('ner'):
    nlp.remove_pipe('ner')

  doc = r"""A possible leak was noticed from the RCP pump 1A.
Rupture of pump bearings likely caused shaft degradation.
Pump power supply seems burnout.
Pump test failed unlikely due to power supply failure.
The valve is about a twenty-nine years old.
It is almost five years since it was replaced.
The pump failed first, then the system failed.
The system failed following the pump failure.
The motor failed while the pump stopped working.
The water leakage is happened above the pump.
The oil puddle is found next to the motor.
The debris is located below the generator.
        """

  #########################
  #  Testing conjecture_entity pipeline
  #########################
  # nlp.add_pipe('conjecture_entity', config={"patterns":patterns})
  nlp.add_pipe('conjecture_entity')
  newDoc = nlp(doc)
  ents = [ent for ent in newDoc.ents if ent.label_=='conjecture']
  print('conjecture:', ents)

  #########################
  #  Testing general_entity pipeline
  #########################
  filename = '~/projects/raven/plugins/SR2ML/src/nlp/data/conjecture_keywords.csv'
  ###################################################################
  conjecturePatterns = CreatePatterns(filename, entLabel='general', nlp=nlp)
  patterns = conjecturePatterns.getPatterns()
  nlp.add_pipe('general_entity', config={"patterns":patterns})
  newDoc = nlp(doc)
  ents = [ent for ent in newDoc.ents if ent.label_=='general']
  entsConjecture = [ent for ent in newDoc.ents if ent.label_=='conjecture']
  print('general:', ents)
  print('conjecture:', entsConjecture)

  #########################
  #  Testing temporal_attribute_entity pipeline
  #########################
  nlp.add_pipe('temporal_attribute_entity')
  newDoc = nlp(doc)
  ents = [ent for ent in newDoc.ents if ent.label_=='temporal_attribute']
  print('temporal_attribute:', ents)


  #########################
  #  Testing temporal_relation_entity pipeline
  #########################
  nlp.add_pipe('temporal_relation_entity')
  newDoc = nlp(doc)
  ents = [ent for ent in newDoc.ents if ent.label_=='temporal_relation_order']
  print('temporal_relation_order:', ents)
  ents = [ent for ent in newDoc.ents if ent.label_=='temporal_relation_reverse_order']
  print('temporal_relation_reverse_order:', ents)
  ents = [ent for ent in newDoc.ents if ent.label_=='temporal_relation_concurrency']
  print('temporal_relation_concurrency:', ents)

  #########################
  #  Testing location_entity pipeline
  #########################
  nlp.add_pipe('location_entity')
  newDoc = nlp(doc)
  ents = [ent for ent in newDoc.ents if ent.label_=='location_proximity']
  print('location_proximity:', ents)
  ents = [ent for ent in newDoc.ents if ent.label_=='location_up']
  print('location_up:', ents)
  ents = [ent for ent in newDoc.ents if ent.label_=='location_down']
  print('location_down:', ents)
