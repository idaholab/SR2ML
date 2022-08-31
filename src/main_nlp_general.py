# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
Created on March, 2022

@author: wangc, mandd
"""

import logging
from nlp.RuleBasedMatcher import RuleBasedMatcher
import spacy

from nlp.SimpleEntityMatcher import SimpleEntityMatcher
from nlp.PhraseEntityMatcher import PhraseEntityMatcher

logging.basicConfig(format='%(asctime)s %(name)-20s %(levelname)-8s %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.DEBUG)
# To enable the logging to both file and console, the logger for the main should be the root,
# otherwise, a function to add the file handler and stream handler need to be created and called by each module.
# logger = logging.getLogger(__name__)
logger = logging.getLogger()
# # create file handler which logs debug messages
fh = logging.FileHandler(filename='main_nlp_general.log', mode='w')
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s %(name)-20s %(levelname)-8s %(message)s')
fh.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh)

if __name__ == "__main__":
  nlp = spacy.load("en_core_web_lg")
  # remove the 'ner' pipeline which is used for ORG and DATE entity recognization
  if nlp.has_pipe('ner'):
    nlp.remove_pipe('ner')

  doc = r"""A leak was noticed from the RCP pump 1A.
RCP pump 1A pressure gauge was found not operating.
RCP pump 1A pressure gauge was found inoperative.
Rupture of pump bearings caused shaft degradation.
Rupture of pump bearings caused shaft degradation and consequent flow reduction.
Pump power supply has been found burnout.
Pump test failed due to power supply failure.
Pump inspection revealed excessive impeller degradation.
Pump inspection revealed excessive impeller degradation likely due to cavitation.
Oil puddle was found in proximity of RCP pump 1A.
Anomalous vibrations were observed for RCP pump 1A.
Several cracks on pump shaft were observed; they could have caused pump failure within few days.
RCP pump 1A  had signs of past leakage.
RCP pump 1A was cavitating and vibrating to some degree during test. This is most likely due to low flow conditions rather than mechanical issues.
Pump flow meter was not responding.
Cavitation was noticed but did not seem severe. The shaft vibration appears to be causing the motor to vibrate as well.
Pump had noise of cavitation which became faint after OPS bled off the air. Low flow conditions most likely causing cavitation.
The shaft deflection is causing the safety cage to rattle. Pumps not experiencing enough flow for the pumps to keep the check valves open during test.
Pump not experiencing enough flow during test.
Shaft made noise. Vibration seems like it is coming from the shaft.
Slight Vibrations noticed - likely from shaft deflection.
Visible shaft deflection in operation.
Pump bearings appear in acceptable condition.
Pump made noises - not nough to affect performance.
Pump shaft has a slight deflection.
        """


  # simple match
  slabel = 'ssc_match'
  terms = [{"LOWER":"pump"}, {"POS":"NOUN"}]
  logger.debug('Add pattern for simple entity matcher')
  nlp.add_pipe('simple_entity_matcher', config={"label": slabel, "terms":terms, "asSpan":True})
  # phrase match
  plabel = 'ssc_phrase_match'
  terms = ["safety cage", "Oil puddle", "OPS"]
  logger.debug('Add pattern for phrase entity matcher')
  nlp.add_pipe('phrase_entity_matcher', config={"label": plabel, "terms":terms, "asSpan":True})
  newDoc = nlp(doc)

  sEnts = [ent for ent in newDoc.ents if ent.label_==slabel]
  pEnts = [ent for ent in newDoc.ents if ent.label_==plabel]
  logger.debug('Simple matches:')
  logger.debug(sEnts)
  logger.debug('Phrase matches:')
  logger.debug(pEnts)
