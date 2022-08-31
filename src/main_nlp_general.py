# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
Created on March, 2022

@author: wangc, mandd
"""

import logging
from nlp.RuleBasedMatcher import RuleBasedMatcher
import spacy

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
  matcher = RuleBasedMatcher(nlp, match=True, phraseMatch=True)

  ##Issue with simple and phrase matcher, if there are duplicate names, callback functions
  ##can not be used, in which case, we can not directly extend doc.ents, which will raise the
  ## error:  Unable to set entity information for token xx which is included in more than
  ## one span in entities, blocked, missing or outside. (https://github.com/explosion/spaCy/discussions/9993)
  ## In this case, we need to use "entity_ruler" to identify the matches
  # simple match
  # name = 'ssc_match'
  # rules = [{"LOWER":"pump"}, {"POS":"NOUN"}]
  # matcher.addPattern(name, rules, callback=None)
  # phrase match
  name = 'ssc_phrase_match'
  # phraseList = ["safety cage", 'pump 1A', 'pump bearings', 'Pump']
  phraseList = ["safety cage", "cage"]
  matcher.addPhrase(name, phraseList, callback=None)
  # depency match
  name = 'ssc_dependency_match'
  dependencyList = [
    {
      "RIGHT_ID":"anchor_found",
      "RIGHT_ATTRS":{"ORTH":"found"}
    },
    {
      "LEFT_ID":"anchor_found",
      "REL_OP":">",
      "RIGHT_ID":"subject",
      "RIGHT_ATTRS":{"DEP":"nsubjpass"}
    },
    {
      "LEFT_ID":"anchor_found",
      "REL_OP":">",
      "RIGHT_ID":"object",
      "RIGHT_ATTRS":{"DEP":"oprd"}
    }
  ]
  # matcher.addDependency(name, dependencyList, callback=None)
  # Entity rule-based match
  name = 'ssc_entity_ruler'
  patterns = [{"label":"pump_comp", "pattern":[{"POS":"NOUN", "OP":"*"}, {"LOWER":"pump"}, {"POS":"NOUN", "OP":"*"}], "id":"SSC"},
              {"label":"pump_comp", "pattern":[{"POS":"NOUN", "OP":"*"}, {"LOWER":"shaft"}, {"POS":"NOUN", "OP":"*"}], "id":"SSC"},
              {"label":"pump_comp", "pattern":[{"POS":"NOUN", "OP":"*"}, {"LOWER":"gauge"}, {"POS":"NOUN", "OP":"*"}], "id":"SSC"},
              {"label":"pump_comp", "pattern":[{"POS":"NOUN", "OP":"*"}, {"LOWER":"bearing"}, {"POS":"NOUN", "OP":"*"}], "id":"SSC"},
              {"label":"pump_comp", "pattern":[{"POS":"NOUN", "OP":"*"}, {"LOWER":"supply"}, {"POS":"NOUN", "OP":"*"}], "id":"SSC"},
              {"label":"pump_comp", "pattern":[{"POS":"NOUN", "OP":"*"}, {"LOWER":"impeller"}, {"POS":"NOUN", "OP":"*"}], "id":"SSC"},
              {"label":"pump_comp", "pattern":[{"POS":"NOUN", "OP":"*"}, {"LOWER":"flow meter"}, {"POS":"NOUN", "OP":"*"}], "id":"SSC"},
              {"label":"pump_comp", "pattern":[{"POS":"NOUN", "OP":"*"}, {"LOWER":"motor"}, {"POS":"NOUN", "OP":"*"}], "id":"SSC"}]
  # matcher.addEntityPattern(name, patterns)

  matcher(doc)
