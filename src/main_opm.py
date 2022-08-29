# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
Created on March, 2022

@author: wangc, mandd
"""

import logging
from nlp.RuleBasedMatcher import RuleBasedMatcher
import spacy
import pandas as pd

import config

import os
import sys
import argparse
# sr2mlPath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
# sys.path.append(sr2mlPath)

# OPL parser to generate object and process lists
from utils.nlpUtils.OPLparser import OPLentityParser


logging.basicConfig(format='%(asctime)s %(name)-20s %(levelname)-8s %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.DEBUG)
# To enable the logging to both file and console, the logger for the main should be the root,
# otherwise, a function to add the file handler and stream handler need to be created and called by each module.
# logger = logging.getLogger(__name__)
logger = logging.getLogger()
# # create file handler which logs debug messages
fh = logging.FileHandler(filename='main.log', mode='w')
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s %(name)-20s %(levelname)-8s %(message)s')
fh.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh)

#####################################################################
# Utils functions

def generatePattern(form, label, id, attr="LOWER"):
  """
    Generate entity pattern
    @ In, form, str or list, the given str or list of lemmas that will be used to generate pattern
    @ In, label, str, the label name for the pattern
    @ In, id, str, the id name for the pattern
    @ In, attr, str, attribute used for the pattern, either "LOWER" or "LEMMA"
    @ Out, pattern, dict, pattern will be used by entity matcher
  """
  # if any of "!", "?", "+", "*" present in the provided string "form", we will treat it as determiter for the form
  if attr.lower() == "lower":
    attr = "LOWER"
    ptn = [{attr:elem} if elem not in ["!", "?", "+", "*"] else {"POS":"DET", "OP":elem} for elem in form.lower().split()]
  elif attr.lower() == "lemma":
    attr = "LEMMA"
    ptn = [{attr:elem} if elem not in ["!", "?", "+", "*"] else {"POS":"DET", "OP":elem} for elem in form]
  else:
    raise IOError(f"Incorrect 'attr={attr}' is provided, valid value for 'attr' is either 'LOWER' or 'LEMMA'")
  pattern = {"label":label, "pattern":ptn, "id": id}
  return pattern

def extractLemma(var):
  """
    Lammatize the variable list
    @ In, var, str, string
    @ Out, lemVar, list, list of lammatized variables
  """
  var = ' '.join(var.split())
  lemVar = [token.lemma_ for token in nlp(var)]
  return lemVar

#####################################################################


if __name__ == "__main__":
  nlp = spacy.load("en_core_web_lg", exclude=[])
  ###################################################################
  # Parse OPM model
  # opmFile = os.path.abspath("./utils/nlpUtils/pump_opl.html")
  # some modifications, bearings --> pump bearings
  # opmFile = os.path.abspath("./nlp/pump_opl.html")
  opmFile = config.nlpConfig['files']['opm_file']
  formList, functionList = OPLentityParser(opmFile)
  for elem in formList:
    print(elem)
  # convert opm formList into matcher patternsOPM
  label = "pump_component"
  id = "SSC"
  patternsOPM = []
  for form in formList:
    pattern = generatePattern(form, label=label, id=id, attr="LOWER")
    patternsOPM.append(pattern)

  ########################################################################
  #  Parse causal keywords, and generate patterns for them
  #  The patterns can be used to identify the causal relationships
  causalLabel = "causal_keywords"
  causalID = "causal"
  patternsCausal = []
  # causalFilename = os.path.join(os.path.dirname(__file__), 'nlp', 'cause_effect_keywords.csv')
  causalFilename = config.nlpConfig['files']['cause_effect_keywords_file']
  ds = pd.read_csv(causalFilename, skipinitialspace=True)
  for col in ds.columns:
    vars = set(ds[col].dropna())
    for var in vars:
      lemVar = extractLemma(var)
      pattern = generatePattern(lemVar, label=causalLabel, id=causalID, attr="LEMMA")
      patternsCausal.append(pattern)

  # text that needs to be processed.
  # TODO: load text from external files
  textFile = config.nlpConfig['files']['text_file']
  with open(textFile, 'r') as ft:
    doc = ft.read()
  # doc = r"""A leak was noticed from the RCP pump 1A.
  #           RCP pump 1A pressure gauge was found not operating.
  #           RCP pump 1A pressure gauge was found inoperative.
  #           RCP pump 1A had signs of past leakage.
  #           The Pump is not experiencing enough flow during test.
  #           Slight Vibrations is noticed - likely from pump shaft deflection.
  #           Pump flow meter was not responding.
  #           Rupture of pump bearings caused pump shaft degradation.
  #           Rupture of pump bearings caused pump shaft degradation and consequent flow reduction.
  #           Power supply has been found burnout.
  #           Pump test failed due to power supply failure.
  #           Pump inspection revealed excessive impeller degradation.
  #           Pump inspection revealed excessive impeller degradation likely due to cavitation.
  #           Oil puddle was found in proximity of RCP pump 1A.
  #           Anomalous vibrations were observed for RCP pump 1A.
  #           Several cracks on pump shaft were observed; they could have caused pump failure within few days.
  #           RCP pump 1A was cavitating and vibrating to some degree during test.
  #           This is most likely due to low flow conditions rather than mechanical issues.
  #           Cavitation was noticed but did not seem severe.
  #           The pump shaft vibration appears to be causing the motor to vibrate as well.
  #           Pump had noise of cavitation which became faint after OPS bled off the air. Low flow conditions most likely causing cavitation.
  #           The pump shaft deflection is causing the safety cage to rattle.
  #           The Pump is not experiencing enough flow for the pumps to keep the check valves open during test.
  #           Pump shaft made noise.
  #           Vibration seems like it is coming from the pump shaft.
  #           Visible pump shaft deflection in operation.
  #           Pump bearings appear in acceptable condition.
  #           Pump made noises - not enough to affect performance.
  #           Pump shaft has a slight deflection.
  #       """
  # doc = """
  #           Pump inspection revealed excessive impeller degradation.
  #           Pump inspection revealed excessive impeller degradation likely due to cavitation.
  #           Pump and pump shaft revealed excessive impeller degradation.
  # """
  # load nlp model and matcher
  nlp = spacy.load("en_core_web_lg", exclude=[])
  name = 'ssc_entity_ruler'
  matcher = RuleBasedMatcher(nlp, match=True, phraseMatch=True)
  matcher.addEntityPattern(name, patternsOPM)

  causalName = 'causal_keywords_entity_ruler'
  matcher.addEntityPattern(causalName, patternsCausal)

  # ##Issue with simple and phrase matcher, if there are duplicate names, callback functions
  # ##can not be used, in which case, we can not directly extend doc.ents, which will raise the
  # ## error:  Unable to set entity information for token xx which is included in more than
  # ## one span in entities, blocked, missing or outside. (https://github.com/explosion/spaCy/discussions/9993)
  # ## In this case, we need to use "entity_ruler" to identify the matches
  # # simple match
  # name = 'ssc_match'
  # rules = [{"LOWER":"pump"}, {"POS":"NOUN"}]
  # matcher.addPattern(name, rules, callback=None)
  # # phrase match
  # name = 'ssc_phrase_match'
  # phraseList = ['pump 1A', 'pump bearings', 'Pump']
  # matcher.addPhrase(name, phraseList, callback=None)
  # # depency match
  # name = 'ssc_dependency_match'
  # dependencyList = [
  #   {
  #     "RIGHT_ID":"anchor_found",
  #     "RIGHT_ATTRS":{"ORTH":"found"}
  #   },
  #   {
  #     "LEFT_ID":"anchor_found",
  #     "REL_OP":">",
  #     "RIGHT_ID":"subject",
  #     "RIGHT_ATTRS":{"DEP":"nsubjpass"}
  #   },
  #   {
  #     "LEFT_ID":"anchor_found",
  #     "REL_OP":">",
  #     "RIGHT_ID":"object",
  #     "RIGHT_ATTRS":{"DEP":"oprd"}
  #   }
  # ]
  # matcher.addDependency(name, dependencyList, callback=None)


  matcher(doc)
