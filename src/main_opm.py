# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
Created on March, 2022

@author: wangc, mandd
"""

import logging
import spacy
import pandas as pd

from nlp.RuleBasedMatcher import RuleBasedMatcher
from nlp import config
from nlp.nlp_utils import generatePatternList

import os
import sys
import argparse
# sr2mlPath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# sys.path.append(sr2mlPath)

# OPL parser to generate object and process lists
from utils.nlpUtils.OPLparser import OPMobject


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
  ###################################################################
  ents = []
  # Parse OPM model
  # some modifications, bearings --> pump bearings
  if 'opm_file' in config.nlpConfig['files']:
    opmFile = config.nlpConfig['files']['opm_file']
    opmObj = OPMobject(opmFile)
    formList = opmObj.returnObjectList()
    functionList = opmObj.returnProcessList()
    attributeList = opmObj.returnAttributeList()
    ents.extend(formList)
  if 'entity_file' in config.nlpConfig['files']:
    entityFile = config.nlpConfig['files']['entity_file']
    entityList = pd.read_csv(entityFile).values.ravel().tolist()
    ents.extend(entityList)
  ents = set(ents)
  # convert opm formList into matcher patternsOPM
  label = "pump_component"
  entId = "SSC"

  patternsOPM = generatePatternList(ents, label=label, id=entId, nlp=nlp, attr="LEMMA")
  ########################################################################
  #  Parse causal keywords, and generate patterns for them
  #  The patterns can be used to identify the causal relationships
  causalLabel = "causal_keywords"
  causalID = "causal"
  patternsCausal = []
  causalFilename = config.nlpConfig['files']['cause_effect_keywords_file']
  ds = pd.read_csv(causalFilename, skipinitialspace=True)
  for col in ds.columns:
    vars = set(ds[col].dropna())
    patternsCausal.extend(generatePatternList(vars, label=causalLabel, id=causalID, nlp=nlp, attr="LEMMA"))

  # text that needs to be processed. either load from file or direct assign
  textFile = config.nlpConfig['files']['text_file']
  with open(textFile, 'r') as ft:
    doc = ft.read()

  # doc = "The Pump is not experiencing enough flow for the pumps to keep the check valves open during test."

  name = 'ssc_entity_ruler'
  matcher = RuleBasedMatcher(nlp, entLabel=entId, causalKeywordLabel=causalID)
  matcher.addEntityPattern(name, patternsOPM)

  causalName = 'causal_keywords_entity_ruler'
  matcher.addEntityPattern(causalName, patternsCausal)

  matcher(doc.lower())
  # matcher(doc)
