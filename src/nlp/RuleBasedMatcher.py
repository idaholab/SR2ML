# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
Created on March, 2022

@author: wangc, mandd
"""
import pandas as pd
import re
import copy
import spacy
from spacy.matcher import Matcher
from spacy.tokens import Token
from spacy.tokens import Span
from spacy import displacy
from spacy.matcher import PhraseMatcher
from spacy.matcher import DependencyMatcher
from collections import deque
# filter_spans is used to resolve the overlap issue in entities
# It gives primacy to longer spans (entities)
from spacy.util import filter_spans
from .nlp_utils import displayNER, resetPipeline, printDepTree
from .nlp_utils import extendEnt
## import pipelines
from .CustomPipelineComponents import normEntities
from .CustomPipelineComponents import initCoref
from .CustomPipelineComponents import aliasResolver
from .CustomPipelineComponents import anaphorCoref
from .CustomPipelineComponents import anaphorEntCoref
from .CustomPipelineComponents import mergePhrase
from .CustomPipelineComponents import pysbdSentenceBoundaries

from .config import nlpConfig


import logging
import os


logger = logging.getLogger(__name__)

## temporary add stream handler
# ch = logging.StreamHandler()
# logger.addHandler(ch)
##

## coreferee module for Coreference Resolution
## Q? at which level to perform coreferee? After NER and perform coreferee on collected sentence
_corefAvail = False
try:
  # check the current version spacy>=3.0.0,<=3.3.0
  from packaging.version import Version
  ver = spacy.__version__
  valid = Version(ver)>=Version('3.0.0') and Version(ver)<=Version('3.3.0')
  if valid:
    # https://github.com/msg-systems/coreferee
    import coreferee
    _corefAvail = True
  else:
    logger.info(f'Module coreferee is not compatible with spacy version {ver}')
except ModuleNotFoundError:
  logger.info('Module coreferee can not be imported')


if not Span.has_extension('health_status'):
  Span.set_extension("health_status", default=None)
if not Token.has_extension('health_status'):
  Token.set_extension("health_status", default=None)
if not Span.has_extension('hs_keyword'):
  Span.set_extension('hs_keyword', default=None)
if not Span.has_extension('ent_status_verb'):
  Span.set_extension('ent_status_verb', default=None)
if not Span.has_extension('conjecture'):
  Span.set_extension('conjecture', default=False)

class RuleBasedMatcher(object):
  """
    Rule Based Matcher Class
  """
  def __init__(self, nlp, entLabel='SSC', causalKeywordLabel='causal', *args, **kwargs):
    """
      Construct
      @ In, nlp, spacy.Language object, contains all components and data needed to process text
      @ In, args, list, positional arguments
      @ In, kwargs, dict, keyword arguments
      @ Out, None
    """
    self.type = self.__class__.__name__
    self.name = self.__class__.__name__
    logger.info(f'Create instance of {self.name}')
    # orders of NLP pipeline: 'ner' --> 'normEntities' --> 'merge_entities' --> 'initCoref'
    # --> 'aliasResolver' --> 'coreferee' --> 'anaphorCoref'
    # pipeline 'merge_noun_chunks' can be used to merge phrases (see also displacy option)
    self.nlp = nlp
    self._causalFile = nlpConfig['files']['cause_effect_keywords_file']
    # SCONJ->Because, CCONJ->so, ADP->as, ADV->therefore
    self._causalPOS = {'VERB':['VERB'], 'NOUN':['NOUN'], 'TRANSITION':['SCONJ', 'CCONJ', 'ADP', 'ADV']}
    # current columns include: "VERB", "NOUN", "TRANSITION", "causal-relator", "effect-relator", "causal-noun", "effect-noun"
    # For relator, such as becaue, therefore, as, etc.
    #   if the column starts with causal, which means causal entity --> keyword --> effect entity
    #   if the column starts with effect, which means effect entity <-- keyword <-- causal entity
    # For NOUN
    #   if the column starts with causal, which means causal entity --> keyword --> effect entity
    #   if the column starts with effect, the relation is depend on the keyword.dep_
    #   First check the right child of the keyword is ADP with dep_ "prep",
    #   Then, check the dep_ of keyword, if it is "dobj", then causal entity --> keyword --> effect entity
    #   elif it is "nsubj" or "nsubjpass" or "attr", then effect entity <-- keyword <-- causal entity
    self._causalKeywords = self.getKeywords(self._causalFile)
    self._statusFile = nlpConfig['files']['status_keywords_file']['all']
    self._statusKeywords = self.getKeywords(self._statusFile)
    self._updateStatusKeywords = False
    self._updateCausalKeywords = False
    self._conjectureFile = nlpConfig['files']['conjecture_keywords_file']
    self._conjectureKeywords = self.getKeywords(self._conjectureFile, columnNames=['conjecture-keywords'])
    ## pipelines "merge_entities" and "merge_noun_chunks" can be used to merge noun phrases and entities
    ## for easier analysis
    if _corefAvail:
      self.pipelines = ['pysbdSentenceBoundaries',
                      'mergePhrase', 'normEntities', 'initCoref', 'aliasResolver',
                      'coreferee','anaphorCoref', 'anaphorEntCoref']
    else:
      self.pipelines = ['pysbdSentenceBoundaries',
                      'mergePhrase','normEntities', 'initCoref', 'aliasResolver',
                      'anaphorCoref', 'anaphorEntCoref']
    # ner pipeline is not needed since we are focusing on the keyword matching approach
    if nlp.has_pipe("ner"):
      nlp.remove_pipe("ner")
    nlp = resetPipeline(nlp, self.pipelines)
    self.nlp = nlp
    self._doc = None
    self.entityRuler = None
    self._entityRuler = False
    self._entityRulerMatches = []
    self._matchedSents = [] # collect data of matched sentences
    self._matchedSentsForVis = [] # collect data of matched sentences to be visualized
    self._visualizeMatchedSents = True
    self._coref = _corefAvail # True indicate coreference pipeline is available
    self._entityLabels = {} # labels for rule-based entities
    self._labelSSC = entLabel
    self._labelCausal = causalKeywordLabel
    self._causalNames = ['cause', 'cause health status', 'causal keyword', 'effect', 'effect health status', 'sentence', 'conjecture']
    self._extractedCausals = [] # list of tuples, each tuple represents one causal-effect, i.e., (cause, cause health status, cause keyword, effect, effect health status, sentence)
    self._causalSentsNoEnts = []
    self._rawCausalList = []
    self._causalSentsOneEnt = []
    self._entHS = None

  def reset(self):
    """
      Reset rule-based matcher
    """
    self._matchedSents = []
    self._matchedSentsForVis = []
    self._extractedCausals = []
    self._causalSentsNoEnts = []
    self._rawCausalList = []
    self._causalSentsOneEnt = []
    self._entHS = None
    self._doc = None

  def getKeywords(self, filename, columnNames=None):
    """
      Get the keywords from given file
      @ In, filename, str, the file name to read the keywords
      @ Out, kw, dict, dictionary contains the keywords
    """
    kw = {}
    if columnNames is not None:
      ds = pd.read_csv(filename, skipinitialspace=True, names=columnNames)
    else:
      ds = pd.read_csv(filename, skipinitialspace=True)
    for col in ds.columns:
      vars = set(ds[col].dropna())
      kw[col] = self.extractLemma(vars)
    return kw

  def extractLemma(self, varList):
    """
      Lammatize the variable list
      @ In, varList, list, list of variables
      @ Out, lemmaList, list, list of lammatized variables
    """
    lemmaList = []
    for var in varList:
      lemVar = [token.lemma_.lower() for token in self.nlp(var) if token.lemma_ not in ["!", "?", "+", "*"]]
      lemmaList.append(lemVar)
    return lemmaList

  def addKeywords(self, keywords, ktype):
    """
      Method to update self._causalKeywords or self._statusKeywords
      @ In, keywords, dict, keywords that will be add to self._causalKeywords or self._statusKeywords
      @ In, ktype, string, either 'status' or 'causal'
    """
    if type(keywords) != dict:
      raise IOError('"addCausalKeywords" method can only accept dictionary, but got {}'.format(type(keywords)))
    if ktype.lower() == 'status':
      for key, val in keywords.items():
        if type(val) != list:
          val = [val]
        val = self.extractLemma(val)
        if key in self._statusKeywords:
          self._statusKeywords[key].append(val)
        else:
          logger.warning('keyword "{}" cannot be accepted, valid keys for the keywords are "{}"'.format(key, ','.join(list(self._statusKeywords.keys()))))
    elif ktype.lower() == 'causal':
      for key, val in keywords.items():
        if type(val) != list:
          val = [val]
        val = self.extractLemma(val)
        if key in self._causalKeywords:
          self._causalKeywords[key].append(val)
        else:
          logger.warning('keyword "{}" cannot be accepted, valid keys for the keywords are "{}"'.format(key, ','.join(list(self._causalKeywords.keys()))))

  def addEntityPattern(self, name, patternList):
    """
      Add entity pattern, to extend doc.ents, similar function to self.extendEnt
      @ In, name, str, the name for the entity pattern.
      @ In, patternList, list, the pattern list, for example:
        {"label": "GPE", "pattern": [{"LOWER": "san"}, {"LOWER": "francisco"}]}
    """
    if not self.nlp.has_pipe('entity_ruler'):
      self.nlp.add_pipe('entity_ruler', before='mergePhrase')
      self.entityRuler = self.nlp.get_pipe("entity_ruler")
    if not isinstance(patternList, list):
      patternList = [patternList]
    # TODO: able to check "id" and "label", able to use "name"
    for pa in patternList:
      label = pa.get('label')
      id = pa.get('id')
      if id is not None:
        if id not in self._entityLabels:
          self._entityLabels[id] = set([label]) if label is not None else set()
        else:
          self._entityLabels[id] = self._entityLabels[id].union(set([label])) if label is not None else set()
    # self._entityLabels += [pa.get('label') for pa in patternList if pa.get('label') is not None]
    self.entityRuler.add_patterns(patternList)
    if not self._entityRuler:
      self._entityRuler = True

  def __call__(self, text):
    """
      Find all token sequences matching the supplied pattern
      @ In, text, string, the text that need to be processed
      @ Out, None
    """
    # Merging Entity Tokens
    # We need to consider how to do this, I sugguest to first conduct rule based NER, then collect
    # all related sentences, then create new pipelines to perform NER with "merge_entities" before the
    # conduction of relationship extraction
    # if self.nlp.has_pipe('merge_entities'):
    #   _ = self.nlp.remove_pipe('merge_entities')
    # self.nlp.add_pipe('merge_entities')
    doc = self.nlp(text)
    self._doc = doc
    ## use entity ruler to identify entity
    # if self._entityRuler:
    #   logger.debug('Entity Ruler Matches:')
    #   print([(ent.text, ent.label_, ent.ent_id_) for ent in doc.ents if ent.label_ in self._entityLabels[self._labelSSC]])

    # First identify coreference through coreferee, then filter it through doc.ents
    if self._coref:
      corefRep = doc._.coref_chains.pretty_representation
      if len(corefRep) != 0:
        logger.debug('Print Coreference Info:')
        print(corefRep)

    matchedSents, matchedSentsForVis = self.collectSents(self._doc)
    self._matchedSents += matchedSents
    self._matchedSentsForVis += matchedSentsForVis
    ## health status
    logger.info('Start to extract health status')
    self.extractHealthStatus(self._matchedSents)
    ## Access health status and output to an ordered csv file
    entList = []
    hsList = []
    svList = []
    kwList = []
    cjList = []
    sentList = []
    for sent in self._matchedSents:
      ents = self.getCustomEnts(sent.ents, self._entityLabels[self._labelSSC])
      elist = [ent.text for ent in ents]
      statusVerb = [ent._.ent_status_verb for ent in ents]
      hs = [ent._.health_status for ent in ents]
      kw = [ent._.hs_keyword for ent in ents]
      cj = [ent._.conjecture for ent in ents]
      sl = [sent.text.strip('\n') for ent in ents]
      entList.extend(elist)
      hsList.extend(hs)
      svList.extend(statusVerb)
      kwList.extend(kw)
      cjList.extend(cj)
      sentList.extend(sl)

    ## include 'root' in the output
    df = pd.DataFrame({'entities':entList, 'root':svList, 'status keywords':kwList, 'health statuses':hsList, 'conjecture':cjList, 'sentence':sentList})
    df.to_csv(nlpConfig['files']['output_health_status_file'], columns=['entities', 'root','status keywords', 'health statuses', 'conjecture', 'sentence'])
    self._entHS = df
    # df = pd.DataFrame({'entities':entList, 'status keywords':kwList, 'health statuses':hsList, 'conjecture':cjList, 'sentence':sentList})
    # df.to_csv(nlpConfig['files']['output_health_status_file'], columns=['entities', 'status keywords', 'health statuses', 'conjecture', 'sentence'])

    logger.info('End of health status extraction!')
    ## causal relation
    logger.info('Start to extract causal relation using OPM model information')
    self.extractRelDep(self._matchedSents)
    dfCausals = pd.DataFrame(self._extractedCausals, columns=self._causalNames)
    dfCausals.to_csv(nlpConfig['files']['output_causal_effect_file'], columns=self._causalNames)
    logger.info('End of causal relation extraction!')
    ## print extracted relation
    # logger.info('Start to use general extraction method to extract causal relation')
    # print(*self.extract(self._matchedSents, predSynonyms=self._causalKeywords['VERB'], exclPrepos=[]), sep='\n')
    # logger.info('End of causal relation extraction using general extraction method!')

  def visualize(self):
    """
      Visualize the processed document
      @ In, None
      @ Out, None
    """
    if self._visualizeMatchedSents:
      # Serve visualization of sentences containing match with displaCy
      # set manual=True to make displaCy render straight from a dictionary
      # (if you're running the code within a Jupyer environment, you can
      # use displacy.render instead)
      # displacy.render(self._matchedSentsForVis, style="ent", manual=True)
      displacy.serve(self._matchedSentsForVis, style="ent", manual=True)

  ##########################
  # methods for relation extraction
  ##########################

  def isPassive(self, token):
    """
      Check the passiveness of the token
      @ In, token, spacy.tokens.Token, the token of the doc
      @ Out, isPassive, True, if the token is passive
    """
    if token.dep_.endswith('pass'): # noun
      return True
    for left in token.lefts: # verb
      if left.dep_ == 'auxpass':
        return True
    return False

  def isConjecture(self, token):
    """
      Check the conjecture of the token
      @ In, token, spacy.tokens.Token, the token of the doc, the token should be the root of the Doc
      @ Out, isConjecture, True, if the token/sentence indicates conjecture
    """
    for left in token.lefts: # Check modal auxiliary verb: can, could, may, might, must, shall, should, will, would
      if left.dep_.startswith('aux') and left.tag_ in ['MD']:
        return True
    if token.pos_ == 'VERB' and token.tag_ == 'VB': # If it is a verb, and there is no inflectional morphology for the verb
      return True
    # check the keywords
    # FIXME: should we use token.subtree or token.children here
    for child in token.subtree:
      if [child.lemma_.lower()] in self._conjectureKeywords['conjecture-keywords']:
        return True
    return False

  def isNegation(self, token):
    """
      Check negation status of given token
      @ In, token, spacy.tokens.Token, token from spacy.tokens.doc.Doc
      @ Out, (neg, text), tuple, the negation status and the token text
    """
    neg = False
    text = ''
    if token.dep_ == 'neg':
      neg = True
      text = token.text
      return neg, text
    # check left for verbs
    for left in token.lefts:
      if left.dep_ == 'neg':
        neg = True
        text = left.text
        return neg, text
    # The following can be used to check the negation status of the sentence
    # # check the subtree
    # for sub in token.subtree:
    #   if sub.dep_ == 'neg':
    #     neg = True
    #     text = sub.text
    #     return neg, text
    return neg, text

  def findVerb(self, doc):
    """
      Find the first verb in the doc
      @ In, doc, spacy.tokens.doc.Doc, the processed document using nlp pipelines
      @ Out, token, spacy.tokens.Token, the token that has VERB pos
    """
    for token in doc:
      if token.pos_ == 'VERB':
        return token
        break
    return None

  def getCustomEnts(self, ents, labels):
    """
      Get the custom entities
      @ In, ents, list, all entities from the processed doc
      @ In, labels, list, list of labels to be used to get the custom entities out of "ents"
      @ Out, customEnts, list, the customEnts associates with the "labels"
    """
    customEnts = [ent for ent in ents if ent.label_ in labels]
    if len(customEnts) == 0:
      customEnts = None
    return customEnts

  def getHealthStatusForPobj(self, ent, include=False):
    """
      Get the status for ent root pos_ 'pobj'
      @ In, ent, Span, the span of entity
      @ In, include, bool, ent will be included in returned status if True
      @ Out, healthStatus, Span or Token, the identified status
    """
    healthStatus = None
    if isinstance(ent, Token):
      root = ent
      start = root.i
      end = start + 1
    elif isinstance(ent, Span):
      root = ent.root
      start = ent.start
      end = ent.end
    if root.dep_ not in ['pobj']:
      return healthStatus
    grandparent = root.head.head
    parent = root.head
    causalStatus = [grandparent.lemma_.lower()] in self._causalKeywords['VERB'] and [grandparent.lemma_.lower()] not in self._statusKeywords['VERB']
    if grandparent.dep_ in ['dobj', 'nsubj', 'nsubjpass', 'pobj']:
      lefts = list(grandparent.lefts)
      if len(lefts) == 0:
        leftInd = grandparent.i
      else:
        leftInd = lefts[0].i
      if not include:
        rights = list(grandparent.rights)
        if grandparent.n_rights > 1 and rights[-1] == parent:
          healthStatus = grandparent.doc[leftInd:rights[-1].i]
        else:
          healthStatus = grandparent.doc[leftInd:grandparent.i+1]
      else:
        healthStatus = grandparent.doc[leftInd:end]
      healthStatus = self.getAmod(healthStatus, healthStatus.start, healthStatus.end, include=True)
    elif grandparent.pos_ in ['VERB'] and causalStatus:
      healthStatus = self.findRightObj(grandparent)
      subtree = list(healthStatus.subtree)
      nbor = self.getNbor(healthStatus)
      if healthStatus is not None and nbor is not None and nbor.dep_ in ['prep'] and subtree[-1].i < root.i:
        healthStatus = grandparent.doc[healthStatus.i:subtree[-1].i+1]
      elif healthStatus is not None and healthStatus.i >= root.i:
        healthStatus = None
    elif grandparent.pos_ in ['VERB'] and grandparent.dep_ in ['ROOT']:
      dobj = [tk for tk in grandparent.rights if tk.dep_ in ['dobj'] and tk.i < start]
      if len(dobj) > 0:
        dobjEnt = root.doc[dobj[0].i:dobj[0].i+1]
        healthStatus = self.getAmod(dobjEnt, dobjEnt.start, dobjEnt.end, include=True)
      else:
        healthStatus = ent
        healthStatus = self.getAmod(ent, start, end, include=include)
    elif grandparent.pos_ in ['NOUN']:
      grandEnt = grandparent.doc[grandparent.i:grandparent.i+1]
      healthStatus = self.getAmod(grandEnt, grandparent.i, grandparent.i+1, include=True)
    elif grandparent.pos_ in ['AUX']:
      healthStatus = grandparent.doc[grandparent.i+1:parent.i]
    else: # search lefts for amod
      healthStatus = self.getAmod(ent, start, end, include)
    return healthStatus

  def getPhrase(self, ent, start, end, include=False):
    """
      Get the phrase for ent with all left children
      @ In, ent, Span, the ent to amend with all left children
      @ In, start, int, the start index of ent
      @ In, end, int, the end index of ent
      @ In, include, bool, include ent in the returned expression if True
      @ Out, healthStatus, Span or Token, the identified status
    """
    leftInd = list(ent.lefts)[0].i
    if not include:
      healthStatus = ent.doc[leftInd:start]
    else:
      healthStatus = ent.doc[leftInd:end]
    return healthStatus

  def getAmod(self, ent, start, end, include = False):
    """
      Get amod tokens for ent
      @ In, ent, Span, the ent to amend with all left children
      @ In, start, int, the start index of ent
      @ In, end, int, the end index of ent
      @ In, include, bool, include ent in the returned expression if True
      @ Out, healthStatus, Span or Token, the identified status
    """
    healthStatus = None
    deps = [tk.dep_ in ['amod'] for tk in ent.lefts]
    if any(deps):
      healthStatus = self.getPhrase(ent, start, end, include)
    else:
      deps = [tk.dep_ in ['compound'] for tk in ent.lefts]
      if any(deps):
        healthStatus = self.getPhrase(ent, start, end, include)
        healthStatus = self.getAmod(healthStatus, healthStatus.start, healthStatus.end, include=True)
    if healthStatus is None and include:
      healthStatus = ent
    return healthStatus

  def getAmodOnly(self, ent):
    """
      Get amod tokens texts for ent
      @ In, ent, Span, the ent to amend with all left children
      @ Out, amod, list, the list of amods for ent
    """
    amod = [tk.text for tk in ent.lefts if tk.dep_ in ['amod']]
    return amod

  def getCompoundOnly(self, headEnt, ent):
    """
      Get the compounds for headEnt except ent
      @ In, headEnt, Span, the head entity to ent
      @ Out, compDes, list, the list of compounds for head ent
    """
    compDes = []
    comp = [tk for tk in headEnt.lefts if tk.dep_ in ['compound'] and tk not in ent]
    if len(comp) > 0:
      for elem in comp:
        des = [tk.text for tk in elem.lefts if tk.dep_ in ['amod', 'compound'] and tk not in ent]
        compDes.extend(des)
        compDes.append(elem.text)
    return compDes

  def getNbor(self, token):
    """
      Method to get the nbor from token, return None if nbor is not exist
      @ In, token, Token, the provided Token to request nbor
      @ Out, nbor, Token, the requested nbor
    """
    nbor = None
    if token is None:
      return nbor
    try:
      nbor = token.nbor()
    except IndexError:
      pass
    return nbor

  def getHealthStatusForSubj(self, ent, entHS, sent, causalStatus, predSynonyms, include=False):
    """
      Get the status for nsubj/nsubjpass ent
      @ In, ent, Span, the nsubj/nsubjpass ent that will be used to search status
      @ In, entHS, Span, the entHS that the status will be associated with
      @ In, sent, Span, the sent that includes the ent, entHS and status
      @ In, causalStatus, bool, the causal status for the ent
      @ In, predSynonyms, list, predicate synonyms
      @ In, include, bool, include ent in the returned expression if True
      @ Out, healthStatus, Span or Token, the identified status
    """
    healthStatus = None
    neg = False
    negText = ''
    entRoot = ent.root
    # root = sent.root
    root = entRoot.head
    causalStatus = [root.lemma_.lower()] in self._causalKeywords['VERB'] and [root.lemma_.lower()] not in self._statusKeywords['VERB']
    if entRoot.dep_ not in ['nsubj', 'nsubjpass']:
      return healthStatus, neg, negText
    if root.pos_ != 'VERB':
      neg, negText = self.isNegation(root)
      if root.pos_ in ['NOUN', 'ADJ']:
        # TODO: search entRoot.lefts for 'amod' and attach to healthStatus
        healthStatus = root
      elif root.pos_ in ['AUX']:
        healthStatus = root.doc[root.i+1:root.i+root.n_rights+1]
      else:
        logger.warning(f'No status identified for "{ent}" in "{sent}"')
    else:
      rights = root.rights
      valid = [tk.dep_ in ['advcl', 'relcl'] for tk in rights if tk.pos_ not in ['PUNCT', 'SPACE']]
      nbor = self.getNbor(root)
      if nbor is not None and (nbor.dep_ in ['cc'] or nbor.pos_ in ['PUNCT']):
        healthStatus = root
      elif len(valid)>0 and all(valid):
        healthStatus = root
      elif not causalStatus:
        if [root.lemma_.lower()] in predSynonyms:
          entHS._.set('hs_keyword', root.lemma_)
        else:
          entHS._.set('ent_status_verb', root.lemma_)
        neg, negText = self.isNegation(root)
        passive = self.isPassive(root)
        # # last is punct, the one before last is the root
        # if root.nbor().pos_ in ['PUNCT']:
        #   healthStatus = root
        healthStatus = self.findRightObj(root)
        if healthStatus and healthStatus.dep_ == 'pobj':
          healthStatus = self.getHealthStatusForPobj(healthStatus, include=True)
        elif healthStatus and healthStatus.dep_ == 'dobj':
          subtree = list(healthStatus.subtree)
          try:
            if healthStatus.nbor().dep_ in ['prep']:
              healthStatus = healthStatus.doc[healthStatus.i:subtree[-1].i+1]
          except IndexError:
            pass
        # no object is found
        if not healthStatus:
          healthStatus = self.findRightKeyword(root)
        # last is punct, the one before last is the root
        # if not healthStatus and root.nbor().pos_ in ['PUNCT']:
        #   healthStatus = root
        if healthStatus is None:
          healthStatus = self.getAmod(ent, ent.start, ent.end, include=include)
        if healthStatus is None:
          extra = [tk for tk in root.rights if tk.pos_ in ['ADP', 'ADJ']]
          # Only select the first ADP and combine with root
          if len(extra) > 0:
            healthStatus = root.doc[root.i:extra[0].i+1]
          else:
            healthStatus = root
      else:
        healthStatus = self.getAmod(ent, ent.start, ent.end, include=include)
      if healthStatus is None:
        healthStatus = root
    return healthStatus, neg, negText

  def getHealthStatusForObj(self, ent, entHS, sent, causalStatus, predSynonyms, include=False):
    """
      Get the status for pobj/dobj ent
      @ In, ent, Span, the pobj/dobj ent that will be used to search status
      @ In, entHS, Span, the entHS that the status will be associated with
      @ In, sent, Span, the sent that includes the ent, entHS and status
      @ In, causalStatus, bool, the causal status for the ent
      @ In, predSynonyms, list, predicate synonyms
      @ In, include, bool, include ent in the returned expression if True
      @ Out, healthStatus, Span or Token, the identified status
    """
    healthStatus = None
    neg = False

    negText = ''
    entRoot = ent.root
    head = entRoot.head
    prep = False
    if head.pos_ in ['VERB']:
      root = head
    elif head.dep_ in ['prep']:
      root = head.head
      prep = True
    else:
      root = head
    causalStatus = [root.lemma_.lower()] in self._causalKeywords['VERB'] and [root.lemma_.lower()] not in self._statusKeywords['VERB']
    if entRoot.dep_ not in ['pobj', 'dobj']:
      return healthStatus, neg, negText
    if root.pos_ != 'VERB':
      neg, negText = self.isNegation(root)
      if root.pos_ in ['ADJ']:
        healthStatus = root
      elif root.pos_ in ['NOUN']:
        if root.dep_ in ['pobj']:
          healthStatus = root.doc[root.head.head.i:root.i+1]
        else:
          healthStatus = root
      elif root.pos_ in ['AUX']:
        leftInd = list(root.lefts)[0].i
        healthStatus = root.doc[leftInd:root.i]
      else:
        logger.warning(f'No status identified for "{ent}" in "{sent}"')
    else:
      if not causalStatus:
        if [root.lemma_.lower()] in predSynonyms:
          entHS._.set('hs_keyword', root.lemma_)
        else:
          entHS._.set('ent_status_verb', root.lemma_)
        passive = self.isPassive(root)
        neg, negText = self.isNegation(root)
        healthStatus = self.findLeftSubj(root, passive)
        if healthStatus is not None and healthStatus.pos_ in ['PRON']:
          # coreference resolution
          passive = self.isPassive(root.head)
          neg, negText = self.isNegation(root.head)
          healthStatus = self.findLeftSubj(root.head, passive)
        if healthStatus is not None:
          healthStatus = self.getAmod(healthStatus, healthStatus.i, healthStatus.i+1, include=True)
        else:
          healthStatus = self.getAmod(ent, ent.start, ent.end, include=include)
          # healthStatus = self.getCompoundOnly(ent, entHS)
        if healthStatus is None:
          rights =[tk for tk in list(root.rights) if tk.pos_ not in ['SPACE', 'PUNCT'] and tk.i >= ent.end]
          if len(rights) > 0 and rights[0].pos_ in ['VERB', 'NOUN', 'ADJ', 'ADV']:
            healthStatus = rights[0]
      else:
        if entRoot.dep_ in ['pobj']:
          healthStatus = self.getHealthStatusForPobj(ent, include=include)
        else:
          healthStatus = self.getAmod(ent, ent.start, ent.end, include=include)
    return healthStatus, neg, negText

  def validSent(self, sent):
    """
      Check if the sentence has valid structure, either contains subject or object
      @ In, sent, Span, sentence from user provided text
      @ Out, valid, bool, False if the sentence has no subject and object.
    """
    foundSubj = False
    foundObj = False
    valid = True
    for tk in sent:
      if tk.dep_.startswith('nsubj'):
        foundSubj = True
      elif tk.dep_.endswith('obj'):
        foundObj = True
    if not foundSubj and not foundObj:
      valid = False
    return valid


  def extractHealthStatus(self, matchedSents, predSynonyms=[], exclPrepos=[]):
    """
      Extract health status and relation
      @ In, matchedSents, list, the matched sentences
      @ In, predSynonyms, list, predicate synonyms
      @ In, exclPrepos, list, exclude the prepositions
    """
    #  first search degradation keywords,
    #  if pobj, then if head.head
    # Ex. 1: an acrid odor in the control room --> entRoot.dep_ in ['pobj'], HS: entRoot.head.head i.e.,
    # let i = entRoot.head.head.i, start = sent.start, nlefts = entRoot.head.head.n_lefts
    # healthStatus = sent[i-start-nlefts:i-start+1]
    # or entRoot.head in 'amod', checking the n_lefts, as above
    # Ex. 2: shaft degradation --> entRoot.dep_ in ['compound'], HS: entRoot.head and entRoot.head.pos_ in in ['NUM']
    # Ex. 3: entRoot.dep_ in ['nsubj', 'nsubjpass']
    # sent.root before ent, search left for nsubj or nsubjpass and any 'amod', 'compound', 'det'
    # sent.root after ent, search right for pobj, and check head.head.dep_ in ['dobj', 'nsubjpass', 'nsubj'], amend it with any 'amod', 'compound', 'det' in its lefts
    # should report both dobj and pobj for the health status
    # if sent.root, check right for 'cc' and 'conj', if next is 'cc', return the root
    #  if entRoot.dep_ in ['conj'], if entRoot.head.dep_ in ['nmod'], return entRoot.head.head
    #  entRoot.dep_ in ['dobj'] and entRoot.head.pos_ in ['VERB'], return the entRoot.head
    # dobj: left children (amod, compound, det) and right childrend, (have signs of overheating)
    # nsubj -- VERB -- pobj :  if pobj is NUM, then everything between VERB and pobj

    predSynonyms = self._statusKeywords['VERB']
    statusNoun = self._statusKeywords['NOUN']
    statusAdj = self._statusKeywords['ADJ']
    causalStatus = False

    for sent in matchedSents:
      valid = self.validSent(sent)
      causalEnts = None
      if self._labelCausal in self._entityLabels:
        causalEnts = self.getCustomEnts(sent.ents, self._entityLabels[self._labelCausal])
      ents = self.getCustomEnts(sent.ents, self._entityLabels[self._labelSSC])
      if ents is None:
        continue
      causalStatus = [sent.root.lemma_.lower()] in self._causalKeywords['VERB'] and [sent.root.lemma_.lower()] not in self._statusKeywords['VERB']
      for ent in ents:
        healthStatus = None        # store health status for identified entities
        healthStatusAmod = None    # store amod for health status
        healthStatusAppend = None  # store some append info for health status (used for short phrase)
        healthStatusAppendAmod = None # store amod info for health status append info
        healthStatusPrepend = None  # store some prepend info
        healthStatusPrependAmod = None # store amod info for prepend info
        healthStatusText = None
        conjecture = False
        passive = False
        entRoot = ent.root
        root = sent.root
        neg = False
        if valid:
          if entRoot.dep_ in ['nsubj', 'nsubjpass']:
            healthStatus, neg, negText = self.getHealthStatusForSubj(ent, ent, sent, causalStatus, predSynonyms)
          elif entRoot.dep_ in ['pobj', 'dobj']:
            if len(ents) == 1 or entRoot.dep_ in ['dobj']:
              healthStatus, neg, negText = self.getHealthStatusForObj(ent, ent, sent, causalStatus, predSynonyms)
              if entRoot.dep_ in ['pobj']:
                # extract append info for health status
                prep = entRoot.head
                healthStatusAppendAmod = self.getCompoundOnly(ent, ent)
                if len(healthStatusAppendAmod) > 0:
                  healthStatusAppendAmod = [prep.text] + healthStatusAppendAmod
                  healthStatusAppend = ent
            else:
              healthStatus = self.getHealthStatusForPobj(ent, include=False)
            if healthStatus is None:
              head = entRoot.head
              if head.dep_ in ['xcomp', 'advcl', 'relcl']:
                for child in head.rights:
                  if child.dep_ in ['ccomp']:
                    healthStatus = child
                    break
          elif entRoot.dep_ in ['compound']:
            head = entRoot.head
            if head.pos_ not in ['SPACE', 'PUNCT']:
              if len(ents) == 1:
                if head.dep_ in ['compound']:
                  head = head.head
                headEnt = head.doc[head.i:head.i+1]
                if head.dep_ in ['nsubj', 'nsubjpass']:
                  healthStatus, neg, negText = self.getHealthStatusForSubj(headEnt, ent, sent, causalStatus, predSynonyms, include=True)
                  if isinstance(healthStatus, Span):
                    if entRoot.i >= healthStatus.start and entRoot.i < healthStatus.end:
                      healthStatus = headEnt
                  if healthStatus is not None:
                    healthStatusPrepend = headEnt
                    healthStatusPrependAmod = self.getAmodOnly(headEnt)
                elif head.dep_ in ['dobj', 'pobj']:
                  healthStatus, neg, negText = self.getHealthStatusForObj(headEnt, ent, sent, causalStatus, predSynonyms, include=False)
                  if healthStatus is not None:
                    if isinstance(healthStatus, Span):
                      if head not in healthStatus:
                        # identify the dobj/pobj, and use it as append info
                        healthStatusAppend = headEnt
                        healthStatusAppendAmod = self.getAmodOnly(headEnt)
                    elif isinstance(healthStatus, Token):
                      if head != healthStatus:
                        # identify the dobj/pobj, and use it as append info
                        healthStatusAppend = headEnt
                        healthStatusAppendAmod = self.getAmodOnly(headEnt)
                if healthStatus is None:
                  healthStatus = headEnt
              else:
                healthStatus = entRoot.head
                healthStatusAmod = self.getAmodOnly(healthStatus)
                if len(healthStatusAmod) == 0:
                  lefts = list(healthStatus.lefts)
                  # remove entity itself
                  for elem in lefts:
                    if elem in ent:
                      lefts.remove(elem)
                  if len(lefts) != 0:
                    healthStatusAmod = [e.text for e in lefts]
                if head.dep_ in ['dobj','pobj','nsubj', 'nsubjpass'] and [root.lemma_.lower()] in predSynonyms:
                  ent._.set('hs_keyword', root.lemma_)
            else:
              healthStatus = self.getAmod(ent, ent.start, ent.end, include=False)

          elif entRoot.dep_ in ['conj']:
            # TODO: recursive function to retrieve non-conj
            healthStatus = self.getAmod(ent, ent.start, ent.end, include=False)
            if healthStatus is None:
              head = entRoot.head
              if head.dep_ in ['conj']:
                head = head.head
              headEnt = head.doc[head.i:head.i+1]
              if head.dep_ in ['nsubj', 'nsubjpass']:
                healthStatus, neg, negText = self.getHealthStatusForSubj(headEnt, ent, sent, causalStatus, predSynonyms)
              elif head.dep_ in ['pobj', 'dobj']:
                healthStatus = self.getHealthStatusForPobj(headEnt, include=False)
                if healthStatus is None:
                  healthStatus, neg, negText = self.getHealthStatusForObj(headEnt, ent, sent, causalStatus, predSynonyms)
          elif entRoot.dep_ in ['ROOT']:
            healthStatus = self.getAmod(ent, ent.start, ent.end, include=False)
            if healthStatus is None:
              rights =[tk for tk in list(entRoot.rights) if tk.pos_ in ['VERB', 'NOUN', 'ADJ', 'ADV'] and tk.i >= ent.end]
              if len(rights) > 0:
                healthStatus = rights[0]
          else:
            logger.warning(f'Entity "{ent}" dep_ is "{entRoot.dep_}" is not among valid list "[nsubj, nsubjpass, pobj, dobj, compound]"')
            if entRoot.head == root:
              headEnt = root.doc[root.i:root.i+1]
              if ent.start < root.i:
                if [root.lemma_.lower()] in predSynonyms:
                  ent._.set('hs_keyword', root.lemma_)
                else:
                  ent._.set('ent_status_verb', root.lemma_)
                neg, negText = self.isNegation(root)
                passive = self.isPassive(root)
                # # last is punct, the one before last is the root
                # if root.nbor().pos_ in ['PUNCT']:
                #   healthStatus = root
                healthStatus = self.findRightObj(root)
                if healthStatus and healthStatus.dep_ == 'pobj':
                  healthStatus = self.getHealthStatusForPobj(healthStatus, include=True)
                elif healthStatus and healthStatus.dep_ == 'dobj':
                  subtree = list(healthStatus.subtree)
                  nbor = self.getNbor(healthStatus)
                  if nbor is not None and nbor.dep_ in ['prep']:
                    healthStatus = healthStatus.doc[healthStatus.i:subtree[-1].i+1]
                # no object is found
                if not healthStatus:
                  healthStatus = self.findRightKeyword(root)
                # last is punct, the one before last is the root
                nbor = self.getNbor(root)
                if not healthStatus and nbor is not None and nbor.pos_ in ['PUNCT']:
                  healthStatus = root
                if healthStatus is None:
                  healthStatus = self.getAmod(ent, ent.start, ent.end, include=False)
                if healthStatus is None:
                  healthStatus = root
              else:
                if [root.lemma_.lower()] in predSynonyms:
                  ent._.set('hs_keyword', root.lemma_)
                else:
                  ent._.set('ent_status_verb', root.lemma_)
                passive = self.isPassive(root)
                neg, negText = self.isNegation(root)
                healthStatus = self.findLeftSubj(root, passive)
                if healthStatus is not None:
                  healthStatus = self.getAmod(healthStatus, healthStatus.i, healthStatus.i+1, include=True)
        else:
          # handle short phrase
          healthStatus = self.getAmod(ent, ent.start, ent.end, include=False)
          # search right
          start = None
          end = None
          if not ent[-1].is_sent_end and ent[-1].dep_ in ['compound']:
            start = ent.end
            end = None
            for i, tk in enumerate(sent.doc[ent.end:sent[-1].i+1]):
              # assume SPACE got removed already
              if tk.pos_ in ['PUNCT']:
                break
              if tk == sent.doc[tk.i-1].head:
                end = tk.i+1
              else:
                break
          if end is not None:
            healthStatusAppend = sent.doc[start:end]
          else:
            if ent[-1].head != ent[-1]:
              ind = ent[-1].head.i
              healthStatusAppend = sent.doc[ind:ind+1]
          if healthStatusAppend is not None:
            healthStatusAppendAmod = self.getAmodOnly(healthStatusAppend)
            if not healthStatusAppendAmod:
              healthStatusAppendAmod = self.getCompoundOnly(healthStatusAppend, ent)
            for elem in healthStatusAppendAmod:
              if elem in ent.text:
                healthStatusAppendAmod.remove(elem)

          if healthStatus is None:
            healthStatus = healthStatusAppend
            healthStatusAmod = healthStatusAppendAmod
            # reset
            healthStatusAppend = None
            healthStatusAppendAmod = None

          # handle conjuncts
          if healthStatus is None and len(ent.conjuncts) > 0:
            conjunct = sent.doc[ent.conjuncts[0].i: ent.conjuncts[0].i+1]
            healthStatus = self.getAmod(conjunct, conjunct.start, conjunct.end, include=False)
            if healthStatus is None:
              ent._.set('health_status',conjunct._.health_status)
              ent._.set('hs_keyword',conjunct._.hs_keyword)
              ent._.set('ent_status_verb',conjunct._.ent_status_verb)
              ent._.set('conjecture',conjunct._.conjecture)
        if healthStatus is None:
          continue

        if isinstance(healthStatus, Span):
          conjecture = self.isConjecture(healthStatus.root.head)
        elif isinstance(healthStatus, Token):
          conjecture = self.isConjecture(healthStatus.head)
        if not neg:
          if isinstance(healthStatus, Span):
            neg, negText = self.isNegation(healthStatus.root)
          else:
            neg, negText = self.isNegation(healthStatus)
        # conjecture = self.isConjecture(healthStatus.head)
        # neg, negText = self.isNegation(healthStatus)

        prependAmodText = ' '.join(healthStatusPrependAmod) if healthStatusPrependAmod is not None else ''
        prependText = healthStatusPrepend.text if healthStatusPrepend is not None else ''
        amodText = ' '.join(healthStatusAmod) if healthStatusAmod is not None else ''
        appendAmodText = ' '.join(healthStatusAppendAmod) if healthStatusAppendAmod is not None else ''
        if healthStatusAppend is not None and healthStatusAppend != ent:
          appText = healthStatusAppend.root.head.text + ' ' + healthStatusAppend.text if healthStatusAppend.root.dep_ in ['pobj'] else healthStatusAppend.text
        else:
          appText = ''

        healthStatusText = ' '.join(list(filter(None, [prependAmodText, prependText, amodText, healthStatus.text, appendAmodText,appText]))).strip()
        if neg:
          healthStatusText = ' '.join([negText,healthStatusText])
        if isinstance(healthStatus, Span):
          if ent.start > healthStatus.start and ent.end < healthStatus.end:
            # remove entity info in healthStatusText
            pn = re.compile(rf'{ent.text}\w*')
            healthStatusText = re.sub(pn, '', healthStatusText).strip()
            # healthStatusText = healthStatusText.replace(ent.text, '')

        logger.debug(f'{ent} health status: {healthStatusText}')
        ent._.set('health_status', healthStatusText)
        ent._.set('conjecture',conjecture)

  def findLeftSubj(self, pred, passive):
    """
      Find closest subject in predicates left subtree or
      predicates parent's left subtree (recursive).
      Has a filter on organizations.
      @ In, pred, spacy.tokens.Token, the predicate token
      @ In, passive, bool, True if passive
      @ Out, subj, spacy.tokens.Token, the token that represent subject
    """
    for left in pred.lefts:
      if passive: # if pred is passive, search for passive subject
        subj = self.findHealthStatus(left, ['nsubjpass', 'nsubj:pass'])
      else:
        subj = self.findHealthStatus(left, ['nsubj'])
      if subj is not None: # found it!
        return subj
    if pred.head != pred and not self.isPassive(pred):
      return self.findLeftSubj(pred.head, passive) # climb up left subtree
    else:
      return None

  def findRightObj(self, pred, deps=['dobj', 'pobj', 'iobj', 'obj', 'obl', 'oprd'], exclPrepos=[]):
    """
      Find closest object in predicates right subtree.
      Skip prepositional objects if the preposition is in exclude list.
      Has a filter on organizations.
      @ In, pred, spacy.tokens.Token, the predicate token
      @ In, exclPrepos, list, list of the excluded prepositions
    """
    for right in pred.rights:
      obj = self.findHealthStatus(right, deps)
      if obj is not None:
        if obj.dep_ == 'pobj' and obj.head.lemma_.lower() in exclPrepos: # check preposition
          continue
        return obj
    return None

  def findRightKeyword(self, pred, exclPrepos=[]):
    """
      Find
      Skip prepositional objects if the preposition is in exclude list.
      Has a filter on organizations.
      @ In, pred, spacy.tokens.Token, the predicate token
      @ In, exclPrepos, list, list of the excluded prepositions
    """
    for right in pred.rights:
      pos = right.pos_
      if pos in ['VERB', 'NOUN', 'ADJ']:
        # skip check to remove the limitation of status only in status keywords list
        # if [right.lemma_.lower()] in self._statusKeywords[pos]:
        return right
    return None

  def findHealthStatus(self, root, deps):
    """
      Return first child of root (included) that matches
      dependency list by breadth first search.
      Search stops after first dependency match if firstDepOnly
      (used for subject search - do not "jump" over subjects)
      @ In, root, spacy.tokens.Token, the root token
      @ In, deps, list, the dependency list
      @ Out, child, token, the token represents the health status
    """
    toVisit = deque([root]) # queue for bfs
    while len(toVisit) > 0:
      child = toVisit.popleft()
      # print("child", child, child.dep_)
      if child.dep_ in deps:
        # to handle preposition
        try:
          nbor = child.nbor()
        except IndexError:
          pass # ignore for now
        # TODO, what else need to be added
        # can not use the first check only, since is nbor is 'during', it will also satisfy the check condition
        # if (nbor.dep_ in ['prep'] and nbor.lemma_.lower() in ['of', 'in']) or nbor.pos_ in ['VERB']:
        #   return self.findRightObj(nbor, deps=['pobj'])
        return child
      elif child.dep_ == 'compound' and \
         child.head.dep_ in deps: # check if contained in compound
        return child
      toVisit.extend(list(child.children))
    return None

  def isValidCausalEnts(self, ent):
    """
      @ In, ent, list, list of entities
      @ Out, valid, bool, valid cansual ent if True
    """
    valid = False
    validDep = ['nsubj', 'nsubjpass', 'nsubj:pass', 'pobj', 'dobj', 'iobj', 'obj', 'obl', 'oprd']
    for e in ent:
      root = e.root
      if root.dep_ in validDep or root.head.dep_ in validDep:
        valid = True
        break
    return valid

  def getIndex(self, ent, entList):
    """
      Get index for ent in entList
      @ In, ent, Span, ent that is used to get index
      @ In, entList, list, list of entities
      @ Out, idx, int, the index for ent
    """
    idx = -1
    for i, e in enumerate(entList):
      if isinstance(e, list):
        if ent in e:
          idx = i
      else:
        if e == ent:
          idx = i
    return idx

  def getSSCEnt(self, entList, index, direction='left'):
    """
      Get the closest group of SSC entities
      @ In, entList, list, list of entities
      @ In, index, int, the start location of entity
      @ In, direction, str, 'left' or 'right', the search direction
      @ Out, ent,
    """
    ent = None
    if direction.lower() == 'left':
      for i in range(index, -1, -1):
        ent = entList[i]
        if isinstance(ent, list):
          # we may check the ent label here
          return ent
    elif direction.lower() == 'right':
      maxInd = len(entList)
      for i in range(index, maxInd):
        ent = entList[i]
        if isinstance(ent, list):
          # we may check the ent label here
          return ent
    return ent

  def extractRelDep(self, matchedSents):
    """
      @ In, matchedSents, list, the list of matched sentences
      @ Out, (subject tuple, predicate, object tuple), generator, the extracted causal relation
    """
    allCauseEffectPairs = []
    for sent in matchedSents:
      if self._labelCausal in self._entityLabels:
        causalEnts = self.getCustomEnts(sent.ents, self._entityLabels[self._labelCausal])
      else:
        continue
      sscEnts = self.getCustomEnts(sent.ents, self._entityLabels[self._labelSSC])
      sscEnts = self.getConjuncts(sscEnts)
      logger.debug(f'Conjuncts pairs: {sscEnts}')
      if causalEnts is None: #  no causal keyword is found, skipping
        continue
      if len(sscEnts) == 0:
        logger.debug(f'No entity is identified in "{sent.text}"')
        self._causalSentsNoEnts.append(sent)
        continue
      if len(sscEnts) == 1:
        logger.debug(f'Entity "({sent.ents})" is identified in "{sent.text}"')
        self._causalSentsOneEnt.append(sent)
        continue
      # shows keywords can be used to identify the causal sentences, but there are some false positive cases
      logger.debug(f'Sentence contains causal keywords: {causalEnts}. \n {sent.text}')

      # grab all ents
      labelList = self._entityLabels[self._labelCausal].union(self._entityLabels[self._labelSSC])
      ents = self.getCustomEnts(sent.ents, labelList)
      mEnts = copy.copy(ents)
      root = sent.root
      i = root.i
      neg, negText = self.isNegation(root)
      conjecture = self.isConjecture(root)
      if neg:
        if conjecture:
          rootTuple = [('conjecture', conjecture), ('negation', neg), ('negation text',negText), root]
        else:
          rootTuple = [('negation', neg), ('negation text',negText), root]
      else:
        if conjecture:
          rootTuple = [('conjecture', conjecture), root]
        else:
          rootTuple = [root]
      idx = -1
      for j, ent in enumerate(ents):
        start = ent.start
        if i < start:
          idx = j
          break
        if i == start:
          rootTuple[-1] = ent
          mEnts[j] = rootTuple
          break
      if idx != -1:
        mEnts.insert(j, rootTuple)
      logger.debug(f'Causal Info: {mEnts}')
      self._rawCausalList.append(mEnts)

      entTuples = [(ent[0].start, ent) for ent in sscEnts] + [(ent.start, ent) for ent in causalEnts]
      orderedEnts = sorted(entTuples, key = lambda x:x[0])
      orderedEnts = [ent[1] for ent in orderedEnts]
      # Loop over causal keywords, make functions, for each of [verb, noun, transition]
      # Define rules for each functions
      causeEffectPair = []
      skipCEnts = []
      rootCause = None
      for i, cEnt in enumerate(causalEnts):
        if cEnt in skipCEnts:
          continue
        cRoot = cEnt.root
        cRootLoc = cRoot.i
        causalEntLemma = [token.lemma_.lower() for token in cEnt if token.lemma_ != "DET"]
        rightSSCEnts = self.getRightSSCEnts(cEnt, orderedEnts)
        leftSSCEnts = self.getLeftSSCEnts(cEnt, orderedEnts)
        validLeftSSCEnts = self.selectValidEnts(leftSSCEnts, cEnt)
        validRightSSCEnts = self.selectValidEnts(rightSSCEnts, cEnt)
        # initial assignment
        causeList = validLeftSSCEnts
        effectList = validRightSSCEnts
        if validLeftSSCEnts is None and validRightSSCEnts is None:
          logger.debug(f'No causal/effect entities exist in "{sent}"')
          continue
        if cRoot.pos_ == 'VERB' and cRoot == sent.root:
          passive = self.isPassive(root)
          conjecture = self.isConjecture(cRoot)
          if causeList is None:
            subj = self.findSubj(cRoot, passive)
            if subj is not None:
              causeList = [[subj]]
          if effectList is None:
            obj = self.findObj(cRoot)
            if obj is not None:
              effectList = [[obj]]
          if passive:
            causeList, effectList = effectList, causeList
          rootCause = (causeList, effectList, conjecture)
        elif cRoot.pos_ == 'VERB' and cRoot != sent.root:
          conjecture = self.isConjecture(cRoot)
          if validRightSSCEnts is None:
            continue
          causeList, effectList = self.identifyCauseEffectForClauseModifier(cRoot, rootCause, validLeftSSCEnts, validRightSSCEnts)
        elif cRoot.pos_ == 'NOUN':
          if causalEntLemma in self._causalKeywords['causal-noun']:
            cRootHead = cRoot.head
            conjecture = self.isConjecture(cRootHead)
            if validRightSSCEnts is None:
              continue
            if cRootHead.dep_ in ['xcomp', 'advcl', 'relcl']:
              causeList, effectList = self.identifyCauseEffectForClauseModifier(cRootHead, rootCause, validLeftSSCEnts, validRightSSCEnts)
            elif cRoot.dep_ in ['attr']:
              causeList, effectList = self.identifyCauseEffectForAttr(self, cRootHead, validLeftSSCEnts, validRightSSCEnts)
              if rootCause is None:
                rootCause = (causeList, effectList, conjecture)
            elif cRoot.dep_ in ['nsubj']:
              causeList, effectList, skip = self.identifyCauseEffectForNsuj(cRoot, i, causalEnts, orderedEnts, validRightSSCEnts, reverse=True)
              skipCEnts.extend(skip)
              if rootCause is None:
                rootCause = (causeList, effectList, conjecture)
            else:
              continue
          elif causalEntLemma in self._causalKeywords['effect-noun']:
            cRootHead = cRoot.head
            conjecture = self.isConjecture(cRootHead)
            if validRightSSCEnts is None:
              continue
            if cRootHead.dep_ in ['xcomp', 'advcl', 'relcl']:
              causeList, effectList = self.identifyCauseEffectForClauseModifier(cRootHead, rootCause, validLeftSSCEnts, validRightSSCEnts, reverse=True)
            elif cRoot.dep_ in ['attr']:
              causeList, effectList = self.identifyCauseEffectForAttr(self, cRootHead, validLeftSSCEnts, validRightSSCEnts, reverse=True)
              if rootCause is None:
                rootCause = (causeList, effectList, conjecture)
            elif cRoot.dep_ in ['nsubj']:
              causeList, effectList, skip = self.identifyCauseEffectForNsuj(cRoot, i, causalEnts, orderedEnts, validRightSSCEnts, reverse=False)
              skipCEnts.extend(skip)
              if rootCause is None:
                rootCause = (causeList, effectList, conjecture)
            else:
              continue
        elif causalEntLemma in self._causalKeywords['causal-relator']:
          if causeList is None:
            causeList = rootCause
        elif causalEntLemma in self._causalKeywords['effect-relator']:
          if validLeftSSCEnts is not None and validRightSSCEnts is not None:
            causeList, effectList = effectList, causeList
          elif validLeftSSCEnts is None and validRightSSCEnts is not None:
            causeList, effectList, skip = self.identifyCauseEffectForNsuj(cRoot, i, causalEnts, orderedEnts, validRightSSCEnts, reverse=False)
            skipCEnts.extend(skip)
            if rootCause is None:
              rootCause = (causeList, effectList, conjecture)
          else:
            continue
        if causeList is None or effectList is None:
          logger.warning(f"Issue found: 'cause list': {causeList}, and 'effect list': {effectList} were identified in sentence '{sent}'")
          continue
        if causeList is None or effectList is None:
          continue
        causeEffectPair.append((causeList, effectList, conjecture))
        if isinstance(causeList, tuple):
          self.collectExtactedCausals(causeList[0], effectList, cEnt, sent, conjecture)
          # effect can be cause for next effect
          self.collectExtactedCausals(causeList[1], effectList, cEnt, sent, conjecture)
        elif isinstance(effectList, tuple):
          self.collectExtactedCausals(causeList, effectList[0], cEnt, sent, conjecture)
          self.collectExtactedCausals(causeList, effectList[1], cEnt, sent, conjecture)
        else:
          self.collectExtactedCausals(causeList, effectList, cEnt, sent, conjecture)
      if len(causeEffectPair) != 0:
        allCauseEffectPairs.append(causeEffectPair)

    # print("Identified Cause-Effect Pairs:")
    # for elem in allCauseEffectPairs:
    #   for i in elem:
    #     print(i)

  def identifyCauseEffectForNsuj(self, cRoot, cEntsIndex, causalEnts, orderedEnts, validRightSSCEnts, reverse=False):
    """
      Identify the cause effect pairs for nsubj
      @ In, cRoot, Token, the root of causal entity
      @ In, cEntsIndex, int, the index for the causal entity
      @ In, causalEnts, list, the list of causal entities
      @ In, orderedEnts, list, the entities ordered by their locations in the Doc
      @ In, validRightSSCEnts, list, the valid list of entities on the right of given causal entity
      @ In, reverse, bool, reverse the cause effect relation if True
      @ Out, cause effect pairs, tuple, (causeList, effectList, skipCEnts)
    """
    causeList = None
    effectList = None
    skipCEnts = []
    if reverse:
      cKey = 'effect-relator'
    else:
      cKey = 'causal-relator'
    causeList, effectList = self.splitEntsFollowingNounCausal(cRoot, validRightSSCEnts)
    if causeList is None:
      obj = self.findObj(cRoot)
      if obj is not None:
        causeList = [[obj]]
    if effectList is None:
      if cEntsIndex < len(causalEnts) - 1:
        nextCEnt = causalEnts[cEntsIndex+1]
        nextCEntLemma = [token.lemma_.lower() for token in nextCEnt if token.lemma_ != "DET"]
        if nextCEntLemma in self._causalKeywords[cKey]:
          ents = self.getRightSSCEnts(nextCEnt, orderedEnts)
          validEnts = self.selectValidEnts(ents, nextCEnt)
          if validEnts is not None:
            effectList = validEnts
          else:
            obj = self.findObj(nextCEnt.root)
            if obj is not None:
              effectList = [[obj]]
          skipCEnts.append(nextCEnt)
      else:
        obj = self.findObj(cRoot.head)
        if obj is not None:
          effectList = [[obj]]
    if reverse:
      causeList, effectList = effectList, causeList
    return causeList, effectList, skipCEnts

  def identifyCauseEffectForAttr(self, cRoot, validLeftSSCEnts, validRightSSCEnts, reverse=False):
    """
      Identify the cause effect pairs for attr
      @ In, cRoot, Token, the root of causal entity
      @ In, validLeftSSCEnts, list, the valid list of entities on the left of given causal entity
      @ In, validRightSSCEnts, list, the valid list of entities on the right of given causal entity
      @ In, reverse, bool, reverse the cause effect relation if True
      @ Out, cause effect pairs, tuple, (causeList, effectList)
    """
    causeList = None
    effectList = None
    if validLeftSSCEnts is not None:
      causeList = validLeftSSCEnts
      effectList = validRightSSCEnts
    else:
      passive = self.isPassive(cRoot)
      subj = self.findSubj(cRoot, passive)
      if subj is not None:
        causeList = [[subj]]
        effectList = validRightSSCEnts
    if reverse:
      return effectList, causeList
    else:
      return causeList, effectList

  def identifyCauseEffectForClauseModifier(self, cRoot, rootCause, validLeftSSCEnts, validRightSSCEnts, reverse=False):
    """
      Identify the cause effect pairs for clause modifier
      @ In, cRoot, Token, the root of causal entity
      @ In, rootCause, tuple,
      @ In, validLeftSSCEnts, list, the valid list of entities on the left of given causal entity
      @ In, validRightSSCEnts, list, the valid list of entities on the right of given causal entity
      @ In, reverse, bool, reverse the cause effect relation if True
      @ Out, cause effect pairs, tuple, (causeList, effectList)
    """
    causeList = None
    effectList = None
    # xcomp: open clausal complement i.e., rendering ..., causing ...
    # advcl: adverbial clause modifier i.e., ... which disabled ...
    # relcl: relative clause modifier i.e. ..., which disabled ...
    if cRoot.dep_ not in ['xcomp', 'advcl', 'relcl']:
      return causeList, effectList
    if rootCause is not None:
      # using rootCause as the cause
      causeList = rootCause
      effectList = validRightSSCEnts
    else:
      if validLeftSSCEnts is not None:
        causeList = validLeftSSCEnts
        effectList = validRightSSCEnts
      else:
        head = cRoot.head
        passive = self.isPassive(head)
        subj = self.findSubj(head, passive)
        if subj is not None:
          causeList = [[subj]]
          effectList = validRightSSCEnts
    if reverse:
      return effectList, causeList
    else:
      return causeList, effectList

  def splitEntsFollowingNounCausal(self, cRoot, validRightSSCEnts):
    """
      Spit the entities into cause, effect
      @ In, cRoot, Token, the root of causal entity
      @ In, validRightSSCEnts, list, the valid list of entities on the right of given causal entity
      @ Out, cause effect pairs, tuple, (cause, effect)
    """
    cause = []
    effect = []
    if validRightSSCEnts is None:
      return None, None
    for ents in validRightSSCEnts:
      root = ents[0].root
      if root in cRoot.subtree:
        cause.append(ents)
      elif root in cRoot.head.subtree:
        effect.append(ents)
    if len(cause) == 0:
      cause = None
    if len(effect) == 0:
      effect = None
    return cause, effect

  def getRightSSCEnts(self, cEnt, orderedEnts):
    """
      Get the SSC ents on the right of causal entity
      @ In, cEnt, Span, causal entity
      @ In, orderedEnts, list, the entities ordered by their locations in the Doc
      @ Out, selEnts, list, list of SSC entities
    """
    cIdx = self.getIndex(cEnt, orderedEnts)
    maxInd = len(orderedEnts)-1
    selEnts = []
    if cIdx == maxInd:
      return None
    for i in range(cIdx+1, maxInd+1):
      if isinstance(orderedEnts[i], list):
        selEnts.append(orderedEnts[i])
      else:
        break
    if len(selEnts) != 0:
      return selEnts
    else:
      return None

  def getLeftSSCEnts(self, cEnt, orderedEnts):
    """
      Get the SSC ents on the left of causal entity
      @ In, cEnt, Span, causal entity
      @ In, orderedEnts, list, the entities ordered by their locations in the Doc
      @ Out, selEnts, list, list of SSC entities
    """
    cIdx = self.getIndex(cEnt, orderedEnts)
    maxInd = len(orderedEnts)-1
    selEnts = []
    if cIdx == 0:
      return None
    for i in range(cIdx-1, -1, -1):
      if isinstance(orderedEnts[i], list):
        selEnts.append(orderedEnts[i])
      else:
        break
    if len(selEnts) != 0:
      return selEnts
    else:
      return None

  def selectValidEnts(self, ents, cEnt):
    """
      Select the valide ents that are within subtree of causal entity
      @ In, ents, list, the list of entities
      @ In, cEnt, Span, causal entity
      @ Out, validEnts, list, list of valid entities
    """
    if ents is None:
      return None
    validEnts = []
    for ent in ents:
      valid = self.isValidCausalEnts(ent)
      if not valid:
        continue
      root = ent[0].root
      if root in cEnt.root.subtree or root in cEnt.root.head.subtree:
        validEnts.append(ent)
    if len(validEnts) == 0:
      return None
    else:
      return validEnts

  def collectExtactedCausals(self, cause, effect, causalKeyword, sent, conjecture=None):
    """
      Collect the extracted causal relations
      @ In, cause, list, list of causes
      @ In, effect, list, list of effects
      @ In, causalKeyword, str, causal keyword
      @ In, sent, spacy.tokens.Span, sentence with identified causal relations
      @ Out, None
    """
    root = sent.root
    if conjecture is None:
      conjecture = self.isConjecture(root)
    for csub in cause:
      for c in csub:
        for esub in effect:
          for e in esub:
            logger.debug(f'({c} health status: {c._.health_status}) "{causalKeyword}" ({e} health status: {e._.health_status}), conjecture: "{conjecture}"')
            self._extractedCausals.append([c, c._.health_status, causalKeyword, e, e._.health_status, sent, conjecture])

  def getConjuncts(self, entList):
    """
      Get a list of conjuncts from entity list
      @ In, entList, list, list of entities
      @ Out, conjunctList, list, list of conjuncts
    """
    ent = entList[0]
    conjunctList = []
    conjuncts = [ent]
    collected = False
    for i, elem in enumerate(entList[1:]):
      # print('elem', elem, elem.conjuncts)
      # print('ent', ent, ent.conjuncts)
      if elem.root not in ent.conjuncts:
        conjunctList.append(conjuncts)
        conjunctList.extend(self.getConjuncts(entList[i+1:]))
        collected = True
        break
      conjuncts.append(elem)
    if not collected:
      conjunctList.append(conjuncts)
    return conjunctList

  ##TODO: how to extend it for entity ruler?
  # @staticmethod
  def collectSents(self, doc):
    """
      collect data of matched sentences that can be used for visualization
      @ In, matcher, spacy.Matcher, the spacy matcher instance
      @ In, doc, spacy.tokens.doc.Doc, the processed document using nlp pipelines
      @ In, i, int, index of the current match (matches[i])
      @ In, matches, List[Tuple[int, int, int]], a list of (match_id, start, end) tuples, describing the matches. A
        match tuple describes a span doc[start:end]
    """
    matchedSents = []
    matchedSentsForVis = []
    for span in doc.ents:
      if span.ent_id_ != self._labelSSC:
        continue
      sent = span.sent
      # Append mock entity for match in displaCy style to matched_sents
      # get the match span by ofsetting the start and end of the span with the
      # start and end of the sentence in the doc
      matchEnts = [{
          "start": span.start_char - sent.start_char,
          "end": span.end_char - sent.start_char,
          "label": span.label_,
      }]
      if sent not in matchedSents:
        matchedSents.append(sent)
      matchedSentsForVis.append({"text": sent.text, "ents": matchEnts})
    return matchedSents, matchedSentsForVis



#############################################################################
# some useful methods, but currently they are not used

  def extract(self, sents, predSynonyms=[], exclPrepos=[]):
    """
      General extraction method
      @ In, sents, list, the list of sentences
      @ In, predSynonyms, list, the list of predicate synonyms
      @ In, exclPrepos, list, the list of exlcuded prepositions
      @ Out, (subject tuple, predicate, object tuple), generator, the extracted causal relation
    """
    for sent in sents:
      root = sent.root
      if root.pos_ == 'VERB' and [root.lemma_.lower()] in predSynonyms:
        passive = self.isPassive(root)
        subj = self.findSubj(root, passive)
        if subj is not None:
          obj = self.findObj(root, deps=['dobj', 'pobj', 'iobj', 'obj', 'obl'], exclPrepos=[])
          if obj is not None:
            if passive: # switch roles
              obj, subj = subj, obj
            yield ((subj), root, (obj))
      else:
        for token in sent:
          if [token.lemma_.lower()] in predSynonyms:
            root = token
            passive = self.isPassive(root)
            subj = self.findSubj(root, passive)
            if subj is not None:
              obj = self.findObj(root, deps=['dobj', 'pobj', 'iobj', 'obj', 'obl'], exclPrepos=[])
              if obj is not None:
                if passive: # switch roles
                  obj, subj = subj, obj
                yield ((subj), root, (obj))

  def bfs(self, root, deps):
    """
      Return first child of root (included) that matches
      entType and dependency list by breadth first search.
      Search stops after first dependency match if firstDepOnly
      (used for subject search - do not "jump" over subjects)
      @ In, root, spacy.tokens.Token, the root token
      @ In, deps, list, list of dependency
      @ Out, child, spacy.tokens.Token, the matched token
    """
    toVisit = deque([root]) # queue for bfs
    while len(toVisit) > 0:
      child = toVisit.popleft()
      if child.dep_ in deps:
        # to handle preposition
        nbor = self.getNbor(child)
        if nbor is not None and nbor.dep_ in ['prep'] and nbor.lemma_.lower() in ['of']:
          obj = self.findObj(nbor, deps=['pobj'])
          return obj
        else:
          return child
      elif child.dep_ == 'compound' and \
         child.head.dep_ in deps: # check if contained in compound
        return child
      toVisit.extend(list(child.children))
    return None

  def findSubj(self, pred, passive):
    """
      Find closest subject in predicates left subtree or
      predicates parent's left subtree (recursive).
      Has a filter on organizations.
      @ In, pred, spacy.tokens.Token, the predicate token
      @ In, passive, bool, True if the predicate token is passive
      @ Out, subj, spacy.tokens.Token, the token that represents subject
    """
    for left in pred.lefts:
      if passive: # if pred is passive, search for passive subject
        subj = self.bfs(left, ['nsubjpass', 'nsubj:pass'])
      else:
        subj = self.bfs(left, ['nsubj'])
      if subj is not None: # found it!
        return subj
    if pred.head != pred and not self.isPassive(pred):
      return self.findSubj(pred.head, passive) # climb up left subtree
    else:
      return None

  def findObj(self, pred, deps=['dobj', 'pobj', 'iobj', 'obj', 'obl'], exclPrepos=[]):
    """
      Find closest object in predicates right subtree.
      Skip prepositional objects if the preposition is in exclude list.
      Has a filter on organizations.
      @ In, pred, spacy.tokens.Token, the predicate token
      @ In, exclPrepos, list, the list of prepositions that will be excluded
      @ Out, obj, spacy.tokens.Token,, the token that represents the object
    """
    for right in pred.rights:
      obj = self.bfs(right, deps)
      if obj is not None:
        if obj.dep_ == 'pobj' and obj.head.lemma_.lower() in exclPrepos: # check preposition
          continue
        return obj
    return None

  def isValidKeyword(self, var, keywords):
    """
      @ In, var, token
      @ In, keywords, list/dict
    """
    if isinstance(keywords, dict):
      for _, vals in keywords.items():
        if var.lemma_.lower() in vals:
          return True
    elif isinstance(keywords, list):
      if var.lemma_.lower() in keywords:
        return True
    return False
#######################################################################################
