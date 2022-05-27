# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
Created on March, 2022

@author: wangc, mandd
"""
import pandas as pd
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
## import pipelines
from .CustomPipelineComponents import normEntities
from .CustomPipelineComponents import initCoref
from .CustomPipelineComponents import aliasResolver
from .CustomPipelineComponents import anaphorCoref
from .CustomPipelineComponents import mergePhrase
from .CustomPipelineComponents import pysbdSentenceBoundaries


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
  # check the current version spacy>=3.1.0,<3.2.0
  from packaging.version import Version
  ver = spacy.__version__
  valid = Version(ver)>=Version('3.1.0') and Version(ver)<Version('3.2.0')
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
if not Span.has_extension('conjecture'):
  Span.set_extension('conjecture', default=False)

class RuleBasedMatcher(object):
  """
    Rule Based Matcher Class
  """

  def __init__(self, nlp, *args, **kwargs):
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

    self._causalFile = os.path.join(os.path.dirname(__file__), 'cause_effect_keywords.csv') # header includes: VERB, NOUN, TRANSITION
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

    self._statusFile = os.path.join(os.path.dirname(__file__), 'health_status_keywords.csv') # header includes: VERB, NOUN, ADJ
    self._statusKeywords = self.getKeywords(self._statusFile)
    self._updateStatusKeywords = False
    self._updateCausalKeywords = False

    # TODO: right now, we are trying to use 'parser' and 'sentencizer' to parse the sentences,
    # But the parse is not always accurate, especially for sentence which endswith "pump 1A."

    # if _corefAvail:
    #   self.pipelines = ['entity_ruler','normEntities', 'merge_entities', 'initCoref', 'aliasResolver', 'coreferee','anaphorCoref', 'expandEntities']
    # else:
    #   self.pipelines = ['entity_ruler','normEntities', 'merge_entities', 'initCoref', 'aliasResolver', 'anaphorCoref', 'expandEntities']

    ## pipelines "merge_entities" and "merge_noun_chunks" can be used to merge noun phrases and entities
    ## for easier analysis

    if _corefAvail:
      self.pipelines = ['pysbdSentenceBoundaries', 'entity_ruler',
                      'mergePhrase', 'normEntities', 'initCoref', 'aliasResolver',
                      'coreferee','anaphorCoref']
    else:
      self.pipelines = ['pysbdSentenceBoundaries', 'entity_ruler',
                      'mergePhrase','normEntities', 'initCoref', 'aliasResolver',
                      'anaphorCoref']
    nlp = resetPipeline(nlp, self.pipelines)
    self.nlp = nlp
    self._doc = None
    self._rules = {}
    self._match = False
    self._phraseMatch = False
    self._dependencyMatch = False
    self._entityRuler = False
    self.matcher = Matcher(nlp.vocab)
    self.phraseMatcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    self.dependencyMatcher = DependencyMatcher(nlp.vocab)
    if nlp.has_pipe("entity_ruler"):
      self.entityRuler = nlp.get_pipe("entity_ruler")
    else:
      self.entityRuler = nlp.add_pipe("entity_ruler")
    self._simpleMatches = []
    self._phraseMatches = []
    self._dependencyMatches = []
    self._entityRulerMatches = []
    self._callbacks = {}
    self._asSpans = True # When True, a list of Span objects using the match_id as the span label will be returned
    self._matchedSents = [] # collect data of matched sentences
    self._matchedSentsForVis = [] # collect data of matched sentences to be visualized
    self._visualizeMatchedSents = True
    self._coref = _corefAvail # True indicate coreference pipeline is available
    self._entityLabels = {} # labels for rule-based entities
    self._labelSSC = 'SSC'
    self._labelCausal = 'causal'

  def getKeywords(self, filename):
    """
      Get the keywords from given file
      @ In, filename, str, the file name to read the keywords
      @ Out, kw, dict, dictionary contains the keywords
    """
    kw = {}
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
      lemVar = [token.lemma_ for token in self.nlp(var) if token.lemma_ not in ["!", "?", "+", "*"]]
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

  def addPattern(self, name, rules, callback=None):
    """
      Add rules
      @ In, name, str, the name for the pattern
      @ In, rules, list, the rules used to match the entities, for example:
        rules = [{"LOWER": "hello"}, {"IS_PUNCT": True}, {"LOWER": "world"}]
    """
    logger.debug('Add rules')
    if not isinstance(rules, list):
      rules = [rules]
    if not isinstance(rules[0], list):
      rules = [rules]
    self._rules[name] = rules
    self._callbacks[name] = callback
    self.matcher.add(name, rules, on_match=callback)
    if not self._match:
      self._match = True

  def addPhrase(self, name, phraseList, callback=None):
    """
      Add phrase patterns
      @ In, name, str, the name for the phrase pattern
      @ In, phraseList, list, the phrase list, for example:
        phraseList = ["hello", "world"]
    """
    logger.debug(f'Add phrase pattern for {name}')
    patterns = [self.nlp.make_doc(text) for text in phraseList]
    self._callbacks[name] = callback
    self._rules[name] = patterns
    self.phraseMatcher.add(name, patterns, on_match=callback)
    if not self._phraseMatch:
      self._phraseMatch = True

  def addDependency(self, name, patternList, callback=None):
    """
      Add dependency pattern
      @ In, name, str, the name for the dependency pattern
      @ In, patternList, list, the dependency pattern list
    """
    logger.debug(f'Add dependency pattern for {name}')
    if not isinstance(patternList, list):
      patternList = [patternList]
    if not isinstance(patternList[0], list):
      patternList = [patternList]
    self._rules[name] = patternList
    self._callbacks[name] = callback
    self.dependencyMatcher.add(name, patternList, on_match=callback)
    if not self._dependencyMatch:
      self._dependencyMatch = True

  def addEntityPattern(self, name, patternList):
    """
      Add entity pattern, to extend doc.ents, similar function to self.extendEnt
      @ In, name, str, the name for the entity pattern.
      @ In, patternList, list, the pattern list, for example:
        {"label": "GPE", "pattern": [{"LOWER": "san"}, {"LOWER": "francisco"}]}
    """
    if not self.nlp.has_pipe('entity_ruler'):
      self.nlp.add_pipe('entity_ruler')
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

    if self._match:
      self._simpleMatches += self.matcher(doc, as_spans = self._asSpans) # <class 'list'>
    if self._phraseMatch:
      self._phraseMatches += self.phraseMatcher(doc, as_spans = self._asSpans) # <class 'list'>
    if self._dependencyMatch:
      self._dependencyMatches = self.dependencyMatcher(doc) # <class 'list'> [tuple(match_id, token_ids)]

    if self._match:
      self.printMatches(self._doc, self._simpleMatches, 'Simple Matches')
    if self._phraseMatch:
      self.printMatches(self._doc, self._phraseMatches, 'Phrase Matches')

    # print dependency matches
    if self._dependencyMatch:
      for (id, tokenIDs) in self._dependencyMatches:
        name = self.nlp.vocab.strings[id]
        for i in range(len(tokenIDs)):
          print(self._rules[name][0][i]["RIGHT_ID"] + ":",doc[tokenIDs[i]].text)

    ## use entity ruler to identify entity
    if self._entityRuler:
      logger.debug('Entity Ruler Matches:')
      print([(ent.text, ent.label_, ent.ent_id_) for ent in doc.ents if ent.label_ in self._entityLabels[self._labelSSC]])

    if self._coref:
      logger.debug('Print Coreference Info:')
      print(doc._.coref_chains.pretty_representation)

    matchedSents, matchedSentsForVis = self.collectSents(self._doc)
    self._matchedSents += matchedSents
    self._matchedSentsForVis += matchedSentsForVis
    # print(self._matchedSents)
    # self.visualize()

    ## TODO: collect and expand entities, then extract health status of the entities

    ## health status
    logger.info('Start to extract health status')
    self.extractHealthStatus(self._matchedSents)
    ## Access health status
    for sent in self._matchedSents:
      ents = self.getCustomEnts(sent.ents, self._entityLabels[self._labelSSC])
      for ent in ents:
        # print(ent)
        print(ent._.health_status)
    logger.info('End of health status extraction!')
    ## causal relation
    logger.info('Start to extract causal relation using OPM model information')
    self.extractRelDep(self._matchedSents)
    logger.info('End of causal relation extraction!')
    ## print extracted relation
    logger.info('Start to use general extraction method to extract causal relation')
    print(*self.extract(self._matchedSents, predSynonyms=self._causalKeywords['VERB'], exclPrepos=[]), sep='\n')
    logger.info('End of causal relation extraction using general extraction method!')

  def printMatches(self, doc, matches, matchType):
    """
      Print the matches
      @ In, doc, spacy.tokens.doc.Doc, the processed document using nlp pipelines
      @ In, matches, list of matches
      @ In, matchType, string, the type for matches
    """
    if not self._asSpans:
      matchList = []
      for id, start, end in matches:
        strID = self.nlp.vocab.strings[id]
        span = doc[start:end]
        matchList.append(span)
    else:
      matchList = matches
    matchText = ', '.join([span.text for span in matchList])
    logger.debug(matchType + ': ' + matchText)

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
    for left in token.lefts: # Check modal auxilliary verb: can, could, may, might, must, shall, should, will, would
      if left.dep_.startswith('aux') and left.tag_ in ['MD']:
        return True
    if token.pos_ == 'VERB' and token.tag_ == 'VB': # If it is a verb, and there is no inflectional morphology for the verb
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
    return customEnts

  def extractHealthStatus(self, matchedSents, predSynonyms=[], exclPrepos=[]):
    """
      Extract health status and relation
      @ In, matchedSents, list, the matched sentences
      @ In, predSynonyms, list, predicate synonyms
      @ In, exclPrepos, list, exclude the prepositions
    """
    predSynonyms = self._statusKeywords['VERB']
    statusNoun = self._statusKeywords['NOUN']
    statusAdj = self._statusKeywords['ADJ']
    causalStatus = False
    for sent in matchedSents:
      conjecture = False
      causalEnts = self.getCustomEnts(sent.ents, self._entityLabels[self._labelCausal])
      ents = self.getCustomEnts(sent.ents, self._entityLabels[self._labelSSC])
      # if len(ents) > 1 and [sent.root.lemma_] in self._causalKeywords['VERB']:
      causalStatus = [sent.root.lemma_] in self._causalKeywords['VERB'] and [sent.root.lemma_] not in self._statusKeywords['VERB']
      causalStatus = causalStatus and len(ents) == 1
      # if causalStatus and len(ents) > 1:
      if (len(ents) > 1 and len(causalEnts) > 0) or causalStatus:
        conjecture = self.isConjecture(sent.root)
        for ent in ents:
          healthStatus = None
          root = ent.root
          if root.dep_ in ['pobj']:
            healthStatus = root.head.head
          elif root.dep_ in ['compound', 'nsubj']:
            healthStatus = root.head
          else:
            continue
          neg, negText = self.isNegation(healthStatus)
          logger.debug(f'{ent} health status: {negText} {healthStatus.text}')
          ent._.set('health_status', ' '.join([negText,healthStatus.text]))
          ent._.set('conjecture',conjecture)
      elif len(ents) == 1:
        healthStatus = None
        root = sent.root
        neg, negText = self.isNegation(root)
        if [root.lemma_] not in predSynonyms and root.pos_ != 'VERB':
          if root.pos_ in ['NOUN', 'ADJ']:
            healthStatus = root
            if self._updateStatusKeywords:
              self.addKeywords({root.pos_:[root]}, 'status')
        elif root.pos_ != 'VERB':
          print('--- root not verb', root.text, root.pos_)
          continue
        else:
          passive = self.isPassive(root)
          conjecture = self.isConjecture(root)
          # # last is punct, the one before last is the root
          # if root.nbor().pos_ in ['PUNCT']:
          #   healthStatus = root
          if ents[0].start < root.i:
            healthStatus = self.findRightObj(root)
            if healthStatus and healthStatus.dep_ == 'pobj':
              # include 'dobj' 'prep' and 'pobj'
              # examples
              # Pump had noise of cavitation
              # RCP pump 1A had signs of past leakage
              if healthStatus.head.head.dep_ == 'dobj':
                healthStatus = healthStatus.doc[healthStatus.head.head.i:healthStatus.i+1]
            # no object is found
            if not healthStatus:
              healthStatus = self.findRightKeyword(root)
            # last is punct, the one before last is the root
            if not healthStatus and root.nbor().pos_ in ['PUNCT']:
              healthStatus = root
          else:
            healthStatus = self.findLeftSubj(root, passive)
        if healthStatus is None:
          continue
        if not neg:
          if isinstance(healthStatus, Span):
            neg, negText = self.isNegation(healthStatus.root)
          else:
            neg, negText = self.isNegation(healthStatus)
        # TODO: may be also report the verb, for example 'RCP pump 1A was cavitating and vibrating to some degree during test.'
        # is not identified properly
        logger.debug(f'{ents[0]} health status: {negText} {healthStatus.text}')
        ents[0]._.set('health_status', ' '.join([negText,healthStatus.text]))
        ents[0]._.set('conjecture', conjecture)

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
        if [right.lemma_] in self._statusKeywords[pos]:
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
        nbor = child.nbor()
        # TODO, what else need to be added
        # can not use the first check only, since is nbor is 'during', it will also satisfy the check condition
        if nbor.dep_ in ['prep'] and nbor.lemma_ in ['of']:
          return self.findRightObj(nbor, deps=['pobj'])
        return child
      elif child.dep_ == 'compound' and \
         child.head.dep_ in deps: # check if contained in compound
        return child
      toVisit.extend(list(child.children))
    return None

  def extractRelDep(self, matchedSents):
    """
      @ In, matchedSents, list, the list of matched sentences
      @ Out, (subject tuple, predicate, object tuple), generator, the extracted causal relation
    """
    # TODO List
    # 1. check causal verbs
    # 2. check causal nouns
    # 3. check causal transitions
    # 4. report missing causal relations (only one ent is present and causal keyword is found)
    # 5. create causal keywords patterns, then use entity_ruler to identify them

    for sent in matchedSents:
      causalEnts = self.getCustomEnts(sent.ents, self._entityLabels[self._labelCausal])
      sscEnts = self.getCustomEnts(sent.ents, self._entityLabels[self._labelSSC])
      sscEnts = self.getConjuncts(sscEnts)
      logger.debug(f'Conjuncts pairs: {sscEnts}')

      if len(causalEnts) == 0: #  no causal keyword is found, skipping
        continue
      elif len(causalEnts) == 1 and len(sscEnts) == 1: # single causal keyword, single entity, report missing causal relation
        causalStatus = [sent.root.lemma_] in self._causalKeywords['VERB'] and [sent.root.lemma_] not in self._statusKeywords['VERB']
        if causalStatus:
          logger.debug(f'missing causal relation: {sent}')
      elif len(sscEnts) == 2: # Two entities and One causal keyword
        if len(causalEnts) == 1:
          root = causalEnts[0].root
          rootLoc = root.i
          causalEntLemma = [token.lemma_ for token in causalEnts[0] if token.lemma_ != "DET"]
          if root.pos_ == 'VERB':
            passive = self.isPassive(root)
            if passive:
              logger.debug(f'({sscEnts[1]} health status: {sscEnts[1][0]._.health_status}) "{causalEnts[0]}" ({sscEnts[0]} health status: {sscEnts[1][0]._.health_status})')
            else:
              logger.debug(f'({sscEnts[0]} health status: {sscEnts[0][0]._.health_status}) "{causalEnts[0]}" ({sscEnts[1]} health status: {sscEnts[1][0]._.health_status})')
          elif root.pos_ == 'NOUN':
            if causalEntLemma in self._causalKeywords['causal-noun']:
              if rootLoc > sscEnts[0][0].start and rootLoc < sscEnts[1][0].start:
                # assert sscEnts[1].root in root.subtree
                logger.debug(f'({sscEnts[0]} health status: {sscEnts[0][0]._.health_status}), ---> "{causalEnts[0]}", ---> ({sscEnts[1]} health status: {sscEnts[1][0]._.health_status})')
              elif rootLoc < sscEnts[0][0].start:
                # assert sscEnts[0].root in root.subtree
                logger.debug(f' "{causalEnts[0]}" ({sscEnts[0]} health status: {sscEnts[0][0]._.health_status}) <--- ({sscEnts[1]} health status: {sscEnts[1][0]._.health_status})')
            elif causalEntLemma in self._causalKeywords['effect-noun']:
              if rootLoc > sscEnts[0][0].start and rootLoc < sscEnts[1][0].start:
                # assert sscEnts[1].root in root.subtree
                if root.dep_ in ['attr']:
                  logger.debug(f'({sscEnts[0]} health status: {sscEnts[0][0]._.health_status}), <--- "{causalEnts[0]}", <--- ({sscEnts[1]} health status: {sscEnts[1][0]._.health_status})')
                elif root.dep_ in ['dobj']:
                  logger.debug(f'({sscEnts[0]} health status: {sscEnts[0][0]._.health_status}), ---> "{causalEnts[0]}", ---> ({sscEnts[1]} health status: {sscEnts[1][0]._.health_status})')
              elif rootLoc < sscEnts[0][0].start:
                # assert sscEnts[0].root in root.subtree
                logger.debug(f' "{causalEnts[0]}" ({sscEnts[0]} health status: {sscEnts[0][0]._.health_status}) ---> ({sscEnts[1]} health status: {sscEnts[1][0]._.health_status})')
          elif causalEntLemma in self._causalKeywords['causal-relator']:
            if rootLoc > sscEnts[0][0].start and rootLoc < sscEnts[1][0].start:
              logger.debug(f'({sscEnts[0]} health status: {sscEnts[0][0]._.health_status}), ---> "{causalEnts[0]}", ---> ({sscEnts[1]} health status: {sscEnts[1][0]._.health_status})')
            else:
              logger.debug(f'Not yet implemented! causal keyword {causalEntLemma}, sentence {sent}')
          elif causalEntLemma in self._causalKeywords['effect-relator']:
            if rootLoc > sscEnts[0][0].start and rootLoc < sscEnts[1][0].start:
              logger.debug(f'({sscEnts[0]} health status: {sscEnts[0][0]._.health_status}), <--- "{causalEnts[0]}", <--- ({sscEnts[1]} health status: {sscEnts[1][0]._.health_status})')
            elif rootLoc < sscEnts[0][0].start:
              logger.debug(f' "{causalEnts[0]}" ({sscEnts[0]} health status: {sscEnts[0][0]._.health_status}) ---> ({sscEnts[1]} health status: {sscEnts[1][0]._.health_status})')
        elif len(causalEnts) == 2:
          ents = self.getCustomEnts(sent.ents, self._entityLabels[self._labelCausal].union(self._entityLabels[self._labelSSC]))
          ents = self.getConjuncts(ents)
          print(ents)
          # TODO: extract more detailed information, the cause directions
        else:
          continue
      # TODO, handle more than two entities
      elif len(causalEnts) == 1 and len(sscEnts) > 2:
        continue

        # The following checks is not complete, swith to different method
        # elif root.pos_ == 'SCONJ':
        #   if rootLoc < sscEnts[0].start:
        #     logger.debug(f' "{causalEnts[0]}" ({sscEnts[0]} health status: {sscEnts[0]._.health_status}) ---> ({sscEnts[1]} health status: {sscEnts[1]._.health_status})')
        #   elif rootLoc > sscEnts[0].start and rootLoc < sscEnts[1].start:
        #     logger.debug(f'({sscEnts[0]} health status: {sscEnts[0]._.health_status}), <--- "{causalEnts[0]}", ({sscEnts[1]} health status: {sscEnts[1]._.health_status})')
        #   else:
        #     logger.debug(f'Unknow relation: {(sscEnts[0].start, sscEnts[0])}, {(sscEnts[1].start, sscEnts[1])}, {(rootLoc, causalEnts[0])}')
        # elif root.pos_ == ' CCONJ':
        #   if rootLoc > sscEnts[0].start and rootLoc < sscEnts[1].start:
        #     logger.debug(f'({sscEnts[0]} health status: {sscEnts[0]._.health_status}), ---> "{causalEnts[0]}", ({sscEnts[1]} health status: {sscEnts[1]._.health_status})')
        #   else:
        #     logger.debug(f'Unknow relation: {(sscEnts[0].start, sscEnts[0])}, {(sscEnts[1].start, sscEnts[1])}, {(rootLoc, causalEnts[0])}')
        # elif root.pos_ == 'ADV' and root.dep_ == 'advmod':
        #   if rootLoc > sscEnts[0].start and rootLoc < sscEnts[1].start:
        #     logger.debug(f'({sscEnts[0]} health status: {sscEnts[0]._.health_status}), ---> "{causalEnts[0]}", ({sscEnts[1]} health status: {sscEnts[1]._.health_status})')
        #   else:
        #     logger.debug(f'Unknow relation: {(sscEnts[0].start, sscEnts[0])}, {(sscEnts[1].start, sscEnts[1])}, {(rootLoc, causalEnts[0])}')
        # elif root.pos_ == 'ADP':
        #   logger.debug('not yet implemented')



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


  ###############
  # methods can be used for callback in "add" method
  ###############
  @staticmethod
  def extendEnt(matcher, doc, i, matches):
    """
      Extend the doc's entity
      @ In, matcher, spacy.Matcher, the spacy matcher instance
      @ In, doc, spacy.tokens.doc.Doc, the processed document using nlp pipelines
      @ In, i, int, index of the current match (matches[i])
      @ In, matches, List[Tuple[int, int, int]], a list of (match_id, start, end) tuples, describing
        the matches. A match tuple describes a span doc[start:end]
    """
    id, start, end = matches[i]
    ent = Span(doc, start, end, label=id)
    logger.debug(ent.text)
    doc.ents = filter_spans(list(doc.ents) +[ent])

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
      if root.pos_ == 'VERB' and [root.lemma_] in predSynonyms:
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
          if [token.lemma_] in predSynonyms:
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
        nbor = child.nbor()
        if nbor.dep_ in ['prep'] and nbor.lemma_ in ['of']:
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
        if var.lemma_ in vals:
          return True
    elif isinstance(keywords, list):
      if var.lemma_ in keywords:
        return True
    return False
#######################################################################################
