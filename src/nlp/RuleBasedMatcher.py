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
    self._entityLabels = []



    self._statusKeyword = ['fail', 'degrade', 'break', 'decline', 'go bad', 'rupture', 'breach', 'reduce', 'increase',
        'decrease', 'fracture', 'aggravate','worsen', 'lose', 'function', 'work', 'operate', 'run', 'find', 'find out',
        'observe', 'detect', 'determine', 'discover', 'get', 'notice', 'become', 'record', 'register', 'show']
    self._causalKeyword = ['cause', 'stimulate', 'make', 'derive', 'trigger', 'result', 'lead', 'increase', 'decrease']


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
      @ Out, lemVar, list, list of lammatized variables
    """
    var = ' '.join(varList)
    lemVar = [token.lemma_ for token in self.nlp(var)]
    return lemVar

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
          self._statusKeywords[key].extend(val)
        else:
          logger.warning('keyword "{}" cannot be accepted, valid keys for the keywords are "{}"'.format(key, ','.join(list(self._statusKeywords.keys()))))
    elif ktype.lower() == 'causal':
      for key, val in keywords.items():
        if type(val) != list:
          val = [val]
        val = self.extractLemma(val)
        if key in self._causalKeywords:
          self._causalKeywords[key].extend(val)
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
    self._entityLabels += [pa.get('label') for pa in patternList if pa.get('label') is not None]
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
      print([(ent.text, ent.label_, ent.ent_id_) for ent in doc.ents if ent.label_ in self._entityLabels])

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
    self.extractHealthStatus(self._matchedSents)
    ## print extracted relation
    print(*self.extractRelDep(self._matchedSents, entID='SSC', predName='causes', predSynonyms=[], exclPrepos=[]), sep='\n')



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

  def isNegation(self, token):
    """
    """
    if token.dep_ == 'neg':
      return True, token.text
    for left in token.lefts:
      if left.dep_ == 'neg':
        return True, left.text
    return False, ''

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
    self.extractStatusFromPredicate(matchedSents)

    # for sent in matchedSents:
    #   ents = list(sent.ents)
    #   predSyns = self._statusKeyword if len(predSynonyms) == 0 else predSynonyms
    #   root = sent.root
    #   if root.lemma_ not in predSyns:
    #     continue
    #   if root.pos_ != 'VERB':
    #     continue
    #   passive = self.isPassive(root)
    #   if len(ents) == 1:
    #     if ents[0].start < root.i:
    #       healthStatus = self.findRightObj(root)
    #     else:
    #       healthStatus = self.findLeftSubj(root, passive)
    #
    #     if healthStatus is None:
    #       continue
    #
    #     logger.debug(f'{ents[0]} health status: {healthStatus.text}')
    #     ents[0]._.set('health_status', healthStatus.text)
    #   else:
    #     logger.debug('Not yet implemented')


  def extractStatusFromPredicate(self, matchedSents, exclPrepos=[]):
    """
      Extract health status
      @ In, matchedSents, list, the matched sentences
      @ In, exclPrepos, list, exclude the prepositions
    """
    predSynonyms = self._statusKeywords['VERB']
    statusNoun = self._statusKeywords['NOUN']
    statusAdj = self._statusKeywords['ADJ']
    for sent in matchedSents:
      ents = self.getCustomEnts(sent.ents, self._entityLabels)
      # ents = list(sent.ents)
      # TODO: multiple entities exist, skip for now
      if len(ents) > 1:
        continue
      root = sent.root
      if root.lemma_ not in predSynonyms:
        if not self._updateStatusKeywords:
          continue
        elif root.pos_ in ['VERB', 'NOUN', 'ADJ']:
          self.addKeywords({root.pos_:[root]}, 'status')
      if root.pos_ != 'VERB':
        print('--- root not verb', root.text, root.pos_)
        continue
      neg, negText = self.isNegation(root)
      passive = self.isPassive(root)
      # # last is punct, the one before last is the root
      # if root.nbor().pos_ in ['PUNCT']:
      #   healthStatus = root
      if ents[0].start < root.i:
        healthStatus = self.findRightObj(root)
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
        neg, negText = self.isNegation(healthStatus)
      logger.debug(f'{ents[0]} health status: {negText} {healthStatus.text}')
      ents[0]._.set('health_status', negText + healthStatus.text)

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
        if right.lemma_ in self._statusKeywords[pos]:
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

  def bfs(self, root, entID, deps, firstDepOnly=False):
    """
      Return first child of root (included) that matches
      entType and dependency list by breadth first search.
      Search stops after first dependency match if firstDepOnly
      (used for subject search - do not "jump" over subjects)
      @ In, root, spacy.tokens.Token, the root token
      @ In, entID, string, the ID for the entity
      @ In, deps, list, list of dependency
      @ In, firstDepOnly, bool, True if only search for the first dependency
      @ Out, child, spacy.tokens.Token, the matched token
    """
    toVisit = deque([root]) # queue for bfs

    while len(toVisit) > 0:
      child = toVisit.popleft()
      # print("child", child, child.dep_)
      if child.dep_ in deps:
        if child.ent_id_ == entID:
          return child
        elif firstDepOnly: # first match (subjects)
          return None
      elif child.dep_ == 'compound' and \
         child.head.dep_ in deps and \
         child.ent_id_ == entID: # check if contained in compound
        return child
      toVisit.extend(list(child.children))
    return None

  def findSubj(self, pred, entID, passive):
    """
      Find closest subject in predicates left subtree or
      predicates parent's left subtree (recursive).
      Has a filter on organizations.
      @ In, pred, spacy.tokens.Token, the predicate token
      @ In, entID, string, the ID for the entity
      @ In, passive, bool, True if the predicate token is passive
      @ Out, subj, spacy.tokens.Token, the token that represents subject
    """
    for left in pred.lefts:
      if passive: # if pred is passive, search for passive subject
        subj = self.bfs(left, entID, ['nsubjpass', 'nsubj:pass'], False)
      else:
        subj = self.bfs(left, entID, ['nsubj'], False)
      if subj is not None: # found it!
        return subj
    if pred.head != pred and not self.isPassive(pred):
      return self.findSubj(pred.head, entID, passive) # climb up left subtree
    else:
      return None

  def findObj(self, pred, entID, exclPrepos):
    """
      Find closest object in predicates right subtree.
      Skip prepositional objects if the preposition is in exclude list.
      Has a filter on organizations.
      @ In, pred, spacy.tokens.Token, the predicate token
      @ In, entID, string, the ID for the entity
      @ In, exclPrepos, list, the list of prepositions that will be excluded
      @ Out, obj, spacy.tokens.Token,, the token that represents the object
    """
    for right in pred.rights:
      obj = self.bfs(right, entID, ['dobj', 'pobj', 'iobj', 'obj', 'obl'])
      if obj is not None:
        if obj.dep_ == 'pobj' and obj.head.lemma_.lower() in exclPrepos: # check preposition
          continue
        return obj
    return None

  def extractRelDep(self, matchedSents, entID, predName='causes', predSynonyms=[], exclPrepos=[]):
    """
      @ In, matchedSents, list, the list of matched sentences
      @ In, entID, string, the ID of entity
      @ In, predName, string, the given name for predicate
      @ In, predSynonyms, list, the list of predicate synonyms
      @ In, exclPrepos, list, the list of exlcuded prepositions
      @ Out, (subject tuple, predicate, object tuple), generator, the extracted causal relation
    """
    for sent in matchedSents:
      if len(set(sent.ents)) < 2:
        continue
      for token in sent:
        predSyns = [token.lemma_] if len(predSynonyms) == 0 else predSynonyms
        if token.pos_ == 'VERB' and token.lemma_ in predSyns:
          pred = token
          passive = self.isPassive(pred)
          subj = self.findSubj(pred, entID, passive)
          if subj is not None:
            obj = self.findObj(pred, entID, exclPrepos)
            if obj is not None:
              if passive: # switch roles
                obj, subj = subj, obj
              yield ((subj._.ref_n, subj._.ref_t), predName,
                     (obj._.ref_n, obj._.ref_t))


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
  @staticmethod
  def collectSents(doc):
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
