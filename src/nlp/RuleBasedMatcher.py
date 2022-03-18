import spacy
from spacy.matcher import Matcher
from spacy.tokens import Span
from spacy import displacy
from spacy.matcher import PhraseMatcher
from spacy.matcher import DependencyMatcher

import logging


logger = logging.getLogger(__name__)

## temporary add stream handler
# ch = logging.StreamHandler()
# logger.addHandler(ch)
##

class RuleBasedMatcher(object):
  """
  """

  def __init__(self, nlp, *args, **kwargs):
    """
      Construct
      @ In, nlp, A spaCy language model object
      @ In, rules, str or list, where to read rules from
      @ In, args, list, positional arguments
      @ In, kwargs, dict, keyword arguments
      @ Out, None
    """
    self.type = self.__class__.__name__
    self.name = self.__class__.__name__
    logger.info(f'Create instance of {self.name}')
    # nlp = spacy.load("en_core_web_sm")
    self.nlp = nlp
    self._rules = {}
    self._match = False
    self._phraseMatch = False
    self._dependencyMatch = False
    self.matcher = Matcher(nlp.vocab)
    self.phraseMatcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    self.dependencyMatcher = DependencyMatcher(nlp.vocab)
    self._callbacks = {}
    self._asSpans = True # When True, a list of Span objects using the match_id as the span label will be returned
    self._matchedSents = [] # collect data of matched sentences to be visualized
    self._visualizeMatchedSents = True

  def addPattern(self, name, rules, callback=None):
    """
      Add rules
      @ In, name, str,
      @ In, rules, list,
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
      @ In, name, str,
      @ In, phraseList, list,
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
      @ In, name, str,
      @ In, patternList, list,
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


  def __call__(self, text):
    """
    """
    doc = self.nlp(text)
    matches = []
    if self._match:
      matches += self.matcher(doc, as_spans = self._asSpans) # <class 'list'>
    if self._phraseMatch:
      matches += self.phraseMatcher(doc, as_spans = self._asSpans) # <class 'list'>
    if self._dependencyMatch:
      depMatches = self.dependencyMatcher(doc) # <class 'list'> [tuple(match_id, token_ids)]

    if self._asSpans:
      for span in matches:
        logger.debug(f'Matches: {span.text}, {span.label_}')
    else:
      for id, start, end in matches:
        strID = self.nlp.vocab.strings[id]
        span = doc[start:end]
        logger.debug(f'Matches: {strID}, {start}, {end}, {span.text}')

    # print dependency matches
    for (id, tokenIDs) in depMatches:
      name = self.nlp.vocab.strings[id]
      for i in range(len(tokenIDs)):
        print(self._rules[name][0][i]["RIGHT_ID"] + ":",doc[tokenIDs[i]].text)



  def visualize():
    """
    """
    if self._visualizeMatchedSents:
      # Serve visualization of sentences containing match with displaCy
      # set manual=True to make displaCy render straight from a dictionary
      # (if you're not running the code within a Jupyer environment, you can
      # use displacy.serve instead)
      displacy.render(self._matchedSents, style="ent", manual=True)

  ###############
  # methods can be used for callback in "add" method
  ###############
  def extendEnt(matcher, doc, i, matches):
    """
      Extend the doc's entity
      @ In, matcher, spacy.Matcher, the spacy matcher instance
      @ In, doc, the document the matcher was used on
      @ In, i, int, index of the current match (matches[i])
      @ In, matches, List[Tuple[int, int, int]], a list of (match_id, start, end) tuples, describing the matches. A
        match tuple describes a span doc[start:end]
    """
    id, start, end = matches[i]
    ent = Span(doc, start, end, label=id)
    doc.ents += (ent,)
    logger.debug(ent.text)

  def collectSents(matcher, doc, i, matches):
    """
      collect data of matched sentences that can be used for visualization
      @ In, matcher, spacy.Matcher, the spacy matcher instance
      @ In, doc, the document the matcher was used on
      @ In, i, int, index of the current match (matches[i])
      @ In, matches, List[Tuple[int, int, int]], a list of (match_id, start, end) tuples, describing the matches. A
        match tuple describes a span doc[start:end]
    """
    id, start, end = matches[i]
    span = doc[start:end]  # Matched span
    sent = span.sent  # Sentence containing matched span
    # Append mock entity for match in displaCy style to matched_sents
    # get the match span by ofsetting the start and end of the span with the
    # start and end of the sentence in the doc
    matchEnts = [{
        "start": span.start_char - sent.start_char,
        "end": span.end_char - sent.start_char,
        "label": "MATCH",
    }]
    self._matchedSents.append({"text": sent.text, "ents": matchEnts})
