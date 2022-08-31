from spacy.matcher import Matcher
from spacy.tokens import Span
from spacy.language import Language
# filter_spans is used to resolve the overlap issue in entities
# It gives primacy to longer spans (entities)
from spacy.util import filter_spans

import logging
logger = logging.getLogger(__name__)

@Language.factory("simple_entity_matcher", default_config={"label": "ssc", "terms":[{"LOWER": "hello"}, {"IS_PUNCT": True}, {"LOWER": "world"}], "asSpan":True})
def create_simple_matcher_component(nlp, name, label, terms, asSpan):
  return SimpleEntityMatcher(nlp, label, terms, asSpan=asSpan)

class SimpleEntityMatcher(object):
  """
    How to use it:
    from SimpleEntityMatcher import SimpleEntityMatcher
    nlp = spacy.load("en_core_web_sm")
    terms = [{"LOWER": "hello"}, {"IS_PUNCT": True}, {"LOWER": "world"}]
    pmatcher = SimpleEntityMatcher(nlp, 'ssc', terms)
    doc = nlp("The shaft deflection is causing the safety cage to rattle. Pumps not experiencing enough flow for the pumps to keep the check valves open during test. Pump not experiencing enough flow during test. Shaft made noise. Vibration seems like it is coming from the shaft.")
    updatedDoc = pmatcher(doc)

    or:

    nlp.add_pipe('simple_entity_matcher', config={"label": "ssc", "terms":[{"LOWER": "hello"}, {"IS_PUNCT": True}, {"LOWER": "world"}], "asSpan":True})
    newDoc = nlp(doc.text)
  """

  def __init__(self, nlp, label, terms, asSpan=True, callback=None):
    """
      @ In, nlp
      @ label, str, the name/label for the patterns in terms
      @ In, terms, list, the rules used to match the entities, for example:
        terms = [{"LOWER": "hello"}, {"IS_PUNCT": True}, {"LOWER": "world"}]
    """
    self.name = 'simple_entity_matcher'
    self.matcher = Matcher(nlp.vocab)
    if not isinstance(terms, list):
      terms = [terms]
    if not isinstance(terms[0], list):
      terms = [terms]
    self.matcher.add(label, terms, on_match=callback)
    self.asSpan = asSpan

  def __call__(self, doc):
    """
      @ In, doc, spacy.tokens.doc.Doc, the processed document using nlp pipelines
    """
    matches = self.matcher(doc, as_spans=self.asSpan)
    spans = []
    if not self.asSpan:
      for label, start, end in matches:
        span = Span(doc, start, end, label=label)
        spans.append(span)
    else:
      spans.extend(matches)
    doc.ents = filter_spans(list(doc.ents)+spans)
    return doc
