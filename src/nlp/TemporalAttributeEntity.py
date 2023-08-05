from spacy.language import Language
from .CreatePatterns import CreatePatterns
# from .config import nlpConfig

import logging
logger = logging.getLogger(__name__)

@Language.factory("temporal_attribute_entity", default_config={"patterns": None,  "asSpan":True})
def create_temporal_attribute_component(nlp, name, patterns, asSpan):
  return TemporalAttributeEntity(nlp, patterns=patterns, asSpan=asSpan)


class TemporalAttributeEntity(object):
  """
    How to use it:
    from TemporalAttributeEntity import TemporalAttributeEntity
    nlp = spacy.load("en_core_web_sm")
    patterns = {'label': 'temporal_attribute', 'pattern': [{'LOWER': 'possible'}], 'id': 'temporal_attribute'}
    cmatcher = ConjectureEntity(nlp, patterns)
    doc = nlp("It is close to 5pm.")
    updatedDoc = cmatcher(doc)

    or:

    nlp.add_pipe('temporal_attribute_entity', config={"patterns": {'label': 'temporal_attribute_entity', 'pattern': [{'LOWER': 'possible'}], 'id': 'temporal_attribute_entity'}, "asSpan":True})
    newDoc = nlp(doc.text)
  """

  def __init__(self, nlp, patterns=None, asSpan=True, callback=None):
    """
      @ In, nlp
      @ label, str, the name/label for the patterns in terms
      @ patterns, list/dict,
    """
    self.name = 'temporal_attribute_entity'
    if patterns is None:
      # update to use config file instead
      # filename = nlpConfig['files']['time_keywords_file']
      filename = '~/projects/raven/plugins/SR2ML/src/nlp/data/time_keywords.csv'
      temporalPatterns = CreatePatterns(filename, entLabel='temporal_attribute', nlp=nlp)
      patterns = temporalPatterns.getPatterns()
    if not isinstance(patterns, list) and isinstance(patterns, dict):
      patterns = [patterns]
    # do we need to pop out other pipes?
    if not nlp.has_pipe('entity_ruler'):
      nlp.add_pipe('entity_ruler')
    self.entityRuler = nlp.get_pipe('entity_ruler')
    self.entityRuler.add_patterns(patterns)
    self.asSpan = asSpan

  def __call__(self, doc):
    """
      @ In, doc, spacy.tokens.doc.Doc, the processed document using nlp pipelines
    """
    doc = self.entityRuler(doc)
    return doc
