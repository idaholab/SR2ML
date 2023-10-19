from spacy.language import Language
import pandas as pd
from nlp.nlp_utils import generatePatternList
# from .config import nlpConfig

import logging
logger = logging.getLogger(__name__)

@Language.factory("temporal_relation_entity", default_config={"patterns": None})
def create_temporal_relation_component(nlp, name, patterns):
  return TemporalRelationEntity(nlp, patterns=patterns)


class TemporalRelationEntity(object):
  """
    How to use it:
    from TemporalRelationEntity import TemporalRelationEntity
    nlp = spacy.load("en_core_web_sm")
    patterns = {'label': 'temporal_relation', 'pattern': [{'LOWER': 'follow'}], 'id': 'temporal_relation'}
    cmatcher = ConjectureEntity(nlp, patterns)
    doc = nlp("The system failed following the pump failure.")
    updatedDoc = cmatcher(doc)

    or:

    nlp.add_pipe('temporal_relation_entity', config={"patterns": {'label': 'temporal_relation', 'pattern': [{'LOWER': 'follow'}], 'id': 'temporal_relation'}})
    newDoc = nlp(doc.text)
  """

  def __init__(self, nlp, patterns=None, callback=None):
    """
      @ In, nlp
      @ label, str, the name/label for the patterns in terms
      @ patterns, list/dict,
    """
    self.name = 'temporal_relation_entity'
    if patterns is None:
      # update to use config file instead
      # filename = nlpConfig['files']['time_relation_file']
      filename = '~/projects/raven/plugins/SR2ML/src/nlp/data/time_relation_keywords.csv'
      entLists = pd.read_csv(filename, header=0)
      orderList = entLists['order'].dropna().values.ravel().tolist()
      reverseOrderList = entLists['reverse-order'].dropna().values.ravel().tolist()
      concurrencyList = entLists['concurrency-coincidence'].dropna().values.ravel().tolist()
      patterns = []
      orderPatterns = generatePatternList(orderList, label='temporal_relation_order', id='temporal_relation', nlp=nlp, attr="LEMMA")
      patterns.extend(orderPatterns)
      reverseOrderPatterns = generatePatternList(reverseOrderList, label='temporal_relation_reverse_order', id='temporal_relation', nlp=nlp, attr="LEMMA")
      patterns.extend(reverseOrderPatterns)
      concurrencyPatterns = generatePatternList(concurrencyList, label='temporal_relation_concurrency', id='temporal_relation', nlp=nlp, attr="LEMMA")
      patterns.extend(concurrencyPatterns)

    if not isinstance(patterns, list) and isinstance(patterns, dict):
      patterns = [patterns]
    # do we need to pop out other pipes?
    if not nlp.has_pipe('entity_ruler'):
      nlp.add_pipe('entity_ruler')
    self.entityRuler = nlp.get_pipe('entity_ruler')
    self.entityRuler.add_patterns(patterns)

  def __call__(self, doc):
    """
      @ In, doc, spacy.tokens.doc.Doc, the processed document using nlp pipelines
    """
    doc = self.entityRuler(doc)
    return doc
