from spacy.language import Language
from .CreatePatterns import CreatePatterns
import pandas as pd
from nlp.nlp_utils import generatePatternList
# from .config import nlpConfig

import logging
logger = logging.getLogger(__name__)

@Language.factory("location_entity", default_config={"patterns": None})
def create_location_component(nlp, name, patterns):
  return LocationEntity(nlp, patterns=patterns)


class LocationEntity(object):
  """
    How to use it:
    from LocationEntity import LocationEntity
    nlp = spacy.load("en_core_web_sm")
    patterns = {'label': 'location', 'pattern': [{'LOWER': 'follow'}], 'id': 'location'}
    cmatcher = ConjectureEntity(nlp, patterns)
    doc = nlp("It is close to 5pm.")
    updatedDoc = cmatcher(doc)

    or:

    nlp.add_pipe('location_entity', config={"patterns": {'label': 'location', 'pattern': [{'LOWER': 'follow'}], 'id': 'location'}})
    newDoc = nlp(doc.text)
  """

  def __init__(self, nlp, patterns=None, callback=None):
    """
      @ In, nlp
      @ label, str, the name/label for the patterns in terms
      @ patterns, list/dict,
    """
    self.name = 'location_entity'
    if patterns is None:
      # update to use config file instead
      # filename = nlpConfig['files']['location_file']
      filename = '~/projects/raven/plugins/SR2ML/src/nlp/data/location_keywords.csv'
      entLists = pd.read_csv(filename, header=0)
      proxList = entLists['proximity'].dropna().values.ravel().tolist()
      upList = entLists['up'].dropna().values.ravel().tolist()
      downList = entLists['down'].dropna().values.ravel().tolist()
      patterns = []
      proxPatterns = generatePatternList(proxList, label='location_proximity', id='location', nlp=nlp, attr="LEMMA")
      patterns.extend(proxPatterns)
      upPatterns = generatePatternList(upList, label='location_up', id='location', nlp=nlp, attr="LEMMA")
      patterns.extend(upPatterns)
      downPatterns = generatePatternList(downList, label='location_down', id='location', nlp=nlp, attr="LEMMA")
      patterns.extend(downPatterns)

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
