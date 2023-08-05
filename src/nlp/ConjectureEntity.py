from spacy.language import Language
from .CreatePatterns import CreatePatterns

import logging
logger = logging.getLogger(__name__)

@Language.factory("conjecture_entity", default_config={"patterns": None,  "asSpan":True})
def create_conjecture_component(nlp, name, patterns, asSpan):
  return ConjectureEntity(nlp, patterns=patterns, asSpan=asSpan)


class ConjectureEntity(object):
  """
    How to use it:
    from ConjectureEntity import ConjectureEntity
    nlp = spacy.load("en_core_web_sm")
    patterns = {'label': 'conjecture', 'pattern': [{'LOWER': 'possible'}], 'id': 'conjecture'}
    cmatcher = ConjectureEntity(nlp, patterns)
    doc = nlp("Vibration seems like it is coming from the shaft.")
    updatedDoc = cmatcher(doc)

    or:

    nlp.add_pipe('conjecture_entity', config={"patterns": {'label': 'conjecture', 'pattern': [{'LOWER': 'possible'}], 'id': 'conjecture'}, "asSpan":True})
    newDoc = nlp(doc.text)
  """

  def __init__(self, nlp, patterns=None, asSpan=True, callback=None):
    """
      @ In, nlp
      @ label, str, the name/label for the patterns in terms
      @ terms, list, the phrase list, for example:
        phraseList = ["hello", "world"]
    """
    self.name = 'conjecture_entity'
    if patterns is None:
      filename = '~/projects/raven/plugins/SR2ML/src/nlp/data/conjecture_keywords.csv'
      conjecturePatterns = CreatePatterns(filename, entLabel='conjecture', nlp=nlp)
      patterns = conjecturePatterns.getPatterns()
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
