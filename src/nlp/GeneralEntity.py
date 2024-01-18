from spacy.language import Language
import logging
logger = logging.getLogger(__name__)



@Language.factory("general_entity", default_config={"patterns": {'label': 'general', 'pattern': [{'LOWER': 'possible'}], 'id': 'general'},  "asSpan":True})
def create_general_component(nlp, name, patterns, asSpan):
  return GeneralEntity(nlp, patterns, asSpan=asSpan)


class GeneralEntity(object):
  """
    How to use it:
    from GeneralEntity import GeneralEntity
    nlp = spacy.load("en_core_web_sm")
    patterns = {'label': 'general', 'pattern': [{'LOWER': 'possible'}], 'id': 'general'}
    cmatcher = generalEntity(nlp, patterns)
    doc = nlp("Vibration seems like it is coming from the shaft.")
    updatedDoc = cmatcher(doc)

    or:

    nlp.add_pipe('general_entity', config={"patterns": {'label': 'general', 'pattern': [{'LOWER': 'possible'}], 'id': 'general'}, "asSpan":True})
    newDoc = nlp(doc.text)
  """

  def __init__(self, nlp, patterns, asSpan=True, callback=None):
    """
      @ In, nlp
      @ label, str, the name/label for the patterns in terms
      @ patterns, list/dict,
    """
    self.name = 'general_entity'
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
