from spacy.matcher import PhraseMatcher
from spacy.tokens import Span
from spacy.language import Language
from quantulum3 import parser
# filter_spans is used to resolve the overlap issue in entities
# It gives primacy to longer spans (entities)
from spacy.util import filter_spans

import logging
logging.getLogger('quantulum3').setLevel(logging.ERROR)
logger = logging.getLogger(__name__)


@Language.factory("unit_entity", default_config={"label": "unit", "asSpan":True})
def create_unit_component(nlp, name, label, asSpan):
  return UnitEntity(nlp, label, asSpan=asSpan)

class UnitEntity(object):
  """
    How to use it:
    from UnitEntity import UnitEntity
    nlp = spacy.load("en_core_web_sm")
    unit = UnitEntity(nlp, 'ssc')
    doc = nlp("The shaft deflection is causing the safety cage to rattle. Pumps not experiencing enough flow for the pumps to keep the check valves open during test. Pump not experiencing enough flow during test. Shaft made noise. Vibration seems like it is coming from the shaft.")
    updatedDoc = unit(doc)

    or:

    nlp.add_pipe('unit_entity', config={"label": "ssc", "asSpan":True})
    newDoc = nlp(doc.text)
  """

  def __init__(self, nlp, label='unit', asSpan=True, callback=None):
    """
      @ In, nlp
      @ label, str, the name/label for the patterns in terms
    """
    self.name = 'unit_entity'
    self.label = label
    self.nlp = nlp
    self.matcher = PhraseMatcher(nlp.vocab, attr='LOWER')
    self.asSpan = asSpan

  def __call__(self, doc):
    """
      @ In, doc, spacy.tokens.doc.Doc, the processed document using nlp pipelines
    """
    text = doc.text
    # print(text)
    quants = parser.parse(text)
    # print(quants)
    # Methods using pattern and matcher to identify the entities
    quants = set([quant.surface.lower().strip() for quant in quants if quant.unit.entity.name not in ['dimensionless', 'time']])
    patterns = [self.nlp.make_doc(quant) for quant in quants]
    self.matcher.add(self.label, patterns)
    matches = self.matcher(doc, as_spans=self.asSpan)

    spans = []
    if not self.asSpan:
      for label, start, end in matches:
        span = Span(doc, start, end, label=label)
        spans.append(span)
    else:
      spans.extend(matches)

    newEnts = []
    for ent in spans:
      check = [True if tk.dep_ in ['prep'] or tk.pos_ in ['ADP'] else False for tk in ent]
      if True not in check:
        newEnts.append(ent)
    doc.ents = filter_spans(list(doc.ents)+newEnts)
    return doc
