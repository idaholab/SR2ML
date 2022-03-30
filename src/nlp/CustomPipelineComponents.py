import spacy
from spacy.language import Language
from spacy.tokens import Span
from spacy.matcher import Matcher
from spacy.tokens import Token
# filter_spans is used to resolve the overlap issue in entities
# It gives primacy to longer spans (entities)
from spacy.util import filter_spans

import logging


logger = logging.getLogger(__name__)

#### Using spacy's Token extensions for coreferee
if Token.has_extension('ref_n'):
  _ = Token.remove_extension('ref_n')
if Token.has_extension('ref_t'):
  _ = Token.remove_extension('ref_t')
if Token.has_extension('ref_t_'):
  _ = Token.remove_extension('ref_t_')
Token.set_extension('ref_n', default='')
Token.set_extension('ref_t', default='')
Span.set_extension("health_status", default=None)

customLabel = ['STRUCTURE', 'COMPONENT', 'SYSTEM']
aliasLookup = {}

# orders of NLP pipeline: 'ner' --> 'normEntities' --> 'merge_entities' --> 'initCoref'
# --> 'aliasResolver' --> 'coreferee' --> 'anaphorCoref'

@Language.component("normEntities")
def normEntities(doc):
  """
    Normalizing Named Entities, remove the leading article and trailing particle
    @ In, doc, spacy.tokens.doc.Doc
    @ Out, doc, spacy.tokens.doc.Doc
  """
  ents = []
  for ent in doc.ents:
    if ent[0].pos_ == "DET": # leading article
      ent = Span(doc, ent.start+1, ent.end, label=ent.label)
    if len(ent) > 0:
      if ent[-1].pos_ == "PART": # trailing particle like 's
        ent = Span(doc, ent.start, ent.end-1, label=ent.label)
      if len(ent) > 0:
        ents.append(ent)
  doc.ents = tuple(ents)
  return doc

@Language.component("initCoref")
def initCoref(doc):
  """
  """
  for e in doc.ents:
    #
    # if e.label_ in customLabel:
    e[0]._.ref_n, e[0]._.ref_t = e.text, e.label_
  return doc

@Language.component("aliasResolver")
def aliasResolver(doc):
  """
    Lookup aliases and store result in ref_t, ref_n
  """
  for ent in doc.ents:
    token = ent[0].text
    if token in aliasLookup:
      aName, aType = aliasLookup[token]
      ent[0]._.ref_n, ent[0]._.ref_t = aName, aType
  return propagateEntType(doc)

def propagateEntType(doc):
  """
    propagate entity type stored in ref_t
  """
  ents = []
  for e in doc.ents:
    if e[0]._.ref_n != '': # if e is a coreference
      e = Span(doc, e.start, e.end, label=e[0]._.ref_t)
    ents.append(e)
  doc.ents = tuple(ents)
  return doc

@Language.component("anaphorCoref")
def anaphorCoref(doc):
  """
    Anaphora resolution using coreferee
    This pipeline need to be added after NER.
    The assumption here is: The entities need to be recognized first, then call
    pipeline "initCoref" to assign initial custom attribute "ref_n" and "ref_t",
    then call pipeline "aliasResolver" to resolve all the aliases used in the text.
    After all these pre-processes, we can use "anaphorCoref" pipeline to resolve the
    coreference.
  """
  if not Token.has_extension('coref_chains'):
    return doc
  for token in doc:
    coref = token._.coref_chains
    # if token is coref and not already dereferenced
    if coref and token._.ref_n == '':
      # check all the references, if "ref_n" is available (determined by NER and initCoref),
      # the value of "ref_n" will be assigned to current totken
      for chain in coref:
        for ref in chain:
          refToken = doc[ref[0]]
          if refToken._.ref_n != '':
            token._.ref_n = refToken._.ref_n
            token._.ref_t = refToken._.ref_t
            break
  return doc

@Language.component("expandEntities")
def expandEntities(doc):
  """
    Expand the current entities, recursive function to extend entity with all previous NOUN
  """
  newEnts = []
  isUpdated = False
  for ent in doc.ents:
    if ent.ent_id_ == "SSC" and ent.start != 0:
      prevToken = doc[ent.start - 1]
      if prevToken.pos_ in ['NOUN']:
        newEnt = Span(doc, ent.start - 1, ent.end, label=ent.label)
        newEnts.append(newEnt)
        isUpdated = True
    else:
      newEnts.append(ent)
  print(newEnts)
  doc.ents = filter_spans(list(doc.ents) +  newEnts)
  if isUpdated:
    doc = expandEntities(doc)
  return doc

# @Language.component("extractHealthStatus")
# def extractHealthStatus(doc):
#   """
#     Extract the health status of identified entities
#   """
#   for ent in doc.ents:
#     if ent.ent_id_ == "SSC":
#       sent = ent.sent
#
#   return doc




#########################################################
# pipelines that can be used in future work

###TODO: update the following pipelines
# @Language.component("extract_person_orgs")
# def extract_person_orgs(doc):
#     person_entities = [ent for ent in doc.ents if ent.label_ == "PERSON"]
#     for ent in person_entities:
#         head = ent.root.head
#         if head.lemma_ == "work":
#             preps = [token for token in head.children if token.dep_ == "prep"]
#             for prep in preps:
#                 orgs = [token for token in prep.children if token.ent_type_ == "ORG"]
#                 print({'person': ent, 'orgs': orgs, 'past': head.tag_ == "VBD"})
#     return doc

# @Language.component("extract_person_orgs")
# def extract_person_orgs(doc):
#     person_entities = [ent for ent in doc.ents if ent.label_ == "PERSON"]
#     for ent in person_entities:
#         head = ent.root.head
#         if head.lemma_ == "work":
#             preps = [token for token in head.children if token.dep_ == "prep"]
#             for prep in preps:
#                 orgs = [t for t in prep.children if t.ent_type_ == "ORG"]
#                 aux = [token for token in head.children if token.dep_ == "aux"]
#                 past_aux = any(t.tag_ == "VBD" for t in aux)
#                 past = head.tag_ == "VBD" or head.tag_ == "VBG" and past_aux
#                 print({'person': ent, 'orgs': orgs, 'past': past})
#     return doc
