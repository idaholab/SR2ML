# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
Created on March, 2022

@author: wangc, mandd
"""
import spacy
from spacy.language import Language
from spacy.tokens import Span
from spacy.matcher import Matcher
from spacy.tokens import Token
# filter_spans is used to resolve the overlap issue in entities
# It gives primacy to longer spans (entities)
from spacy.util import filter_spans

# use pysbd as a sentencizer component for spacy
import pysbd

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
if not Token.has_extension('ref_ent'):
  Token.set_extension("ref_ent", default=None)

customLabel = ['STRUCTURE', 'COMPONENT', 'SYSTEM']
aliasLookup = {}

# orders of NLP pipeline: 'ner' --> 'normEntities' --> 'merge_entities' --> 'initCoref'
# --> 'aliasResolver' --> 'coreferee' --> 'anaphorCoref'

@Language.component("normEntities")
def normEntities(doc):
  """
    Normalizing Named Entities, remove the leading article and trailing particle
    @ In, doc, spacy.tokens.doc.Doc, the processed document using nlp pipelines
    @ Out, doc, spacy.tokens.doc.Doc, the document after the normalizing named entities
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
    Initialize the coreference, assign text and label to custom extension "ref_n" and "ref_t"
    @ In, doc, spacy.tokens.doc.Doc, the processed document using nlp pipelines
    @ Out, doc, spacy.tokens.doc.Doc, the document after the initializing coreference
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
    @ In, doc, spacy.tokens.doc.Doc, the processed document using nlp pipelines
    @ Out, doc, spacy.tokens.doc.Doc, the document after the alias lookup
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
    @ In, doc, spacy.tokens.doc.Doc, the processed document using nlp pipelines
    @ Out, doc, spacy.tokens.doc.Doc, the document after entity type extension
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
    @ In, doc, spacy.tokens.doc.Doc, the processed document using nlp pipelines
    @ Out, doc, spacy.tokens.doc.Doc, the document after the anaphora resolution using coreferee
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

@Language.component("anaphorEntCoref")
def anaphorEntCoref(doc):
  """
    Anaphora resolution using coreferee for Entities
    This pipeline need to be added after NER.
    The assumption here is: The entities need to be recognized first, then call
    pipeline "initCoref" to assign initial custom attribute "ref_n" and "ref_t",
    then call pipeline "aliasResolver" to resolve all the aliases used in the text.
    After all these pre-processes, we can use "anaphorEntCoref" pipeline to resolve the
    coreference.
    @ In, doc, spacy.tokens.doc.Doc, the processed document using nlp pipelines
    @ Out, doc, spacy.tokens.doc.Doc, the document after the anaphora resolution using coreferee
  """
  if not Token.has_extension('coref_chains'):
    return doc

  for ent in doc.ents:
    for token in ent:
      coref = token._.coref_chains
      if not coref:
        continue
      for chain in coref:
        for ref in chain:
          for index in ref:
            refToken = doc[index]
            if refToken._.ref_ent is None:
              refToken._.ref_ent = ent
  return doc



@Language.component("expandEntities")
def expandEntities(doc):
  """
    Expand the current entities, recursive function to extend entity with all previous NOUN
    @ In, doc, spacy.tokens.doc.Doc, the processed document using nlp pipelines
    @ Out, doc, spacy.tokens.doc.Doc, the document after expansion of current entities
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

@Language.component("mergePhrase")
def mergePhrase(doc):
  """
    Expand the current entities
    This method will keep "DET" or "PART", using pipeline "normEntities" after this pipeline to remove them
    @ In, doc, spacy.tokens.doc.Doc, the processed document using nlp pipelines
    @ Out, doc, spacy.tokens.doc.Doc, the document after merge phrase
  """
  def isNum(nounChunks):
    for elem in nounChunks:
      if elem.pos_ == 'NUM':
        return True, elem
        break
    return False, None

  with doc.retokenize() as retokenizer:
    for np in doc.noun_chunks:
      # skip ents since ents are recognized by OPM model and entity_ruler
      # TODO: we may expand the ents, combined with pipeline "expandEntities"
      if len(list(np.ents)) > 1:
        continue
      elif len(list(np.ents)) == 1:
        if np.ents[0].label_ not in ['causal_keywords', 'ORG', 'DATE']:
          # print(np.ents[0].label_)
          continue
      # When a number is provided, we will merge it, but keep the attributes from the number
      num, elem = isNum(np)
      if not num:
        attrs = {
            "tag": np.root.tag_,
            "lemma": np.root.lemma_,
            "pos": np.root.pos_,
            "ent_type": np.root.ent_type_,
            "_": {
                  "ref_n": np.root._.ref_n,
                  "ref_t": np.root._.ref_t,
                  },
        }
      else:
        attrs = {
            "tag": elem.tag_,
            "lemma": elem.lemma_,
            "pos":elem.pos_,
            "ent_type": np.root.ent_type_,
            "_": {
                  "ref_n": np.root._.ref_n,
                  "ref_t": np.root._.ref_t,
                  },
        }
      retokenizer.merge(np, attrs=attrs)
  return doc





@Language.component("pysbdSentenceBoundaries")
def pysbdSentenceBoundaries(doc):
  """
    Use pysbd as a sentencizer component for spacy
    @ In, doc, spacy.tokens.doc.Doc, the processed document using nlp pipelines
    @ Out, doc, spacy.tokens.doc.Doc, the document after process
  """
  seg = pysbd.Segmenter(language="en", clean=False, char_span=True)
  sentsCharSpans = seg.segment(doc.text)
  charSpans = [doc.char_span(sentSpan.start, sentSpan.end, alignment_mode="contract") for sentSpan in sentsCharSpans]
  startTokenIds = [span[0].idx for span in charSpans if span is not None]
  for token in doc:
      token.is_sent_start = True if token.idx in startTokenIds else False
  return doc
