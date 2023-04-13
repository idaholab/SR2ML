# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
Created on March, 2022

@author: wangc, mandd
"""
import pandas as pd
import spacy
from spacy.tokens import Span
# filter_spans is used to resolve the overlap issue in entities
# It gives primacy to longer spans (entities)
from spacy.util import filter_spans
import logging

import networkx as nx
import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)

###########################################################################

def displayNER(doc, includePunct=False):
  """
    Generate data frame for visualization of spaCy doc with custom attributes.
    @ In, doc, spacy.tokens.doc.Doc, the processed document using nlp pipelines
    @ In, includePunct, bool, True if the punctuaction is included
    @ Out, df, pandas.DataFrame, data frame contains attributes of tokens
  """
  rows = []
  for i, t in enumerate(doc):
    if not t.is_punct or includePunct:
      row = {'token': i,
             'text': t.text, 'lemma': t.lemma_,
             'pos': t.pos_, 'dep': t.dep_, 'ent_type': t.ent_type_,
             'ent_iob_': t.ent_iob_}
      if doc.has_extension('coref_chains'):
        if t.has_extension('coref_chains') and t._.coref_chains: # neuralcoref attributes
          row['coref_chains'] = t._.coref_chains.pretty_representation
        else:
          row['coref_chains'] = None
      if t.has_extension('ref_n'): # referent attribute
        row['ref_n'] = t._.ref_n
        row['ref_t'] = t._.ref_t
      if t.has_extension('ref_ent'): # ref_n/ref_t
        row['ref_ent'] = t._.ref_ent
      rows.append(row)
  df = pd.DataFrame(rows).set_index('token')
  df.index.name = None

  return df

def resetPipeline(nlp, pipes):
  """
    remove all custom pipes, and add new pipes
    @ In, nlp, spacy.Language object, contains all components and data needed to process text
    @ In, pipes, list, list of pipes that will be added to nlp pipeline
    @ Out, nlp, spacy.Language object, contains updated components and data needed to process text
  """
  customPipes = [pipe for (pipe, _) in nlp.pipeline
                  if pipe not in ['tagger', 'parser', 'ner',
                                  'tok2vec', 'attribute_ruler', 'lemmatizer']]
  for pipe in customPipes:
    _ = nlp.remove_pipe(pipe)
  # re-add specified pipes
  for pipe in pipes:
    if pipe in ['pysbdSentenceBoundaries']:
      # nlp.add_pipe(pipe, before='parser')
      nlp.add_pipe(pipe, first=True)
    else:
      nlp.add_pipe(pipe)
  logger.info(f"Model: {nlp.meta['name']}, Language: {nlp.meta['lang']}")
  logger.info('Available pipelines:'+', '.join([pipe for (pipe,_) in nlp.pipeline]))
  return nlp

def printDepTree(doc, skipPunct=True):
  """
    Utility function to pretty print the dependency tree.
    @ In, doc, spacy.tokens.doc.Doc, the processed document using nlp pipelines
    @ In, skipPunct, bool, True if skip punctuactions
    @ Out, None
  """
  def printRecursive(root, indent, skipPunct):
    if not root.dep_ == 'punct' or not skipPunct:
      print(" "*indent + f"{root} [{root.pos_}, {root.dep_}]")
    for left in root.lefts:
      printRecursive(left, indent=indent+4, skipPunct=skipPunct)
    for right in root.rights:
      printRecursive(right, indent=indent+4, skipPunct=skipPunct)

  for sent in doc.sents: # iterate over all sentences in a doc
    printRecursive(sent.root, indent=0, skipPunct=skipPunct)


def plotDAG(edges, colors='k'):
  """
  @ In, edges, list of tuples, [(subj, conj), (..,..)] or [(subj, conj, {"color":"blue"}), (..,..)]
  @ In, colors, str or list, list of colors
  """
  g = nx.MultiDiGraph()
  g.add_edges_from(edges)
  nx.draw_networkx(g, edge_color=colors)
  ax = plot.gca()
  plt.axis("off")
  plot.show()


###############
# methods can be used for callback in "add" method
###############
def extendEnt(matcher, doc, i, matches):
  """
    Extend the doc's entity
    @ In, matcher, spacy.Matcher, the spacy matcher instance
    @ In, doc, spacy.tokens.doc.Doc, the processed document using nlp pipelines
    @ In, i, int, index of the current match (matches[i])
    @ In, matches, List[Tuple[int, int, int]], a list of (match_id, start, end) tuples, describing
      the matches. A match tuple describes a span doc[start:end]
  """
  id, start, end = matches[i]
  ent = Span(doc, start, end, label=id)
  logger.debug(ent.label_ + ' ' + ent.text)
  doc.ents = filter_spans(list(doc.ents) +[ent])
