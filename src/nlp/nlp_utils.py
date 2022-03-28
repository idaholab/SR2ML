import pandas as pd
import spacy
from spacy.tokens import Span
import logging


logger = logging.getLogger(__name__)

###########################################################################

def displayNER(doc, includePunct=False):
  """
    Generate data frame for visualization of spaCy doc with custom attributes.
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
  """
  customPipes = [pipe for (pipe, _) in nlp.pipeline
                  if pipe not in ['tagger', 'parser', 'ner',
                                  'tok2vec', 'attribute_ruler', 'lemmatizer']]
  for pipe in customPipes:
    _ = nlp.remove_pipe(pipe)
  # re-add specified pipes
  for pipe in pipes:
    nlp.add_pipe(pipe)
  logger.info(f"Model: {nlp.meta['name']}, Language: {nlp.meta['lang']}")
  logger.info('Available pipelines:'+', '.join([pipe for (pipe,_) in nlp.pipeline]))
  return nlp

def printDepTree(doc, skipPunct=True):
  """
    Utility function to pretty print the dependency tree.
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
