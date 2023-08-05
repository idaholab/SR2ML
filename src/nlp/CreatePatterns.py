import logging
import spacy
import pandas as pd
from nlp.nlp_utils import generatePatternList


class CreatePatterns(object):

  def __init__(self, filename, entLabel, entID=None, nlp=None, *args, **kwargs):
    """

    """
    self.filename = filename
    self.label = entLabel
    if entID is None:
      self.id = entLabel
    else:
      self.id = entID
    self.entities = self.readFile()
    if nlp is None:
      self.nlp = spacy.load("en_core_web_lg", exclude=[])
    else:
      self.nlp = nlp
    self.patterns = self.generatePatterns()


  def readFile(self):
    """
    """
    # assume one column without column name for the csv file
    entList = pd.read_csv(self.filename).values.ravel().tolist()
    return entList


  def generatePatterns(self):
    """
    """
    patterns = generatePatternList(self.entities, label=self.label, id=self.id, nlp=self.nlp, attr="LEMMA")
    return patterns

  def getPatterns(self):
    """
    """
    return self.patterns
