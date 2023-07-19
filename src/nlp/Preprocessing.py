# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
Created on October, 2022

@author: dgarrett622
"""
from cytoolz import functoolz
import re
import textacy.preprocessing as preprocessing
from numerizer import numerize
import spacy
from spacy.vocab import Vocab
from contextualSpellCheck.contextualSpellCheck import ContextualSpellCheck
import autocorrect
import itertools
from .similarity.simUtils import wordsSimilarity
from nltk.corpus import wordnet as wn
import os
import numpy as np
import pandas as pd

# list of available preprocessors in textacy.preprocessing.normalize
textacyNormalize = ['bullet_points',
                    'hyphenated_words',
                    'quotation_marks',
                    'repeating_chars',
                    'unicode',
                    'whitespace']
# list of available preprocessors in textacy.preprocessing.remove
textacyRemove = ['accents',
                 'brackets',
                 'html_tags',
                 'punctuation']
# list of available preprocessors in textacy.preprocessing.replace
textacyReplace = ['currency_symbols',
                  'emails',
                  'emojis',
                  'hashtags',
                  'numbers',
                  'phone_numbers',
                  'urls',
                  'user_handles']
# list of available preprocessors from numerizer
numerizer = ['numerize']

class Preprocessing(object):
  """
    NLP Preprocessing class
  """

  def __init__(self, preprocessorList, preprocessorOptions):
    """
      Preprocessing object constructor
      @ In, preprocessorList, list, list of preprocessor names as strings
      @ In, preprocessorOptions, dict, dictionary of dictionaries containing optional arguments for preprocessors
                                       top level key is name of preprocessor
      @ Out, None
    """
    self.functionList = [] # list of preprocessor functions
    self.preprocessorNames = textacyNormalize + textacyRemove + textacyReplace + numerizer

    # collect preprocessor functions in a list
    for name in preprocessorList:
      # strip out options for preprocessor
      if name in preprocessorOptions:
        options = preprocessorOptions[name]
      else:
        options = {}
      # build the function to do the preprocessing
      if name in textacyNormalize:
        self.createTextacyNormalizeFunction(name, options)
      elif name in textacyRemove:
        self.createTextacyRemoveFunction(name, options)
      elif name in textacyReplace:
        self.createTextacyReplaceFunction(name, options)
      elif name in numerizer:
        # create function to store in functionList
        self.functionList.append(lambda x: numerize(x, ignore=['a ', 'A', 'second']))
      else:
        print(f'{name} is ignored! \nAvailable preprocessors: {self.preprocessorNames}')

    # create the preprocessor pipeline (composition of functionList)
    self.pipeline = functoolz.compose_left(*self.functionList)

  def createTextacyNormalizeFunction(self, name, options):
    """
      Creates a function from textacy.preprocessing.normalize such that only argument is a string
      and adds it to the functionList
      @ In, name, str, name of the preprocessor
      @ In, options, dict, dictionary of preprocessor options
      @ Out, None
    """
    # check for optional arguments
    useChars, useMaxn, useForm = False, False, False
    # options for repeating_chars
    if 'chars' in options and isinstance(options['chars'], str):
      # if chars is not str, it gets ignored
      useChars = True
    if 'maxn' in options and isinstance(options['maxn'], int):
      # if maxn is not int, it gets ignored
      useMaxn = True
    # option for unicode
    if 'form' in options and isinstance(options['form'], str):
      # if form is not str, it gets ignored
      useForm = True

    # build function for the pipeline
    if useChars or useMaxn or useForm:
      # include optional arguments
      f = lambda x: getattr(preprocessing.normalize, name)(x, **options)
    else:
      # no options need to be included
      f = lambda x: getattr(preprocessing.normalize, name)(x)

    # add to the functionList
    self.functionList.append(f)

  def createTextacyRemoveFunction(self, name, options):
    """
      Creates a function from textacy.preprocessing.remove such that the only argument is a string
      and adds it to the functionList
      @ In, name, str, name of the preprocessor
      @ In, options, dict, dictionary of preprocessor options
      @ Out, None
    """
    # check for optional arguments
    useFast, useOnly = False, False
    # option for accents
    if 'fast' in options and isinstance(options['fast'], bool):
      # if fast is not bool, it gets ignored
      useFast = True
    # option for brackets and punctuation
    if 'only' in options and isinstance(options['only'], (str, list, tuple)):
      # if only is not str, list, or tuple, it gets ignored
      useOnly = True

    # build function for the pipeline
    if useFast or useOnly:
      # include optional arguments
      f = lambda x: getattr(preprocessing.remove, name)(x, **options)
    else:
      # no options need to be included
      f = lambda x: getattr(preprocessing.remove, name)(x)

    # add to the functionList
    self.functionList.append(f)

  def createTextacyReplaceFunction(self, name, options):
    """
      Creates a function from textacy.preprocessing.replace such that the only argument is a string
      and adds it to the functionList
      @ In, name, str, name of the preprocessor
      @ In, options, dict, dictionary of preprocessor options
      @ Out, None
    """
    # check for optional arguments
    useRepl = False
    if 'repl' in options and isinstance(options['repl'], str):
      # if repl is not str, it gets ignored
      useRepl = True

    # build function for the pipeline
    if useRepl:
      # include optional argument
      f = lambda x: getattr(preprocessing.replace, name)(x, **options)
    else:
      # no options need to be included
      f = lambda x: getattr(preprocessing.replace, name)(x)

    # add to the functionList
    self.functionList.append(f)

  def __call__(self, text):
    """
      Performs the preprocessing
      @ In, text, str, string of text to preprocess
      @ Out, processed, str, string of processed text
    """
    processed = self.pipeline(text)

    return processed

class SpellChecker(object):
  """
    Object to find misspelled words and automatically correct spelling
  """

  def __init__(self, checker='autocorrect'):
    """
      SpellChecker object constructor
      @ In, checker, str, optional, spelling corrector to use ('autocorrect' or 'ContextualSpellCheck')
      @ Out, None
    """
    self.checker = checker.lower()
    # get included and additional dictionary words and update speller dictionary
    if self.checker == 'autocorrect':
      self.speller = autocorrect.Speller()
      self.includedWords = []
      file2open = os.path.join(os.path.dirname(__file__) , 'data' , 'ac_additional_words.txt')
      with open(file2open, 'r') as file:
        tmp = file.readlines()
      self.addedWords = list({x.replace('\n', '') for x in tmp})
      self.speller.nlp_data.update({x: 1000000 for x in self.addedWords})
    else:
      name = 'contextual spellcheck'
      self.nlp = spacy.load('en_core_web_sm')
      self.speller = ContextualSpellCheck(self.nlp, name)
      self.includedWords = list(self.speller.BertTokenizer.get_vocab().keys())
      file2open = os.path.join(os.path.dirname(__file__) , 'data' , 'csc_additional_words.txt')
      with open(file2open, 'r') as file:
        tmp = file.readlines()
      self.addedWords = [x.replace('\n', '') for x in tmp]
      self.speller.vocab = Vocab(strings=self.includedWords+self.addedWords)

  def addWordsToDictionary(self, words):
    """
      Adds a list of words to the spell check dictionary
      @ In, words, list, list of words to add to the dictionary
      @ Out, None
    """
    if self.checker == 'autocorrect':
      self.speller.nlp_data.update({word: 1000000 for word in words})
    else:
      self.speller.vocab = Vocab(strings=self.includedWords+self.addedWords+words)

  def getMisspelledWords(self, text):
    """
      Returns a list of words that are misspelled according to the dictionary used
      @ In, None
      @ Out, misspelled, list, list of misspelled words
    """
    if self.checker == 'autocorrect':
      corrected = self.speller(text.lower())
      original = re.findall(r'[^\s!,.?":;-]+', text)
      auto = re.findall(r'[^\s!,.?":;-]+', corrected)
      misspelled = list({w1 if w1.lower() != w2.lower() else None for w1, w2 in zip(original, auto)})
      if None in misspelled:
        misspelled.remove(None)
    else:
      doc = self.nlp(text)
      doc = self.speller(doc)
      misspelled = list({str(x) for x in doc._.suggestions_spellCheck.keys()})

    return misspelled

  def correct(self, text):
    """
      Performs automatic spelling correction and returns corrected text
      @ In, None
      @ Out, corrected, str, spelling corrected text
    """
    if self.checker == 'autocorrect':
      corrected = self.speller(text)
    else:
      doc = self.nlp(text)
      doc = self.speller(doc)
      corrected = doc._.outcome_spellCheck

    return corrected

  def handleAbbreviations(self, abbrDatabase, text, type):
    """
      Performs automatic correction of abbreviations and returns corrected text
      This method relies on a database of abbreviations located at:
      src/nlp/data/abbreviations.xlsx
      This database contains the most common abbreviations collected from literarture and
      it provides for each abbreviation its corresponding full word(s); an abbreviation might
      have multple words associated. In such case the full word that makes more sense given the
      context is chosen (see findOptimalOption method)
      @ In, abbrDatabase, pandas dataframe, dataframe containing library of abbreviations
                                            and their correspoding full expression
      @ In, text, str, string of text that will be analyzed
      @ In, type, string, type of abbreviation method ('spellcheck','hard','mixed') that are employed
                          to determine which words are abbreviations that nned to be expanded
                          * spellcheck: in this case spellchecker is used to identify words that
                                        are not recognized
                          * hard: here we directly search for the abbreviations in the provided
                                  sentence
                          * mixed: here we perform first a "hard" search followed by a "spellcheck"
                                   search
      @ Out, options, list, list of corrected text options
    """
    abbreviationSet = set(abbrDatabase['Abbreviation'].values)
    if type == 'spellcheck':
      unknowns = self.getMisspelledWords(text)
    elif type == 'hard' or type=='mixed':
      unknowns = []
      splitSent = text.split()
      for word in splitSent:
        if word.lower() in abbreviationSet:
          unknowns.append(word)
      if type=='mixed':
        set1 = set(self.getMisspelledWords(text))
        set2 = set(unknowns)
        unknowns = list(set1.union(set2))

    corrections={}
    for word in unknowns:
      if word.lower() in abbrDatabase['Abbreviation'].values:
        locs = list(abbrDatabase['Abbreviation'][abbrDatabase['Abbreviation']==word.lower()].index.values)
        if locs:
          corrections[word] = abbrDatabase['Full'][locs].values.tolist()
        else:
          print(word)
      else:
        # Here we are addressing the fact that the abbreviation database will never be complete
        # Given an abbreviation that is not part of the abbreviation database, we are looking for a
        # a subset of abbreviations the abbreviation database that are close enough (and consider
        # them as possible candidates
        from difflib import SequenceMatcher
        corrections[word] = []
        abbreviationDS = abbrDatabase['Abbreviation'].values
        for index,abbr in enumerate(abbreviationDS):
          if SequenceMatcher(None, word, abbr).ratio()>0.8:
            corrections[word].append(abbrDatabase['Full'].values.tolist()[index])
      if not corrections[word]:
        corrections.pop(word)

    combinations = list(itertools.product(*list(corrections.values())))
    options = []
    for comb in combinations:
      corrected = text
      for index,key in enumerate(corrections.keys()):
        corrected = re.sub(r"\b%s\b" % str(key) , comb[index], corrected)
      options.append(corrected)

    if not options:
      return text
    else:
      bestOpt = self.findOptimalOption(options)
      return bestOpt

  def generateAbbrDict(self, abbrDatabase):
    """
      Generates a AbbrDict that can be used by handleAbbreviationsDict
      @ In, abbrDatabase, pandas dataframe, dataframe containing library of abbreviations
                                            and their correspoding full expression
      @ Out, abbrDict, dictionary, a abbreviations dictionary
    """
    abbrDict = {}
    #There may be a more efficient way to do the following
    for row in abbrDatabase.itertuples():
      abbrs = abbrDict.get(row.Abbreviation,[])
      abbrs.append(row.Full)
      abbrDict[row.Abbreviation] = abbrs
    return abbrDict

  def handleAbbreviationsDict(self, abbrDict, text, type):
    """
      Performs automatic correction of abbreviations and returns corrected text
      This method relies on a database of abbreviations located at:
      src/nlp/data/abbreviations.xlsx
      This database contains the most common abbreviations collected from literarture and
      it provides for each abbreviation its corresponding full word(s); an abbreviation might
      have multple words associated. In such case the full word that makes more sense given the
      context is chosen (see findOptimalOption method)
      @ In, abbrDict, dictionary, dictionary containing library of abbreviations
                                            and their correspoding full expression
      @ In, text, str, string of text that will be analyzed
      @ In, type, string, type of abbreviation method ('spellcheck','hard','mixed') that are employed
                          to determine which words are abbreviations that nned to be expanded
                          * spellcheck: in this case spellchecker is used to identify words that
                                        are not recognized
                          * hard: here we directly search for the abbreviations in the provided
                                  sentence
                          * mixed: here we perform first a "hard" search followed by a "spellcheck"
                                   search
      @ Out, options, list, list of corrected text options
    """
    if type == 'spellcheck':
      unknowns = self.getMisspelledWords(text)
    elif type == 'hard' or type=='mixed':
      unknowns = []
      splitSent = text.split()
      for word in splitSent:
        if word.lower() in abbrDict.keys():
          unknowns.append(word)
      if type=='mixed':
        set1 = set(self.getMisspelledWords(text))
        set2 = set(unknowns)
        unknowns = list(set1.union(set2))

    corrections={}
    for word in unknowns:
      if word.lower() in abbrDict.keys():
        if len(abbrDict[word.lower()]) > 0:
          corrections[word] = abbrDict[word.lower()]
      else:
        # Here we are addressing the fact that the abbreviation database will never be complete
        # Given an abbreviation that is not part of the abbreviation database, we are looking for a
        # a subset of abbreviations the abbreviation database that are close enough (and consider
        # them as possible candidates
        from difflib import SequenceMatcher
        corrections[word] = []
        abbreviationDS = list(abbrDict)
        for index,abbr in enumerate(abbreviationDS):
          val=0
          newVal = SequenceMatcher(None, word, abbr).ratio()
          if newVal>=0.75 and newVal>val:
            corrections[word] = abbrDict[abbr]
            val = newVal
      if not corrections[word]:
        corrections.pop(word)

    combinations = list(itertools.product(*list(corrections.values())))
    options = []
    for comb in combinations:
      corrected = text
      for index,key in enumerate(corrections.keys()):
        corrected = re.sub(r"\b%s\b" % str(key) , comb[index], corrected)
      options.append(corrected)

    if not options:
      return text
    else:
      bestOpt = self.findOptimalOption(options)
      return bestOpt

  def findOptimalOption(self,options):
    """
      Method to handle abbreviation with multiple meanings
      @ In, options, list, list of sentence options
      @ Out, optimalOpt, string, option from the provided options list that fits more the
                                 possible
    """
    nOpt = len(options)
    combScore = np.zeros(nOpt)
    for index,opt in enumerate(options):
      listOpt = opt.split()
      for i,word in enumerate(listOpt):
        for j in range(i+1,len(listOpt)):
          combScore[index] = combScore[index] + wordsSimilarity(word,listOpt[j])
    optIndex = np.argmax(combScore)
    optimalOpt = options[optIndex]
    return optimalOpt


class AbbrExpander(object):
  """
    Class to expand abbreviations
  """

  def __init__(self, abbreviationsFilename, checkerType='autocorrect', abbrType='mixed'):
    """
      Abbrviation expander constructor
      @ In, abbreviationsFilename, string, filename of abbreviations data
      @ Out, None
    """
    self.abbrType = abbrType
    self.checkerType = checkerType

    self.abbrList = pd.read_excel(abbreviationsFilename)
    self.preprocessorList = ['hyphenated_words',
                             'whitespace',
                             'numerize']
    self.preprocess = Preprocessing(self.preprocessorList, {})
    self.checker = SpellChecker(checker=self.checkerType)
    self.abbrDict = self.checker.generateAbbrDict(self.abbrList)


  def abbrProcess(self, text, splitToList=False):
    """
      Expands the abbreviations in text
      @ In, text, string, the text to expand
      @ Out, expandedText, string, the text with abbreviations expanded
    """
    text = self.preprocess(text)
    if not splitToList:
      expandedText = self.checker.handleAbbreviationsDict(self.abbrDict, text.lower(), type=self.abbrType)
    else:
      text = text.replace("\n", "")
      textList = [t.strip() for t in text.split('.')]
      expandedText = []
      for t in textList:
        cleanedText = self.checker.handleAbbreviationsDict(self.abbrDict, t.lower(), type=self.abbrType)
        expandedText.append(cleanedText)
      expandedText = '. '.join(expandedText)
    return expandedText
