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

  def __init__(self, text, checker='autocorrect'):
    """
      SpellChecker object constructor
      @ In, text, str, string of text that will be analyzed
      @ In, checker, str, optional, spelling corrector to use ('autocorrect' or 'ContextualSpellCheck')
      @ Out, None
    """
    # store text
    self.text = text
    self.checker = checker.lower()
    # get included and additional dictionary words and update speller dictionary
    if self.checker == 'autocorrect':
      self.speller = autocorrect.Speller()
      self.includedWords = []
      with open('../data/ac_additional_words.txt', 'r') as file:
        tmp = file.readlines()
      self.addedWords = list({x.replace('\n', '') for x in tmp})
      self.speller.nlp_data.update({x: 1000000 for x in self.addedWords})
    else:
      name = 'contextual spellcheck'
      self.nlp = spacy.load('en_core_web_sm')
      self.speller = ContextualSpellCheck(self.nlp, name)
      self.includedWords = list(self.speller.BertTokenizer.get_vocab().keys())
      with open('../data/csc_additional_words.txt', 'r') as file:
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

  def getMisspelledWords(self):
    """
      Returns a list of words that are misspelled according to the dictionary used
      @ In, None
      @ Out, misspelled, list, list of misspelled words
    """
    if self.checker == 'autocorrect':
      corrected = self.speller(self.text)
      original = re.findall(r'[^\s!,.?":;-]+', self.text)
      auto = re.findall(r'[^\s!,.?":;-]+', corrected)
      misspelled = list({w1 if w1 != w2 else None for w1, w2 in zip(original, auto)})
      if None in misspelled:
        misspelled.remove(None)
    else:
      doc = self.nlp(self.text)
      doc = self.speller(doc)
      misspelled = list({str(x) for x in doc._.suggestions_spellCheck.keys()})

    return misspelled

  def correct(self):
    """
      Performs automatic spelling correction and returns corrected text
      @ In, None
      @ Out, corrected, str, spelling corrected text
    """
    if self.checker == 'autocorrect':
      corrected = self.speller(self.text)
    else:
      doc = self.nlp(self.text)
      doc = self.speller(doc)
      corrected = doc._.outcome_spellCheck

    return corrected
