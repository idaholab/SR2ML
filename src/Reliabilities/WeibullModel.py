"""
Created on Feb. 7, 2020

@author: wangc, mandd
"""
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import abc
import sys
import os
import numpy as np
from scipy.stats import weibull_min as weibull
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from utils import mathUtils as utils
from utils import InputData, InputTypes
from .ScipyStatsModelBase import ScipyStatsModelBase
#Internal Modules End--------------------------------------------------------------------------------

class WeibullModel(ScipyStatsModelBase):
  """
    Weibull reliability models
  """

  @classmethod
  def getInputSpecification(cls):
    """
      Collects input specifications for this class.
      @ In, None
      @ Out, inputSpecs, InputData, specs
    """
    inputSpecs = super(WeibullModel, cls).getInputSpecification()
    inputSpecs.addSub(InputData.parameterInputFactory('alpha', contentType=InputTypes.InterpretedListType))
    inputSpecs.addSub(InputData.parameterInputFactory('beta', contentType=InputTypes.InterpretedListType))
    return inputSpecs

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    ScipyStatsModelBase.__init__(self)
    self._alpha = None
    self._beta = 1
    self._modelClass = weibull

  def _localHandleInput(self, paramInput):
    """
      Function to read the portion of the parsed xml input that belongs to this specialized class
      and initialize some stuff based on the inputs got
      @ In, paramInput, InputData.ParameterInput, the parsed xml input
      @ Out, None
    """
    ScipyStatsModelBase._localHandleInput(self, paramInput)
    for child in paramInput.subparts:
      if child.getName().lower() == 'alpha':
        self._alpha = self.setVariable(child.value)
        self._variableDict['_alpha'] = self._alpha
      elif child.getName().lower() == 'beta':
        self._beta = self.setVariable(child.value)
        self._variableDict['_beta'] = self._beta
      elif child.getName().lower() == 'tm':
        self._tm = self.setVariable(child.value)
        self._variableDict['_tm'] = self._tm

  def initialize(self, inputDict):
    """
      Method to initialize this plugin
      @ In, inputDict, dict, dictionary of inputs
      @ Out, None
    """
    ScipyStatsModelBase.initialize(self, inputDict)
    if self._alpha <= 0:
      raise IOError('alpha should be postive, provided value is {}'.format(self._alpha))
    self._model = self._modelClass(self._alpha, loc=self._loc, scale=self._beta)

  def _failureRateFunction(self):
    """
      Function to calculate probability
      @ In, None
      @ Out, ht, float/numpy.array, the calculated failure rate value(s)
    """
    # Numerical Solution
    # ht = self._probabilityFunction()/self._reliabilityFunction()
    # Analytic Solution
    ht = None
    if self._tm > self._loc:
      ht = self._alpha/self._beta * np.power((self._tm-self._loc)/self._beta,self._alpha -1.)
    else:
      ht = np.zeros(len(self._tm))
    return ht
