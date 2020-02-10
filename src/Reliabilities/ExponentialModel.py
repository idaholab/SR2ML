"""
Created on Jan. 30 2020

@author: wangc, mandd
"""
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import abc
import sys
import os
from scipy.stats import expon
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from utils import mathUtils as utils
from utils import InputData, InputTypes
from .ScipyStatsModelBase import ScipyStatsModelBase
#Internal Modules End--------------------------------------------------------------------------------

class ExponentialModel(ScipyStatsModelBase):
  """
    Exponential reliability models
  """

  @classmethod
  def getInputSpecification(cls):
    """
      Collects input specifications for this class.
      @ In, None
      @ Out, inputSpecs, InputData, specs
    """
    inputSpecs = super(ExponentialModel, cls).getInputSpecification()
    inputSpecs.addSub(InputData.parameterInputFactory('lambda', contentType=InputTypes.InterpretedListType))
    return inputSpecs

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    ScipyStatsModelBase.__init__(self)
    self._lambda = None
    self._modelClass = expon

  def _localHandleInput(self, paramInput):
    """
      Function to read the portion of the parsed xml input that belongs to this specialized class
      and initialize some stuff based on the inputs got
      @ In, paramInput, InputData.ParameterInput, the parsed xml input
      @ Out, None
    """
    ScipyStatsModelBase._localHandleInput(self, paramInput)
    for child in paramInput.subparts:
      if child.getName().lower() == 'lambda':
        self._lambda = self.setVariable(child.value)
        self._variableDict['_lambda'] = self._lambda
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
    self._model = self._modelClass(loc=self._loc, scale=1./self._lambda)

  def _failureRateFunction(self):
    """
      Function to calculate probability
      @ In, None
      @ Out, ht, float/numpy.array, the calculated failure rate value(s)
    """
    # both numerical and analytic failure rate can be used.
    # the parent class is using numerical method, and here we implemented the analytic method
    mask = self._tm > self._loc
    ht = np.zeros(len(self._tm))
    ht = ma.array(ht, mask=mask)
    ht = ht.filled(self._lambda[0])
    return ht
