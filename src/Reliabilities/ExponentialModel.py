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
from .ReliabilityBase import ReliabilityBase
#Internal Modules End--------------------------------------------------------------------------------

class ExponentialModel(ReliabilityBase):
  """
    Exponential reliability models
  """

  @classmethod
  def getInputSpecification(cls):
    """
      Collects input specifications for this class.
      @ In, None
      @ Out, input_specs, InputData, specs
    """
    inputSpecs = super(ExponentialModel, cls).getInputSpecification()
    inputSpecs.addSub(InputData.parameterInputFactory('lambda', contentType=InputTypes.InterpretedListType))
    inputSpecs.addSub(InputData.parameterInputFactory('Tm', contentType=InputTypes.InterpretedListType))
    return inputSpecs

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    ReliabilityBase.__init__(self)
    self.type = self.__class__.__name__
    self.name = self.__class__.__name__
    # If True the metric needs to be able to handle dynamic data
    self._dynamicHandling    = False
    self._lambda = None
    self._tm = None
    self._expon = None

  def _localHandleInput(self, paramInput):
    """
      Function to read the portion of the xml input that belongs to this specialized class
      and initialize some stuff based on the inputs got
      @ In, paramInput, InputData.ParameterInput, the parsed xml input
      @ Out, None
    """
    for child in paramInput.subparts:
      if child.getName().lower() == 'lambda':
        self._lambda = self.setVariable(child.value)
        if utils.isAString(self._lambda):
          self._variableDict['_lambda'] = self._lambda
      elif child.getName().lower() == 'tm':
        self._tm = self.setVariable(child.value)
        if utils.isAString(self._tm):
          self._variableDict['_tm'] = self._tm
      elif child.getName() == 'outputVariables':
        self._outputList = child.value

  def _probabilityFunction(self):
    """
    """
    return self._expon.pdf(self._tm)

  def _cumulativeFunction(self):
    """
    """
    return self._expon.cdf(self._tm)

  def _reliabilityFunction(self):
    """
    """
    return self._expon.sf(self._tm)

  def _failureRateFunction(self):
    """
    """
    ht = self._probabilityFunction()/self._reliabilityFunction()
    return ht

  def initialize(self):
    """
      Method to initialize this plugin
      @ In, None
      @ Out, None
    """
    ReliabilityBase.initialize(self)
    self._expon = expon(loc=0, scale=1./self._lambda)
