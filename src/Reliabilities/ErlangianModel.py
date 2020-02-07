"""
Created on Feb. 6, 2020

@author: wangc, mandd
"""
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import abc
import sys
import os
from scipy.stats import erlang
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from utils import mathUtils as utils
from utils import InputData, InputTypes
from .ExponentialModel import ExponentialModel
#Internal Modules End--------------------------------------------------------------------------------

class ErlangianModel(ExponentialModel):
  """
    Erlangian reliability models
  """

  @classmethod
  def getInputSpecification(cls):
    """
      Collects input specifications for this class.
      @ In, None
      @ Out, inputSpecs, InputData, specs
    """
    inputSpecs = super(ErlangianModel, cls).getInputSpecification()
    inputSpecs.addSub(InputData.parameterInputFactory('k', contentType=InputTypes.InterpretedListType))
    return inputSpecs

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    ExponentialModel.__init__(self)
    # If True the metric needs to be able to handle dynamic data
    self._k = 1
    self._modelClass = erlang

  def _localHandleInput(self, paramInput):
    """
      Function to read the portion of the parsed xml input that belongs to this specialized class
      and initialize some stuff based on the inputs got
      @ In, paramInput, InputData.ParameterInput, the parsed xml input
      @ Out, None
    """
    ExponentialModel._localHandleInput(self, paramInput)
    nodeK = paramInput.findFirst('k')
    if nodeK is not None:
      self._k = self.setVariable(nodeK.value)
      self._variableDict['_k'] = self._k

  def initialize(self):
    """
      Method to initialize this plugin
      @ In, None
      @ Out, None
    """
    if self._lambda <= 0:
      raise IOError('lambda should be postive, provided value is {}'.format(self._lambda))
    self._model = self._modelClass(self._k, loc=self._loc, scale=1./self._lambda)

  def _failureRateFunction(self):
    """
      Function to calculate probability
      @ In, None
      @ Out, ht, float/numpy.array, the calculated failure rate value(s)
    """
    return self._probabilityFunction()/self._reliabilityFunction()
