# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
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
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from utils import mathUtils as utils
from utils import InputData, InputTypes
from .ReliabilityBase import ReliabilityBase
#Internal Modules End--------------------------------------------------------------------------------

class ScipyStatsModelBase(ReliabilityBase):
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
    inputSpecs = super(ScipyStatsModelBase, cls).getInputSpecification()
    inputSpecs.addSub(InputData.parameterInputFactory('Tm', contentType=InputTypes.InterpretedListType,
      descr='Mission time'))
    return inputSpecs

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    ReliabilityBase.__init__(self)
    # variable to store mission time
    self._tm = None

  def _localHandleInput(self, paramInput):
    """
      Function to read the portion of the parsed xml input that belongs to this specialized class
      and initialize some stuff based on the inputs got
      @ In, paramInput, InputData.ParameterInput, the parsed xml input
      @ Out, None
    """
    ReliabilityBase._localHandleInput(self, paramInput)

  def _checkInputParams(self, needDict):
    """
      Method to check input parameters
      @ In, needDict, dict, dictionary of required parameters
      @ Out, None
    """
    ReliabilityBase._checkInputParams(self, needDict)

  def _probabilityFunction(self):
    """
      Function to calculate probability
      @ In, None
      @ Out, _probabilityFunction, float/numpy.array, the calculated pdf value(s)
    """
    return self._model.pdf(self._tm)

  def _cumulativeFunction(self):
    """
      Function to calculate cumulative value
      @ In, None
      @ Out, _cumulativeFunction, float/numpy.array, the calculated cdf value(s)
    """
    return self._model.cdf(self._tm)

  def _reliabilityFunction(self):
    """
      Function to calculate reliability value
      @ In, None
      @ Out, _reliabilityFunction, float/numpy.array, the calculated reliability value(s)
    """
    return self._model.sf(self._tm)

  def _failureRateFunction(self):
    """
      Function to calculate failure rate
      @ In, None
      @ Out, ht, float/numpy.array, the calculated failure rate value(s)
    """
    return self._probabilityFunction()/self._reliabilityFunction()
