# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
Created on Jan. 30 2020

@author: wangc, mandd
"""
#External Modules------------------------------------------------------------------------------------
import abc
import numpy as np
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from utils import mathUtils as utils
from utils import InputData, InputTypes
from SR2ML.src.Bases import ModelBase
#Internal Modules End--------------------------------------------------------------------------------

class ReliabilityBase(ModelBase):
  """
    Base class for reliability models
  """

  @classmethod
  def getInputSpecification(cls):
    """
      Collects input specifications for this class.
      @ In, None
      @ Out, inputSpecs, InputData, specs
    """
    inputSpecs = super().getInputSpecification()
    inputSpecs.addSub(InputData.parameterInputFactory('Td', contentType=InputTypes.InterpretedListType,
        descr='The time delay before the onset of failure'))
    return inputSpecs

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    super().__init__()
    self._dynamicHandling    = True
    # location parameter, i.e. time delay/shift
    self._loc = np.array([0])
    # variable stores cdf value(s)
    self._cdf = None
    # variable stores pdf value(s)
    self._pdf = None
    # variable stores reliability distribution function value(s)
    self._rdf = None
    # variable stores failure rate function value(s)
    self._frf = None

  def _handleInput(self, paramInput):
    """
      Function to process the parsed xml input
      @ In, paramInput, InputData.ParameterInput, the parsed xml input
      @ Out, None
    """
    super()._handleInput(paramInput)
    td = paramInput.findFirst('Td')
    if td is not None:
      self._loc = self.setVariable(td.value)
      self._variableDict['_loc'] = self._loc

  def _checkInputParams(self, needDict):
    """
      Method to check input parameters
      @ In, needDict, dict, dictionary of required parameters
      @ Out, None
    """
    super()._checkInputParams(needDict)
    for key, val in needDict.items():
      if key == '_tm' or key == '_loc':
        if np.any(val<0.):
          raise IOError('Variable "Tm" should be nonnegative, but provided value is "{}"!'.format(val))
      else:
       if key != '_c' and np.any(val<=0.):
        raise IOError('Variable "{}" should be postive, but provided value is "{}"!'.format(key.strip('_'),val))
       if len(val) > 1:
         raise IOError('Multiple values "{}" are provided for variable {}, this is not allowed now!'.format(val, key.strip('_')))
    if '_tm' in needDict and '_loc' in needDict:
      if len(needDict['_tm']) != len(needDict['_loc']) and len(needDict['_loc']) != 1:
        raise IOError('Variable "{}" and "{}" should have the same length, but "{}" != "{}"!'.format('Tm', 'Td', len(needDict['_tm']), len(needDict['_loc'])))
      # if needDict['_tm'] < needDict['_loc']:
      #   raise IOError('Variable "{}" with value "{}" is less than variable "{}" with value "{}", this is not allowed!'.format('_tm',needDict['_tm'],'_loc',needDict['_loc']))

  def getCDF(self):
    """
      get calculated cdf value
      @ In, None
      @ Out, self._cdf, float/numpy.array, the calculated cdf value(s)
    """
    return self._cdf

  def getPDF(self):
    """
      get calculated pdf value
      @ In, None
      @ Out, self._pdf, float/numpy.array, the calculated pdf value(s)
    """
    return self._pdf

  def getRDF(self):
    """
      get calculated reliability distribution function value
      @ In, None
      @ Out, self._rdf, float/numpy.array, the calculated reliablity value(s)
    """
    return self._rdf

  def getFRF(self):
    """
      get calculated failure rate function value
      @ In, None
      @ Out, self._frf, float/numpy.array, the calculated failure rate value(s)
    """
    return self._frf

  @abc.abstractmethod
  def _probabilityFunction(self):
    """
      Function to calculate probability
      @ In, None
      @ Out, _probabilityFunction, float/numpy.array, the calculated pdf value(s)
    """

  @abc.abstractmethod
  def _cumulativeFunction(self):
    """
      Function to calculate cumulative distribution function value
      @ In, None
      @ Out, _cumulativeFunction, float/numpy.array, the calculated cdf value(s)
    """

  @abc.abstractmethod
  def _reliabilityFunction(self):
    """
      Function to calculate reliability distribution function value
      @ In, None
      @ Out, _reliabilityFunction, float/numpy.array, the calculated reliability value(s)
    """

  @abc.abstractmethod
  def _failureRateFunction(self):
    """
      Function to calculate failure rate function value
      @ In, None
      @ Out, _failureRateFunction, float/numpy.array, the calculated failure rate value(s)
    """

  def run(self):
    """
      Method to calculate reliability related quantities
      @ In, None
      @ Out, None
    """
    self._pdf = self._probabilityFunction()
    self._cdf = self._cumulativeFunction()
    self._rdf = self._reliabilityFunction()
    self._frf = self._failureRateFunction()
