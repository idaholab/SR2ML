"""
Created on Jan. 30 2020

@author: wangc, mandd
"""
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import os
import abc
import sys
import numpy as np
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from utils import mathUtils as utils
from utils import InputData, InputTypes
#Internal Modules End--------------------------------------------------------------------------------

class ReliabilityBase:
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
    inputSpecs = InputData.parameterInputFactory('ReliabilityModel')
    inputSpecs.addParam('type', param_type=InputTypes.StringType,
        descr='The reliablity model object identifier')
    inputSpecs.addSub(InputData.parameterInputFactory('Td', contentType=InputTypes.InterpretedListType,
        descr='The time delay before the onset of failure'))
    return inputSpecs

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    self.type = self.__class__.__name__
    self.name = self.__class__.__name__
    # not used yet
    # True indicates reliablity model could accept time series input data, and returns
    # time-dependent reliablity data (Default True)
    self._dynamicHandling    = True
    # dictionary: keys all required input parameters, and values either user provided values or
    # variables determined by raven
    self._variableDict = {}
    # instance of reliability model
    self._model = None
    # class of reliablity model
    self._modelClass = None
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

  def handleInput(self, xmlNode):
    """
      Function to read the portion of the xml input that belongs to this specialized class
      and initialize some stuff based on the inputs got
      @ In, xmlNode, xml.etree.ElementTree.Element, Xml element node
      @ Out, None
    """
    paramInput = self.getInputSpecification()()
    paramInput.parseNode(xmlNode)
    td = paramInput.findFirst('Td')
    if td is not None:
      self._loc = self.setVariable(td.value)
      self._variableDict['_loc'] = self._loc
    self._localHandleInput(paramInput)

  @abc.abstractmethod
  def _localHandleInput(self, paramInput):
    """
      Function to process the parsed xml input
      @ In, paramInput, InputData.ParameterInput, the parsed xml input
      @ Out, None
    """

  def initialize(self, inputDict):
    """
      Method to initialize
      @ In, inputDict, dict, dictionary of inputs
      @ Out, None
    """
    needDict = self.getParams()
    needDict = self.loadVariables(needDict, inputDict)
    self.setParams(needDict)
    self._checkInputParams(needDict)

  def _checkInputParams(self, needDict):
    """
      Method to check input parameters
      @ In, needDict, dict, dictionary of required parameters
      @ Out, None
    """
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

  def isDynamic(self):
    """
      This method is utility function that tells if the reliability model is able to
      treat dynamic data on its own or not
      @ In, None
      @ Out, isDynamic, bool, True if the reliability model is able to treat dynamic data, False otherwise
    """
    return self._dynamicHandling

  def setVariable(self, value):
    """
      Set value if a float/int/list is provided in the node text, othersise treat the provided value as RAVEN variable
      @ In, value, str or float or list, the value of given variable
      @ Out, ret, str or float or numpy.array, the recasted value
    """
    ret = None
    # multi-entry or single-entry?
    if len(value) == 1:
      if not utils.isAFloatOrInt(value[0]):
        ret = value[0]
      else:
        ret = np.atleast_1d(value)
    else:
      # should be floats; InputData assures the entries are the same type already
      if not utils.isAFloatOrInt(value[0]):
        raise IOError('Multiple non-number entries are found, but require either a single variable name or multiple float entries: {}'.format(value))
      ret = np.asarray(value)
    return ret

  def loadVariables(self, need, inputDict):
    """
      Load the values of variables that is generated by RAVEN
      @ In, need, dict, the dict of parameters
      @ In, inputDict, dict, the dict of parameters that is provided from other sources
      @ Out, need, dict, the dict of parameters updated with variables
    """
    # load variable values from variables as needed
    for key, val in need.items():
      if utils.isAString(val):
        value = inputDict.get(val, None)
        if value is None:
          raise KeyError('Looking for variable "{}" to fill "{}" but not found among variables!'.format(val, key))
        need[key] = np.atleast_1d(value)
    return need

  def getParams(self):
    """
      Get the parameters
      @ In, None
      @ Out, self._variableList, dict, list of variables
    """
    return self._variableDict

  def setParams(self, paramDict):
    """
      Set the parameters from a given dictionary.
      @ In, paramDict, dict, settings
      @ Out, None
    """
    for key, val in paramDict.items():
      if key in self.__dict__.keys():
        setattr(self, key, val)
      else:
        print('WARNING: Variable "{}" is not defined in class "{}"!'.format(key, self.name))

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
