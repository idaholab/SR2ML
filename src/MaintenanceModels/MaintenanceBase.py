# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
Created on April 20 2020

@author: mandd,wangc
"""

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

class MaintenanceBase:
  """
    Base class for maintenance models
  """

  @classmethod
  def getInputSpecification(cls):
    """
      Collects input specifications for this class.
      @ In, cls, class instance
      @ Out, inputSpecs, InputData, specs
    """
    inputSpecs = InputData.parameterInputFactory('MaintenanceModel')
    inputSpecs.addParam('type', param_type=InputTypes.StringType, descr='The maintenance model object identifier')
    return inputSpecs

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    self.type = self.__class__.__name__
    self.name = self.__class__.__name__
    # dictionary: keys all required input parameters, and values either user provided values or
    # variables determined by raven
    self._variableDict = {}
    # instance of maintenance model
    self._model = None
    # class of maintenance model
    self._modelClass = None
    # variable stores unavailability value
    self._unavail = None
    # variable stores availability value
    self._avail = None

  def handleInput(self, xmlNode):
    """
      Function to read the portion of the xml input that belongs to this specialized class
      and initialize some stuff based on the inputs got
      @ In, xmlNode, xml.etree.ElementTree.Element, Xml element node
      @ Out, None
    """
    paramInput = self.getInputSpecification()()
    paramInput.parseNode(xmlNode)
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
    pass

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
      Load the values of variables that are generated by RAVEN
      @ In, need, dict, the dict of parameters
      @ In, inputDict, dict, the dict of parameters that is provided from other sources
      @ Out, need, dict, the dict of parameters updated with variables
    """
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

  def getAvail(self):
    """
      get calculated availability value
      @ In, None
      @ Out, self._avail, float/numpy.array, the calculated availability value
    """
    return self._avail

  def getUnavail(self):
    """
      get calculated unavailability value
      @ In, None
      @ Out, self._unavail, float/numpy.array, the calculated unavailability value
    """
    return self._unavail

  def run(self,inputDict):
    """
      Method to calculate availability/unavailability related quantities
      @ In, None
      @ Out, None
    """
    self._avail   = self._availabilityFunction(inputDict)
    self._unavail = self._unavailabilityFunction(inputDict)

  @abc.abstractmethod
  def _availabilityFunction(self, inputDict):
    """
      Method to calculate availability value
      @ In, inputDict, dict, dictionary of inputs
      @ Out, availability, float, value of unavailability for the considered model
    """
    pass

  @abc.abstractmethod
  def _unavailabilityFunction(self, inputDict):
    """
      Method to calculate unavailability value
      @ In, inputDict, dict, dictionary of inputs
      @ Out, availability, float, value of availability for the considered model
    """
    pass
