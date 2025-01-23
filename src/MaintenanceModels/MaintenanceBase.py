# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
Created on April 20 2020

@author: mandd,wangc
"""

#External Modules------------------------------------------------------------------------------------
import abc
import numpy as np
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from ravenframework.utils import mathUtils as utils
from ravenframework.utils import InputData, InputTypes
from ...src.Bases import ModelBase
#Internal Modules End--------------------------------------------------------------------------------

class MaintenanceBase(ModelBase):
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
    inputSpecs = super().getInputSpecification()
    return inputSpecs

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    super().__init__()
    self._unavail = None
    # variable stores availability value
    self._avail = None

  def _checkInputParams(self, needDict):
    """
      Method to check input parameters
      @ In, needDict, dict, dictionary of required parameters
      @ Out, None
    """
    super()._checkInputParams(needDict)

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
