# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
Created on May 13 2021

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
from SR2ML.src.Bases import ModelBase
#Internal Modules End--------------------------------------------------------------------------------

class MarginBase(ModelBase):
  """
    Base class for margin models
  """

  @classmethod
  def getInputSpecification(cls):
    """
      Collects input specifications for this class.
      @ In, cls, class instance
      @ Out, inputSpecs, InputData, specs
    """
    inputSpecs = super().getInputSpecification()
    inputSpecs.addParam('type', param_type=InputTypes.StringType, descr='The margin model object identifier')
    return inputSpecs

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    super().__init__()
    # variable stores margin value
    self._margin = None

  def _checkInputParams(self, needDict):
    """
      Method to check input parameters
      @ In, needDict, dict, dictionary of required parameters
      @ Out, None
    """
    super()._checkInputParams(needDict)

  def run(self,inputDict):
    """
      Method to calculate margin values
      @ In, None
      @ Out, None
    """
    self._margin = self._marginFunction(inputDict)

  @abc.abstractmethod
  def _marginFunction(self, inputDict):
    """
      Method to calculate margin value
      @ In, inputDict, dict, dictionary of inputs
      @ Out, margin, float, value of margin for the considered model
    """
    pass

