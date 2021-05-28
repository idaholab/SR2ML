# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
Created on Feb. 7, 2020

@author: wangc, mandd
"""
#External Modules------------------------------------------------------------------------------------
import numpy as np
import numpy.ma as ma
from scipy.stats import exponweib
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from utils import mathUtils as utils
from utils import InputData, InputTypes
from .ScipyStatsModelBase import ScipyStatsModelBase
#Internal Modules End--------------------------------------------------------------------------------

class ExponWeibullModel(ScipyStatsModelBase):
  """
    Exponential Weibull reliability models
  """

  @classmethod
  def getInputSpecification(cls):
    """
      Collects input specifications for this class.
      @ In, None
      @ Out, inputSpecs, InputData, specs
    """
    inputSpecs = super(ExponWeibullModel, cls).getInputSpecification()
    inputSpecs.description = r"""
      Exponential Weibull reliability models
      """
    inputSpecs.addSub(InputData.parameterInputFactory('alpha', contentType=InputTypes.InterpretedListType,
      descr='Shape parameter of exponentiation'))
    inputSpecs.addSub(InputData.parameterInputFactory('beta', contentType=InputTypes.InterpretedListType,
      descr='Scale parameter'))
    inputSpecs.addSub(InputData.parameterInputFactory('gamma', contentType=InputTypes.InterpretedListType,
      descr='Shape parameter of the non-exponentiation Weibull law'))

    return inputSpecs

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    super().__init__()
    # Shape parameter of exponentiation
    self._alpha = None
    # Scale parameter
    self._beta = 1
    # Shape parameter of the non-exponentiation Weibull law
    self._gamma = None
    # class of exponweib
    self._modelClass = exponweib

  def _handleInput(self, paramInput):
    """
      Function to read the portion of the parsed xml input that belongs to this specialized class
      and initialize some stuff based on the inputs got
      @ In, paramInput, InputData.ParameterInput, the parsed xml input
      @ Out, None
    """
    super()._handleInput(paramInput)
    for child in paramInput.subparts:
      if child.getName().lower() == 'alpha':
        self._alpha = self.setVariable(child.value)
        self._variableDict['_alpha'] = self._alpha
      elif child.getName().lower() == 'beta':
        self._beta = self.setVariable(child.value)
        self._variableDict['_beta'] = self._beta
      elif child.getName().lower() == 'gamma':
        self._gamma = self.setVariable(child.value)
        self._variableDict['_gamma'] = self._gamma
      elif child.getName().lower() == 'tm':
        self._tm = self.setVariable(child.value)
        self._variableDict['_tm'] = self._tm

  def initialize(self, inputDict):
    """
      Method to initialize this plugin
      @ In, inputDict, dict, dictionary of inputs
      @ Out, None
    """
    super().initialize(inputDict)
    self._model = self._modelClass(self._gamma, self._alpha, loc=self._loc, scale=self._beta)
