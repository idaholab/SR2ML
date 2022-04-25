# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
Created on Feb. 7, 2020

@author: wangc, mandd
"""
#External Modules------------------------------------------------------------------------------------
import numpy as np
import numpy.ma as ma
from scipy.stats import weibull_min as weibull
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from ravenframework.utils import mathUtils as utils
from ravenframework.utils import InputData, InputTypes
from ...src.Reliabilities.ScipyStatsModelBase import ScipyStatsModelBase
#Internal Modules End--------------------------------------------------------------------------------

class WeibullModel(ScipyStatsModelBase):
  """
    Weibull reliability models
  """

  @classmethod
  def getInputSpecification(cls):
    """
      Collects input specifications for this class.
      @ In, None
      @ Out, inputSpecs, InputData, specs
    """
    inputSpecs = super(WeibullModel, cls).getInputSpecification()
    inputSpecs.description = r"""
      Weibull reliability models
      """
    inputSpecs.addSub(InputData.parameterInputFactory('alpha', contentType=InputTypes.InterpretedListType,
      descr='Shape parameter'))
    inputSpecs.addSub(InputData.parameterInputFactory('beta', contentType=InputTypes.InterpretedListType,
      descr='Scale parameter'))
    return inputSpecs

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    super().__init__()
    self._alpha = None # shape parameter
    self._beta = 1 # scale parameter
    self._modelClass = weibull # instance of weibull model from scipy.stats

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
        self.setVariable('_alpha', child.value)
      elif child.getName().lower() == 'beta':
        self.setVariable('_beta', child.value)
      elif child.getName().lower() == 'tm':
        self.setVariable('_tm', child.value)

  def initialize(self, inputDict):
    """
      Method to initialize this plugin
      @ In, inputDict, dict, dictionary of inputs
      @ Out, None
    """
    super().initialize(inputDict)
    self._model = self._modelClass(self._alpha, loc=self._loc, scale=self._beta)

  def _failureRateFunction(self):
    """
      Function to calculate failure rate function value
      @ In, None
      @ Out, ht, float/numpy.array, the calculated failure rate value(s)
    """
    # Numerical Solution
    # ht = self._probabilityFunction()/self._reliabilityFunction()
    # Analytic Solution
    # ht = self._alpha/self._beta * np.power((self._tm-self._loc)/self._beta,self._alpha -1.)
    mask = self._tm < self._loc
    dt = ma.array(self._tm-self._loc, mask=mask)
    ht = self._alpha/self._beta * np.power(dt/self._beta,self._alpha -1.)
    ht = ht.filled(0)
    return ht
