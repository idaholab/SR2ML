# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
Created on Jan. 30 2020

@author: wangc, mandd
"""

#External Modules------------------------------------------------------------------------------------
import numpy as np
import numpy.ma as ma
from scipy.stats import expon
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from utils import mathUtils as utils
from utils import InputData, InputTypes
from .ScipyStatsModelBase import ScipyStatsModelBase
#Internal Modules End--------------------------------------------------------------------------------

class ExponentialModel(ScipyStatsModelBase):
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
    inputSpecs = super(ExponentialModel, cls).getInputSpecification()
    inputSpecs.description = r"""
      Exponential reliability models
      """
    inputSpecs.addSub(InputData.parameterInputFactory('lambda', contentType=InputTypes.InterpretedListType,
      descr='The mean failure rate'))
    return inputSpecs

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    super().__init__()
    # mean failure rate
    self._lambda = None
    # class of expon from scipy.stats
    self._modelClass = expon

  def _handleInput(self, paramInput):
    """
      Function to read the portion of the parsed xml input that belongs to this specialized class
      and initialize some stuff based on the inputs got
      @ In, paramInput, InputData.ParameterInput, the parsed xml input
      @ Out, None
    """
    super()._handleInput(paramInput)
    for child in paramInput.subparts:
      if child.getName().lower() == 'lambda':
        self.setVariable('_lambda', child.value)
      elif child.getName().lower() == 'tm':
        self.setVariable('_tm', child.value)

  def initialize(self, inputDict):
    """
      Method to initialize this plugin
      @ In, inputDict, dict, dictionary of inputs
      @ Out, None
    """
    super().initialize(inputDict)
    self._model = self._modelClass(loc=self._loc, scale=1./self._lambda)

  def _failureRateFunction(self):
    """
      Function to calculate failure rate
      @ In, None
      @ Out, ht, float/numpy.array, the calculated failure rate value(s)
    """
    # both numerical and analytic failure rate can be used.
    # the parent class is using numerical method, and here we implemented the analytic method
    mask = self._tm > self._loc
    ht = np.zeros(len(self._tm))
    ht = ma.array(ht, mask=mask)
    ht = ht.filled(self._lambda[0])
    return ht
