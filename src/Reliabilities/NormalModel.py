# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
Created on Feb. 10, 2020

@author: wangc, mandd
"""

#External Modules------------------------------------------------------------------------------------
from scipy.stats import norm
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from ravenframework.utils import mathUtils as utils
from ravenframework.utils import InputData, InputTypes
from ...src.Reliabilities.ScipyStatsModelBase import ScipyStatsModelBase
#Internal Modules End--------------------------------------------------------------------------------

class NormalModel(ScipyStatsModelBase):
  """
    Normal reliability models
  """

  @classmethod
  def getInputSpecification(cls):
    """
      Collects input specifications for this class.
      @ In, None
      @ Out, inputSpecs, InputData, specs
    """
    inputSpecs = super(NormalModel, cls).getInputSpecification()
    inputSpecs.description = r"""
      Normal reliability models
      """
    inputSpecs.addSub(InputData.parameterInputFactory('sigma', contentType=InputTypes.InterpretedListType,
      descr='Shape parameter'))
    return inputSpecs

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    super().__init__()
    # Shape parameter
    self._sigma = 1
    # class of norm from scipy.stats
    self._modelClass = norm

  def _handleInput(self, paramInput):
    """
      Function to read the portion of the parsed xml input that belongs to this specialized class
      and initialize some stuff based on the inputs got
      @ In, paramInput, InputData.ParameterInput, the parsed xml input
      @ Out, None
    """
    super()._handleInput(paramInput)
    for child in paramInput.subparts:
      if child.getName().lower() == 'sigma':
        self.setVariable('_sigma', child.value)
      elif child.getName().lower() == 'tm':
        self.setVariable('_tm', child.value)

  def initialize(self, inputDict):
    """
      Method to initialize this plugin
      @ In, inputDict, dict, dictionary of inputs
      @ Out, None
    """
    super().initialize(inputDict)
    self._model = self._modelClass(loc=self._loc, scale=self._sigma)
