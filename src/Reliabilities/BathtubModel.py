# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
Created on Feb. 7, 2020

@author: wangc, mandd
"""

#External Modules------------------------------------------------------------------------------------
import numpy as np
import numpy.ma as ma
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from utils import mathUtils as utils
from utils import InputData, InputTypes
from .ReliabilityBase import ReliabilityBase
#Internal Modules End--------------------------------------------------------------------------------

class BathtubModel(ReliabilityBase):
  """
    Bathtub reliability models from:
    B. S. Dhillon, "A Hazard Rate Model," IEEE Trans. Rel. 29, 150 (1979)
  """

  @classmethod
  def getInputSpecification(cls):
    """
      Collects input specifications for this class.
      @ In, None
      @ Out, inputSpecs, InputData, specs
    """
    inputSpecs = super(BathtubModel, cls).getInputSpecification()
    inputSpecs.description = r"""
      Bathtub reliability model, see reference "B. S. Dhillon, "A Hazard Rate Model," IEEE Trans. Rel. 29, 150 (1979)"
      """
    inputSpecs.addSub(InputData.parameterInputFactory('alpha', contentType=InputTypes.InterpretedListType,
      descr='Shape parameter'))
    inputSpecs.addSub(InputData.parameterInputFactory('beta', contentType=InputTypes.InterpretedListType,
      descr='Scale parameter'))
    inputSpecs.addSub(InputData.parameterInputFactory('c', contentType=InputTypes.InterpretedListType,
      descr='Weight parameter, 0<=c<=1'))
    inputSpecs.addSub(InputData.parameterInputFactory('theta', contentType=InputTypes.InterpretedListType,
      descr='Scale parameter'))
    inputSpecs.addSub(InputData.parameterInputFactory('rho', contentType=InputTypes.InterpretedListType,
      descr='Shape parameter'))
    inputSpecs.addSub(InputData.parameterInputFactory('Tm', contentType=InputTypes.InterpretedListType,
      descr='Mission time'))
    return inputSpecs

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    super().__init__()
    # shape parameter
    self._alpha = None
    # scale parameter
    self._beta = 1
    # c \in [0,1], weight parameter
    self._c = 1
    # scale parameter
    self._theta = 1
    # shape parameter
    self._rho = 1.

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
      elif child.getName().lower() == 'tm':
        self._tm = self.setVariable(child.value)
        self._variableDict['_tm'] = self._tm
      elif child.getName().lower() == 'theta':
        self._theta = self.setVariable(child.value)
        self._variableDict['_theta'] = self._theta
      elif child.getName().lower() == 'rho':
        self._rho = self.setVariable(child.value)
        self._variableDict['_rho'] = self._rho
      elif child.getName().lower() == 'c':
        self._c = self.setVariable(child.value)
        self._variableDict['_c'] = self._c

  def initialize(self, inputDict):
    """
      Method to initialize this plugin
      @ In, inputDict, dict, dictionary of inputs
      @ Out, None
    """
    super().initialize(inputDict)

  def _checkInputParams(self, needDict):
    """
      Method to check input parameters
      @ In, needDict, dict, dictionary of required parameters
      @ Out, None
    """
    super()._checkInputParams(needDict)
    if '_c' in needDict:
      if np.any(needDict['_c']<0) or np.any(needDict['_c']>1):
        raise IOError('Variable "{}" should be between [0,1], but "{}" is provided!'.format('_c',needDict['_c']))

  def _probabilityFunction(self):
    """
      Function to calculate probability
      @ In, None
      @ Out, pdf, float/numpy.array, the calculated pdf value(s)
    """
    alpha = self._alpha
    beta = self._beta
    c = self._c
    rho = self._rho
    theta = self._theta
    mask = self._tm < self._loc
    tm = ma.array(self._tm-self._loc, mask=mask)
    term1 = c * alpha * np.power(tm/beta, alpha - 1.)
    term2 = (1.-c) * rho * np.power(tm/theta, rho -1.) * np.exp(np.power(tm/theta, alpha))
    term3 = -c * beta * np.power(tm/beta, alpha)
    term4 = -(1.-c) *(np.exp(np.power(tm/theta, rho)) - 1.)
    pdf = (term1+term2) * np.exp(term3+term4)
    pdf = pdf.filled(0.)
    return pdf

  def _cumulativeFunction(self):
    """
      Function to calculate cumulative value
      @ In, None
      @ Out, cdf, float/numpy.array, the calculated cdf value(s)
    """
    alpha = self._alpha
    beta = self._beta
    c = self._c
    rho = self._rho
    theta = self._theta
    mask = self._tm < self._loc
    tm = ma.array(self._tm-self._loc, mask=mask)
    term3 = -c * beta * np.power(tm/beta, alpha)
    term4 = -(1.-c) *(np.exp(np.power(tm/theta, rho)) - 1.)
    cdf = 1. - np.exp(term3+term4)
    cdf = cdf.filled(0.)
    return cdf

  def _reliabilityFunction(self):
    """
      Function to calculate reliability value
      @ In, None
      @ Out, rdf, float/numpy.array, the calculated reliability value(s)
    """
    alpha = self._alpha
    beta = self._beta
    c = self._c
    rho = self._rho
    theta = self._theta
    mask = self._tm < self._loc
    tm = ma.array(self._tm-self._loc, mask=mask)
    term3 = -c * beta * np.power(tm/beta, alpha)
    term4 = -(1.-c) *(np.exp(np.power(tm/theta, rho)) - 1.)
    rdf = np.exp(term3+term4)
    rdf = rdf.filled(1.0)
    return rdf

  def _failureRateFunction(self):
    """
      Function to calculate hazard rate/failure rate
      @ In, None
      @ Out, frf, float/numpy.array, the calculated failure rate value(s)
    """
    alpha = self._alpha
    beta = self._beta
    c = self._c
    rho = self._rho
    theta = self._theta
    mask = self._tm < self._loc
    tm = ma.array(self._tm-self._loc, mask=mask)
    term1 = c * alpha * np.power(tm/beta, alpha - 1.)
    term2 = (1.-c) * rho * np.power(tm/theta, rho -1.) * np.exp(np.power(tm/theta, alpha))
    frf = term1 + term2
    frf = frf.filled(0.)
    return frf
