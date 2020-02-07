"""
Created on Feb. 7, 2020

@author: wangc, mandd
"""
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import abc
import sys
import os
import numpy as np

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
    inputSpecs.addSub(InputData.parameterInputFactory('alpha', contentType=InputTypes.InterpretedListType))
    inputSpecs.addSub(InputData.parameterInputFactory('beta', contentType=InputTypes.InterpretedListType))
    inputSpecs.addSub(InputData.parameterInputFactory('c', contentType=InputTypes.InterpretedListType))
    inputSpecs.addSub(InputData.parameterInputFactory('theta', contentType=InputTypes.InterpretedListType))
    inputSpecs.addSub(InputData.parameterInputFactory('rho', contentType=InputTypes.InterpretedListType))
    inputSpecs.addSub(InputData.parameterInputFactory('Tm', contentType=InputTypes.InterpretedListType))
    return inputSpecs

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    ReliabilityBase.__init__(self)
    self._alpha = None
    self._beta = 1
    self._c = 1
    self._theta = 1
    self._rho = 0.5

  def _localHandleInput(self, paramInput):
    """
      Function to read the portion of the parsed xml input that belongs to this specialized class
      and initialize some stuff based on the inputs got
      @ In, paramInput, InputData.ParameterInput, the parsed xml input
      @ Out, None
    """
    ReliabilityBase._localHandleInput(self, paramInput)
    for child in paramInput.subparts:
      if child.getName().lower() == 'alpha':
        self._alpha = self.setVariable(child.value)
        if utils.isAString(self._alpha):
          self._variableDict['_alpha'] = self._alpha
      elif child.getName().lower() == 'beta':
        self._beta = self.setVariable(child.value)
        if utils.isAString(self._beta):
          self._variableDict['_beta'] = self._beta
      elif child.getName().lower() == 'tm':
        self._tm = self.setVariable(child.value)
        if utils.isAString(self._tm):
          self._variableDict['_tm'] = self._tm
      elif child.getName().lower() == 'theta':
        self._theta = self.setVariable(child.value)
        if utils.isAString(self._theta):
          self._variableDict['_theta'] = self._theta
      elif child.getName().lower() == 'rho':
        self._rho = self.setVariable(child.value)
        if utils.isAString(self._rho):
          self._variableDict['_rho'] = self._rho
      elif child.getName().lower() == 'c':
        self._c = self.setVariable(child.value)
        if utils.isAString(self._c):
          self._variableDict['_c'] = self._c

  def initialize(self, inputDict):
    """
      Method to initialize this plugin
      @ In, inputDict, dict, dictionary of inputs
      @ Out, None
    """
    ReliabilityBase.initialize(self, inputDict)
    if self._alpha <= 0:
      raise IOError('alpha should be postive, provided value is {}'.format(self._alpha))

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
    tm = self._tm - self._loc
    term1 = c * alpha * np.power(tm/beta, alpha - 1.)
    term2 = (1.-c) * rho * np.power(tm/theta, rho -1.) * np.exp(np.power(tm/theta, alpha))
    term3 = -c * beta * np.power(tm/beta, alpha)
    term4 = -(1.-c) *(np.exp(np.power(tm/theta, rho)) - 1.)
    pdf = (term1+term2) * np.exp(term3+term4)
    return pdf

  def _cumulativeFunction(self):
    """
      Function to calculate probability
      @ In, None
      @ Out, cdf, float/numpy.array, the calculated cdf value(s)
    """
    alpha = self._alpha
    beta = self._beta
    c = self._c
    rho = self._rho
    theta = self._theta
    tm = self._tm - self._loc
    term3 = -c * beta * np.power(tm/beta, alpha)
    term4 = -(1.-c) *(np.exp(np.power(tm/theta, rho)) - 1.)
    cdf = 1. - np.exp(term3+term4)
    return cdf

  def _reliabilityFunction(self):
    """
      Function to calculate probability
      @ In, None
      @ Out, rdf, float/numpy.array, the calculated reliability value(s)
    """
    alpha = self._alpha
    beta = self._beta
    c = self._c
    rho = self._rho
    theta = self._theta
    tm = self._tm - self._loc
    term3 = -c * beta * np.power(tm/beta, alpha)
    term4 = -(1.-c) *(np.exp(np.power(tm/theta, rho)) - 1.)
    rdf = np.exp(term3+term4)
    return rdf

  def _failureRateFunction(self):
    """
      Function to calculate probability
      @ In, None
      @ Out, frf, float/numpy.array, the calculated failure rate value(s)
    """
    alpha = self._alpha
    beta = self._beta
    c = self._c
    rho = self._rho
    theta = self._theta
    tm = self._tm - self._loc
    term1 = c * alpha * np.power(tm/beta, alpha - 1.)
    term2 = (1.-c) * rho * np.power(tm/theta, rho -1.) * np.exp(np.power(tm/theta, alpha))
    frf = term1 + term2
    return frf
