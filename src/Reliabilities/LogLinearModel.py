# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
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
import numpy.ma as ma
from scipy.integrate import quad
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from utils import mathUtils as utils
from utils import InputData, InputTypes
from .ReliabilityBase import ReliabilityBase
#Internal Modules End--------------------------------------------------------------------------------

class LogLinearModel(ReliabilityBase):
  """
    Log Linear reliability models: lambda(t) = exp(alpha + (t-t0)*(beta))
    This is called Cox-Lewis model.
  """

  @classmethod
  def getInputSpecification(cls):
    """
      Collects input specifications for this class.
      @ In, None
      @ Out, inputSpecs, InputData, specs
    """
    inputSpecs = super(LogLinearModel, cls).getInputSpecification()
    inputSpecs.description = r"""
      Log Linear reliability models: lambda(t) = exp(alpha + (t-t0)*(beta))
      This is called Cox-Lewis model.
      """
    inputSpecs.addSub(InputData.parameterInputFactory('alpha', contentType=InputTypes.InterpretedListType,
      descr='Shape parameter'))
    inputSpecs.addSub(InputData.parameterInputFactory('beta', contentType=InputTypes.InterpretedListType,
      descr='The inverse is the scale parameter'))
    inputSpecs.addSub(InputData.parameterInputFactory('Tm', contentType=InputTypes.InterpretedListType,
      descr='Mission time'))
    return inputSpecs

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    ReliabilityBase.__init__(self)
    # Shape parameter
    self._alpha = 1.
    # The inverse is the scale parameter
    self._beta = 1.

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
        self._variableDict['_alpha'] = self._alpha
      elif child.getName().lower() == 'beta':
        self._beta = self.setVariable(child.value)
        self._variableDict['_beta'] = self._beta
      elif child.getName().lower() == 'tm':
        self._tm = self.setVariable(child.value)
        self._variableDict['_tm'] = self._tm

  def initialize(self, inputDict):
    """
      Method to initialize this plugin
      @ In, inputDict, dict, dictionary of inputs
      @ Out, None
    """
    ReliabilityBase.initialize(self, inputDict)

  def _probabilityFunction(self):
    """
      Function to calculate probability
      @ In, None
      @ Out, pdf, float/numpy.array, the calculated pdf value(s)
    """
    alpha = self._alpha
    beta = self._beta
    mask = self._tm < self._loc
    tm = ma.array(self._tm-self._loc, mask=mask)
    lambdaT = _exponLaw(tm, alpha, beta)
    integration = self._integration(tm, alpha, beta)
    expT = np.exp(-integration)
    pdf = lambdaT * expT
    pdf = pdf.filled(0.)
    return pdf

  def _cumulativeFunction(self):
    """
      Function to calculate cumulative
      @ In, None
      @ Out, cdf, float/numpy.array, the calculated cdf value(s)
    """
    alpha = self._alpha
    beta = self._beta
    mask = self._tm < self._loc
    tm = ma.array(self._tm-self._loc, mask=mask)
    integration = self._integration(tm, alpha, beta)
    cdf = 1. - np.exp(-integration)
    return cdf

  def _reliabilityFunction(self):
    """
      Function to calculate reliability for given time or time series
      @ In, None
      @ Out, rdf, float/numpy.array, the calculated reliability value(s)
    """
    alpha = self._alpha
    beta = self._beta
    mask = self._tm < self._loc
    tm = ma.array(self._tm-self._loc, mask=mask)
    integration = self._integration(tm, alpha, beta)
    rdf = np.exp(-integration)
    return rdf

  def _failureRateFunction(self):
    """
      Function to calculate failure rate
      @ In, None
      @ Out, frf, float/numpy.array, the calculated failure rate value(s)
    """
    alpha = self._alpha
    beta = self._beta
    mask = self._tm < self._loc
    tm = ma.array(self._tm-self._loc, mask=mask)
    frf = _exponLaw(tm, alpha, beta)
    frf = frf.filled(0.)
    return frf

  @staticmethod
  def _integration(t, alpha, beta):
    """
      Power Law model
      @ In, t, numpy.array, the mission time
      @ In, alpha, numpy.array, coefficient of power law model
      @ In, beta, numpy.array, coefficient of power law model
      @ Out, integration, float/numpy.array, the integration of failure rate
    """
    integration = np.zeros(len(t))
    for index, tm in enumerate(t):
      if tm is not ma.masked:
        integration[index] = quad(_exponLaw, 0, tm, args=(alpha, beta))[0]
    return integration

def _exponLaw(t, alpha, beta):
  """
    Power Law model
    @ In, t, numpy.array, the mission time
    @ In, alpha, numpy.array, coefficient of power law model
    @ In, beta, numpy.array, coefficient of power law model
    @ Out, _exponLaw, float/numpy.array, the failure rate from exponential law model
  """
  return np.exp(alpha+beta*t)
