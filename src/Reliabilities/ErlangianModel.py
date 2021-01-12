# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
Created on Feb. 6, 2020

@author: wangc, mandd
"""
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import abc
import sys
import os
from scipy.stats import erlang
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from utils import mathUtils as utils
from utils import InputData, InputTypes
from .ScipyStatsModelBase import ScipyStatsModelBase
#Internal Modules End--------------------------------------------------------------------------------

class ErlangianModel(ScipyStatsModelBase):
  """
    Erlangian (or homogeneous poisson process) reliability models
  """

  @classmethod
  def getInputSpecification(cls):
    """
      Collects input specifications for this class.
      @ In, None
      @ Out, inputSpecs, InputData, specs
    """
    inputSpecs = super(ErlangianModel, cls).getInputSpecification()
    inputSpecs.description = r"""
      Erlangian (or homogeneous poisson process) reliability models
      """
    inputSpecs.addSub(InputData.parameterInputFactory('lambda', contentType=InputTypes.InterpretedListType,
      descr='Mean failure rate for each event'))
    inputSpecs.addSub(InputData.parameterInputFactory('k', contentType=InputTypes.InterpretedListType,
      descr='Shape parameter (integer). Note that this restriction is not enforced.'))
    return inputSpecs

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    ScipyStatsModelBase.__init__(self)
    # Mean failure rate for each event
    self._lambda = None
    # Shape parameter, integer but not enforced
    self._k = 1
    # class of reliability model from scipy.stats
    self._modelClass = erlang

  def _localHandleInput(self, paramInput):
    """
      Function to read the portion of the parsed xml input that belongs to this specialized class
      and initialize some stuff based on the inputs got
      @ In, paramInput, InputData.ParameterInput, the parsed xml input
      @ Out, None
    """
    ScipyStatsModelBase._localHandleInput(self, paramInput)
    for child in paramInput.subparts:
      if child.getName().lower() == 'lambda':
        self._lambda = self.setVariable(child.value)
        self._variableDict['_lambda'] = self._lambda
      elif child.getName().lower() == 'tm':
        self._tm = self.setVariable(child.value)
        self._variableDict['_tm'] = self._tm
      elif child.getName() == 'k':
        self._k = self.setVariable(child.value)
        self._variableDict['_k'] = self._k

  def initialize(self, inputDict):
    """
      Method to initialize this plugin
      @ In, inputDict, dict, dictionary of inputs
      @ Out, None
    """
    ScipyStatsModelBase.initialize(self, inputDict)
    self._model = self._modelClass(self._k, loc=self._loc, scale=1./self._lambda)
