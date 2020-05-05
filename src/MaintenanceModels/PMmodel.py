"""
Created on May 1 2020

@author: mandd,wangc
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
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from utils import mathUtils as utils
from utils import InputData, InputTypes
#Internal Modules End--------------------------------------------------------------------------------

class PMmodel(MaintenanceBase):
  """
    Basic Preventive Maintenance (PM) model
  """

  @classmethod
  def getInputSpecification(cls):
    """
      Collects input specifications for this class.
      @ In, None
      @ Out, inputSpecs, InputData, specs
    """
    inputSpecs = super(PMmodel, cls).getInputSpecification()
    inputSpecs.description = r"""
      Preventive maintenance reliability models
      """
    inputSpecs.addSub(InputData.parameterInputFactory('type', contentType=InputTypes.InterpretedListType, descr='Type of SSC considered: stand-by or operating'))
    return inputSpecs

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    MaintenanceBase.__init__(self)
    # Component type
    self._type = None
    self._outageTime = None
    self._rho = None
    self._tau = None

  def _localHandleInput(self, paramInput):
    """
      Function to read the portion of the parsed xml input that belongs to this specialized class
      and initialize some stuff based on the inputs got
      @ In, paramInput, InputData.ParameterInput, the parsed xml input
      @ Out, None
    """
    MaintenanceBase._localHandleInput(self, paramInput)
    for child in paramInput.subparts:
      if child.getName().lower() == 'type':
        self._type = self.setVariable(child.value)
        self._variableDict['_type'] = self._type
      if child.getName().lower() == 'outageTime':
        self._outageTime = self.setVariable(child.value)
        self._variableDict['_outageTime'] = self._outageTime  
      if child.getName().lower() == 'rho':
        self._rho = self.setVariable(child.value)
        self._variableDict['_rho'] = self._rho  
      if child.getName().lower() == 'tau':
        self._tau = self.setVariable(child.value)
        self._variableDict['_tau'] = self._tau    

  def initialize(self, inputDict):
    """
      Method to initialize this plugin
      @ In, inputDict, dict, dictionary of inputs
      @ Out, None
    """
    MaintenanceBase.initialize(self, inputDict)

  def _availabilityFunction(self, inputDict):
    if self._type = 'standby':
      availability = 1.0 - vaurioModelStandby(self._rho, self._outageTime, inputDict['T'], inputDict['lambda'])
    else:
      availability = 1.0 - vaurioModelOperating(self._tau, self._outageTime, inputDict['T'], inputDict['lambda'])
    return availability

  def _unavailabilityFunction(self, inputDict):
    if self._type = 'standby':
      unavailability = vaurioModelStandby(self._rho, self._outageTime, inputDict['T'], inputDict['lambda'])
    else:
      availability = 1.0 - vaurioModelOperating(self._tau, self._outageTime, inputDict['T'], inputDict['lambda'])
    return unavailability

  def vaurioModelStandby(rho, delta, T, lamb):
    u = rho+delta/T+0.5*lamb*T
    return u

  def vaurioModelOperating(tau, delta, T, lamb):
    rho = lamb*tau/(1.0+lamb*tau)
    u = rho+delta/T+0.5*lamb*T
    return u