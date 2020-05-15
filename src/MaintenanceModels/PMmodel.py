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

class PMModel(MaintenanceBase):
  """
    Basic Preventive Maintenance (PM) model
    Reference: 
      D. Kancev, M. Cepin 148
      Evaluation of risk and cost using an age-dependent unavailability modelling of test and maintenance for standby components
      Journal of Loss Prevention in the Process Industries 24 (2011) pp. 146-155.
  """

  @classmethod
  def getInputSpecification(cls):
    """
      Collects input specifications for this class.
      @ In, None
      @ Out, inputSpecs, InputData, specs
    """
    inputSpecs = super(PMModel, cls).getInputSpecification()
    inputSpecs.description = r"""
      Preventive maintenance reliability models
      """
    inputSpecs.addSub(InputData.parameterInputFactory('type', contentType=InputTypes.InterpretedListType, descr='Type of SSC considered: stand-by or operating'))
    inputSpecs.addSub(InputData.parameterInputFactory('rho',  contentType=InputTypes.InterpretedListType, descr='Failure probability on demand'))
    inputSpecs.addSub(InputData.parameterInputFactory('Tpm',  contentType=InputTypes.InterpretedListType, descr='Time required to perform PM activities'))
    inputSpecs.addSub(InputData.parameterInputFactory('Tr',   contentType=InputTypes.InterpretedListType, descr='Average repair time'))
    inputSpecs.addSub(InputData.parameterInputFactory('Tt',   contentType=InputTypes.InterpretedListType, descr='Average test duration'))
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
    self._rho  = None
    self._Tpm  = None
    self._Tr   = None
    self._Tt   = None

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
      if child.getName().lower() == 'rho':
        self._rho = self.setVariable(child.value)
        self._variableDict['_rho'] = self._rho  
      if child.getName().lower() == 'Tpm':
        self._Tpm = self.setVariable(child.value)
        self._variableDict['_Tpm'] = self._Tpm   
      if child.getName().lower() == 'Tr':
        self._Tr = self.setVariable(child.value)
        self._variableDict['_Tr'] = self._Tr    
      if child.getName().lower() == 'Tt':
        self._Tt = self.setVariable(child.value)
        self._variableDict['_Tt'] = self._Tt  

  def initialize(self, inputDict):
    """
      Method to initialize this plugin
      @ In, inputDict, dict, dictionary of inputs
      @ Out, None
    """
    MaintenanceBase.initialize(self, inputDict)

  def _availabilityFunction(self, inputDict):
    """
      Method to calculate component availability
      @ In, inputDict, dict, dictionary of inputs
      @ Out, availability, float, compoennt availability
    """
    if self._type == 'standby':
      availability = 1.0 - standbyModel(self._rho, inputDict['Ti'], self._Tr, self._Tt, self._Tpm, inputDict['Tm'], inputDict['lambda'])
    else:
      availability = 1.0 - operatingModel(self._Tr, self._Tpm, inputDict['Tm'], inputDict['lambda'])
    return availability

  def _unavailabilityFunction(self, inputDict):
    """
      Method to calculate component unavailability
      @ In, inputDict, dict, dictionary of inputs
      @ Out, availability, float, compoennt unavailability
    """
    if self._type == 'standby':
      unavailability = standbyModel(self._rho, inputDict['Ti'], self._Tr, self._Tt, self._Tpm, inputDict['Tm'], inputDict['lambda'])
    else:
      unavailability = operatingModel(self._tau, self._Tpm, inputDict['Tm'], inputDict['lambda'])
    return unavailability

  def standbyModel(rho, Ti, Tr, Tt, Tpm, Tm, lamb):
    """
      Method to calculate unavailability for a component in a stand-by configuration 
      @ In, rho, float, failure probability per demand
      @ In, Ti,  float, surveillance test interval
      @ In, Tr,  float, mean time to repair
      @ In, Tt,  float, test duration
      @ In, Tpm, float, mean time to perform preventive maintenance
      @ In, Tm,  float, preventive maintenance interval
      @ Out, unavailability, float, component unavailability
    """
    u = rho + 0.5*lamb*Ti + Tt/Ti + (rho+lamb*Ti)*Tr/Ti + Tpm/Tm
    return u

  def operatingModel(Tr, Tpm, Tm, lamb):
    """
      Method to calculate unavailability for a component which is continuosly operating 
      @ In, Tr,  float, mean time to repair
      @ In, Tpm, float, mean time to perform preventive maintenance
      @ In, Tm,  float, preventive maintenance interval
      @ Out, unavailability, float, component unavailability
    """
    u = lamb*Tr/(1.0+lamb*Tr) + Tpm/Tm
    return u