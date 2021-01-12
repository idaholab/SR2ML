# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
Created on May 1 2020

@author: mandd,wangc
"""

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
from .MaintenanceBase import MaintenanceBase
#Internal Modules End--------------------------------------------------------------------------------

class PMModel(MaintenanceBase):
  """
    Basic reference for Preventive Maintenance (PM) modeling
    Reference:
      D. Kancev, M. Cepin 148
      Evaluation of risk and cost using an age-dependent unavailability modelling of test and maintenance for standby components
      Journal of Loss Prevention in the Process Industries 24 (2011) pp. 146-155.
  """

  @classmethod
  def getInputSpecification(cls):
    """
      Collects input specifications for this class.
      @ In, cls, class instance
      @ Out, inputSpecs, InputData, specs
    """
    typeEnum = InputTypes.makeEnumType('PMType', 'PMTypeType', ['standby','operating'])
    inputSpecs = super(PMModel, cls).getInputSpecification()
    inputSpecs.description = r"""
      Preventive maintenance reliability models
      """
    inputSpecs.addSub(InputData.parameterInputFactory('type',  contentType=typeEnum,                       descr='Type of SSC considered: stand-by or operating'))
    inputSpecs.addSub(InputData.parameterInputFactory('rho',   contentType=InputTypes.InterpretedListType, descr='Failure probability on demand'))
    inputSpecs.addSub(InputData.parameterInputFactory('Tpm',   contentType=InputTypes.InterpretedListType, descr='Time required to perform PM activities'))
    inputSpecs.addSub(InputData.parameterInputFactory('Tr',    contentType=InputTypes.InterpretedListType, descr='Average repair time'))
    inputSpecs.addSub(InputData.parameterInputFactory('Tt',    contentType=InputTypes.InterpretedListType, descr='Average test duration'))
    inputSpecs.addSub(InputData.parameterInputFactory('Lambda',contentType=InputTypes.InterpretedListType, descr='Component failure rate'))
    inputSpecs.addSub(InputData.parameterInputFactory('Tm',    contentType=InputTypes.InterpretedListType, descr='Preventive maintenance interval'))
    inputSpecs.addSub(InputData.parameterInputFactory('Ti',    contentType=InputTypes.InterpretedListType, descr='Surveillance test interval'))
    return inputSpecs

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    MaintenanceBase.__init__(self)
    # Component type
    self.type = None
    self.rho  = None
    self.Tpm  = None
    self.Tr   = None
    self.Tt   = None
    self.Lambda = None
    self.Tm   = None
    self.Ti   = None

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
        self.type = child.value
      if child.getName().lower() == 'rho':
        self.rho = self.setVariable(child.value)
        self._variableDict['rho'] = self.rho
      if child.getName().lower() == 'tpm':
        self.Tpm = self.setVariable(child.value)
        self._variableDict['Tpm'] = self.Tpm
      if child.getName().lower() == 'tr':
        self.Tr = self.setVariable(child.value)
        self._variableDict['Tr'] = self.Tr
      if child.getName().lower() == 'tt':
        self.Tt = self.setVariable(child.value)
        self._variableDict['Tt'] = self.Tt
      if child.getName().lower() == 'lambda':
        self.Lambda = self.setVariable(child.value)
        self._variableDict['Lambda'] = self.Lambda
      if child.getName().lower() == 'tm':
        self.Tm = self.setVariable(child.value)
        self._variableDict['Tm'] = self.Tm
      if child.getName().lower() == 'ti':
        self.Ti = self.setVariable(child.value)
        self._variableDict['Ti'] = self.Ti

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
      @ Out, availability, float, component availability
    """
    if self.type == 'standby':
      availability = 1.0 - self.standbyModel(self.rho, self.Ti, self.Tr, self.Tt, self.Tpm, self.Tm, self.Lambda)
    else:
      availability = 1.0 - self.operatingModel(self.Tr, self.Tpm, self.Tm, self.Lambda)
    return availability

  def _unavailabilityFunction(self, inputDict):
    """
      Method to calculate component unavailability
      @ In, inputDict, dict, dictionary of inputs
      @ Out, availability, float, component unavailability
    """
    if self.type == 'standby':
      unavailability = self.standbyModel(self.rho, self.Ti, self.Tr, self.Tt, self.Tpm, self.Tm, self.Lambda)
    else:
      unavailability = self.operatingModel(self.Tr, self.Tpm, self.Tm, self.Lambda)
    return unavailability

  def standbyModel(self, rho, Ti, Tr, Tt, Tpm, Tm, lamb):
    """
      Method to calculate unavailability for a component in a stand-by configuration
      @ In, rho, float, failure probability per demand
      @ In, Ti,  float, surveillance test interval
      @ In, Tr,  float, mean time to repair
      @ In, Tt,  float, test duration
      @ In, Tpm, float, mean time to perform preventive maintenance
      @ In, Tm,  float, preventive maintenance interval
      @ In, lamb,float, component failure rate
      @ Out, unavailability, float, component unavailability
    """
    u = rho + 0.5*lamb*Ti + Tt/Ti + (rho+lamb*Ti)*Tr/Ti + Tpm/Tm
    return u

  def operatingModel(self, Tr, Tpm, Tm, lamb):
    """
      Method to calculate unavailability for a component which is continuosly operating
      @ In, Tr,  float, mean time to repair
      @ In, Tpm, float, mean time to perform preventive maintenance
      @ In, Tm,  float, preventive maintenance interval
      @ In, lamb,float, component failure rate
      @ Out, unavailability, float, component unavailability
    """
    u = lamb*Tr/(1.0+lamb*Tr) + Tpm/Tm
    return u
