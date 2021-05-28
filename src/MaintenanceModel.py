# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
Created on March 23 2020

@author: mandd, wangc
"""

#External Modules------------------------------------------------------------------------------------
import abc
import sys
import os
# import logging
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from SR2ML.src import MaintenanceModels
from utils import mathUtils as utils
from utils import InputData
from utils import InputTypes
from PluginBaseClasses.ExternalModelPluginBase import ExternalModelPluginBase
#Internal Modules End--------------------------------------------------------------------------------


class MaintenanceModel(ExternalModelPluginBase):
  """
     RAVEN ExternalModel for Maintenance Models
  """

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    ExternalModelPluginBase.__init__(self)
    self.type = self.__class__.__name__
    self.name = self.__class__.__name__
    self._dynamicHandling    = False
    self._model = None
    self._modelType = None
    self._modelXMLInput = None

  def _readMoreXML(self, container, xmlNode):
    """
      Method to read the portion of the XML that belongs to this plugin
      @ In, container, object, self-like object where all the variables can be stored
      @ In, xmlNode, xml.etree.ElementTree.Element, XML node that needs to be read
      @ Out, None
    """
    variables = xmlNode.find('variables')
    delimiter = ',' if ',' in variables.text else None
    container.variables = [var.strip() for var in variables.text.split(delimiter)]
    self._modelXMLInput = xmlNode.find('MaintenanceModel')
    self._modelType = self._modelXMLInput.get('type')
    if self._modelType is None:
      raise IOError("Required attribute 'type' for node 'MaintenanceModel' is not provided!")
    self._model = MaintenanceModels.returnInstance(self._modelType)

  def initialize(self, container, runInfoDict, inputFiles):
    """
      Method to initialize the Maintenance Model
      @ In, container, object, self-like object where all the variables can be stored
      @ In, runInfoDict, dict, dictionary containing all the RunInfo parameters (XML node <RunInfo>)
      @ In, inputFiles, list, list of input files (if any)
      @ Out, None
    """
    pass

  def run(self, container, inputDict):
    """
      This is a simple example of the run method in a plugin.
      @ In, container, object, self-like object where all the variables can be stored
      @ In, inputDict, dict, dictionary of inputs from RAVEN
      @ Out, None
    """
    self._model.handleInput(self._modelXMLInput)
    self._model.initialize(inputDict)
    self._model.run(inputDict)
    outputDict = {}
    outputDict['avail']   = self._model.getAvail()
    outputDict['unavail'] = self._model.getUnavail()
    for key, val in outputDict.items():
      setattr(container, key, val)
