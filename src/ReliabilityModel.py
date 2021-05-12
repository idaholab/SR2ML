# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
Created on Jan. 30 2020

@author: wangc, mandd
"""
#External Modules------------------------------------------------------------------------------------
import abc
import sys
import os
# import logging
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from SR2ML.src import Reliabilities
from utils import mathUtils as utils
from utils import InputData
from utils import InputTypes
from PluginBaseClasses.ExternalModelPluginBase import ExternalModelPluginBase
#Internal Modules End--------------------------------------------------------------------------------

## option to use logging
# logging.basicConfig(format='%(asctime)s %(name)-20s %(levelname)-8s %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.DEBUG)
# logger = logging.getLogger()
# fh = logging.FileHandler(filename='logos.log', mode='w')
# fh.setLevel(logging.DEBUG)
# formatter = logging.Formatter('%(asctime)s %(name)-20s %(levelname)-8s %(message)s')
# fh.setFormatter(formatter)
# logger.addHandler(fh)

class ReliabilityModel(ExternalModelPluginBase):
  """
     RAVEN ExternalModel for reliability analysis
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
    self._modelXMLInput = xmlNode.find('ReliabilityModel')
    self._modelType = self._modelXMLInput.get('type')
    if self._modelType is None:
      raise IOError("Required attribute 'type' for node 'ReliabilityModel' is not provided!")
    self._model = Reliabilities.returnInstance(self._modelType)

  def isDynamic(self):
    """
      This method is utility function that tells if the metric is able to
      treat dynamic data on its own or not
      @ In, None
      @ Out, isDynamic, bool, True if the metric is able to treat dynamic data, False otherwise
    """
    return self._dynamicHandling

  def initialize(self, container, runInfoDict, inputFiles):
    """
      Method to initialize the Reliability Model
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
      @ Out,
    """
    self._model.handleInput(self._modelXMLInput)
    self._model.initialize(inputDict)
    self._model.run()
    outputDict = {}
    outputDict['cdf_F'] = self._model.getCDF()
    outputDict['pdf_f'] = self._model.getPDF()
    outputDict['rdf_R'] = self._model.getRDF()
    outputDict['frf_h'] = self._model.getFRF()
    for key, val in outputDict.items():
      setattr(container, key, val)
