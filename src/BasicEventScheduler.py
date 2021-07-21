# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED

"""
Created on June 24, 2020

@author: mandd
"""

#External Modules---------------------------------------------------------------
import numpy as np
import xarray as xr
import pandas as pd
#External Modules End-----------------------------------------------------------

#Internal Modules---------------------------------------------------------------
from PluginBaseClasses.ExternalModelPluginBase import ExternalModelPluginBase
#Internal Modules End-----------------------------------------------------------

class BasicEventScheduler(ExternalModelPluginBase):
  """
    This class is designed to create a Maintenance Scheduler model
  """
  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    ExternalModelPluginBase.__init__(self)

  def initialize(self, container, runInfoDict, inputFiles):
    """
      Method to initialize the Basic Event Scheduler model
      @ In, container, object, self-like object where all the variables can be stored
      @ In, runInfoDict, dict, dictionary containing all the RunInfo parameters (XML node <RunInfo>)
      @ In, inputFiles, list, list of input files (if any)
      @ Out, None
    """

  def _readMoreXML(self, container, xmlNode):
    """
      Method to read the portion of the XML that belongs to the Basic Event Scheduler model
      @ In, container, object, self-like object where all the variables can be stored
      @ In, xmlNode, xml.etree.ElementTree.Element, XML node that needs to be read
      @ Out, None
    """
    container.basicEvents = {}

    for child in xmlNode:
      if child.tag == 'BE':
        container.basicEvents[child.text.strip()] = [child.get('tin'),child.get('tfin')]
      elif child.tag == 'timeID':
        container.timeID = child.text.strip()
      elif child.tag == 'variables':
        variables = [str(var.strip()) for var in child.text.split(",")]
      else:
        raise IOError("basicEventScheduler: xml node " + str(child.tag) + " is not allowed")

  def run(self, container, inputs):
    """
      This method generate an historySet from the a pointSet which contains initial and final time of the
      basic events. This method is generating the time series variable basicEventHistorySet which is passed
      to RAVEN through the container.__dict__ container
      @ In, inputs, dict, dictionary of inputs from RAVEN
      @ In, container, object, self-like object where all the variables can be stored
    """
    dataDict = {}
    for key in container.basicEvents.keys():
      dataDict[key] = []
      dataDict[key].append(inputs[container.basicEvents[key][0]][0])
      dataDict[key].append(inputs[container.basicEvents[key][1]][0])

    inputDataset = pd.DataFrame.from_dict(dataDict, orient='index',columns=['tin', 'tfin'])

    timeArray = np.concatenate([inputDataset['tin'],inputDataset['tfin']])
    timeArraySorted = np.sort(timeArray,axis=0)
    timeArrayCleaned = np.unique(timeArraySorted)

    keys = list(container.basicEvents.keys())
    dataVars={}
    for key in keys:
      dataVars[key]=(['RAVEN_sample_ID',container.timeID],np.zeros((1,timeArrayCleaned.shape[0])))

    basicEventHistorySet = xr.Dataset(data_vars = dataVars,
                                      coords    = dict(time=timeArrayCleaned,
                                      RAVEN_sample_ID=np.zeros(1)))

    for key in container.basicEvents.keys():
      tin  = inputs[container.basicEvents[key][0]][0]
      tend = inputs[container.basicEvents[key][1]][0]
      indexes = np.where(np.logical_and(timeArrayCleaned>tin,timeArrayCleaned<=tend))
      basicEventHistorySet[key][0][indexes] = 1.0

      container.__dict__[key] = basicEventHistorySet[key].values[0]

    container.__dict__[container.timeID] = timeArrayCleaned

