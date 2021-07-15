# Copyright 2017 Battelle Energy Alliance, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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

class basicEventScheduler(ExternalModelPluginBase):
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
      basic events
      @ In, inputDataset, dict, dictionary of inputs from RAVEN
      @ In, container, object, self-like object where all the variables can be stored
      @ Out, basicEventHistorySet, Dataset, xarray dataset which contains time series for each basic event
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

    print(container.__dict__)

