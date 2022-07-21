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
import itertools
import math
import xarray as xr
import copy
#External Modules End-----------------------------------------------------------

#Internal Modules---------------------------------------------------------------
from ravenframework.PluginBaseClasses.ExternalModelPluginBase import ExternalModelPluginBase
from ..src.PostProcessors.MCSimporter import mcsReader
#Internal Modules End-----------------------------------------------------------

class MCSSolver(ExternalModelPluginBase):
  """
    This class is designed to create a MCS solver model
  """
  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    ExternalModelPluginBase.__init__(self)
    self.solver = {}
    self.timeDepData   = None  # This variable contains the basic event temporal profiles as xr.Dataset
    self.topEventTerms = {}    # Dictionary containing, for each order, a list of terms containing the union of MCSs
    self.mcsList = None        # List containing all the MCSs; each MCS is a list of basic events
    self.solver['setType'] = None # Type of sets provided path sets (path) or cut sets (cut)

  def initialize(self, container, runInfoDict, inputFiles):
    """
      Method to initialize the MCS solver model
      @ In, container, object, self-like object where all the variables can be stored
      @ In, runInfoDict, dict, dictionary containing all the RunInfo parameters (XML node <RunInfo>)
      @ In, inputFiles, list, list of input files (if any)
      @ Out, None
    """

  def _readMoreXML(self, container, xmlNode):
    """
      Method to read the portion of the XML that belongs to the MCS solver model
      @ In, container, object, self-like object where all the variables can be stored
      @ In, xmlNode, xml.etree.ElementTree.Element, XML node that needs to be read
      @ Out, None
    """
    container.filename   = None # ID of the file containing the list of MCSs
    container.topEventID = None # ID of the Top Event (which is the union of the MCSs)
    container.fileFrom   = None # the source of the provided MCSs file, i.e., saphire
    container.timeID     = None # ID of the temporal variable
    container.mapping    = {}   # mapping between RAVEN variables and BEs contained in the MCSs
    container.invMapping = {}   # mapping between BEs contained in the MCSss and RAVEN variable

    # variables required for the TD calculation from PS
    container.tInitial   = None  # ID of the variable containing the initial time of the BEs
    container.tEnd       = None  # ID of the variable containing the final time of the BEs
    container.beId       = None  # ID of the variable containing the IDs of the BEs
    container.tdFromPS   = False # boolean variable which flags when TD calculation is generated from PS

    metricOrder = {'0':0, '1':1, '2':2, 'inf':np.inf}
    setTypes = ['path','cut']

    for child in xmlNode:
      if child.tag == 'topEventID':
        container.topEventID = child.text.strip()
      elif child.tag == 'fileFrom':
        container.fileFrom = child.text.strip().lower()
      elif child.tag == 'timeID':
        container.timeID = child.text.strip()
      elif child.tag == 'tInitial':
        container.tInitial = child.text.strip()
      elif child.tag == 'tEnd':
        container.tEnd = child.text.strip()
      elif child.tag == 'BE_ID':
        container.beId = child.text.strip()
      elif child.tag == 'solver':
        self.solver['type'] = child.get('type')
        for childChild in child:
          if childChild.tag == 'solverOrder':
            try:
              self.solver['solverOrder'] = int(childChild.text.strip())
            except:
              raise IOError("MCSSolver: xml node solverOrder must contain an integer value")
          elif childChild.tag == 'metric':
            metricValue = childChild.text.strip()
            if metricValue not in metricOrder.keys():
              raise IOError("MCSSolver: value in xml node metric is not allowed (0,1,2,inf)")
            self.solver['metric'] = metricOrder[metricValue]
          elif childChild.tag == 'setType':
            setType = childChild.text.strip()
            if setType not in setTypes:
              raise IOError("MCSSolver: set type in xml node setType is not allowed (cut,or path)")
            self.solver['setType'] = setType
      elif child.tag == 'variables':
        variables = [str(var.strip()) for var in child.text.split(",")]
      elif child.tag == 'map':
        container.mapping[child.get('var')]      = child.text.strip()
        container.invMapping[child.text.strip()] = child.get('var')
      else:
        raise IOError("MCSSolver: xml node " + str(child.tag) + " is not allowed")


  def createNewInput(self, container, inputs, samplerType, **kwargs):
    """
      This function has been added for this model in order to generate the terms in each order
      @ In, container, object, self-like object where all the variables can be stored
      @ In, inputs, list, the inputs (list) to start from to generate the new one
      @ In, samplerType, string, is the type of sampler that is calling to generate a new input
      @ In, **kwargs, dict,  is a dictionary that contains the information coming from the sampler,
           a mandatory key is the sampledVars'that contains a dictionary {'name variable':value}
      @ Out, kwargs, dict, dictionary which contains the information coming from the sampler
    """
    if len(inputs) > 2:
      raise IOError("MCSSolver: More than one file has been passed to the MCS solver")

    for input in inputs:
      if input.type == 'HistorySet':
        self.timeDepData = input.asDataset()
      elif input.type == 'PointSet':
        self.timeDepData = self.generateHistorySetFromSchedule(container,input.asDataset())
        container.tdFromPS = True
      else:
        mcsIDs, probability, self.mcsList, self.beList = mcsReader(input.getFilename(), type=container.fileFrom)

    # mcsList is supposed to be a list of lists
    # E.g., if the MCS are ABC CD and AE --> MCS1=['A','B','C'], MCS2=['D','C'], MCS3=['A','E']
    #       then mcsList = [MCS1,MCS2,MCS3] =
    #                    = [['A', 'B', 'C'], ['D', 'C'], ['A', 'E']]
    # Top event should be:   ABC + CD + AE +
    #                      - ABCD - ABCE - ACDE
    #                      + ABCDE

    if self.solver['type'] == 'probability':
      for order in range(1,self.solver['solverOrder']+1):
        self.topEventTerms[order]=[]
        terms = list(itertools.combinations(self.mcsList,order))
        # terms is a list of tuples
        # E.g., for order=2: [ (['A', 'B', 'C'], ['D', 'C']),
        #                      (['A', 'B', 'C'], ['A', 'E']),
        #                      (['D', 'C'], ['A', 'E']) ]

        basicEventCombined = list(set(itertools.chain.from_iterable(term)) for term in terms)
        self.topEventTerms[order] = basicEventCombined

    return kwargs

  def run(self, container, inputs):
    """
      This method performs the calculation of the TopEvent of the FT provided the status of its Basic Events.
      Depending on the nature of the problem it performs either a static of time dependent calculation.
      @ In, container, object, self-like object where all the variables can be stored
      @ In, inputs, dict, dictionary of inputs from RAVEN
      @ Out, None
    """

    if self.timeDepData is None:
      self.runStatic(container, inputs)
    else:
      self.runDynamic(container, inputs)

  def runStatic(self, container, inputs):
    """
      This method determines the probability or margin of the TopEvent of the FT provided the
      status of its Basic Events for a static calculation
      @ In, container, object, self-like object where all the variables can be stored
      @ In, inputs, dict, dictionary of inputs from RAVEN
      @ Out, None
    """
    inputForSolver = {}
    for key in container.invMapping.keys():
      inputForSolver[key] = inputs[container.invMapping[key]]

    if self.solver['type'] == 'probability':
      topEventValue = self.mcsSolverProbability(inputForSolver)
    else:
      topEventValue = self.mcsSolverMargin(inputForSolver)
      sensitivities = self.marginSensitivities(topEventValue, inputForSolver)
      for key in sensitivities:
        keyID = "sens_" + str(container.invMapping[key])
        container.__dict__[keyID] = sensitivities[key]

    container.__dict__[container.topEventID] = np.asarray(float(topEventValue))
    print(container.__dict__)


  def runDynamic(self, container, inputs):
    """
      This method determines the probability or margin of the TopEvent of the FT provided the
      status of its Basic Events for a time dependent calculation
      @ In, container, object, self-like object where all the variables can be stored
      @ In, inputs, dict, dictionary of inputs from RAVEN
      @ Out, None
    """
    topEventValue = np.zeros([self.timeDepData[container.timeID].shape[0]])
    if self.solver['type'] == 'margin':
      sensitivities = {}
      for key in inputs:
        sensitivities[key] = np.zeros([self.timeDepData[container.timeID].shape[0]])

    for index,t in enumerate(self.timeDepData[container.timeID]):
      inputForSolver = {}
      for key in container.invMapping.keys():
        if key in self.timeDepData.data_vars and self.timeDepData[key][0].values[index]>0:
          inputForSolver[key] = 1.0
        else:
          inputForSolver[key] = inputs[container.invMapping[key]]

      if self.solver['type'] == 'probability':
        topEventValue[index] = self.mcsSolverProbability(inputForSolver)
      else:
        topEventValue[index] = self.mcsSolverMargin(inputForSolver)

        sensValues = self.marginSensitivities(topEventValue, inputForSolver)
        for key in sensitivities:
          sensitivities[key][index] = sensValues[key]

    if container.tdFromPS:
      for key in container.invMapping.keys():
        container.__dict__[key] = self.timeDepData[key][0].values

    container.__dict__[container.timeID]     = self.timeDepData[container.timeID].values
    container.__dict__[container.topEventID] = topEventValue

    if self.solver['type'] == 'margin':
      for key in sensitivities:
        keyID = "sens_" + str(container.invMapping[key])
        container.__dict__[keyID] = sensitivities[key]


  def marginSensitivities(self, MsysBase, inputDict):
    """
      This method calculates the sensitivity (derivative based) of the top event margin vs. basic event margin
      @ In, inputs, inputDict, dictionary containing the probability  value of all basic events
      @ In, MsysBase, float, base top event margin
      @ Out, sensDict, dict, dictionary containing the sensitivity values of all basic events
    """
    sensDict={}
    epsilon = 0.01
    for key in inputDict:
      tempDict = copy.deepcopy(inputDict)
      tempDict[key] = tempDict[key] * (1.-epsilon)
      deltaMsys = self.mcsSolverMargin(tempDict)
      sensDict[key] = (MsysBase - deltaMsys) / (inputDict[key] - tempDict[key])

    return sensDict


  def mcsSolverProbability(self, inputDict):
    """
      This method determines the probability of the TopEvent of the FT provided the probability of its Basic Events
      @ In, inputs, inputDict, dictionary containing the probability  value of all basic events
      @ Out, teProbability, float, probability value of the top event
    """
    teProbability = 0.0
    multiplier = 1.0

    # perform probability calculation for each order level
    for order in range(1,self.solver['solverOrder']+1):
      orderProbability=0
      for term in self.topEventTerms[order]:
        # map the sampled values of the basic event probabilities to the MCS basic events ID
        termValues = list(map(inputDict.get,term))
        orderProbability = orderProbability + np.prod(termValues)
      teProbability = teProbability + multiplier * orderProbability
      multiplier = -1.0 * multiplier

    return float(teProbability)

  def mcsSolverMargin(self, inputDict):
    """
      This method determines the margin of the TopEvent of the FT provided the margin of its Basic Events
      for a time dependent calculation
      @ In, inputs, inputDict, dictionary containing the margin value of all basic events
      @ Out, teMargin, float, margin value of the top event
    """
    mcsMargins = np.zeros(len(self.mcsList))
    for index,mcs in enumerate(self.mcsList):
      termValues = list(map(inputDict.get,mcs))
      if self.solver['setType']=='cut':
        mcsMargins[index] = np.linalg.norm(termValues, ord=self.solver['metric'])
      else:
        mcsMargins[index] = np.amin(termValues)

    if self.solver['setType']=='cut':
      teMargin = np.amin(mcsMargins)
    else:
      teMargin = np.linalg.norm(mcsMargins, ord=self.solver['metric'])

    return teMargin

  def generateHistorySetFromSchedule(self, container, inputDataset):
    """
      This method generate an historySet from the a pointSet which contains initial and final time of the
      basic events
      @ In, inputDataset, dict, dictionary of inputs from RAVEN
      @ In, container, object, self-like object where all the variables can be stored
      @ Out, basicEventHistorySet, Dataset, xarray dataset which contains time series for each basic event
    """
    timeArray = np.concatenate([inputDataset[container.tInitial],inputDataset[container.tEnd]])
    timeArraySorted = np.sort(timeArray,axis=0)
    timeArrayCleaned = np.unique(timeArraySorted)

    keys = list(container.invMapping.keys())
    dataVars={}
    for key in keys:
      dataVars[key]=(['RAVEN_sample_ID',container.timeID],np.zeros((1,timeArrayCleaned.shape[0])))

    basicEventHistorySet = xr.Dataset(data_vars = dataVars,
                                      coords    = dict(time=timeArrayCleaned,
                                                       RAVEN_sample_ID=np.zeros(1)))

    for index,key in enumerate(inputDataset[container.beId].values):
      tin  = inputDataset[container.tInitial][index].values
      tend = inputDataset[container.tEnd][index].values
      indexes = np.where(np.logical_and(timeArrayCleaned>tin,timeArrayCleaned<=tend))
      basicEventHistorySet[key][0][indexes] = 1.0

    return basicEventHistorySet
