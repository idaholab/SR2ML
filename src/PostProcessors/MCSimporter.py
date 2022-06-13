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
Created on Nov 1, 2019

@author: mandd
"""

#External Modules---------------------------------------------------------------
import pandas as pd
import numpy as np
#External Modules End-----------------------------------------------------------

#Internal Modules---------------------------------------------------------------
from ravenframework.PluginBaseClasses.PostProcessorPluginBase import PostProcessorPluginBase
from ravenframework.utils import InputData, InputTypes
#Internal Modules End-----------------------------------------------------------

class MCSImporter(PostProcessorPluginBase):
  """
    This is the base class of the PostProcessor that imports Minimal Cut Sets (MCSs) into RAVEN as a PointSet
  """
  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.printTag  = 'POSTPROCESSOR MCS IMPORTER'
    self.expand    = None  # option that controls the structure of the ET. If True, the tree is expanded so that
                           # all possible sequences are generated. Sequence label is maintained according to the
                           # original tree
    self.validDataType = ['PointSet'] # The list of accepted types of DataObject
    ## Currently, we have used both DataObject.addRealization and DataObject.load to
    ## collect the PostProcessor returned outputs. DataObject.addRealization is used to
    ## collect single realization, while DataObject.load is used to collect multiple realizations
    ## However, the DataObject.load can not be directly used to collect single realization
    self.outputMultipleRealizations = True
    self.fileFrom = None # the source of the provided MCSs file, i.e., saphire

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for the class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    inputSpecification = super().getInputSpecification()
    inputSpecification.addSub(InputData.parameterInputFactory("expand",       contentType=InputTypes.BoolType))
    inputSpecification.addSub(InputData.parameterInputFactory("BElistColumn", contentType=InputTypes.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("fileFrom", contentType=InputTypes.StringType))
    return inputSpecification

  def _handleInput(self, paramInput):
    """
      Method that handles PostProcessor parameter input block.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    super()._handleInput(paramInput)
    expand = paramInput.findFirst('expand')
    self.expand = expand.value
    fileFrom = paramInput.findFirst('fileFrom')
    if fileFrom is not None:
      self.fileFrom = fileFrom.value
    if self.expand:
      beListColumn = paramInput.findFirst('BElistColumn')
      self.beListColumn = beListColumn.value

  def run(self, inputIn):
    """
      This method executes the PostProcessor action.
      @ In,  inputIn, dict, dictionary contains the input data and input files, i.e.,
          {'Data':[DataObjects.asDataset('dict')], 'Files':[FileObject]}, only 'Files'
          will be used by this PostProcessor
      @ Out, mcsPointSet, dict, dictionary of outputs, i.e.,
          {'data':dict of realizations, 'dim':{}}
    """
    inputs = inputIn['Files']
    mcsFileFound = False
    beFileFound  = False

    for file in inputs:
      if file.getType()=="MCSlist":
        if mcsFileFound:
          self.raiseAnError(IOError, 'MCSImporterPostProcessor Post-Processor ' + self.name + ', Multiple files with type=MCSlist have been found')
        else:
          mcsListFile = file
          mcsFileFound = True
      if file.getType()=="BElist":
        if self.expand==False:
          self.raiseAnError(IOError, 'MCSImporterPostProcessor Post-Processor ' + self.name + ', A file with type=BElist has been found but expand is set to False')
        if beFileFound:
          self.raiseAnError(IOError, 'MCSImporterPostProcessor Post-Processor ' + self.name + ', Multiple files with type=BElist have been found')
        else:
          BElistFile = file
          beFileFound  = True

    if beFileFound==False and self.expand==True:
      self.raiseAnError(IOError, 'MCSImporterPostProcessor Post-Processor ' + self.name + ', Expand is set to False but no file with type=BElist has been found')

    self.mcsIDs, self.probability, self.mcsList, self.beList = mcsReader(mcsListFile.getFilename(), type=self.fileFrom)

    if self.expand:
      beData = pd.read_csv(BElistFile.getFilename())
      self.beList = beData[self.beListColumn].values.tolist()

    mcsPointSet = {}

    # MCS Input variables
    mcsPointSet['probability'] = self.probability
    mcsPointSet['MCS_ID']      = self.mcsIDs
    mcsPointSet['out']         = np.ones((self.probability.size))

    # MCS Output variables
    for be in self.beList:
      mcsPointSet[be]= np.zeros(self.probability.size)
    counter=0
    for mcs in self.mcsList:
      for be in mcs:
        mcsPointSet[be][counter] = 1.0
      counter = counter+1
    mcsPointSet = {'data': mcsPointSet, 'dims': {}}
    return mcsPointSet

def mcsReader(mcsListFile, type=None):
  """
    Function designed to read a file containing the set of MCSs and to save it as list of list
    @ In, mcsListFile, string, name of the file containing the set of MCSs
    @ In, type, string, the type of MCS file, it can be generated by Saphire, or User provided csv file
    @ Out, mcsIDs, np array, array containing the ID associated to each MCS
    @ Out, probability, np array, array containing the probability associated to each MCS
    @ Out, mcsList, list, list of MCS, each element is also a list containing the basic events of each MCS
    @ Out, beList, list, list of all basic events contained in the MCSs
  """
  mcsList=[]
  beList=set()
  probability = np.zeros((0))
  mcsIDs = np.zeros((0))

  # construct the list of MCSs and the list of BE
  with open(mcsListFile, 'r') as file:
    next(file) # skip header
    lines = file.read().splitlines()
    if type is None:
      for l in lines:
        elementsList = l.split(',')
        mcsIDs = np.append(mcsIDs,elementsList[0])
        elementsList.pop(0)
        probability=np.append(probability,float(elementsList[0]))
        elementsList.pop(0)
        mcsList.append(list(element.rstrip('\n') for element in elementsList))
        beList.update(elementsList)
    elif type.lower() == 'saphire':
      lines = lines[1:] # skip additional description line
      for l in lines:
        elementsList = l.split(',')
        # skip empty line
        if elementsList[0].strip() == '':
          continue
        mcsIDs = np.append(mcsIDs,elementsList[0])
        probability=np.append(probability, float(elementsList[1]))
        # skip column 3 which is the fraction to the total
        mcs = list(element.strip() for element in elementsList[3:])
        mcsList.append(mcs)
        beList.update(mcs)

  return mcsIDs, probability, mcsList, beList
