# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
Created on May 13 2021

@author: mandd,wangc
"""

#External Modules------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import copy
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from utils import InputData, InputTypes
from .MarginBase import MarginBase
from sklearn.metrics import pairwise_distances
#Internal Modules End--------------------------------------------------------------------------------


class PointSetMarginModel(MarginBase):

  @classmethod
  def getInputSpecification(cls):
    """
      Collects input specifications for this class.
      @ In, cls, class instance
      @ Out, inputSpecs, InputData, specs
    """
    inputSpecs = super(PointSetMarginModel, cls).getInputSpecification()
    inputSpecs.description = """ PointSet Margin Model """
    inputSpecs.addSub(InputData.parameterInputFactory('failedDataFileID', contentType=InputTypes.InterpretedListType, descr='failed data file'))
    
    inputSpecs.addSub(InputData.parameterInputFactory('marginID', contentType=InputTypes.InterpretedListType, descr='ID of the margin variable'))

    mapping = InputData.parameterInputFactory('map', contentType=InputTypes.InterpretedListType, descr='ID of the column of the csv containing failed data')
    mapping.addParam("var", InputTypes.StringType)
    inputSpecs.addSub(mapping)   
    
    return inputSpecs


  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    MarginBase.__init__(self)

    self.failedDataFileID = None  # name of the file containing the failed data 
    self.mapping = {}             # dictionary containing mapping between actual and failed data
    self.InvMapping = {}          # dictionary containing mapping between failed and actual data
    self.marginID = None          # ID of the calculated margin variable
    self.dimensionality = None    # dimensionality of the point set

  def _handleInput(self, paramInput):
    """
      Function to read the portion of the parsed xml input that belongs to this specialized class
      and initialize some stuff based on the inputs got
      @ In, paramInput, InputData.ParameterInput, the parsed xml input
      @ Out, None
    """
    super()._handleInput(paramInput)
    for child in paramInput.subparts:
      if child.getName() == 'failedDataFileID':
        self.failedDataFileID = self.setVariable(child.value)
      if child.getName() == 'marginID':
        self.marginID = self.setVariable(child.value)
      elif child.getName() == 'map':
        self.mapping[child.parameterValues.get('var')]      = self.setVariable(child.value)
        self.InvMapping[child.value[0]] = child.parameterValues.get('var')

    self.failedData = pd.read_csv(self.failedDataFileID)[self.mapping.keys()]
    
    self.dimensionality = len(self.mapping.keys())
    

  def initialize(self, inputDict):
    """
      Method to initialize this class
      @ In, inputDict, dict, dictionary of inputs
      @ Out, None
    """
    super().initialize(inputDict)


  def _marginFunction(self, inputDict):
    """
      Method to calculate margin value
      @ In, inputDict, dict, dictionary of inputs
      @ Out, margin, float, value of margin for the considered model
    """
    actualData = pd.DataFrame(inputDict)
    actualData = actualData.rename(columns=self.InvMapping)

    distMatrix = pairwise_distances(self.failedData.values, actualData.values, metric=customDist)
    distMatrix[distMatrix<0] = 0
    margin = np.mean(distMatrix)
    
    zeroPoint = copy.deepcopy(actualData)
    zeroPoint[:] = 0.0

    distMatrix2 = pairwise_distances(self.failedData.values, zeroPoint.values, metric=customDist)
    distMatrix2[distMatrix2<0] = 0
    margin2 = np.mean(distMatrix2)

    outputDict = {}
    outputDict[self.marginID] = margin/margin2

    return outputDict

def customDist(pointSet,refPoint):
  """
    Method to calculate distance between two vectors
    @ In, pointSet, np array, first numpy array
    @ In, refPoint, np array, second numpy array
    @ Out, distance, float, distance between vector a and b
  """
  distance = np.linalg.norm(pointSet - refPoint)
  return distance


