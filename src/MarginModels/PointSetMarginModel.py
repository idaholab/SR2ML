# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
Created on May 13 2021

@author: mandd,wangc
"""

#External Modules------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from utils import mathUtils as utils
from utils import InputData, InputTypes
from .MarginBase import MarginBase
from scipy.stats import wasserstein_distance
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
    inputSpecs.addSub(InputData.parameterInputFactory('failedDataFile',    contentType=InputTypes.InterpretedListType, descr='failed data file'))
    return inputSpecs


  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    MarginBase.__init__(self)

    self.failedDataFileID = None
    self.columnID = None
    self.actualDataID = None


  def _handleInput(self, paramInput):
    """
      Function to read the portion of the parsed xml input that belongs to this specialized class
      and initialize some stuff based on the inputs got
      @ In, paramInput, InputData.ParameterInput, the parsed xml input
      @ Out, None
    """
    super()._handleInput(paramInput)
    for child in paramInput.subparts:
      if child.getName().lower() == 'failedDataFileID':
        self.failedDataFileID = child.value
      if child.getName().lower() == 'columnID':
        self.columnID = child.value
      if child.getName().lower() == 'actualDataID':
        self.actualDataID = child.value

    self.failedData = pd.read_csv(self.failedDataFileID)[self.columnID]

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
    actualData = inputDict[self.actualDataID]

    zeroPoint = np.zeros(1)

    distMatrix = pairwise_distances(self.failedData.reshape(-1,1), actualData.reshape(-1,1), metric=customDist)
    distMatrix[distMatrix<0] = 0
    margin = np.mean(distMatrix)

    distMatrix2 = pairwise_distances(self.failedData.reshape(-1,1), zeroPoint.reshape(-1,1), metric=customDist)
    distMatrix2[distMatrix2<0] = 0
    margin2 = np.mean(distMatrix2)

    normalizedMargin = margin/margin2

    return normalizedMargin

def customDist(a,b):
  return a-b


