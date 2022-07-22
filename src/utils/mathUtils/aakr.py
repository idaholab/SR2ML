# Copyright 2020, Battelle Energy Alliance, LLC
"""
Created on Dec 20, 2020

@author: mandd
"""
# External Imports
import numpy as np
import pandas as pd
from sklearn.preprocessing    import StandardScaler
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics.pairwise import pairwise_kernels
# Internal Imports

class AAKR():
  
  def __init__(self, metric):
    """
      This method initializes the AAKR class
      @ In, metric, string, type of metric to be employed in the distance calculation
    """
    self.metric = metric
    
        
  def train(self, trainData):
    """
      This method load the training data into the AAKR class
      @ In, trainData, pandas DataFrame, dataframe containing the training dataset, i.e., X^{obs_NC}
    """
    if isinstance(trainData,pd.DataFrame):
      self.trainingData = trainData.to_numpy()
    else:
      self.trainingData = trainData
    
    # Z-Normalize data
    self.scaler = StandardScaler()
    self.scaler.fit(self.trainingData)
    self.trainingData = self.scaler.transform(self.trainingData)


  def fit(self, timeSeries, batchSize=None, **Kwargs):
    """
      This method performs partition the provided timeSeries in batches before performing the regression. 
      This is useful when training dataset and timeSeries are very big.
      @ In, timeSeries, pandas DataFrame, time series of actual recorded data
      @ In, Kwargs, dict, parameters for the chosen kernel
      @ In, batchSize, int, number of partitions of the timeSeries to perform the regression
      @ Out, reconstructedData, pandas DataFrame, reconstructed timeSeries
      @ Out, residual, pandas DataFrame, residual: timeSeries - reconstructedData
    """
    if batchSize is None:
      return self.reconstruct(timeSeries, **Kwargs)
    else:
      batches = np.array_split(timeSeries, batchSize)
      reconstructedDataList = [None] * batchSize
      residualDataList = [None] * batchSize
      counter = 0
      for batch in batches:
        print("serving batch: " + str(counter))
        reconstructedDataBatch, residualDataBatch = self.reconstruct(batch, **Kwargs)
        reconstructedDataList[counter] = reconstructedDataBatch
        residualDataList[counter] = residualDataBatch
        counter = counter + 1
      reconstructedData = pd.concat(reconstructedDataList)
      residualData = pd.concat(residualDataList)

      return reconstructedData, residualData
    
  def reconstruct(self, timeSeries, **Kwargs): 
    """
      This method performs the regression of the provided timeSeries for one single batch 
      using the training data X^{obs_NC}
      @ In, timeSeries, pandas DataFrame, time series of actual recorded data
      @ In, Kwargs, dict, parameters for the chosen kernel
      @ Out, reconstructedData, pandas DataFrame, reconstructed timeSeries
      @ Out, residual, pandas DataFrame, residual: timeSeries - reconstructedData
    """
    recData = {}
    resData = {}
    keys = timeSeries.keys()
    
    # Normalize actual data
    timeSeriesNorm = self.scaler.transform(timeSeries.to_numpy())

    distanceMatrix = pairwise_distances(X = self.trainingData, 
                                        Y = timeSeriesNorm, 
                                        metric = self.metric)
    
    weights = 1.0/np.sqrt(2.0*3.14159*Kwargs['bw']**2.0) * np.exp(-distanceMatrix**2.0/(2.0*Kwargs['bw']**2.0))
    weightSum = np.sum(weights,axis=0)
    weightsClean = np.where(sum==0, 1, weightSum)[:, None]

    recDataRaw = weights.T.dot(self.trainingData)
    recDataRaw = recDataRaw/weightsClean
    
    recDataRaw = self.scaler.inverse_transform(recDataRaw) 
    
    for index,key in enumerate(keys):
      recData[key] = recDataRaw[:,index]
      resData[key] = recDataRaw[:,index] - timeSeries.to_numpy()[:,index]
    
    reconstructedData = pd.DataFrame(recData,  index=timeSeries.index)
    residualData      = pd.DataFrame(resData,  index=timeSeries.index)
    
    return reconstructedData, residualData
