# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
Created on Mar 29, 2021

@author: mandd
"""
# External Imports
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_kernels
# Internal Imports

class AAKR():
  """
  Class for the Auto Associative Kernel Regression
  
  Reference: P. Baraldi, F. Di Maio, P. Turati, E. Zio, "Robust signal reconstruction for condition monitoring 
             of industrial components via a modified AutoAssociative Kernel Regression method," Mechanical Systems 
             and Signal Processing, 60-61, pp. 29–44 (2015).
  """
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
    self.trainingData = trainData
  

  def fit(self, timeSeries, **Kwargs):
    """
      This method performs the regression of the provided timeSeries using the training data X^{obs_NC}
      @ In, timeSeries, pandas DataFrame, time series af actual recorded data
      @ In, Kwargs, dict, parameters for the chosen kernel
      @ Out, reconstructedData, pandas DataFrame, reconstructed timeSeries
      @ Out, residual, pandas DataFrame, residual of ||timeSeries - reconstructedData||
    """
    recData = {}
    residual = {}
    for var in timeSeries:
      if var in self.trainingData.keys():
        distanceMatrix = pairwise_kernels(X=timeSeries[var].values.reshape(-1, 1), Y=self.trainingData[var].values.reshape(-1, 1), metric=self.metric, **Kwargs)
        numerator = np.dot(distanceMatrix,self.trainingData[var].values.reshape(-1, 1)).T[0]
        denominator = distanceMatrix.sum(axis=1)
        recData[var] = np.divide(numerator,denominator)
        residual[var] = (timeSeries[var].values.reshape(-1, 1) - recData[var])[0]
      else:
        print('error')
    
    reconstructedData = pd.DataFrame(recData,  index=timeSeries.index.to_numpy())
    residual          = pd.DataFrame(residual, index=timeSeries.index.to_numpy())
    
    return reconstructedData, residual
  

import matplotlib.pyplot as plt

trainData = np.loadtxt('train.dat')
valData   = np.loadtxt('test_3.dat')
timeTrain = pd.date_range('1/1/2000', periods=4000, freq='H')
timeVal   = pd.date_range('1/1/2000', periods=1752, freq='d')

trainDF = pd.DataFrame({'var1':trainData[:,5]}, index=timeTrain)
trainDF.index.name = 'time'
valDF   = pd.DataFrame({'var1':valData[:,5]} , index=timeVal)
valDF.index.name = 'time'

aakr = AAKR(metric='rbf')
aakr.train(trainDF)
reconstructData, residual = aakr.fit(valDF, gamma=.5)

ax = valDF.plot(linewidth=0.4, label="measured")
reconstructData.plot(ax=ax,linewidth=0.4, label="reconstructed")

ax.legend(["measured", "reconstructed"])

plt.show()

    