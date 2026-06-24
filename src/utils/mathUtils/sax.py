# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
Created on Mar 17, 2021

@author: mandd
"""
# External Imports
import numpy as np
import pandas as pd
from scipy.stats import norm
import string
# Internal Imports

class SAX():
  """
  Class containing the algorithm which performs symbolic conversion of time series using the SAX algorithm
  
  Reference: Lin, J., Keogh, E., Wei, L. and Lonardi, S. (2007). 
             Experiencing SAX: a Novel Symbolic Representation of Time Series. 
             Data Mining and Knowledge Discovery Journal.
  
  Link: https://www.cs.ucr.edu/~eamonn/SAX.htm
  """
  
  def __init__(self, freq, alphabetSizeDict, timeID=None):
    """
      This method initializes the SAX class
      @ In, alphabetSizeDict, dict, discretization size for each dimensions
      @ In, timeWindows, int, discretization of the time axis
    """
    self.freq = freq
    self.alphabetSizeDict = alphabetSizeDict
    self.timeID = timeID
  
  def fit(self, data, normalization=True):
    """
      This method performs symbolic conversion of time series using the SAX algorithm
      @ In, data, pandas DataFrame, time series that needs to be converted
      @ In, normalization, bool, parameter that set if time series normalization is required (True) or not (False)
      @ Out, symbolicTS, pandas DataFrame, symbolic conversion of provided time series
      @ Out, varCutPoints, dict, dictionary containing the discretization points for each dimension
    """
    # Normalize data
    if normalization:  
      normalizedData, normalizationData = self.timeSeriesNormalization(data)
    
    # PAA process
    paaData = self.piecewiseAggregateApproximation(normalizedData)
    
    symbolicData,varCutPoints = self.ndTS2String(paaData)
    
    for var in varCutPoints:
      varCutPoints[var] = varCutPoints[var]*normalizationData[var][1]+normalizationData[var][0]
  
    return symbolicData, varCutPoints
    
  def piecewiseAggregateApproximation(self, data):
    print(data)
    paa = data.resample(self.freq, on='time').mean().reset_index()
    return paa
    
  def piecewiseAggregateApproximationOLD(self, data):
    """
      This method performs Piecewise Aggregate Approximation of the given time series
      @ In, data, pandas DataFrame, time series to be discretized
      @ Out, paa, pandas DataFrame, discretized time series
    """
    nTimeVals, nVars = data.shape   
    paaData = {}
    for var in self.alphabetSizeDict.keys():
      res = np.zeros(self.timeWindows)
      if (nTimeVals % self.timeWindows == 0):
        inc = nTimeVals // self.timeWindows
        for i in range(0, nTimeVals):
          idx = i // inc
          res[idx] = res[idx] + data[var].to_numpy()[i]
        paaData[var] = res / inc
      else:
        for i in range(0, self.timeWindows * nTimeVals):
          idx = i // nTimeVals
          pos = i // self.timeWindows
          res[idx] = res[idx] + data[var].to_numpy()[pos]
        paaData[var] = res / nTimeVals
    
    paa = pd.DataFrame(paaData)

    return paa
  
  
  def timeSeriesNormalization(self, data):
    """
      This method performs the Z-normalization of a given time series
      @ In, data, pandas DataFrame, time series to be normalized
      @ Out, data, pandas DataFrame, normalized time series
      @ Out, normalizationData, dict, dictionary containing mean and std-dev of each dimension of the time series
    """
    normalizationData = {}
    normalizedData = {}
    
    for var in self.alphabetSizeDict.keys():
      if var!=self.timeID:
        normalizationData[var] = [np.mean(data[var].values),np.std(data[var].values)]
        normalizedData[var] = (data[var].values-normalizationData[var][0])/normalizationData[var][1]  
        
    normalizedData[self.timeID] = data[self.timeID].values
    normalizedDataDF = pd.DataFrame(normalizedData)
    return normalizedDataDF, normalizationData
  
  
  def ndTS2String(self, paaTimeSeries):  
    """
      This method performs the symbolic conversion of a given time series
      @ In, data, pandas DataFrame, multi-variate time series to be converted into string
      @ Out, paaTimeSeries, pandas DataFrame, symbolic converted time series
      @ Out, varCutPoints, dict, dictionary containing cuts data for each dimension
    """
    varCutPoints = {}
    
    for var in paaTimeSeries:
      if var!=self.timeID:
        varCutPoints[var] = norm.ppf(np.linspace(0.0, 1.0, num=self.alphabetSizeDict[var]+1),loc=0., scale=1.)
        paaTimeSeries[var] = self.ts2String(paaTimeSeries[var], varCutPoints[var])
    
    return paaTimeSeries, varCutPoints
        
  def ts2String(self, series, cuts): 
    """
      This method performs the symbolic conversion of a single time series
      @ In, series, pandas DataFrame, uni-variate time series to be converted into string
      @ In, cuts, dict, dictionary containing cuts data for the considered time series
      @ Out, charArray, np.array, symbolic converted time series
    """
    alphabetString = string.ascii_uppercase
    alphabetList = list(alphabetString) 
  
    series = np.array(series)
    charArray = np.chararray(series.shape[0],unicode=True) 
    
    for i in range(series.shape[0]):
      j=0
      while cuts[j]<series[i]:
        j=j+1
      charArray[i] = alphabetList[j-1]
  
    return charArray

