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
  
  def __init__(self, timeWindows, alphabetSizeDict):
    """
      This method initializes the SAX class
      @ In, alphabetSizeDict, dict, discretization size for each dimensions
      @ In, timeWindows, int, discretization of the time axis
    """
    self.timeWindows = timeWindows
    self.alphabetSizeDict = alphabetSizeDict
  
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
    """
      This method performs Piecewise Aggregate Approximation of the given time series
      @ In, data, pandas DataFrame, time series to be discretized
      @ Out, paa, pandas DataFrame, discretized time series
    """
    nTimeVals, nVars = data.shape
    
    newDate = pd.date_range(data.index.to_numpy()[0], data.index.to_numpy()[-1], periods=self.timeWindows)
    
    paaData = {}
    for var in data:
      res = np.zeros(self.timeWindows)
      if (nTimeVals % self.timeWindows == 0):
        inc = nTimeVals // self.timeWindows
        for i in range(0, nTimeVals):
          idx = i // inc
          res[idx] = res[idx] + data[var][i]
        paaData[var] = res / inc
      else:
        for i in range(0, self.timeWindows * nTimeVals):
          idx = i // nTimeVals
          pos = i // self.timeWindows
          res[idx] = res[idx] + data[var][pos]
        paaData[var] = res / nTimeVals
    
    paa = pd.DataFrame(paaData, index=newDate)
  
    return paa
  
  
  def timeSeriesNormalization(self, data):
    """
      This method performs the Z-normalization of a given time series
      @ In, data, pandas DataFrame, time series to be normalized
      @ Out, data, pandas DataFrame, normalized time series
      @ Out, normalizationData, dict, dictionary containing mean and std-dev of each dimension of the time series
    """
    normalizationData = {}
    
    for var in data:
      normalizationData[var] = [np.mean(data[var].values),np.std(data[var].values)]
      data[var] = (data[var]-normalizationData[var][0])/normalizationData[var][1]  
    
    return data, normalizationData
  
  
  def ndTS2String(self, paaTimeSeries):  
    """
      This method performs the symbolic conversion of a given time series
      @ In, data, pandas DataFrame, multi-variate time series to be converted into string
      @ Out, paaTimeSeries, pandas DataFrame, symbolic converted time series
      @ Out, varCutPoints, dict, dictionary containing cuts data for each dimension
    """
    varCutPoints = {}
    
    for var in paaTimeSeries:
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



''' testing '''
import matplotlib.pyplot as plt

data = {}
data['var1'] = np.random.randn(1000)
data['date'] = pd.date_range('1/1/2000', periods=1000)
df = pd.DataFrame({'var1':data['var1']}, index=data['date'])
df.index.name = 'date'
df = df.cumsum()

df.to_csv('data.csv')

df.plot()

alphabetSizeDict={}
alphabetSizeDict['var1']=5
saxConverter = SAX(20,alphabetSizeDict)
sax,cuts = saxConverter.fit(df, normalization=True)
print(sax)

for val in cuts['var1']:
  if val not in ['-inf','inf']:
    plt.axhline(y=val,color='red', linewidth=0.2)
for timeVal in sax.index.to_numpy():
  plt.axvline(x=timeVal,color='red', linewidth=0.2)

print(cuts)
plt.show()



   
    
    
    