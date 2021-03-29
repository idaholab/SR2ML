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

def piecewiseAggregateApproximation(data,w):
  """
    This method performs Piecewise Aggregate Approximation of the given time series
    @ In, data, pandas DataFrame, time series to be discretized
    @ In, w, int, number of discretization points
    @ Out, paa, pandas DataFrame, discretized time series

    Reference: J. Lin, E. Keogh, L. Wei, and S. Lonardi 
               Experiencing SAX: a Novel Symbolic Representation of Time Series. 
               Data Mining and Knowledge Discovery Journal (2007).
    
    Link: https://www.cs.ucr.edu/~eamonn/SAX.htm
  """
  nTimeVals, nVars = data.shape
  
  newDate = pd.date_range(data.index.to_numpy()[0], data.index.to_numpy()[-1], periods=w)
  
  paaData = {}
  for var in data:
    res = np.zeros(w)
    if (nTimeVals % w == 0):
      inc = nTimeVals // w
      for i in range(0, nTimeVals):
        idx = i // inc
        res[idx] = res[idx] + data[var][i]
      paaData[var] = res / inc
    else:
      for i in range(0, w * nTimeVals):
        idx = i // nTimeVals
        pos = i // w
        res[idx] = res[idx] + data[var][pos]
      paaData[var] = res / nTimeVals
  
  paa = pd.DataFrame(paaData, index=newDate)

  return paa


def timeSeriesNormalization(data):
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


def ndTS2String(paaTimeSeries, alphabetSizeDict):  
  """
    This method performs the symbolic conversion of a given time series
    @ In, data, pandas DataFrame, multi-variate time series to be converted into string
    @ Out, paaTimeSeries, pandas DataFrame, symbolic converted time series
    @ Out, varCutPoints, dict, dictionary containing cuts data for each dimension
  """
  varCutPoints = {}
  
  for var in paaTimeSeries:
    varCutPoints[var] = norm.ppf(np.linspace(0.0, 1.0, num=alphabetSizeDict[var]+1),loc=0., scale=1.)
    paaTimeSeries[var] = ts2String(paaTimeSeries[var], varCutPoints[var])
  
  return paaTimeSeries, varCutPoints
      
def ts2String(series, cuts): 
  """
    This method performs the symbolic conversion of a single time series
    @ In, data, pandas DataFrame, univariate time series to be converted into string
    @ In, cuts, dict, dictionary containing cuts data for the considered time series
    @ Out, sax, np.array, symbolic converted time series
  """
  alphabetString = string.ascii_uppercase
  alphabetList = list(alphabetString) 
  
  series = np.array(series)
  cutSize = len(cuts)
  sax = np.chararray(series.shape[0]) 

  for i in range(series.shape[0]):
    num = series[i]
    if num>=0:
      j = cutSize - 1
      while j>0 and cuts[j]>=num:
        j = j - 1
      sax[i] = alphabetList[j] 
    else:
      j = 1
      while j<cutSize and cuts[j]<=num:
        j = j + 1
      sax[i] = alphabetList[j-1] 

  return sax


def SAXtimePoints(data, alphabetSizeDict, symbolicSeriesLength, normalization=True):
  """
    This method perform symbolic conversion of time series using the SAX algorithm
    @ In, data, pandas DataFrame, 
    @ In, alphabetSize, dict, discretization size for each dimensions
    @ In, timeSeriesLength, int, discretization of the time axis
    @ Out, symbolicTS, pandas DataFrame, symbolic conversion of provided time series

    Reference: Lin, J., Keogh, E., Wei, L. and Lonardi, S. (2007). 
               Experiencing SAX: a Novel Symbolic Representation of Time Series. 
               Data Mining and Knowledge Discovery Journal.
    
    Link: https://www.cs.ucr.edu/~eamonn/SAX.htm
  """
  
  # Normalize data
  if normalization:  
    normalizedData, normalizationData = timeSeriesNormalization(data)
  
  # PAA process
  paaData = piecewiseAggregateApproximation(normalizedData,symbolicSeriesLength)
  
  symbolicData,varCutPoints = ndTS2String(paaData, alphabetSizeDict)
  
  return symbolicData,varCutPoints,normalizationData


''' testing '''
from datetime import datetime
import matplotlib.pyplot as plt

data = {}
data['var1'] = np.random.randn(1000)
data['date'] = pd.date_range('1/1/2000', periods=1000)
df = pd.DataFrame({'var1':data['var1']}, index=data['date'])
df.index.name = 'date'
df = df.cumsum()

df.plot()

alphabetSizeDict={}
alphabetSizeDict['var1']=10
sax,cuts,normData = SAXtimePoints(df, alphabetSizeDict, 20, normalization=True)
print(sax)

for val in cuts['var1']:
  if val not in ['-inf','inf']:
    val = val*normData['var1'][1]+normData['var1'][0]
    plt.axhline(y=val,color='red')
for timeVal in sax.index.to_numpy():
  plt.axvline(x=timeVal,color='red')

print(cuts)
plt.show()



   
    
    
    