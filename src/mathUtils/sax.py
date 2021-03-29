# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
Created on Mar 17, 2021

@author: mandd
"""
# External Imports
import numpy as np
import math
import copy
import pandas as pd
from scipy.stats import norm
from scipy import stats
import string
# Internal Imports

def piecewiseAggregateApproximation(data,w,timeID):
  """
    This method ...
    @ In, data, pandas DataFrame, 
    @ In, w, int, []
    @ Out, c, pandas DataFrame, 

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
  print(paa)
  return paa


def timeSeriesNormalization(data, timeID):
  normalizationData = {}
  for var in data:
    normalizationData[var] = [np.mean(data[var].values),np.std(data[var].values)]
    data[var] = (data[var]-normalizationData[var][0])/normalizationData[var][1]  
  return data, normalizationData


def ndTS2String(paaTimeSeries, alphabetSizeDict, timeID):  
  varCutPoints = {}
  
  for var in paaTimeSeries:
    varCutPoints[var] = norm.ppf(np.linspace(0.0, 1.0, num=alphabetSizeDict[var]+1),loc=0., scale=1.)
    paaTimeSeries[var] = ts2String(paaTimeSeries[var], varCutPoints[var])
  
  return paaTimeSeries, varCutPoints
      
def ts2String(series, cuts): 
  alphabetString = string.ascii_uppercase
  alphabetList = list(alphabetString) 
  
  series = np.array(series)
  a_size = len(cuts)
  sax = list()

  for i in range(series.shape[0]):
      num = series[i]
      if num >= 0:
          j = a_size - 1
          while j > 0 and cuts[j] >= num:
              j = j - 1
          sax.append(alphabetList[j])
      else:
          j = 1
          while j < a_size and cuts[j] <= num:
              j = j + 1
          sax.append(alphabetList[j-1])

  return sax


def SAXtimePoints(data, timeID, alphabetSizeDict, symbolicSeriesLength, normalization=True):
  """
    This method ...
    @ In, data, pandas DataFrame, 
    @ In, alphabetSize, dict, discretization parameters for each dimensions
    @ In, timeSeriesLength, int,
    @ Out, symbolicTS, pandas DataFrame, 

    Reference: Lin, J., Keogh, E., Wei, L. and Lonardi, S. (2007). 
               Experiencing SAX: a Novel Symbolic Representation of Time Series. 
               Data Mining and Knowledge Discovery Journal.
    
    Link: https://www.cs.ucr.edu/~eamonn/SAX.htm
  """
  
  # Normalize data
  if normalization:  
    normalizedData, normalizationData = timeSeriesNormalization(data, timeID)
  
  # PAA process
  paaData = piecewiseAggregateApproximation(normalizedData,symbolicSeriesLength,timeID)
  
  symbolicData,varCutPoints = ndTS2String(paaData, alphabetSizeDict, timeID)
  
  return symbolicData,varCutPoints,normalizationData
  
    
def SAXtimeInterval(data, alphabetSize, timeSeriesLength):
  pass 


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
alphabetSizeDict['var1']=6
sax,cuts,normData = SAXtimePoints(df, 'date', alphabetSizeDict, 20, normalization=True)
print(sax)

for val in cuts['var1']:
  if val not in ['-inf','inf']:
    val = val*normData['var1'][1]+normData['var1'][0]
    plt.axhline(y=val,color='red')
for timeVal in sax.index.to_numpy():
  plt.axvline(x=timeVal,color='red')

print(cuts)
plt.show()



   
    
    
    