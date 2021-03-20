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
import statsmodels.api as sm
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
  
  #dateReformat = pd.to_datetime((data[timeID].astype(np.int64)//10**9 * 10**9).astype('datetime64[ns]'))
  newDate = pd.date_range(data[timeID][0], data[timeID].values[-1], periods=w)
  
  paa = pd.DataFrame(newDate, columns=['date'])
  
  for var in data:
    if var != timeID:
      res = np.zeros(w)
      if (nTimeVals % w == 0):
        inc = nTimeVals // w
        for i in range(0, nTimeVals):
          idx = i // inc
          res[idx] = res[idx] + data[var][i]
        paa[var] = res / inc
      else:
        for i in range(0, w * nTimeVals):
          idx = i // nTimeVals
          pos = i // w
          res[idx] = res[idx] + data[var][pos]
        paa[var] = res / nTimeVals
  return paa


def timeSeriesNormalization(data, timeID):
  normalizationData = {}
  for var in data:
    if var is not timeID:  
        normalizationData[var] = [np.mean(data[var]),np.std(data[var])]
        data[var] = (data[var]-normalizationData[var][0])/normalizationData[var][1]  
  return data, normalizationData


def ndTS2String(paaTimeSeries, alphabetSizeDict, timeID):
  cutPoints = {}
  cutPoints['2']  = np.array([-np.inf,  0.0])
  cutPoints['3']  = np.array([-np.inf, -0.43,  0.43])
  cutPoints['4']  = np.array([-np.inf, -0.67,  0,     0.67])
  cutPoints['5']  = np.array([-np.inf, -0.84, -0.25,  0.25,  0.84])
  cutPoints['6']  = np.array([-np.inf, -0.97, -0.43,  0.0,   0.43,  0.97])
  cutPoints['7']  = np.array([-np.inf, -1.07, -0.57, -0.18,  0.18,  0.57,  1.07])
  cutPoints['8']  = np.array([-np.inf, -1.15, -0.67, -0.32,  0.0,   0.32,  0.67,  1.15])
  cutPoints['9']  = np.array([-np.inf, -1.22, -0.76, -0.43, -0.14,  0.14,  0.43,  0.76,  1.22,])
  cutPoints['10'] = np.array([-np.inf, -1.28, -0.84, -0.52, -0.25,  0.0,   0.25,  0.52,  0.84,  1.28])
  cutPoints['11'] = np.array([-np.inf, -1.34, -0.91, -0.60, -0.35, -0.11,  0.11,  0.35,  0.60,  0.91, 1.34])
  cutPoints['12'] = np.array([-np.inf, -1.38, -0.97, -0.67, -0.43, -0.21,  0.0,   0.21,  0.43,  0.67, 0.97, 1.38])
  cutPoints['13'] = np.array([-np.inf, -1.43, -1.02, -0.74, -0.50, -0.29, -0.1,   0.1,   0.29,  0.50, 0.74, 1.02, 1.43])
  cutPoints['14'] = np.array([-np.inf, -1.47, -1.07, -0.79, -0.57, -0.37, -0.18,  0.0,   0.18,  0.37, 0.57, 0.79, 1.07, 1.47])
  cutPoints['15'] = np.array([-np.inf, -1.50, -1.11, -0.84, -0.62, -0.43, -0.25, -0.08,  0.08,  0.25, 0.43, 0.62, 0.84, 1.11, 1.5])
  cutPoints['16'] = np.array([-np.inf, -1.53, -1.15, -0.89, -0.67, -0.49, -0.32, -0.16,  0.0,   0.16, 0.32, 0.49, 0.67, 0.89, 1.15, 1.53])
  cutPoints['17'] = np.array([-np.inf, -1.56, -1.19, -0.93, -0.72, -0.54, -0.38, -0.22, -0.07,  0.07, 0.22, 0.38, 0.54, 0.72, 0.93, 1.19, 1.56])
  cutPoints['18'] = np.array([-np.inf, -1.59, -1.22, -0.97, -0.76, -0.59, -0.43, -0.28, -0.14,  0.0,  0.14, 0.28, 0.43, 0.59, 0.76, 0.97, 1.22, 1.59])
  cutPoints['19'] = np.array([-np.inf, -1.62, -1.25, -1.0,  -0.80, -0.63, -0.48, -0.34, -0.20, -0.07, 0.07, 0.20, 0.34, 0.48, 0.63, 0.80, 1.0,  1.25, 1.62])
  cutPoints['20'] = np.array([-np.inf, -1.64, -1.28, -1.04  -0.84, -0.67, -0.52, -0.39, -0.25, -0.13, 0.0,  0.13, 0.25, 0.39, 0.52, 0.67, 0.84, 1.04, 1.28, 1.64])
  
  # cutPoints = norm.ppf(np.linspace(0.0, 1.0, num=alphabetSizeDict+1),loc=0., scale=1.)
  
  simbolicTimeSeries = copy.deepcopy(paaTimeSeries)
  
  for var in simbolicTimeSeries:
    if var != timeID:
      varCutPoints = cutPoints[str(alphabetSizeDict[var])]
      simbolicTimeSeries = ts2String(paaTimeSeries[var], varCutPoints)
  
  return simbolicTimeSeries
      
def ts2String(series, cuts): 
  # Create list of 
  alphabetString = string.ascii_uppercase
  alphabetList = list(alphabetString) 
  
  series = np.array(series)
  a_size = len(cuts)
  sax = list()

  for i in range(series.shape[0]):
      num = series[i]
      # If the number is below 0, start from the bottom, otherwise from the top
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
  
  symbolicData = ndTS2String(paaData, alphabetSizeDict, timeID)
  
  return symbolicData
  
  '''
  normalizationData = {}
  
  for var in data:
    if var is not timeID:
      #Perform Z-normalization
      if normalization:
        normalizationData[var] = [np.mean(data[var]),np.std(data[var])]
        data[var] = (data[var]-normalizationData[var][0])/normalizationData[var][1]
      
      # 1 discretize value axis
      # PAA conversion
      interfaceParam = data[timeID].shape[0]/symbolicSeriesLength
      
      if (data[timeID].shape[0]%symbolicSeriesLength):
        # TS length is NOT divisible by symbolicSeriesLength
        
      else:
        # TS length is divisible by symbolicSeriesLength
        windows = np.split(data[var],symbolicSeriesLength)
        means = np.zeros(symbolicSeriesLength)
        for idx,window in enumerate(windows):
          means[idx] = np.mean(window)
      
      for t in range(symbolicSeriesLength):
        if t==0:
          window = data[var][1:(int(interfaceParam)+1)]
          window[-1] = window[-1]*math.modf(interfaceParam)[0]
          data[0] = np.mean(window)
        elif t == symbolicSeriesLength:
          window = data[var][math.modf(t*interfaceParam)[0]:-1]
          window[0] = 
          data[0] = np.mean(window)
        else:
              
      cdfAxis = np.linspace(0.0, 1.0, num=alphabetSize[var])
      kde = sm.nonparametric.KDEUnivariate(data[var])
      kde.fit()
      varAxis = kde.icdf(cdfAxis)
    else:
      # B discretize time axis
      data[timeID] = np.linspace(data[timeID][0], data[timeID][-1], num=timeSeriesLength)
'''
    
def SAXtimeInterval(data, alphabetSize, timeSeriesLength):
  pass 


''' testing '''
from datetime import datetime
date_rng = pd.date_range(start='1/1/2018', end='1/08/2018', freq='H')
df = pd.DataFrame(date_rng, columns=['date'])
df['var1'] = np.random.randint(0,100,size=(len(date_rng)))

alphabetSizeDict={}
alphabetSizeDict['var1']=15
sax = SAXtimePoints(df, 'date', alphabetSizeDict, 20, normalization=True)
print(sax)


   
    
    
    