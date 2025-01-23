"""
  This Module performs Unit Tests for the mcsReader inside MCSImporter methods
  It cannot be considered part of the active code but of the regression test system
"""

import os,sys
import numpy as np
filePath = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.normpath(os.path.join(filePath,os.pardir,os.pardir,os.pardir))) #Plugins (including SR2ML)
sys.path.append(os.path.normpath(os.path.join(filePath,os.pardir,os.pardir,os.pardir, os.pardir))) # RAVEN
# for regression test (RAVEN and SR2ML in the same directory)
sys.path.append(os.path.normpath(os.path.join(filePath,os.pardir,os.pardir,os.pardir, 'raven')))

from SR2ML.src._utils import get_raven_loc
frameworkDir = get_raven_loc()

from ravenframework.utils.utils import find_crow
find_crow(frameworkDir)

from SR2ML.src.PostProcessors.MCSimporter import mcsReader

results = {"pass":0,"fail":0}

def checkFloat(comment,value,expected,tol=1e-10,update=True):
  """
    This method is aimed to compare two floats given a certain tolerance
    @ In, comment, string, a comment printed out if it fails
    @ In, value, float, the value to compare
    @ In, expected, float, the expected value
    @ In, tol, float, optional, the tolerance
    @ In, update, bool, optional, if False then don't update results counter
    @ Out, res, bool, True if same
  """
  if np.isnan(value) and np.isnan(expected):
    res = True
  elif np.isnan(value) or np.isnan(expected):
    res = False
  else:
    res = abs(value - expected) <= tol
  if update:
    if not res:
      print("checking float",comment,'|',value,"!=",expected)
      results["fail"] += 1
    else:
      results["pass"] += 1
  return res

def checkSame(comment,value,expected,update=True):
  """
    This method is aimed to compare two identical things
    @ In, comment, string, a comment printed out if it fails
    @ In, value, float, the value to compare
    @ In, expected, float, the expected value
    @ In, update, bool, optional, if False then don't update results counter
    @ Out, res, bool, True if same
  """
  res = value == expected
  if update:
    if res:
      results["pass"] += 1
    else:
      print("checking string",comment,'|',value,"!=",expected)
      results["fail"] += 1
  return res

def checkArray(comment,first,second,dtype,tol=1e-10,update=True):
  """
    This method is aimed to compare two arrays
    @ In, comment, string, a comment printed out if it fails
    @ In, value, float, the value to compare
    @ In, expected, float, the expected value
    @ In, tol, float, optional, the tolerance
    @ In, update, bool, optional, if False then don't update results counter
    @ Out, res, bool, True if same
  """
  res = True
  if len(first) != len(second):
    res = False
    print("checking answer",comment,'|','lengths do not match:',len(first),len(second))
  else:
    for i in range(len(first)):
      if dtype == float:
        pres = checkFloat('',first[i],second[i],tol,update=False)
      elif dtype.__name__ in ('str','unicode'):
        pres = checkSame('',first[i],second[i],update=False)
      if not pres:
        print('checking array',comment,'|','entry "{}" does not match: {} != {}'.format(i,first[i],second[i]))
        res = False
  if update:
    if res:
      results["pass"] += 1
    else:
      results["fail"] += 1
  return res

fileName = 'MCSlist.csv'
mcsIDs, probability, mcsList, beList = mcsReader(fileName)
desired = ['1', '2', '5', '4']
checkArray('check MCS IDs', mcsIDs, desired, dtype=str)
desired = [1.86E-02, 4.00E-03, 4.00E-03, 4.00E-04]
checkArray('check MCS probabilities', probability, desired, dtype=float)
desired = [['ACCUMLATOR 1 DISCHARGE CKV 001 FAILS TO OPEN','480 VAC BUS 1A1 FAILS'],
            ['ACCUMLATOR 2 DISCHARGE CKV 002 FAILS TO OPEN','480 VAC BUS 1A2 FAILS'],
            ['ACCUMLATOR 3 DISCHARGE CKV 003 FAILS TO OPEN','480 VAC BUS 1A3 FAILS'],
            ['ACCUMULATOR CKVS 2 OF 3 FAIL FROM COMMON CAUSE TO OPEN']]
for i in range(len(desired)):
  checkArray('check MCS list', mcsList[i], desired[i], dtype=str)
desired = {'ACCUMLATOR 1 DISCHARGE CKV 001 FAILS TO OPEN','480 VAC BUS 1A1 FAILS',
          'ACCUMLATOR 2 DISCHARGE CKV 002 FAILS TO OPEN','480 VAC BUS 1A2 FAILS',
          'ACCUMLATOR 3 DISCHARGE CKV 003 FAILS TO OPEN','480 VAC BUS 1A3 FAILS',
          'ACCUMULATOR CKVS 2 OF 3 FAIL FROM COMMON CAUSE TO OPEN'}
checkArray('check BE list', sorted(beList), sorted(desired), dtype=str)

fileName = 'cutset_saphire.txt'
mcsIDs, probability, mcsList, beList = mcsReader(fileName, type='saphire')
desired = ['1', '4', '9', '10', '11', '12']
checkArray('check MCS IDs', mcsIDs, desired, dtype=str)
desired = [2.113E-2, 2.536E-5, 2.400E-6, 2.113E-6, 2.050E-6, 2.015E-6]
checkArray('check MCS probabilities', probability, desired, dtype=float)
desired = [['S-DGN-FR-B'],
            ['C-PMP-FS-B','S-DGN-FR-A'],
            ['S-TNK-FC-T1'],
            ['C-CKV-CC-B',  'S-DGN-FR-A'],
            ['C-CKV-CF'],
            ['C-PMP-FR-B',  'S-DGN-FS-A']]
for i in range(len(desired)):
  checkArray('check MCS list', mcsList[i], desired[i], dtype=str)
desired = {'S-DGN-FR-B','C-PMP-FS-B','S-DGN-FR-A', 'S-TNK-FC-T1', 'C-CKV-CC-B','C-CKV-CF', 'C-PMP-FR-B',  'S-DGN-FS-A'}
checkArray('check BE list', sorted(beList), sorted(desired), dtype=str)
print(results)

sys.exit(results["fail"])

"""
  <TestInfo>
    <name>src/PostProcessors.MCSimporter.mcsReader</name>
    <author>wangc</author>
    <created>2022-02-02</created>
    <classesTested>MCSimporter.mcsReader</classesTested>
    <description>
       This test performs Unit Tests for the MCSimporter.mcsReader.
       It cannot be considered part of the active code but of the regression test system
    </description>
  </TestInfo>
"""
