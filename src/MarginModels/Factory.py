# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
Created on May 18th, 2021
@author: mandd,wangc
"""

from .PointSetMarginModel import PointSetMarginModel

"""
 Interface Dictionary (factory) (private)
"""
__base = 'MarginBase'
__interfaceDict = {}
__interfaceDict['PointSetMarginModel'] = PointSetMarginModel

def knownTypes():
  """
    Returns a list of strings that define the types of instantiable objects for
    this base factory.
    @ In, None
    @ Out, knownTypes, list, the known types
  """
  return __interfaceDict.keys()

def returnInstance(classType):
  """
    Attempts to create and return an instance of a particular type of object
    available to this factory.
    @ In, classType, string, string should be one of the knownTypes.
    @ Out, returnInstance, instance, subclass object constructed with no arguments
  """
  return returnClass(classType)()

def returnClass(classType):
  """
    Attempts to return a particular class type available to this factory.
    @ In, classType, string, string should be one of the knownTypes.
    @ Out, returnClass, class, reference to the subclass
  """
  try:
    return __interfaceDict[classType]
  except KeyError:
    raise IOError(__name__ + ': unknown ' + __base + ' type ' + classType)
