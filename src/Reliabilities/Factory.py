#
#
#
"""
Created on Feb. 21, 2019
@author: wangc, mandd
"""

from .ExponentialModel import ExponentialModel
from .ErlangianModel import ErlangianModel
from .GammaModel import GammaModel
from .LognormalModel import LognormalModel
from .WeibullModel import WeibullModel
from .FatigueLifeModel import FatigueLifeModel
from .NormalModel import NormalModel
from .BathtubModel import BathtubModel
from .PowerLawModel import PowerLawModel

"""
 Interface Dictionary (factory) (private)
"""
__base = 'ReliabilityBase'
__interfaceDict = {}
__interfaceDict['exponential'] = ExponentialModel
__interfaceDict['erlangian'] = ErlangianModel
__interfaceDict['gamma'] = GammaModel
__interfaceDict['lognorm'] = LognormalModel
__interfaceDict['weibull'] = WeibullModel
__interfaceDict['fatiguelife'] = FatigueLifeModel
__interfaceDict['normal'] = NormalModel
__interfaceDict['bathtub'] = BathtubModel
__interfaceDict['powerlaw'] = PowerLawModel

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
    return __interfaceDict[classType.lower()]
  except KeyError:
    raise IOError(__name__ + ': unknown ' + __base + ' type ' + classType)
