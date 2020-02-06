#_________________________________________________________________
#
#
#_________________________________________________________________
"""
Created on Feb. 4, 2020
@author: wangc, mandd
"""

from __future__ import absolute_import

try:
  from SR2ML.src.Reliabilities.ExponentialModel import ExponentialModel
  from SR2ML.src.Reliabilities.Factory import knownTypes, returnInstance, returnClass
except ImportError:
  from .ExponentialModel import ExponentialModel
  from .Factory import knownTypes, returnInstance, returnClass

__all__ = ['ExponentialModel']
