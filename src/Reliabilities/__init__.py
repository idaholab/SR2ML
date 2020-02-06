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
  from .ExponentialModel import ExponentialModel
  from .Factory import knownTypes
  from .Factory import returnInstance
  from .Factory import returnClass
except ImportError:
  from . import ExponentialModel
  from .Factory import knownTypes, returnInstance, returnClass

__all__ = ['ExponentialModel']
