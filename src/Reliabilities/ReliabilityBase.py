"""
Created on Jan. 30 2020

@author: wangc, mandd
"""
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import os
import abc
import sys
#External Modules End--------------------------------------------------------------------------------

raven_path= os.path.abspath(os.path.dirname(__file__)) + '/../../../raven/framework'
sys.path.append(raven_path) #'~/projects/raven/framework') # TODO generic RAVEN location
#Internal Modules------------------------------------------------------------------------------------
from utils import mathUtils as utils
from utils import InputData, InputTypes
#Internal Modules End--------------------------------------------------------------------------------

class ReliabilityBase:
  """
    Base class for reliability models
  """

  @classmethod
  def getInputSpecification(cls):
    """
      Collects input specifications for this class.
      @ In, None
      @ Out, input_specs, InputData, specs
    """
    inputSpecs = InputData.parameterInputFactory('ReliabilityModel')
    inputSpecs.addParam('type', param_type=InputTypes.StringType)
    inputSpecs.addSub(InputData.parameterInputFactory('outputVariables', contentType=InputTypes.StringListType))
    return inputSpecs

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    ExternalModelPluginBase.__init__(self)
    self.type = self.__class__.__name__
    self.name = self.__class__.__name__
    # If True the metric needs to be able to handle dynamic data
    self._dynamicHandling    = False
    self._outputList = []
    self._variableDict = {}
    self._cdf = None
    self._pdf = None
    self._rdf = None
    self._frf = None

  def handleInput(self, xmlNode):
    """
      Function to read the portion of the xml input that belongs to this specialized class
      and initialize some stuff based on the inputs got
      @ In, xmlNode, xml.etree.ElementTree.Element, Xml element node
      @ Out, None
    """
    paramInput = self.getInputSpecification()()
    paramInput.parseNode(xmlNode)
    self._localHandleInput(self, paramInput)

  @abc.abstractmethod
  def _localHandleInput(self, paramInput):
    """
      Function to read the portion of the xml input that belongs to this specialized class
      and initialize some stuff based on the inputs got
      @ In, paramInput, InputData.ParameterInput, the parsed xml input
      @ Out, None
    """

  def isDynamic(self):
    """
      This method is utility function that tells if the reliability model is able to
      treat dynamic data on its own or not
      @ In, None
      @ Out, isDynamic, bool, True if the reliability model is able to treat dynamic data, False otherwise
    """
    return self._dynamicHandling

  def setVariable(self, value):
    """
      Set variable
      @ In, value, str or float or list, the value of given variable
      @ Out, ret, str or float or numpy.array, the recasted value
    """
    ret = None
    # multi-entry or single-entry?
    if len(value) == 1:
      # single entry should be either a float or string (raven variable)
      ret = value[0]
    else:
      # should be floats; InputData assures the entries are the same type already
      if not utils.isAFloatOrInt(value[0]):
        raise IOError('Multiple non-number entries are found, but require either a single variable name or multiple float entries: {}'.format(value))
      ret = np.asarray(value)
    return ret

  def loadVariables(self, need, inputDict):
    """
      Load the values of parameters from variables
      @ In, need, dict, the dict of parameters
      @ In, inputDict, dict, the dict of parameters that is provided from other sources
      @ Out, need, dict, the dict of parameters updated with variables
    """
    # load variable values from variables as needed
    for key, val in need.items():
      if utils.isAString(val):
        value = inputDict.get(val, None)
        if value is None:
          raise KeyError('Looking for variable "{}" to fill "{}" but not found among variables!'.format(val, key))
        need[key] = np.atleast_1d(value)
    return need

  def getParams(self):
    """
      Get the parameter value
      @ In, None
      @ Out, self._variableList, dict, list of variables
    """
    return self._variableDict

  def setParams(self, paramDict):
    """
      Sets the settings from a dictionary, instead of via an input file.
      @ In, paramDict, dict, settings
      @ Out, None
    """
    for key, val in paramDict.items():
      if key in self.__dict__.keys():
        setattr(self, key, val)
      else:
        print('WARNING: Variable "{}" is not defined in class "{}"!'.format(key, self.name))

  def getCDF(self):
    """
      get calculated cdf value
      @ In, None
      @ Out, self._cdf,
    """
    return self._cdf

  def getPDF(self):
    """
      get calculated cdf value
      @ In, None
      @ Out, self._pdf,
    """
    return self._pdf

  def getRDF(self):
    """
      get calculated cdf value
      @ In, None
      @ Out, self._cdf,
    """
    return self._rdf

  def getFDF(self):
    """
      get calculated cdf value
      @ In, None
      @ Out, self._cdf,
    """
    return self._fdf

  @abc.abstractmethod
  def _probabilityFunction(self):
    """
    """

  @abc.abstractmethod
  def _cumulativeFunction(self):
    """
    """

  @abc.abstractmethod
  def _reliabilityFunction(self):
    """
    """

  @abc.abstractmethod
  def _failureRateFunction(self):
    """
    """

  def initialize(self):
    """
      Method to initialize
      @ In, None
      @ Out, None
    """
    pass

  def run(self):
    """
      @ In, None
      @ Out, None
    """
    self._pdf = self._probabilityFunction()
    self._cdf = self._cumulativeFunction()
    self._rdf = self._reliabilityFunction()
    self._frf = self._failureRateFunction()
