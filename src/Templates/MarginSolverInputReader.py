# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
Created on January 20, 2022

@author: wangc, mandd

Process Margin Solver Input XML file
"""

import xml.etree.ElementTree as ET

ravenTemplateDir = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'framework')
sys.path.append(ravenTemplateDir)

from BaseClasses import InputDataUser
from utils import InputData, InputTypes

class MarginSolverInputReader(InputDataUser):
  """
  """

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    spec = InputData.parameterInputFactory('SR2ML', ordered=False, descr=r""" """)
    # Run_settings
    sen = InputData.parameterInputFactory('sensitivity', contentType=InputTypes.BoolType, descr=r""" """)
    mode = InputData.parameterInputFactory('mode', contentType=InputTypes.StringType, descr=r""" """)
    settings = InputData.parameterInputFactory('Run_settings', descr=r""" """)
    settings.addSub(sen)
    settings.addSub(mode)
    spec.addsub(settings)
    # Model
    mcsFile = InputData.parameterInputFactory('MCS_file', contentType=InputTypes.StringType, descr=r""" """)
    teID = InputData.parameterInputFactory('top_event_ID', contentType=InputTypes.StringType, descr=r""" """)
    model = InputData.parameterInputFactory('Model', descr=r""" """)
    model.addSub(mcsFile)
    model.addSub(teID)
    spec.addSub(model)
    # Basic_event
    be = InputData.parameterInputFactory('Basic_event', descr=r""" """)
    be.addSub(InputData.parameterInputFactory('value', contentType=InputTypes.FloatType, descr=r""" """))
    dist = InputData.parameterInputFactory('distribution', descr=r""" """)
    dist.addParam('type', param_type=InputTypes.StringType, required=True, descr=r""" """)
    dist.addSub(InputData.parameterInputFactory('param1', contentType=InputTypes.FloatType, descr=r""" """))
    dist.addSub(InputData.parameterInputFactory('param2', contentType=InputTypes.FloatType, descr=r""" """))
    be.addSub(dist)
    spec.addSub(be)

    return spec

  def __init__(self, fileObj):
    """
      Construct.
      @ In, None
      @ Out, None
    """
    super().__init__()
    xml = self.processFile(fileObj)
    paramInput = self.parseXML(xml)

  def processFile(self, fileObj):
    """
    """
    if type(fileObj) == str:
      tree = ET.parse(fileObj)
      root = tree.getroot()
    elif isinstance(fileObj, ET.ElementTree):
      root = fileObj.getroot()
    elif isinstance(fileObj, ET.Element):
      root = fileObj
    else:
      raise IOError('Unsupported type of input is provided: ' + str(type(fileObj)))
    return root

  def parseXML(self, xml):
    """
      Parse XML into input parameters
      @ In, xml, xml.etree.ElementTree.Element, XML element node
      @ Out, paramInput, InputData.ParameterInput, the parsed input
    """
    paramInput = super().parseXML(xml)
    return paramInput

  def handleInput(self, paramInput, **kwargs):
    """
      Handles the input from the user.
      @ In, InputData.ParameterInput, the parsed input
      @ In, kwargs, dict, keyword arguments
      @ Out, None
    """
    super().handleInput(paramInput, **kwargs)
