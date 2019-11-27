# Copyright 2017 Battelle Energy Alliance, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Created on November 13, 2019

@author: mandd
"""

from __future__ import division, print_function , unicode_literals, absolute_import
import warnings
warnings.simplefilter('default', DeprecationWarning)

#External Modules---------------------------------------------------------------
import numpy as np
#External Modules End-----------------------------------------------------------

#Internal Modules---------------------------------------------------------------
from PluginsBaseClasses.ExternalModelPluginBase import ExternalModelPluginBase
from PostProcessors.FTStructure import FTStructure
#Internal Modules End-----------------------------------------------------------


class TDfailureRateReliabiltyModel(ExternalModelPluginBase):
  """
    This class is designed to create a time dependent failure rate reliability model
  """
  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    ExternalModelPluginBase.__init__(self)

  def _readMoreXML(self, container, xmlNode):
    """
      Method to read the portion of the XML that belongs to the time dependent failure rate reliability model
      @ In, container, object, self-like object where all the variables can be stored
      @ In, xmlNode, xml.etree.ElementTree.Element, XML node that needs to be read
      @ Out, None
    """
    container.mapping    = {}
    container.InvMapping = {}
    allowedTypesParams   = {'linear':{'alpha','integrationTimeStep'},
                            'weibull':{'alpha','beta','integrationTimeStep'},
                            'customLambda':{'fileName','skipFirstRow','timeColumn','reliabilityDataColumn','integrationTimeStep'},
                            'customPfailure':{'fileName','skipFirstRow','timeColumn',
                                              'reliabilityDataColumn','integrationTimeStep','customType'}
                            }

    for child in xmlNode:
      if child.tag == 'type':
        container.type = child.text.strip()
      if child.tag == 'alpha':
        container.alpha = child.text.strip()
      if child.tag == 'beta':
        container.beta = child.text.strip()
      if child.tag == 'fileName':
        container.fileName = child.text.strip()
      if child.tag == 'timeColumn':
        container.timeColumn = child.text.strip()
      if child.tag == 'reliabilityDataColumn':
        container.reliabilityDataColumn = child.text.strip()
      if child.tag == 'skipFirstRow':
        container.skipFirstRow = child.text.strip()
      if child.tag == 'integrationTimeStep':
        container.integrationTimeStep = child.text.strip()
      if child.tag == 'customType':
        container.customType = child.text.strip()
      else:
        raise IOError("TDfailureRateReliabiltyModel: xml node " + str (child.tag) + " is not allowed")

  def initialize(self, container, runInfoDict, inputFiles):
    """
      Method to initialize this plugin
      @ In, container, object, self-like object where all the variables can be stored
      @ In, runInfoDict, dict, dictionary containing all the RunInfo parameters (XML node <RunInfo>)
      @ In, inputFiles, list, list of input files (if any)
      @ Out, None
    """
    # if container.type in {'customLambda','customPfailure'}:
      # read file
      #self.dataFilename = os.path.join(self.workingDir,container.fileName)


  def run(self, container, Inputs):
    """
      This method determines the status of the TopEvent of the FT provided the status of its Basic Events
      @ In, container, object, self-like object where all the variables can be stored
      @ In, Inputs, dict, dictionary of inputs from RAVEN
    """
