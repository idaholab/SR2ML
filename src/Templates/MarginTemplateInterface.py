# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
Created on January 19, 2022

@author: wangc, mandd

This module is the Margin Template interface, which use the user provided input XML
file to construct the corresponding RAVEN workflows i.e. RAVEN input XML file.
"""

import os
import sys
import copy
import logging
import argparse
import numpy as np
import xml.etree.ElementTree as ET
try:
  from SR2MLTemplate import SR2MLTemplate
  import ravenXMLNodeUtils as ravenET
  from MarginSolverInputReader import MarginSolverInputReader
except ImportError:
  from SR2ML.src.Template.SR2MLTemplate import SR2MLTemplate
  import SR2ML.src.Template.ravenXMLNodeUtils as ravenET
  from SR2ML.src.Template.MarginSolverInputReader import MarginSolverInputReader

ravenTemplateDir = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'framework')
sys.path.append(ravenTemplateDir)
from utils import xmlUtils

logging.basicConfig(format='%(asctime)s %(name)-20s %(levelname)-8s %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.DEBUG)
# To enable the logging to both file and console, the logger for the main should be the root,
# otherwise, a function to add the file handler and stream handler need to be created and called by each module.
# logger = logging.getLogger(__name__)
logger = logging.getLogger()
# # create file handler which logs debug messages
fh = logging.FileHandler(filename='MarginSolver.log', mode='w')
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s %(name)-20s %(levelname)-8s %(message)s')
fh.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh)

class MarginTemplateInterface(object):
  """
    Capital Budgeting Template Interface
  """

  def __init__(self, filename):
    """
      Constructor.
      @ In, filename, the filename of Margin Solver input
      @ Out, None
    """
    self._filename = filename
    self._marginInputObj = MarginSolverInputReader(filename) # Use MarginSolverInputReader to parse the input
    self._paramInput = self._marginInputObj.getParamInput() # store the parsed input of ET tree of SR2ML input
    self._inputVarList = [] # user provided input variable list
    self._externalModelInputDict = {} # dictionary stores the inputs for each external model i.e. {externalModelName: list of inputs}
    self._externalModelOutputDict = {} # dictionary stores the outputs for each external model i.e. {externalModelName: list of outputs}
    self._distList = [] # list of constructed Distributions (ET element)
    self._sampVarList = [] # list of constructed sampled variables (ET element)
    self._ensembleModelList = [] # list of constructed models that are connected by EnsembleModel (ET element)
    self._dsList = [] # list of constructed data objects that are used for EnsembleModel (ET element)
    self._printList = [] # list of constructed output streams (ET element)
    self._variableGroupsList = [] # list of constructed variable groups (ET element)
    self._inputDict = {} # dictionary stores the user provided information, constructed from self._miscDict
    self._miscDict =  {'WorkingDir':'.',
                       'limit':'20'} # dictionary stores some default values and required inputs
    self._modelList = []

  def getOutput(self):
    """
      get the processed outputs from this class: MarginTemplateInterface
      @ In, None
      @ Out, (outputDict, miscDict), tuple, first dictionary contains the whole element that need to be appended in
        the templated input, while the second dictionary contains only the values that need to be replaced.
    """
    outputDict = {'Models': self._modelList,
                  'EnsembleModel': self._ensembleModelList,
                  'MonteCarlo': self._sampVarList,
                  'Distributions': self._distList,
                  'DataObjects': self._dsList,
                  'OutStreams': self._printList,
                  'Simulations': self._variableGroupsList}
    miscDict = {'WorkingDir': self._inputDict['WorkingDir'],
                'limit': self._inputDict['limit']}
    return outputDict, miscDict

  def readInput(self):
    """
      Read the SR2ML input files, and construct corresponding ET elements
      @ In, None
      @ Out, None
    """

    # process run settings
    runSettings = self._marginInputObj.getSub('Run_settings')
    self.readRunSettings(runSettings)
    # process Model
    modelNode = self._marginInputObj.getSub('Model')
    self.readModel(modelNode)
    bes = self._marginInputObj.findAll('Basic_event')
    self.readBasicEvents(bes)
    self.checkInput()

    self.buildMCSSolverModel()
    # self.buildModelListForEnsembleModel()

  def readRunSettings(self, paramInputs):
    """
      Read the GlobalSettings XML node from the input root
      @ In, paramInputs, InputData.ParameterInput, the parsed parameter input
      @ Out, None
    """
    logger.info("Start to read 'Run_settings' node")
    names = list(self._miscDict.keys())
    nodeDict, notFound = paramInputs.findNodesAndExtractValues(names)
    assert(not notFound)
    for key, val in self._miscDict.items():
      if key in nodeDict:
        self._inputDict[key] = xmlUtils.newNode(key, val)


  def readVariables(self, paramInputs):
    """
      Read the Variables XML node from the input XML
      @ In, paramInputs, InputData.ParameterInput, the parsed parameter input
      @ Out, None
    """
    logger.info("Start to read 'Variables' node")
    for node in paramInputs.subparts:
      self.readSingleVariable(node)

  def readSingleVariable(self, paramInputs):
    """
      Read the single Variable XML node from the input XML
      @ In, paramInputs, InputData.ParameterInput, the parsed parameter input
      @ Out, None
    """
    logger.debug("Start to read single variable: %s", paramInputs.getName())
    varName = paramInputs.getName()
    self._inputVarList.append(varName)
    distName = varName + '_dist'
    dist = paramInputs.getSub('Uniform')


    # distNode = copy.deepcopy(self.uniformDistNode)
    # distNode.attrib['name'] = distName
    # distNode.find('lowerBound').text = str(lower)
    # distNode.find('upperBound').text = str(upper)
    # # variable node for Samplers
    # sampNode = copy.deepcopy(self.samplerVarNode)
    # sampNode.attrib['name'] = varName
    # sampNode.find('distribution').text = distName
    # self._distList.append(distNode)
    # self._sampVarList.append(sampNode)
    logger.debug("End reading single variable: %s", paramInputs.getName())

  def readModel(self, paramInputs):
    """
      Read the Projects XML node from the input XML
      @ In, paramInputs, InputData.ParameterInput, the parsed parameter input
      @ Out, None
    """
    logger.info("Start to read %s node", paramInputs.getName())






  def checkInput(self):
    """
      Check the consistency of user provided inputs
      @ In, None
      @ Out, None
    """
    logger.info("Checking the consistency of user provided input")

  def buildCapitalInvestmentModel(self, templateModel):
    """
      Build the LOGOS.CapitalInvestmentModel based on the SR2ML input
      @ In, templateModel, xml.etree.ElementTree.Element, templated LOGOS.CapitalInvestmentModel
      @ Out, None
    """


  def buildModelListForEnsembleModel(self):
    """
      Build list of models for RAVEN EnsembleModel based on the SR2ML input
      @ In, None
      @ Out, None
    """
    totalOutput = []
    for model in self._modelList:
      key = model.attrib['name']
      inputDS = 'input_' + key
      outputDS = 'output_' + key
      # build ensemble model list
      ensembleModelNode = self.buildModelForEnsembleModel(key, inputDS, outputDS)
      self._ensembleModelList.append(ensembleModelNode)
      # build corresponding data object list
      inputVars = ','.join(self._externalModelInputDict[key])
      pointSet = self.buildPointSet(inputDS, inputVars, 'OutputPlaceHolder')
      self._dsList.append(pointSet)
      outputVars = ','.join(self._externalModelOutputDict[key])
      totalOutput.extend(self._externalModelOutputDict[key])
      pointSet = self.buildPointSet(outputDS, inputVars, outputVars)
      self._dsList.append(pointSet)
    # build main point set for whole ensemble model
    inputVars = ','.join(self._inputVarList)
    outputVars = ','.join(totalOutput)
    outputDS = "main_ps"
    pointSet = self.buildPointSet(outputDS, inputVars, outputVars)
    self._dsList.append(pointSet)



if __name__ == '__main__':
  logger.info('Welcome to the SR2ML Templated RAVEN Runner!')
  parser = argparse.ArgumentParser(description='SR2ML Templated RAVEN Runner')
  parser.add_argument('-i', '--input', nargs=1, required=True, help='SR2ML input filename')
  parser.add_argument('-t', '--template', nargs=1, required=True, help='SR2ML template filename')
  parser.add_argument('-o', '--output', nargs=1, help='SR2ML output filename')

  args = parser.parse_args()
  args = vars(args)
  inFile = args['input'][0]
  logger.info('SR2ML input file: %s', inFile)
  tempFile = args['template'][0]
  logger.info('SR2ML template file: %s', tempFile)
  if args['output'] is not None:
    outFile = args['output'][0]
    logger.info('SR2ML output file: %s', outFile)
  else:
    outFile = 'raven_' + inFile.strip()
    logger.warning('Output file is not specifies, default output file with name ' + outFile + ' will be used')

  # read capital budgeting input file
  templateInterface = MarginTemplateInterface(inFile)
  templateInterface.readInput()
  outputDict, miscDict = templateInterface.getOutput()

  # create template class instance
  templateClass = SR2MLTemplate()
  # load template
  here = os.path.abspath(os.path.dirname(tempFile))
  logger.info(' ... working directory: %s', here)
  tempFile = os.path.split(tempFile)[-1]
  templateClass.loadTemplate(tempFile, here)
  logger.info(' ... workflow successfully loaded ...')
  # create modified template
  template = templateClass.createWorkflow(outputDict, miscDict)
  logger.info(' ... workflow successfully modified ...')
  # write files
  here = os.path.abspath(os.path.dirname(inFile))
  templateClass.writeWorkflow(template, os.path.join(here, outFile), run=False)
  logger.info('')
  logger.info(' ... workflow successfully created and run ...')
  logger.info(' ... Complete!')
