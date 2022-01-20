# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
Created on January 19, 2022

@author: wangc, mandd


"""

import logging

ravenTemplateDir = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'framework')
sys.path.append(ravenTemplateDir)
from utils import xmlUtils

logger = logging.getLogger(__name__)


def buildUniformDistNode(name, lower, upper):
  """
    Build Uniform Distribution XML node
    @ In, name, str, the name for the XML node
    @ In, lower, float, the lower bound for the distribution
    @ In, upper, float, the upper bound for the distribution
    @ out, distNode, xml.etree.ElementTree.Element, the constructed XML node
  """
  distNode = xmlUtils.newNode(tag='Uniform', attrib={'name':name})
  distNode.append(xmlUtils.newNode(tag='lowerBound', text=lower))
  distNode.append(xmlUtils.newNode(tag='upperBound', text=upper))
  return distNode

def buildNormDistNode(name, mean, sigma):
  """
    Build Normal Distribution XML node
    @ In, name, str, the name for the XML node
    @ In, mean, float, the mean value
    @ In, sigma, float, the stardard deviation
    @ out, distNode, xml.etree.ElementTree.Element, the constructed XML node
  """
  distNode = xmlUtils.newNode(tag='Normal', attrib={'name':name})
  distNode.append(xmlUtils.newNode(tag='mean', text=mean))
  distNode.append(xmlUtils.newNode(tag='sigma', text=sigma))
  return distNode

def buildLogNormDistNode(name, mean, sigma):
  """
    Build LogNormal Distribution XML node
    @ In, name, str, the name for the XML node
    @ In, mean, float, the mean value
    @ In, sigma, float, the stardard deviation
    @ out, distNode, xml.etree.ElementTree.Element, the constructed XML node
  """
  distNode = xmlUtils.newNode(tag='LogNormal', attrib={'name':name})
  distNode.append(xmlUtils.newNode(tag='mean', text=mean))
  distNode.append(xmlUtils.newNode(tag='sigma', text=sigma))
  return distNode

def buildLogUniformDistNode(name, lower, upper, base='decimal'):
  """
    Build LogUniform Distribution XML node
    @ In, name, str, the name for the XML node
    @ In, lower, float, the lower bound for the distribution
    @ In, upper, float, the upper bound for the distribution
    @ In, base, str, the base for the LogUniform distribution
    @ out, distNode, xml.etree.ElementTree.Element, the constructed XML node
  """
  distNode = xmlUtils.newNode(tag='LogUniform', attrib={'name':name})
  distNode.append(xmlUtils.newNode(tag='lowerBound', text=lower))
  distNode.append(xmlUtils.newNode(tag='upperBound', text=upper))
  distNode.append(xmlUtils.newNode(tag='base', text=base))
  return distNode

def buildGammaDistNode(name, low, alpha, beta):
  """
    Build Gamma Distribution XML node
    @ In, name, str, the name for the XML node
    @ In, low, float, the lower domain boundary
    @ In, alpha, float, shape parameter
    @ In, beta, float, the inverse scale parameter
    @ out, distNode, xml.etree.ElementTree.Element, the constructed XML node
  """
  distNode = xmlUtils.newNode(tag='Gamma', attrib={'name':name})
  distNode.append(xmlUtils.newNode(tag='low', text=low))
  distNode.append(xmlUtils.newNode(tag='alpha', text=alpha))
  distNode.append(xmlUtils.newNode(tag='beta', text=beta))
  return distNode

def buildBetaDistNode(name, low, high, alpha, beta):
  """
    Build Beta Distribution XML node
    @ In, name, str, the name for the XML node
    @ In, low, float, the lower domain boundary
    @ In, high, float, the upper domain boundary
    @ In, alpha, float, shape parameter
    @ In, beta, float, shape parameter
    @ out, distNode, xml.etree.ElementTree.Element, the constructed XML node
  """
  distNode = xmlUtils.newNode(tag='Beta', attrib={'name':name})
  distNode.append(xmlUtils.newNode(tag='low', text=low))
  distNode.append(xmlUtils.newNode(tag='high', text=high))
  distNode.append(xmlUtils.newNode(tag='alpha', text=alpha))
  distNode.append(xmlUtils.newNode(tag='beta', text=beta))
  return distNode

def buildTriangDistNode(name, min, apex, max):
  """
    Build Triangular Distribution XML node
    @ In, name, str, the name for the XML node
    @ In, min, float, the domain lower boundary
    @ In, max, float, the domain upper boundary
    @ In, apex, float, the peak location
    @ out, distNode, xml.etree.ElementTree.Element, the constructed XML node
  """
  distNode = xmlUtils.newNode(tag='Triangular', attrib={'name':name})
  distNode.append(xmlUtils.newNode(tag='min', text=min))
  distNode.append(xmlUtils.newNode(tag='apex', text=apex))
  distNode.append(xmlUtils.newNode(tag='max', text=max))
  return distNode

def buildPoissonDistNode(name, mu):
  """
    Build Poisson Distribution XML node
    @ In, name, str, the name for the XML node
    @ In, mu, float, the mean rate
    @ out, distNode, xml.etree.ElementTree.Element, the constructed XML node
  """
  distNode = xmlUtils.newNode(tag='Poisson', attrib={'name':name})
  distNode.append(xmlUtils.newNode(tag='mu', text=mu))
  return distNode

def buildBinomialDistNode(name, n, p):
  """
    Build Binomial Distribution XML node
    @ In, name, str, the name for the XML node
    @ In, n, integer, the number of experiments
    @ In, p, float, the probability of success
    @ out, distNode, xml.etree.ElementTree.Element, the constructed XML node
  """
  distNode = xmlUtils.newNode(tag='Binomial', attrib={'name':name})
  distNode.append(xmlUtils.newNode(tag='n', text=n))
  distNode.append(xmlUtils.newNode(tag='p', text=p))
  return distNode

def buildBernoulliDistNode(name, p):
  """
    Build Bernoulli Distribution XML node
    @ In, name, str, the name for the XML node
    @ In, p, float, the probability of success
    @ out, distNode, xml.etree.ElementTree.Element, the constructed XML node
  """
  distNode = xmlUtils.newNode(tag='Bernoulli', attrib={'name':name})
  distNode.append(xmlUtils.newNode(tag='p', text=p))
  return distNode

def buildGeometricDistNode(name, p):
  """
    Build Geometric Distribution XML node
    @ In, name, str, the name for the XML node
    @ In, p, float, the success fraction for the trials
    @ out, distNode, xml.etree.ElementTree.Element, the constructed XML node
  """
  distNode = xmlUtils.newNode(tag='Geometric', attrib={'name':name})
  distNode.append(xmlUtils.newNode(tag='p', text=p))
  return distNode

def buildLogisticDistNode(name, loc, scale):
  """
    Build Logistic Distribution XML node
    @ In, name, str, the name for the XML node
    @ In, loc, float, the distribution mean
    @ In, scale, float, scale parameter that is proportional to the standard deviation
    @ out, distNode, xml.etree.ElementTree.Element, the constructed XML node
  """
  distNode = xmlUtils.newNode(tag='Logistic', attrib={'name':name})
  distNode.append(xmlUtils.newNode(tag='location', text=loc))
  distNode.append(xmlUtils.newNode(tag='scale', text=scale))
  return distNode

def buildLaplaceDistNode(name, loc, scale):
  """
    Build Laplace Distribution XML node
    @ In, name, str, the name for the XML node
    @ In, loc, float, determines the location or shift of the distribution
    @ In, scale, float, must be greater than 0, and determines how spread out the distribution is
    @ out, distNode, xml.etree.ElementTree.Element, the constructed XML node
  """
  distNode = xmlUtils.newNode(tag='Laplace', attrib={'name':name})
  distNode.append(xmlUtils.newNode(tag='location', text=loc))
  distNode.append(xmlUtils.newNode(tag='scale', text=scale))
  return distNode

def buildExponentialDistNode(name, param):
  """
    Build Exponential Distribution XML node
    @ In, name, str, the name for the XML node
    @ In, param, float, rate parameter
    @ out, distNode, xml.etree.ElementTree.Element, the constructed XML node
  """
  distNode = xmlUtils.newNode(tag='Exponential', attrib={'name':name})
  distNode.append(xmlUtils.newNode(tag='lambda', text=param))
  return distNode

def buildWeibullDistNode(name, k, param, low=0.0):
  """
    Build Weibull Distribution XML node
    @ In, name, str, the name for the XML node
    @ In, k, float, shape parameter
    @ In, param, float, scale parameter
    @ In, low, float, the lower domain boundary
    @ out, distNode, xml.etree.ElementTree.Element, the constructed XML node
  """
  distNode = xmlUtils.newNode(tag='Weibull', attrib={'name':name})
  distNode.append(xmlUtils.newNode(tag='k', text=k))
  distNode.append(xmlUtils.newNode(tag='lambda', text=param))
  distNode.append(xmlUtils.newNode(tag='low', text=low))
  return distNode

def buildVariableGroupNode(name, vars):
  """
    Build VariableGroup XML node
    @ In, name, str, the name for the XML node
    @ In, vars, str, the list of variable names
    @ out, node, xml.etree.ElementTree.Element, the constructed XML node
  """
  node = xmlUtils.newNode(tag='Group', attrib={'name':name}, text= vars)
  return node

def buildSamplerVarNode(name, distName):
  """
    Build Sampler Variable XML node
    @ In, name, str, the name for the XML node
    @ In, distName, str, the name for the distribution associated with the variable
    @ out, samplerVarNode, xml.etree.ElementTree.Element, the constructed XML node
  """
  samplerVarNode = xmlUtils.newNode(tag='variable', attrib={'name':name})
  samplerVarNode.append(xmlUtils.newNode(tag='distribution', text=distName))
  return samplerVarNode

def buildSamplerGridVarNode(name, distName, gridConst, gridType, gridVals, step=None):
  """
    Build Sampler Grid Variable XML node
    @ In, name, str, the name for the XML node
    @ In, distName, str, the name for the distribution associated with the variable
    @ In, gridConst, str, the construction method for the grid, i.e. 'custom' or 'equal'
    @ In, gridType, str, the type for the construction, i.e., 'CDF' or 'value'
    @ In, gridVals, str, the list of values for the grid
    @ In, step, int, the number of steps for the grid
    @ out, gridVals, xml.etree.ElementTree.Element, the constructed XML node
  """
  gridVarNode = buildSamplerVarNode(name, distName)
  if gridConst == 'equal':
    gridVarNode.append(xmlUtils.newNode(tag='grid', attrib={'construction':gridConst, 'steps':step, 'type':gridType}, text=gridVals))
  else:
    gridVarNode.append(xmlUtils.newNode(tag='grid', attrib={'construction':gridConst, 'type':gridType}, text=gridVals))
  return gridVals

def buildSamplerConstantNode(name, val):
  """
    Build Sampler Constant XML node
    @ In, name, str, the name for the XML node
    @ In, val, float, the value for the constant
    @ out, constNode, xml.etree.ElementTree.Element, the constructed XML node
  """
  constNode = xmlUtils.newNode(tag='constant', attrib={'name':name}, text=val)
  return constNode

def buildExternalModelNode(name, subType, inputs, outputs):
  """
    Build ExternalModel XML node
    @ In, name, str, the name for the XML node
    @ In, subType, str, the type of the external model
    @ In, inputs, str, list of input variables
    @ In, outputs, str, list of output variables
    @ out, externalModelNode, xml.etree.ElementTree.Element, the constructed XML node
  """
  externalModelNode = xmlUtils.newNode(tag='ExternalModel', attrib={'name':name, 'subType':subType})
  externalModelNode.append(xmlUtils.newNode(tag='inputs', text=inputs))
  externalModelNode.append(xmlUtils.newNode(tag='outputs', text=outputs))
  return externalModelNode

def buildOutStreamPrintNode(name, dsName):
  """
    Build OutStream Print XML node
    @ In, name, str, the name for the XML node
    @ In, dsName, str, the name for the DataSet
    @ out, printNode, xml.etree.ElementTree.Element, the constructed XML node
  """
  printNode = xmlUtils.newNode(tag='Print', attrib={'name':name})
  printNode.append(xmlUtils.newNode(tag='type', text='csv'))
  printNode.append(xmlUtils.newNode(tag='source', text=dsName))
  printNode.append(xmlUtils.newNode(tag='what', text='input, output'))
  return printNode


def buildFilesInputNode(name, filename, type=None):
  """
    Build Files Input XML node
    @ In, name, str, the name for the XML node
    @ In, type, str, the type for the file
    @ In, filename, str, the name for the file
    @ out, fileNode, xml.etree.ElementTree.Element, the constructed XML node
  """
  fileNode = xmlUtils.newNode(tag='Input', attrib={'name':name, 'type':type}, text=filename)
  return fileNode

def buildModelForEnsembleModel(modelName, inputDS, outputDS):
  """
    Build single model that used by EnsembleModel
    @ In, modelName, str, the name of model used as tag in xml.etree.ElementTree.Element
    @ In, inputDS, str, the name of input data object
    @ In, outputDS, str, the name of output data object
    @ Out, ensembleModelNode, xml.etree.ElementTree.Element, the ET element of Models under EnsembleModel
  """
  # ensemble model model node
  attribDict = {'class':'Models', 'type':'ExternalModel'}
  ensembleModelNode = xmlUtils.newNode(tag='Model', text=modelName, attrib=attribDict)
  attribDict = {'class':'DataObjects', 'type':'PointSet'}
  ensembleModelNode.append(xmlUtils.newNode(tag='Input', text=inputDS, attrib=attribDict))
  ensembleModelNode.append(xmlUtils.newNode(tag='TargetEvaluation', text=outputDS, attrib=attribDict))
  return ensembleModelNode

def buildPointSet(name, inputs, outputs):
  """
    Build single PointSet XML node
    @ In, name, str, the name for the PointSet
    @ In, inputs, str, string that contains the list of input variables
    @ In, outputs, str, string that contains the list of output variables
    @ out, pointSet, xml.etree.ElementTree.Element, the constructed PointSet XML node
  """
  pointSet = xmlUtils.newNode(tag='PointSet', attrib={'name':name})
  pointSet.append(xmlUtils.newNode(tag='Input', text=inputs))
  pointSet.append(xmlUtils.newNode(tag='Output', text=outputs))
  return pointSet

def buildHistorySet(name, inputs, outputs, pivot):
  """
    Build single HistorySet XML node
    @ In, name, str, the name for the PointSet
    @ In, inputs, str, string that contains the list of input variables
    @ In, outputs, str, string that contains the list of output variables
    @ In, pivot, str, the pivotParameter for HistorySet
    @ out, ds, xml.etree.ElementTree.Element, the constructed HistorySet XML node
  """
  ds = xmlUtils.newNode(tag='HistorySet', attrib={'name':name})
  ds.append(xmlUtils.newNode(tag='Input', text=inputs))
  ds.append(xmlUtils.newNode(tag='Output', text=outputs))
  option = xmlUtils.newNode(tag='options')
  option.append(xmlUtils.newNode(tag='pivotParameter', text=pivot))
  ds.append(option)
  return ds
