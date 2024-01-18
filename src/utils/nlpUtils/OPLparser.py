'''
Created on May 3, 2021

@author: mandd
'''

# External Imports
import xml.etree.ElementTree as ET
# import codecs
from bs4 import BeautifulSoup
# import nltk
# import unicodedata
import re
import networkx as nx
import matplotlib.pyplot as plt
import spacy
# Internal Import

class OPMobject(object):
  def __init__(self, filename):
    self.filename = filename
    self.objectList  = []
    self.processList = []
    self.edge_colors = []
    self.node_colors = []
    self.sentences   = None
    self.opmGraph    = None
    self.links2OPMs  = []
    self.acronyms    = {}

    self.OPLentityParser()
    self.OPLtextParser()
    self.OPLparser()


  def OPLentityParser(self):
    '''
    This method extracts all the form and function entities out of the OPL html file and it puts them in two separate lists:
    - self.objectList
    - self.processList
    This process is performed by parsing the html file and identify color-coded entities.
    '''
    with open(self.filename) as fp:
      soup = BeautifulSoup(fp, "html.parser")
      elements = soup.find("font").findChildren()
      for element in elements:
        if element.has_attr('color'):
          elem = element.string.replace("\n", " ")
          elem = checkAcronym(elem)
          if elem[1] is not None:
            self.acronyms[elem[1]] = elem[0]
          elem = elem[0]
          if element['color'] == '#006d00':
            if elem.lower() not in self.objectList:
              self.objectList.append(elem.lower())
          elif element['color'] == '#000078':
            if elem.lower() not in self.processList:
              self.processList.append(elem.lower())


  def OPLtextParser(self):
    '''
    This method extracts all the sentences out of the OPL html file and it puts them in a list (self.sentences)
    '''
    objects = {}
    functions = {}
    with open(self.filename) as fp:
      soup = BeautifulSoup(fp, "html.parser")

      # kill all script and style elements
      for script in soup(["script", "style"]):
          script.extract()    # rip it out
      # get text
      text = soup.get_text()
      # break into lines and remove leading and trailing space on each
      lines = (line.strip() for line in text.splitlines())
      # break multi-headlines into a line each
      chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
      # drop blank lines
      text = '\n'.join(chunk for chunk in chunks if chunk)
      self.sentences = text.split(".")

      for index, sentence in enumerate(self.sentences):
        self.sentences[index] = self.sentences[index].replace("\n", " ")
        self.sentences[index] = self.sentences[index].replace("\xa0", "")
        self.sentences[index] = self.sentences[index].lstrip().lower()

      self.sentences.remove('')


  def OPLparser(self):
    '''
    This method translates all the sentences (see self.sentences) and it create a graph structure (self.opmGraph)
    '''
    self.opmGraph = nx.MultiDiGraph()

    # These are 4 sets of OPL keywords
    OPLattributes = ['environmental','physical','informatical']
    OPLkeywordsDefinition = ['is an instance of ','is an','is']
    OPLkeywordsObjects = ['consists of']
    OPLkeywordsProcess = ['consumes','yields','requires','affects', 'feed']
    OPLkeywordsStates = ['can be']
    OPLkeywordsAttributes = ['exhibits']
    OPLkeywordsAction = ['changes']

    colorMatches = {'consists of':'r',
                    'consumes'   :'b' ,
                    'yields'     :'k',
                    'requires'   :'m',
                    'affects'    :'g',
                    'feed'       :'y'}

    for sentence in self.sentences:
      sentence = removeAcronym(sentence)
      # create new elements in the graph from each sentence
      for elem in OPLkeywordsObjects+OPLkeywordsProcess:
        if elem in sentence:
          partitions = sentence.partition(elem)
          subj = partitions[0]
          conjs = re.split('and |, ',partitions[2])
          if '' in conjs:
            conjs.remove('')
          for conj in conjs:
            self.opmGraph.add_edge(subj.strip(), conj.strip(), color=colorMatches[elem], key=elem)
            self.edge_colors.append(colorMatches[elem])

      # remove elements from "exhibits"
      if OPLkeywordsAttributes[0] in sentence:
        partitions = sentence.partition(OPLkeywordsAttributes[0])
        subj = partitions[0]
        conjs = re.split('and |, ',partitions[2])
        for conj in conjs:
          self.opmGraph.add_node(conj.strip(), color='g', key='attribute')
          self.objectList.remove(conj.strip())
          self.opmGraph.add_edge(subj.strip(), conj.strip(), color='b', key='exhibits')

      # address "changes"
      if OPLkeywordsAction[0] in sentence:
        partitions = sentence.partition(OPLkeywordsAction[0])
        subj = partitions[0]
        conj = partitions[2].partition(' from ')
        self.opmGraph.add_edge(subj.strip(), conj[0].strip(), color='g', key='changes')

      # address 'is instance of a'
      if 'is instance of a' in sentence:
        partitions = sentence.partition('is instance of a')
        subj = partitions[0]
        conj = partitions[2]
        self.opmGraph.add_edge(subj.strip(), conj.strip(), color='c', key='changes')
        self.links2OPMs.append(conj.strip())

    for elem in self.objectList:
      self.opmGraph.add_node(elem, color='m', key='object')
    for elem in self.processList:
      self.opmGraph.add_node(elem, color='k', key='process')

  def returnsExternalLinks(self):
    '''
    This method returns the links to other external OPM models
    '''
    return self.links2OPMs

  def returnGraph(self):
    '''
    This method returns the networkx graph
    '''
    return self.opmGraph

  def returnObjectList(self):
    '''
    This method returns the the list of objects
    '''
    objectNodes = [x for x,y in self.opmGraph.nodes(data=True) if y['key']=='object']
    return objectNodes

  def returnProcessList(self):
    '''
    This method returns the the list of processes
    '''
    processNodes = [x for x,y in self.opmGraph.nodes(data=True) if y['key']=='process']
    return processNodes

  def returnAttributeList(self):
    '''
    This method returns the the list of attributes
    '''
    attributeNodes = [x for x,y in self.opmGraph.nodes(data=True) if y['key']=='attribute']
    return attributeNodes

  def returnAcronym(self):
    return self.acronyms

def checkAcronym(s):
  '''
  This method separates an OPM object if an acronym is defined
  E.g.: 'travelling screen (TWS)'  --> ('travelling screen', 'TWS')
        'travelling screen'        --> ('travelling screen', None)
  '''
  if '(' in s:
      acronym = s[s.find("(")+1:s.find(")")]
      label = s[0:s.find("(")]
      return (label.strip(),acronym)
  else:
      return (s,None)

def removeAcronym(s):
  '''
  This method returns only the OPM object if an acronym is defined
  E.g.: 'travelling screen (TWS) failed'  --> 'travelling screen failed'
  '''
  if '(' in s:
      acronym = s[s.find("(")+1:s.find(")")]
      cleaned = s.replace("("+acronym+")", '')
      return cleaned
  else:
      return s
