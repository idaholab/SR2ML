'''
Created on May 3, 2021

@author: mandd
'''

# External Imports
import xml.etree.ElementTree as ET
import codecs
from bs4 import BeautifulSoup
import nltk
# Internal Import

def OPLentityParser(filename):
  objectList = []
  processList = []
  with open(filename) as fp:
    soup = BeautifulSoup(fp, "html.parser")
    elements = soup.find("font").findChildren()
    for element in elements:
      if element.has_attr('color'):
        if element['color'] == '#006d00':
          elem = element.string.replace("\n", " ")
          if elem not in objectList:
            objectList.append(elem)
        elif element['color'] == '#000078':
          elem = element.string.replace("\n", " ")
          if elem not in processList:
            processList.append(elem)          
  return objectList, processList

def OPLtextParser(filename):
  objects = {}
  functions = {}
  #fileObj = codecs.open(filename, 'r')
  #print(fileObj.read())
  with open(filename) as fp:
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
    sentences = text.split(".")
    for index, sentence in enumerate(sentences):
      sentences[index] = sentence.replace("\n", " ")
    return sentences

def parseInstantiation(sentence,processDict,objectDict):
  pass

def parseSpecializationProcess(sentence,processDict):
  pass

def parseSpecializationObject(sentence,processDict,objectDict):
  pass
  
def OPLparser(sentences,objectList,processList): 
  objectDict  = {}
  processDict = {}
  
  attributes = ['environmental','physical','informatical']
  OPLkeywordsDefinition = ['is an instance of ','is an','is']
  OPLkeywordsObjects = ['consists of'] 
  OPLkeywordsProcess = ['consumes','yields','requires','affects', 'changes']
  
  print(sentences)
  
  for sentence in sentences:
    #tokenizedSentence = nltk.word_tokenize(sentence)
    if OPLkeywordsDefinition[0] in sentence:
      processDict,objectDict = parseInstantiation(sentence,processDict,objectDict)
    elif OPLkeywordsDefinition[1] in sentence:
      processDict = parseSpecializationProcess(sentence,processDict)
    elif OPLkeywordsDefinition[2] in sentence:
      objectDict= parseSpecializationObject(sentence,processDict,objectDict)
  return sentences  

objectList, processList = OPLentityParser('pump_OPL.html')
sentences = OPLtextParser('pump_OPL.html')
sentences = OPLparser(sentences,objectList,processList)

