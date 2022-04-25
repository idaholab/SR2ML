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

def OPLentityParser(filename):
  '''
  This method extracts all the form and function entities out of the OPL html file and it puts them in two separate lists
  @in: filename, file name containing the OPL text
  @out: objectList, list, list of form elements contained in the OPL file
  @out: processList, list, list of function elements contained in the OPL file
  '''
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
  '''
  This method extracts all the sentences out of the OPL html file and it puts them in a list
  @in: filename, file name containing the OPL text
  @out: objectList, sentences, list of sentenced contained in the OPL file
  '''
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
      sentences[index] = sentences[index].replace("\n", " ")
      sentences[index] = sentences[index].replace("\xa0", "")
      sentences[index] = sentences[index].lstrip()

    sentences.remove('')

    return sentences

def listLemmatization(functionList):
  '''
  This method is designed to lemmatize all words contained in the list wordList
  @in: wordList, list, list containing the words to lemmatize
  @out: lemmatizedWords, list, list containing the lemmatized words
  '''
  nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
  lemmatizer = nlp.get_pipe("lemmatizer")
  lemmatizedWords = []
  for elem in functionList:
    lemmatizedWords.append([token.lemma_ for token in nlp(elem)][0])
  return lemmatizedWords

def OPLparser(sentences):
  '''
  This method translates all the sentences create a graph structure
  @in: filename: file name containing the OPL text
  @out: objectList: sentences, list of sentenced contained in the OPL file
  '''
  opmGraph = nx.MultiDiGraph()


  # These are 4 sets of OPL keywords
  OPLattributes = ['environmental','physical','informatical']
  OPLkeywordsDefinition = ['is an instance of ','is an','is']
  OPLkeywordsObjects = ['consists of']
  OPLkeywordsProcess = ['consumes','yields','requires','affects', 'changes']

  colorMatches = {'consists of':0.1,
                  'consumes'   :1 ,
                  'yields'     :2,
                  'requires'   :3,
                  'affects'    :4,
                  'changes'    :5}

  edge_colors = []

  for sentence in sentences:
    # create new elements in the graph from each sentence
    for elem in OPLkeywordsObjects+OPLkeywordsProcess:
      if elem in sentence:
        partitions = sentence.partition(elem)
        subj = partitions[0]
        conjs = re.split('and |, ',partitions[2])
        if '' in conjs:
          conjs.remove('')

        for conj in conjs:
          opmGraph.add_edge(subj, conj, key=elem)
          edge_colors.append(colorMatches[elem])

  return opmGraph,edge_colors

'''Testing workflow '''
formList, functionList = OPLentityParser('pump_OPL.html')

lemmatizedFunctionList = listLemmatization(functionList)
print(lemmatizedFunctionList)

sentences = OPLtextParser('pump_OPL.html')
opmGraph,edge_colors = OPLparser(sentences)

nx.draw_networkx(opmGraph,edge_color=edge_colors)
ax = plt.gca()
plt.axis("off")
plt.show()

