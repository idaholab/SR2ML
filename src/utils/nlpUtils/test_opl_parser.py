'''
Created on April 25, 2022

@author: mandd, wangc
'''

import networkx as nx
import matplotlib.pyplot as plt
from OPLparser import OPLentityParser, listLemmatization, OPLtextParser, OPLparser

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
