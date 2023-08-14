# How to install NLP libraries for SR2ML
- conda create -n nlp_libs python=3.9
- conda activate nlp_libs
- pip install spacy==3.1 textacy matplotlib nltk coreferee beautifulsoup4 networkx pysbd tomli numerizer autocorrect pywsd openpyxlquantulum3 numpy

# download language model from spacy (can not use INL network)
- python -m spacy download en_core_web_lg

# You may need to install stemming for some of unit parsing
- pip install stemming
