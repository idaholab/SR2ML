# How to install NLP libraries for SR2ML
- conda create -n nlp_libs python=3.9
- conda activate nlp_libs
- pip install spacy==3.1 textacy matplotlib nltk coreferee beautifulsoup4 networkx pysbd tomli numerizer autocorrect pywsd openpyxl quantulum3[classifier] numpy scikit-learn=1.2.2

# scikit-learn 1.2.2 is required for quantulum3

# download language model from spacy (can not use INL network)
- python -m spacy download en_core_web_lg

# Different approach when there is a issue with SSLError
1. Download en_core_web_lg-3.1.0.tar.gz from https://github.com/explosion/spacy-models/releases/tag/en_core_web_lg-3.1.0

2. python -m pip install ./en_core_web_lg-3.1.0.tar.gz

# You may need to install stemming for some of unit parsing
- pip install stemming

# Windows machine have issue with pydantic
See https://github.com/explosion/spaCy/issues/12659
Installing typing_extensions<4.6
pip install typing_extensions=4.5.*
