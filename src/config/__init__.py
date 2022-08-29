# __init__.py

import pathlib
import tomli
import os

path = pathlib.Path(os.path.join(pathlib.Path(__file__).parent, 'nlp_config.toml'))
with path.open(mode="rb") as fp:
  nlpConfig = tomli.load(fp)
  for file in nlpConfig['files']:
    if file != 'status_keywords_file':
      nlpConfig['files'][file] = os.path.join(os.path.dirname(__file__), nlpConfig['files'][file])
    else:
      for sub in nlpConfig['files'][file]:
        nlpConfig['files'][file][sub] = os.path.join(os.path.dirname(__file__), nlpConfig['files'][file][sub])
