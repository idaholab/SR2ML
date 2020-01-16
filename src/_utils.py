"""
  utilities for use within SR2ML
"""
import os
import sys
import importlib
import xml.etree.ElementTree as ET

def get_raven_loc():
  """
    Return RAVEN location
    hopefully this is read from heron/.ravenconfig.xml
    @ In, None
    @ Out, loc, string, absolute location of RAVEN
  """
  config = os.path.abspath(os.path.join(os.path.dirname(__file__),'..','.ravenconfig.xml'))
  if not os.path.isfile(config):
    raise IOError('SR2ML config file not found at "{}"! Has SR2ML been installed as a plugin in a RAVEN installation?'
                  .format(config))
  loc = ET.parse(config).getroot().find('FrameworkLocation').text
  return loc

if __name__ == '__main__':
  action = sys.argv[1]
  if action == 'get_raven_loc':
    print(get_raven_loc())
  else:
    raise IOError('Unrecognized action: "{}"'.format(action))
