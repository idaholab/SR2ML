# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
import numpy as np

def run(self,inputDict):
  # intput: None
  # output: time
  t0 = inputDict['T0'][0]
  tf  = inputDict['Tf'][0]
  num = inputDict['steps'][0]
  self.time  = np.linspace(t0, tf, num=num)
  self.tm = np.linspace(t0, tf, num=num)
