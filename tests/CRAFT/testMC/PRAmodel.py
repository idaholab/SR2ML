# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
import random


def run(self,Input):
  # intput:
  # output:

  IEfreq = 1.E-3
  numberDiscretizations = 100

  self.time = np.linspace(0,Input['T'],numberDiscretizations)
