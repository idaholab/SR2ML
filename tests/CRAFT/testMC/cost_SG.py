# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
import random

def initialize(self, runInfo, inputs):
  seed = 9491
  random.seed(seed)

def run(self,Input):
  # intput:
  # output:

  if self.outcome_SG == 0:
    self.cost_SG = 0.
  else:
    numberDaysSD = float(random.randint(30,60))
    costPerDay   = 0.8 + 0.4 * random.random()
    self.cost_SG = numberDaysSD * costPerDay

  self.p_SG_cost = self.p_SG_ET
  self.t_SG_cost = self.t_SG_ET
