# Copyright 2019, Intel Corporation

import numpy as np
from tools.cost_model.constants import *

def extract_feature_line(line):
  '''
  Extract the features from a line
  '''
  start = line.find(FEATURE_HEAD)
  if start < 0:
    raise Exception('Not a feature line: ' + line)
  feature_line = line[start + len(FEATURE_HEAD):]
  sub_strs = feature_line.split()
  direction = int(sub_strs[0])
  num_buffer = int(sub_strs[1])
  total_size = int(sub_strs[2])
  num_constraints = int(sub_strs[3])
  model_key = (direction, num_buffer)
  features = [int(n) for n in sub_strs[4:]]
  return model_key, np.array(features)
