# Copyright 2019, Intel Corporation

import numpy as np
from tools.cost_model.constants import *

def extract_feature(feature_str):
  '''
  Extract the feature from a string
  '''
  feature = []
  sub_strs = feature_str.split(';')
  for sub_str in sub_strs:
    numbers = [int(n) for n in sub_str.split()]
    feature.append(numbers)
  return feature

def extract_feature_line(line):
  '''
  Extract the features from a line
  '''
  start = line.find(FEATURE_HEAD)
  if start < 0:
    raise Exception('Not a feature line: ' + line)
  feature_line = line[start + len(FEATURE_HEAD):]
  sub_strs = feature_line.split('.')
  stmts_str = sub_strs[0]
  outer_str = sub_strs[1]
  inner_str = sub_strs[2]
  stmts_feature = [int(n) for n in stmts_str.split()]
  outer_feature = np.array(extract_feature(outer_str))
  inner_feature = np.array(extract_feature(inner_str))
  model_key = tuple(stmts_feature) + outer_feature.shape
  outer_feature = outer_feature.flatten()
  inner_feature = inner_feature.flatten()
  return model_key, np.concatenate((outer_feature, inner_feature))
