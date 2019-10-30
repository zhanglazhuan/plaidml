# Copyright 2019, Intel Corporation
'''
Train the model for cost model prediction
'''
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from tools.cost_model.constants import *

def read_data(train_dir):
  '''
  Read training data
  '''
  # We store the data according to feature shapes
  features = []
  performance = []
  # For every two lines, the first line is the feature
  # while the second line is the execution time
  filename = os.path.join(train_dir, MEASURE_FILE)
  with open(filename, 'r') as fp:
    while True:
      feature_str = fp.readline()
      if feature_str == '':
        break
      exec_time = float(fp.readline())
      pos = feature_str.find(FEATURE_HEAD)
      if pos < 0:
        raise Exception('Wrong feature string : %s' % feature_str)
      features.append(feature_str)
      performance.append(exec_time)
  return features, performance

def preprocess(train_dir, features, performance):
  '''
  Main pre-processing procedure
  '''
  feature_map = {}
  for i in range(len(features)):
    feature_str = features[i]
    header_start = feature_str.find(FEATURE_HEAD)
    feature_start = header_start + len(FEATURE_HEAD)
    run_nums = feature_str[:header_start].split()
    workgroups = int(run_nums[0])
    loops = int(run_nums[1])
    feature_strs = feature_str[feature_start:].split()
    feature_nums = [int(n) for n in feature_strs]
    direction = feature_nums[0]
    num_buffer = feature_nums[1]
    total_size = feature_nums[2]
    cost = performance[i] / (workgroups * loops * total_size)
    feature = [direction, num_buffer] + feature_nums[3:]
    key = tuple(feature)
    if not key in feature_map:
      feature_map[key] = []
    feature_map[key].append(cost)
 
  for feature_str in feature_map:
    mean = np.mean(feature_map[feature_str])
    std = np.std(feature_map[feature_str])
    print(feature_str, feature_map[feature_str], mean, std, mean - std, mean + std)

#  filename = os.path.join(train_dir, PREPROCESSED_FILE)
#  with open(filename, 'w') as fp:
#    for i in range(len(features)):
#      fp.write(feature_str)
#      fp.write('{} {}\n'.format(performance[i], cost))

if __name__ == '__main__':
  train_dir = os.environ[CM_TRAIN_DIR]
  if train_dir == '':
    raise Exception('Need environment variable %s for saving the processed data' % CM_TRAIN_DIR)

  # Read training data
  features, performance = read_data(train_dir)

  # Pre-process
  preprocess(train_dir, features, performance)
