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
  features = {}
  performance = {}
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
      case_id = int(feature_str[:pos])
      if not case_id in features:
        features[case_id] = []
        performance[case_id] = []
      features[case_id].append(feature_str)
      performance[case_id].append(exec_time)
  return features, performance

def preprocess_kmeans(train_dir, features, performance):
  '''
  Main pre-processing procedure
  '''
  filename = os.path.join(train_dir, PREPROCESSED_FILE)
  with open(filename, 'w') as fp:
    for case_id in performance:
      feature = features[case_id]
      perf = np.expand_dims(np.array(np.log10(performance[case_id])), axis = 1)
      cluster = KMeans(n_clusters = NUM_CLASSES, random_state = 0).fit(perf)
      num = len(perf)
      centers = np.squeeze(cluster.cluster_centers_, axis = 1)
      new_index = np.argsort(np.argsort(centers))
      print(cluster.cluster_centers_, new_index)
      count = [0] * len(cluster.cluster_centers_)
      orig_perf = performance[case_id]
      for i in range(num):
        fp.write(feature[i])
        fp.write(str(orig_perf[i]) + ' ' + str(new_index[cluster.labels_[i]]) + '\n')
        count[new_index[cluster.labels_[i]]] += 1
      print(count)

def preprocess_gmm(train_dir, features, performance):
  '''
  Main pre-processing procedure
  '''
  filename = os.path.join(train_dir, PREPROCESSED_FILE)
  with open(filename, 'w') as fp:
    for case_id in performance:
      feature = features[case_id]
      perf = np.expand_dims(np.array(np.log10(performance[case_id])), axis = 1)
      cluster = GaussianMixture(n_components = NUM_CLASSES, covariance_type='full')
      classes = cluster.fit_predict(perf)
      num = len(perf)
      # calculate centers
      centers = [0.0] * NUM_CLASSES
      count = [0] * NUM_CLASSES
      orig_perf = performance[case_id]
      for i in range(num):
        centers[classes[i]] += orig_perf[i]
        count[classes[i]] += 1
      for i in range(NUM_CLASSES):
        if count[i] > 0:
          centers[i] /= count[i]
      new_index = np.argsort(np.argsort(centers))
      print(centers, new_index)
      for i in range(num):
        fp.write(feature[i])
        fp.write(str(orig_perf[i]) + ' ' + str(new_index[classes[i]]) + '\n')
      print(count)

if __name__ == '__main__':
  train_dir = os.environ[CM_TRAIN_DIR]
  if train_dir == '':
    raise Exception('Need environment variable %s for saving the processed data' % CM_TRAIN_DIR)

  # Read training data
  features, performance = read_data(train_dir)

  # Pre-process
  preprocess_kmeans(train_dir, features, performance)
