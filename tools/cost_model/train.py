# Copyright 2019, Intel Corporation
'''
Train the model for cost model prediction
'''
import os
import numpy as np
import xgboost as xgb
from tools.cost_model.constants import *
from tools.cost_model.feature import extract_feature_line
from tools.cost_model.perf_learner import PerfLearner

def read_data(train_dir):
  '''
  Read training data
  '''
  # We store the data according to feature shapes
  features = {}
  performance = {}
  # For every two lines, the first line is the feature
  # while the second line is the execution time
  filename = os.path.join(train_dir, PREPROCESSED_FILE)
  with open(filename, 'r') as fp:
    while True:
      feature_str = fp.readline()
      if feature_str == '':
        break
      second_line = fp.readline().split()
      exec_time = float(second_line[0])
      cost = float(second_line[1])
      model_key, feature = extract_feature_line(feature_str)
      if not model_key in features:
        features[model_key] = []
        performance[model_key] = []
      features[model_key].append(feature)
      performance[model_key].append(np.log(cost))
  return features, performance

def train(train_dir, features, performance):
  '''
  Main training procedure
  '''
  models = PerfLearner()
  for model_key in features:
    params = {'booster' : 'gbtree',
              'max_depth' : 16,
              'min_child_weight': 1,
              'objective' : 'reg:squarederror',
              'eta' : 0.1,
              'subsample' : 0.5,
              'alpha' : 0.1,
              'colsample_bytree' : 1.0,
              'eval_metric' : 'rmse',
              'grow_policy' : 'depthwise'}
    dtrain = xgb.DMatrix(features[model_key], performance[model_key])
    dtest = xgb.DMatrix(features[model_key], performance[model_key])
    num_boost_round = 1024
    model = xgb.train(params,
                      dtrain,
                      num_boost_round = num_boost_round,
                      evals = [(dtest, 'Test')],
    )
    models.set_model(model_key, model)
  models.save(train_dir)

if __name__ == '__main__':
  train_dir = os.environ[CM_TRAIN_DIR]
  if train_dir == '':
    raise Exception('Need environment variable %s for saving the models' % CM_TRAIN_DIR)

  # Read training data
  features, performance = read_data(train_dir)

  # Start training
  train(train_dir, features, performance) 
