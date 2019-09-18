# Copyright 2019, Intel Corporation
'''
Test the trained model for cost model prediction
'''
import os
import numpy as np
from tools.cost_model.feature import extract_feature_line
from tools.cost_model.perf_regressor import PerfRegressor

def read_data(data_fn):
  data = []
  with open(data_fn, 'r') as fp:
    for line in fp:
      model_key, feature = extract_feature_line(line)
      data.append([model_key, np.array(feature)])
  return data

def test(models, data):
  for case in data:
    model_key = case[0]
    model = models.get_model(model_key)
    feature = case[1]
    results = model.predict([feature])
    print(results)

def test_file(model_dir, data_fn):
  models = PerfRegressor()
  models.load(model_dir)
  data = read_data(data_fn)
  test(models, data)

if __name__ == '__main__':
  train_dir = os.environ[CM_TRAIN_DIR]
  if train_dir == '':
    raise Exception('Need environment variable %s for loading the models', CM_TRAIN_DIR)
  data_fn = os.environ['TEST_DATA']
  if data_fn == '' or not os.path.exists(data_fn):
    raise Exception('Need environment variable TEST_DATA for the testing file')
  test_file(model_dir, os.path.join(model_dir, data_fn));
