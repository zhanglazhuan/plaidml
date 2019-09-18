# Copyright 2019, Intel Corporation

import os
import pickle
import numpy as np

ModelListFilename = 'model_list'

class PerfLearner:
  '''
  A wrapper for the models for cost model prediction
  '''
  def __init__(self):
    self.models = {}

  def set_model(self, model_key, model):
    self.models[model_key] = model

  def get_model(self, model_key):
    if self.model_exists(model_key):
      return self.models[model_key]
    raise Exception('Model with key {} does not exist'.format(model_key))

  def model_exists(self, model_key):
    return model_key in self.models

  def load(self, model_dir):
    with open(model_dir + '/' + ModelListFilename, 'r') as fp:
      while True:
        key_str = fp.readline()
        if key_str == '':
          break
        key_str = key_str.strip('\n')
        model_fn = fp.readline()
        model_path = os.path.join(model_dir, model_fn.strip('\n'))
        model = pickle.load(open(model_path, "rb"))
        model_key = tuple([int(n) for n in key_str.split()])
        self.set_model(model_key, model)

  def save(self, model_dir):
    # Create the model directory
    if not os.path.exists(model_dir):
      try:
        os.mkdir(model_dir)
      except OSError:
        print ("Creation of the directory %s failed" % model_dir)

    with open(model_dir + '/' + ModelListFilename, 'w') as fp:
      count = 0
      for (model_key, model) in self.models.items():
        model_fn = "%d.dat" % count
        model_path = os.path.join(model_dir, model_fn)
        count += 1
        model.save_model(model_path)
        key_str = ''
        for n in model_key:
          key_str += str(n) + ' '
        fp.write('%s\n%s\n' % (key_str.strip(), model_fn))
