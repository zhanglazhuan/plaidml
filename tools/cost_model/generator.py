# Copyright 2019, Intel Corporation

'''
Generate the training data for machine learning-based cost model
'''

import plaidml

import functools
import numpy as np
import operator
import os
import sys
import time

from tools.cost_model.constants import *
from multiprocessing import Process
from plaidml.keras import backend as pkb
from keras.backend import floatx

def m(*args, **kwargs):
  dtype = kwargs.get('dtype', floatx())
  '''Makes a test matrix whose dimensions are the supplied arguments.'''
  total = functools.reduce(operator.mul, args, 1)
  arr = np.array(range(-2, total - 2), dtype=dtype)
  if (arr.dtype in [
            "float16",
            "float32",
            "float64",
  ]):
    arr = arr / 2.0
  arr = np.reshape(arr, args)
  return arr

def _conv_inp(IN, IC, OC, IS, KS, strides, padding = 'valid', data_format = None):
  kernel_mat_np = m(*(KS + [IC, OC]))
  if data_format == 'channels_first':
    input_mat_np = m(*([IN] + [IC] + IS))
  else:
    input_mat_np = m(*([IN] + IS + [IC]))
  inputMat = input_mat_np
  kernelMat = kernel_mat_np
  return [inputMat, kernelMat, {'strides': strides, 'padding': padding, 'data_format': data_format}]

def TotalTiles():
  count = 0;
  with open(os.path.join(os.environ[CM_TRAIN_DIR], PLAN_FILE), 'r') as fp:
    for line in fp:
      count += 1;
  return count

def LastTestedTile():
  last_tested_path = os.path.join(os.environ[CM_TRAIN_DIR], LAST_TESTED_FILE);
  if not os.path.exists(last_tested_path):
    return -1;
  with open(last_tested_path, 'r') as fp:
    return int(fp.readline())

def LastBuiltTile():
  last_built_path = os.path.join(os.environ[CM_TRAIN_DIR], LAST_BUILT_FILE);
  if not os.path.exists(last_built_path):
    return -1;
  with open(last_built_path, 'r') as fp:
    return int(fp.readline())

def RemoveTmpFiles(files):
  train_dir = os.environ[CM_TRAIN_DIR]
  for fn in files:
    path = os.path.join(train_dir, fn)
    if os.path.exists(path):
      os.remove(path)

def run_plaidml_backend(data,
                        test_func,
                        backend,
                        num_iterations,
                        dtype,
                        shapes):
  if shapes:
    x = [backend.placeholder(shape = t) for t in shapes]
  else:
    x = [backend.placeholder(shape = t.shape) for t in data if hasattr(t, 'shape')]
  xv = [backend.variable(t, dtype = dtype) for t in data if hasattr(t, 'shape')]
  ps = [t for t in data if not hasattr(t, 'shape')]
  funcs = test_func(*(xv), **ps[0])
  tot_time = 0
  result = []
  for it_counter in range(num_iterations):
    start = time.time()
    # Evaluate forward operation
    result = funcs.eval()
    end = time.time()
    tot_time += (end - start)
  tot_time /= num_iterations
  print("    Testing took: %s sec." % (tot_time))

def run_test_cases(test_func,
                   in_data,
                   input_shapes = None,
                   dtype = floatx(),
                   num_iterations = 1):
  count = 0;
  for didx, data in enumerate(in_data):
    shapes = None
    if input_shapes:
      shapes = input_shapes[didx]
    count += 1
    print('    running: {}/{}'.format(count, len(in_data)))
    sys.stdout.flush()
    # Remove the status and plan files for the first part
    RemoveTmpFiles([PLAN_FILE, LAST_TESTED_FILE, LAST_BUILT_FILE, FIRST_GENERATED_FILE, FAILED_TILE_FILE])
    total_tiles = -1
    part = 1
    # A test case may be splitted into parts
    while True:
      print("    Building and running tile plans part #{} of case #{}".format(part, count))
      new_process = Process(target = run_plaidml_backend,
                            args=(data, test_func, pkb, num_iterations, dtype, shapes))
      new_process.start()
      new_process.join()
      if new_process.exitcode < 0:
        # If failed, try again. PlaidML will handle failure cases internally
        last_built = LastBuiltTile()
        print("    Failed to build tile plan #{}".format(last_built + 1))
        continue
      part += 1
      # Test if this part finished
      if total_tiles < 0:
        total_tiles = TotalTiles()
      last_tested = LastTestedTile()
      if last_tested >= total_tiles - 1:
        # We have tested all tiles
        break;

class TilePlanGenerator(object):
  def __init__(self, backend):
    self.backend_ = backend

  def testDot(self):
    data = [
      [m(16, 16), m(16, 16), {}],
      [m(32, 32), m(32, 32), {}],
      [m(64, 64), m(64, 64), {}],
      [m(512, 512), m(512, 512), {}],
      [m(1024, 1024), m(1024, 1024), {}],
      [m(2048, 2048), m(2048, 2048), {}],
      [m(517, 121), m(121, 517), {}],
      [m(512, 128), m(128, 512), {}],
      [m(67, 33), m(33, 67), {}],
      [m(64, 16), m(16, 64), {}],
      [m(13, 17), m(17, 13), {}],
    ]
    run_test_cases(self.backend_.dot, data)

  def testConv1d(self):
    data = [
      _conv_inp(IN = 1, IC = 3, OC = 1, IS = [64], KS = [2],
          strides = (1), padding = 'same', data_format = 'channels_last'),
      _conv_inp(IN = 1, IC = 1, OC = 3, IS = [128], KS = [2],
          strides = (1), padding = 'same', data_format = 'channels_first'),
      _conv_inp(IN = 2, IC = 4, OC = 4, IS = [224], KS = [2],
          strides = (1), padding = 'valid', data_format = 'channels_last'),
      _conv_inp(IN = 1, IC = 3, OC = 1, IS = [64], KS = [3],
          strides = (1), padding = 'valid', data_format = 'channels_last'),
      _conv_inp(IN = 1, IC = 1, OC = 3, IS = [128], KS = [3],
          strides = (1), padding = 'same', data_format = 'channels_first'),
      _conv_inp(IN = 2, IC = 4, OC = 4, IS = [224], KS = [3],
          strides = (1), padding = 'valid', data_format = 'channels_last'),
      _conv_inp(IN = 1, IC = 3, OC = 1, IS = [64], KS = [5],
          strides = (1), padding = 'valid', data_format = 'channels_last'),
      _conv_inp(IN = 1, IC = 1, OC = 3, IS = [128], KS = [5],
          strides = (1), padding = 'valid', data_format = 'channels_first'),
      _conv_inp(IN = 2, IC = 4, OC = 4, IS = [224], KS = [5],
          strides = (1), padding = 'same', data_format = 'channels_last'),
    ]
    run_test_cases(self.backend_.conv1d, data)

  def testConv2d_4_6(self):
    data = [
      _conv_inp(IN = 1, IC = 3, OC = 64, IS = [224, 224], KS = [7, 7],
          strides = (2, 2), padding = 'valid', data_format = 'channels_last'),
      _conv_inp(IN = 1, IC = 64, OC = 64, IS = [56, 56], KS = [3, 3],
          strides = (1, 1), padding = 'same', data_format = 'channels_last'),
      _conv_inp(IN = 1, IC = 128, OC = 128, IS = [28, 28], KS = [3, 3],
          strides = (1, 1), padding = 'same', data_format = 'channels_last'),
      _conv_inp(IN = 1, IC = 256, OC = 256, IS = [14, 14], KS = [3, 3],
          strides = (1, 1), padding = 'same', data_format = 'channels_last'),
      _conv_inp(IN = 1, IC = 512, OC = 512, IS = [7, 7], KS = [3, 3],
          strides = (1, 1), padding = 'same', data_format = 'channels_last'),
    ]
    run_test_cases(self.backend_.conv2d, data)

  def testConv2d_4_4(self):
    data = [
      _conv_inp(IN = 1, IC = 64, OC = 64, IS = [56, 56], KS = [1, 1],
          strides = (1, 1), padding = 'valid', data_format = 'channels_last'),
      _conv_inp(IN = 1, IC = 64, OC = 256, IS = [56, 56], KS = [1, 1],
          strides = (1, 1), padding = 'valid', data_format = 'channels_last'),
      _conv_inp(IN = 1, IC = 256, OC = 64, IS = [56, 56], KS = [1, 1],
          strides = (1, 1), padding = 'valid', data_format = 'channels_last'),
      _conv_inp(IN = 1, IC = 256, OC = 128, IS = [56, 56], KS = [1, 1],
          strides = (2, 2), padding = 'same', data_format = 'channels_last'),
      _conv_inp(IN = 1, IC = 256, OC = 512, IS = [56, 56], KS = [1, 1],
          strides = (2, 2), padding = 'same', data_format = 'channels_last'),
      _conv_inp(IN = 1, IC = 128, OC = 512, IS = [28, 28], KS = [1, 1],
          strides = (1, 1), padding = 'same', data_format = 'channels_last'),
      _conv_inp(IN = 1, IC = 512, OC = 128, IS = [28, 28], KS = [1, 1],
          strides = (1, 1), padding = 'same', data_format = 'channels_last'),
      _conv_inp(IN = 1, IC = 512, OC = 256, IS = [28, 28], KS = [1, 1],
          strides = (2, 2), padding = 'same', data_format = 'channels_last'),
      _conv_inp(IN = 1, IC = 512, OC = 1024, IS = [28, 28], KS = [1, 1],
          strides = (2, 2), padding = 'same', data_format = 'channels_last'),
      _conv_inp(IN = 1, IC = 256, OC = 1024, IS = [14, 14], KS = [1, 1],
          strides = (1, 1), padding = 'same', data_format = 'channels_last'),
      _conv_inp(IN = 1, IC = 1024, OC = 256, IS = [14, 14], KS = [1, 1],
          strides = (1, 1), padding = 'same', data_format = 'channels_last'),
      _conv_inp(IN = 1, IC = 1024, OC = 512, IS = [14, 14], KS = [1, 1],
          strides = (2, 2), padding = 'same', data_format = 'channels_last'),
      _conv_inp(IN = 1, IC = 1024, OC = 2048, IS = [14, 14], KS = [1, 1],
          strides = (2, 2), padding = 'same', data_format = 'channels_last'),
      _conv_inp(IN = 1, IC = 512, OC = 2048, IS = [7, 7], KS = [1, 1],
          strides = (1, 1), padding = 'same', data_format = 'channels_last'),
      _conv_inp(IN = 1, IC = 2048, OC = 512, IS = [7, 7], KS = [1, 1],
          strides = (1, 1), padding = 'same', data_format = 'channels_last'),
    ]
    run_test_cases(self.backend_.conv2d, data)

  def testConv3d(self):
    data = [
      _conv_inp(IN = 3, IC = 3, OC = 1, IS = [32, 32, 32], KS = [2, 2, 2],
          strides = (1, 1, 1), padding = 'same', data_format = 'channels_last'),
      _conv_inp(IN = 1, IC = 1, OC = 3, IS = [56, 28, 16], KS = [2, 2, 2],
          strides = (1, 1, 1), padding = 'valid', data_format = 'channels_first'),
      _conv_inp(IN = 2, IC = 4, OC = 2, IS = [112, 16, 4], KS = [2, 2, 2],
          strides = (1, 1, 1), padding = 'valid', data_format = 'channels_first'),
      _conv_inp(IN = 3, IC = 3, OC = 1, IS = [32, 32, 32], KS = [3, 3, 3],
          strides = (1, 1, 1), padding = 'valid', data_format = 'channels_last'),
      _conv_inp(IN = 1, IC = 1, OC = 3, IS = [56, 28, 16], KS = [3, 3, 3],
          strides = (1, 1, 1), padding = 'same', data_format = 'channels_first'),
      _conv_inp(IN = 2, IC = 4, OC = 2, IS = [112, 16, 4], KS = [3, 3, 3],
          strides = (1, 1, 1), padding = 'valid', data_format = 'channels_first'),
      _conv_inp(IN = 3, IC = 3, OC = 1, IS = [32, 32, 32], KS = [5, 5, 5],
          strides = (1, 1, 1), padding = 'valid', data_format = 'channels_last'),
      _conv_inp(IN = 1, IC = 1, OC = 3, IS = [56, 28, 16], KS = [5, 5, 5],
          strides = (1, 1, 1), padding = 'valid', data_format = 'channels_first'),
      _conv_inp(IN = 2, IC = 4, OC = 2, IS = [112, 16, 4], KS = [5, 5, 5],
          strides = (1, 1, 1), padding = 'same', data_format = 'channels_first'),
    ]
    run_test_cases(self.backend_.conv3d, data)

  def runAll(self):
    for func in dir(self):
      if func.startswith('test'):
        to_call = getattr(self, func)
        print('Testing ', func)
        sys.stdout.flush()
        to_call()

if __name__ == '__main__':
  if CM_TRAIN_DIR not in os.environ:
    raise Exception('Need environment variable %s for the filename of the test result' % CM_TRAIN_DIR)
  train_dir = os.environ[CM_TRAIN_DIR]
  if os.path.exists(train_dir):
    for fn in os.listdir(train_dir):
      file_path = os.path.join(train_dir, fn)
      try:
        if os.path.isfile(file_path):
          os.unlink(file_path)
      except Exception as e:
        print(e)
  else:
    try:
      os.mkdir(train_dir)
    except OSError:
      print ("Creation of the directory %s failed" % train_dir)
  if 'VERBOSE' in os.environ:
    verbose = os.environ['VERBOSE']
    if verbose != "":
      plaidml._internal_set_vlog(int(verbose))
  ttp = TilePlanGenerator(pkb)
  #ttp.runAll()
  ttp.testConv2d_4_4()
  ttp.testConv2d_4_6()
