#include "../include/vta/vta_api.h"
#include "../include/vta/hw_spec.h"
#include "../include/vta/runtime.h"
#include <iostream>
Tensor* inp = NULL;
Tensor* wgt = NULL;
Tensor* res = NULL;

int kernel1(void* ptr) {
  for (int i = 0; i < 2; i++) {
    VTAUopLoopBegin(res->shape[i], res->shape[i + 1], 0, 0);
  }

  VTAUopPush(0, 1, 0, 0, 0, 0, 0, 0);

  for (int i = 0; i < 2; i++) {
    VTAUopLoopEnd();
  }

  return 0;
}

int kernel2(void* ptr) {
  for (int i = 0; i < 2; i++) {
    if (i == 0)
      VTAUopLoopBegin(res->shape[i], res->shape[i + 1], 1, 0);
    else
      VTAUopLoopBegin(res->shape[i], res->shape[i + 1], 0, 1);
  }

  VTAUopPush(0, 0, 0, 0, 0, 0, 0, 0);

  for (int i = 0; i < 2; i++) {
    VTAUopLoopEnd();
  }
  return 0;
}

void preproc(Tensor* ts, vector<int> factor) {
  // Padding
  vector<int> newShape;
  int newDataLen = 1;
  bool realloc = false;

  for (int i = 0; i < ts->shape.size(); i++) {
    int si = ts->shape[i];
    int newSi = (si + (factor[i] - 1)) & ~(factor[i] - 1);
    newShape.push_back(newSi);
    newDataLen *= newSi;
    cout << si << ' ' << newSi << endl;
  }

  if (newDataLen != ts->dataLen) {
    int* newData = (int*)calloc(newDataLen, sizeof(int));
    int* pNew = newData;
    int* pOld = ts->data;

    for (int i = 0; i < ts->shape[0]; i++) {
      memcpy(pNew, pOld, sizeof(int) * ts->shape[1]);

      pNew += newShape[1];
      pOld += ts->shape[1];
    }

    ts->oldData = ts->data;
    ts->oldShape = ts->shape;
    ts->oldDataLen = ts->dataLen;

    ts->data = newData;
    ts->shape = newShape;
    ts->dataLen = newDataLen;
  }

  // Reshape
  ts->reshape({ts->shape[0] / factor[0], factor[0], ts->shape[1] / factor[1], factor[1]});

  // Transpose
  ts->transpose({0, 2, 1, 3});
}

void postProc(Tensor *ts) {
  // Transpose
  ts->transpose({0, 2, 1, 3});

  // Reshape
  ts->reshape({ts->shape[0] * ts->shape[1], ts->shape[2] * ts->shape[3]});

  // Restore
  if (ts->oldData) {
    int* pNew = ts->data;
    int* pOld = ts->oldData;
    for (int i = 0; i < ts->oldShape[0]; i++) {
      memcpy(pOld, pNew, sizeof(int) * ts->oldShape[1]);

      pNew += ts->shape[1];
      pOld += ts->oldShape[1];
    }

    free(ts->data);
    ts->data = ts->oldData;
    ts->shape = ts->oldShape;
    ts->dataLen = ts->oldDataLen;

    ts->oldData = NULL;
    ts->oldShape = {};
    ts->oldDataLen = -1;
  }
}

Tensor* gemm(Tensor* t1, Tensor* t2) {
  vector<int> shape1 = t1->shape;
  vector<int> shape2 = t2->shape;

  Tensor *t3 = new Tensor({shape1[0], shape2[0]});

  inp = t1;
  wgt = t2;
  res = t3;

  assert(shape1.size() == 2 && shape2.size() == 2);

  assert(shape1[1] == shape2[1]);

  preproc(inp, {VTA_BATCH, VTA_BLOCK_IN});
  preproc(wgt, {VTA_BLOCK_OUT, VTA_BLOCK_IN});
  preproc(res, {VTA_BATCH, VTA_BLOCK_OUT});

#if 1
  inp->dump();
  wgt->dump();
#endif

  VTACommandHandle handle = VTATLSCommandHandle();
  int inpSize = inp->dataLen * sizeof(int);
  int wgtSize = wgt->dataLen * sizeof(int);
  int resSize = res->dataLen * sizeof(int);

  int* inpBuf = (int*)VTABufferAlloc(inpSize);
  int* wgtBuf = (int*)VTABufferAlloc(wgtSize);
  int* resBuf = (int*)VTABufferAlloc(resSize);

  VTABufferCopy(inp->data, 0, inpBuf, 0, inpSize, 1);
  VTABufferCopy(wgt->data, 0, wgtBuf, 0, wgtSize, 1);

  void* pKernel1 = NULL;
  VTAPushGEMMOp(&pKernel1, kernel1, NULL, 0);

  // ko: kernel outer factor
  assert(inp->shape[1] == wgt->shape[1]);
  // ki: kernel inner factor
  assert(inp->shape[3] == wgt->shape[3]);

  void* pKernel2 = NULL;
  for (int ko = 0; ko < inp->shape[1]; ko++) {
    VTALoadBuffer2D(handle, inpBuf, ko, 1, inp->shape[0], inp->shape[1], 0, 0, 0, 0, 0, 2);
    VTALoadBuffer2D(handle, wgtBuf, ko, 1, wgt->shape[0], wgt->shape[1], 0, 0, 0, 0, 0, 1);

    VTAPushGEMMOp(&pKernel2, kernel2, NULL, 0);
  }

  VTAStoreBuffer2D(handle, 0, 4, resBuf, 0, res->shape[0], res->shape[1], res->shape[0]);
  VTASynchronize(handle, 0x80000000);

  VTABufferCopy(resBuf, 0, res->data, 0, resSize, 2);

  postProc(res);
  res->dump();
  
  inp = wgt = res = NULL;
  return res;
}

void* gemm(void* data1, vector<int> shape1, void* data2, vector<int> shape2) {
  Tensor t1(shape1, data1);
  Tensor t2(shape2, data2);

  Tensor* t3 = gemm(&t1, &t2);

  t3->dump();
}
