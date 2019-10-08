#ifndef VTA_API_H_
#define VTA_API_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdlib.h>

#ifdef __cplusplus
}
#endif
#include <iostream>
#include <vector>

using namespace std;

class Tensor {
 public:
  Tensor(vector<int> inShape, void* inData = NULL) {
    shape = inShape;
    dataLen = 1;

    for (int i = 0; i < shape.size(); i++) {
      dataLen *= shape[i];
    }

    if (!inData) {
      data = (int*)malloc(dataLen * sizeof(int));

      for (int i = 0; i < dataLen; i++) {
        data[i] = i;
      }
    } else
      data = (int*)inData;
  }

  void reshape(vector<int> inShape) {
    int newLen = 1;
    for (int i = 0; i < inShape.size(); i++) {
      newLen *= inShape[i];
    }

    assert(dataLen == newLen);

    shape = inShape;
  }

  void transpose1(vector<int> dimIdx, vector<int>& tDims, vector<int>& newShape, int depth, int* newData) {
    if (depth == shape.size()) {
      vector<int> newDimIdx;

      for (int i = 0; i < tDims.size(); i++) {
        newDimIdx.push_back(dimIdx[tDims[i]]);
      }

      int newDataIdx = 0;
      int factor = 1;
      for (int i = newShape.size() - 1; i >= 0; i--) {
        newDataIdx += newDimIdx[i] * factor;
        factor *= newShape[i];
      }

      int oldDataIdx = 0;
      factor = 1;
      for (int i = shape.size() - 1; i >= 0; i--) {
        oldDataIdx += dimIdx[i] * factor;
        factor *= shape[i];
      }

      newData[newDataIdx] = data[oldDataIdx];

      return;
    }

    for (int i = 0; i < shape[depth]; i++) {
      dimIdx.push_back(i);

      transpose1(dimIdx, tDims, newShape, depth + 1, newData);

      dimIdx.pop_back();
    }
  }

  void transpose(vector<int> tDims) {
    vector<int> newShape;

    assert(tDims.size() == shape.size());

    for (int i = 0; i < tDims.size(); i++) {
      assert(tDims[i] < shape.size());
      newShape.push_back(shape[tDims[i]]);
    }

    int* newData = (int*)malloc(dataLen * sizeof(int));

    transpose1({}, tDims, newShape, 0, newData);

    shape = newShape;

    free(data);

    data = newData;
  }

  void dump1(int depth, int& dataIdx) {
    if (depth == shape.size()) {
      cout << data[dataIdx++] << ' ';
      return;
    }

    cout << '[';
    for (int i = 0; i < shape[depth]; i++) {
      dump1(depth + 1, dataIdx);
    }
    cout << ']' << endl;
  }
  void dump() {
    cout << "shape:";
    for (int i = 0; i < shape.size(); i++) {
      cout << shape[i] << ' ';
    }

    cout << endl;

    int idx = 0;
    dump1(0, idx);
  }

  void dumpFlat() {
    cout << "length:" << dataLen << endl;
    for (int i = 0; i < dataLen; i++) {
      cout << data[i] << ' ';
    }
    cout << endl;
  }

  void clear() { memset(data, 0, sizeof(int) * dataLen); }

  int* data;
  int dataLen;
  vector<int> shape;
};

Tensor* gemm(Tensor* t1, Tensor* t2);
void* gemm(void* data1, vector<int> shape1, void* data2, vector<int> shape2);

#endif  // VTA_API_H_
