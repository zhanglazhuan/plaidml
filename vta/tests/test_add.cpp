#include "../include/vta/runtime.h"
#include <iostream>
#include <stdio.h>

#define LEN (64 * 16)
#define SIZE (LEN * 4)

using namespace ::std;

int foo(void* ptr) {
  VTAUopLoopBegin(64, 1, 1, 0);
  VTAUopPush(1, 0, 0, 64, 0, 2, 0, 0);
  VTAUopLoopEnd();

  return 0;
}

void initialize32(int* buf, int size, int val) {
  for (int i = 0; i < size; i++) {
    buf[i] = val;
  }
}

void initialize8(char* buf, int size, int val) {
  for (int i = 0; i < size; i++) {
    buf[i] = val;
  }
}

void dump32(int* buf, int size) {
  for (int i = 0; i < size; i++) {
    cout << buf[i] << ' ';
    if ((i + 1) % 16 == 0) cout << "\n";
  }
}

void dump8(char* buf, int size) {
  for (int i = 0; i < size; i++) {
    printf("%d ", buf[i]);
    if ((i + 1) % 16 == 0) cout << "\n";
  }
}

int main() {
  VTACommandHandle handle = VTATLSCommandHandle();

  int* host1 = (int*)malloc(SIZE);
  int* host2 = (int*)malloc(SIZE);
  int* host3 = (int*)malloc(SIZE);
  int* buf1 = (int*)VTABufferAlloc(SIZE);
  int* buf2 = (int*)VTABufferAlloc(SIZE);
  int* buf3 = (int*)VTABufferAlloc(SIZE);

  initialize32(host1, SIZE, 2);
  initialize32(host2, SIZE, 8);
  initialize32(host3, SIZE, 0);

  VTABufferCopy(host1, 0, buf1, 0, SIZE, 1);
  VTABufferCopy(host2, 0, buf2, 0, SIZE, 1);
  VTABufferCopy(host3, 0, buf3, 0, SIZE, 1);

  VTALoadBuffer2D(handle, buf1, 0, 64, 1, 64, 0, 0, 0, 0, 0, 3);
  VTALoadBuffer2D(handle, buf2, 0, 64, 1, 64, 0, 0, 0, 0, 64, 3);

  void* pKernel = NULL;
  VTAPushALUOp(&pKernel, foo, NULL, 0);

  // VTADepPush(handle, 2, 3);
  // VTADepPop(handle, 2, 3);

  VTAStoreBuffer2D(handle, 0, 4, buf3, 0, 64, 1, 64);
  VTASynchronize(handle, 0x80000000);

  dump32(host3, 64);
  VTABufferCopy(buf3, 0, host3, 0, SIZE, 2);

  dump32(host1, 64);
  dump32(host2, 64);
  dump32(host3, 64);
}
