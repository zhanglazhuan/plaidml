#include "../include/vta/vta_api.h"

int main() {
  Tensor* inp = new Tensor({2, 16});
  Tensor* wgt = new Tensor({16, 16});
  // Tensor *inp = new Tensor({2, 96});
  // Tensor *wgt = new Tensor({48, 96});

  // Tensor *inp = new Tensor({2, 5});
  // Tensor *wgt = new Tensor({4, 5});

  // inp->dump();
  gemm(inp, wgt);
}
