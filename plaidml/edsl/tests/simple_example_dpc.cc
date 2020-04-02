// Copyright 2020 Intel Corporation

#include <map>
#include <iostream>
#include <variant>

#include "plaidml/edsl/autodiff.h"
#include "plaidml/edsl/edsl.h"
#include "plaidml/exec/exec.h"
#include "pmlc/util/env.h"

using namespace plaidml;
using namespace plaidml::edsl;


// 2D Matrix Multiply
template <typename T>
tensor<T, 2> MatMul(const tensor<T, 2>& X, const tensor<T, 2>& Y) {
  tidx i, j, k;
  auto R = tensor<T, 2>(X.d0, Y.d2);
  R(i, j) += X(i, k) * Y(k, j);
  return R;
}


int main(int argc, char **argv) {
  // Initialization moves into DPC++ context

  std::vector<float> input = {
      1.0f, 2.0f, 3.0f,  //
      4.0f, 5.0f, 6.0f,  //
      7.0f, 8.0f, 9.0f,  //
  };

  std::vector<float> expected = {
      30.0f,  36.0f,  42.0f,   //
      66.0f,  81.0f,  96.0f,   //
      102.0f, 126.0f, 150.0f,  //
  };

  queue q;

  auto A = tensor<float, 2>({3, 3}, input.data());
  auto B = tensor<float, 2>({3, 3}, input.data());
  auto C = tensor<float, 2>();
  
  kernel_set k = q.compile([] {
    auto C = MatMul(A, B);
    return {C};
  })

  q.sumit(k)
  
  
  // Compile the EDSL matmul program that produces output C into
  // target-independent MLIR (tile dialect)
  auto program = ProgramBuilder("matmul", {C}).compile();
  std::cout << "Tile dialect code: " << std::endl << program << std::endl;
  
  // Lower the code via the path expected by the configured device
  auto binder = exec::Binder(program);
  auto executable = binder.compile();

  // Bind the placeholder inputs to actual data
  binder.input(A).copy_from(input.data());
  binder.input(B).copy_from(input.data());

  // Async launch of program, will compute output C in a separate thread
  executable->run();

  // mmap'ing the output implies waiting for the computation to complete
  auto outview = reinterpret_cast<float*>(binder.output(C).mmap_current().data());
  for (int i = 0; i < expected.size(); i++) {
    auto val = outview[i];
    auto exp = expected[i];
    if (val != exp) return -1;
    std::cout << "val: " << val << " exp: " << exp << std::endl;
  }
  return 0;
}