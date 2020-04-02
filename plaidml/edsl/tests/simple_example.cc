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


// 2D Matrix Multiply specified in the tile EDSL
Tensor MatMul(const Tensor& X, const Tensor& Y) {
  // Symbolic names for tensor dimensions
  TensorDim I, J, K;
  TensorIndex i("i"), j("j"), k("k");
  X.bind_dims(I, K);
  Y.bind_dims(K, J);
  auto R = TensorOutput(I, J);
  R(i, j) += X(i, k) * Y(k, j);
  return R;
}


int main(int argc, char **argv) {
  // Initialize modules and select the llvm cpu deice
  plaidml::init(); plaidml::edsl::init(); plaidml::exec::init();
  plaidml::Settings::set("PLAIDML_DEVICE", "llvm_cpu.0");
  plaidml::Settings::set("PLAIDML_TARGET", "llvm_cpu");

  std::vector<float> input = {
      1.0f, 2.0f, 3.0f,  //
      4.0f, 5.0f, 6.0f,  //
      7.0f, 8.0f, 9.0[f,  //
  };

  std::vector<float> expected = {
      30.0f,  36.0f,  42.0f,   //
      66.0f,  81.0f,  96.0f,   //
      102.0f, 126.0f, 150.0f,  //
  };

  auto A = Placeholder(DType::FLOAT32, {3, 3});
  auto B = Placeholder(DType::FLOAT32, {3, 3});
  auto C = MatMul(A, B);
  
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
