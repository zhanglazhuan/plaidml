// Copyright 2018, Intel Corporation

#pragma once

#include "tile/codegen/alias.h"
#include "tile/codegen/codegen.pb.h"
#include "tile/codegen/compile_pass.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {

class GeneratePlansPass final : public CompilePass {
 public:
  explicit GeneratePlansPass(const proto::GeneratePlansPass& options) : options_{options} {}
  void Apply(CompilerState* state) const final;

 private:
  proto::GeneratePlansPass options_;
};

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
