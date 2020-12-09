// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plaidml_ops.hpp"
#include "plaidml_util.hpp"

#include "ngraph/opsets/opset.hpp"
#include "ngraph/opsets/opset1.hpp"

#include "plaidml/op/op.h"

using namespace plaidml;          // NOLINT[build/namespaces]
using namespace InferenceEngine;  // NOLINT[build/namespaces]

namespace {

template <typename T>
std::vector<T> cast_constant_operand(size_t operand_idx, ngraph::Node* layer) {
  auto ngraph_const =
      std::dynamic_pointer_cast<ngraph::op::Constant>(layer->input_value(operand_idx).get_node_shared_ptr());
  if (ngraph_const) {
    return ngraph_const->cast_vector<T>();
  } else {
    THROW_IE_EXCEPTION << "Dynamically-sized Range operation not currently supported by PlaidML plugin; all of start, "
                          "stop, and step must be Constants.";
  }
}

}  // namespace

namespace PlaidMLPlugin {

static OpRegistration reg("range", [](const Context& ctx) {
  return edsl::make_tuple(ctx.operands.at(0) + ctx.operands.at(1));

  // auto* layer = dynamic_cast<ngraph::opset1::Range*>(ctx.layer);
  // auto type = to_plaidml(layer->get_element_type());
  // if (type != plaidml::DType::FLOAT32) {
  //   THROW_IE_EXCEPTION << "PlaidML plugin currenlty only supports fp32 for Range op";
  // }
  // auto start = cast_constant_operand<float>(0, layer)[0];
  // auto stop = cast_constant_operand<float>(1, layer)[0];
  // auto step = cast_constant_operand<float>(2, layer)[0];
  // std::vector<float> range_data;
  // if (step == 0) {
  //   THROW_IE_EXCEPTION << "Range requires non-zero step value";
  // }
  // if (step > 0) {
  //   float curr_val = start;
  //   while (curr_val < stop) {
  //     range_data.push_back(curr_val);
  //     curr_val += step;
  //   }
  // } else {
  //   float curr_val = start;
  //   while (curr_val > stop) {
  //     range_data.push_back(curr_val);
  //     curr_val += step;
  //   }
  // }
  // std::vector<int64_t> dims(1, range_data.size());
  // TensorShape ts(type, dims);
  // Buffer buffer(ctx.device, ts);
  // buffer.copy_from(range_data.data());
  // return edsl::make_tuple(edsl::Constant(type, buffer, dims, layer->get_friendly_name()));
});

}  // namespace PlaidMLPlugin
