#pragma once

#include <string>
#include <unordered_map>
#include <xgboost/c_api.h>

#include "tile/util/features.h"

namespace vertexai {
namespace tile {
namespace util {

class XGBInvoker {

public:
  XGBInvoker(const std::string& model_list_fn, const std::string& model_dir);

  ~XGBInvoker();

  bool ModelKeyExists(const ModelKey& model_key);

  // Important: Do not release the returned array. Otherwise it will cause double free.
  const float* Predict(const ModelKey& model_key, const float* features,
                       size_t n_rows, size_t n_columns);

private:
  std::unordered_map<ModelKey, BoosterHandle, ModelKeyHash> boosters_;
};

}  // namespace util
}  // namespace tile
}  // namespace vertexai
