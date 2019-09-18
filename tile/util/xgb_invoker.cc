#include "tile/util/xgb_invoker.h"
#include "tile/util/features.h"

namespace vertexai {
namespace tile {
namespace util {

#define safe_xgboost(call) {                                               \
int err = (call);                                                          \
if (err != 0) {                                                            \
  throw std::runtime_error(str(boost::format("%s:%d: error in %s: %s\n") % __FILE__ % __LINE__ % #call % XGBGetLastError()));                      \
}                                                                          \
}

XGBInvoker::XGBInvoker(const std::string& model_list_fn, const std::string& model_dir) {
  std::ios_base::openmode mode = std::ios_base::in;
  std::ifstream ifs(model_dir + "/" + model_list_fn, mode);
  if (ifs.fail()) {
    throw std::runtime_error(str(boost::format("Unable to open file \"%1%\"") % model_list_fn));
  }
  std::string line;
  while (std::getline(ifs, line)) {
    ModelKey model_key = ExtractModelKeyFromString(line);
    std::string model_fn;
    std::getline(ifs, model_fn);
    BoosterHandle booster;
    safe_xgboost(XGBoosterCreate(0, 0, &booster));
    safe_xgboost(XGBoosterLoadModel(booster, (model_dir + "/" + model_fn).c_str()));
    boosters_.emplace(model_key, booster);
  }
}

XGBInvoker::~XGBInvoker() {
  for (auto& it : boosters_) {
    XGBoosterFree(it.second);
  }
}

bool XGBInvoker::ModelKeyExists(const ModelKey& model_key) {
  return boosters_.find(model_key) != boosters_.end();
}

const float* XGBInvoker::Predict(const ModelKey& model_key,
                           const float* features,
                           size_t n_rows,
                           size_t n_columns) {
  auto it = boosters_.find(model_key);
  if (it == boosters_.end()) {
    return nullptr;
  }
  BoosterHandle& booster = it->second;
  const float* result = NULL;
  size_t len = 0;
  DMatrixHandle test_data;
  safe_xgboost(XGDMatrixCreateFromMat(features, n_rows, n_columns, 0, &test_data));
  safe_xgboost(XGBoosterPredict(booster, test_data, 0, 0, &len, &result)); 
  safe_xgboost(XGDMatrixFree(test_data));
  return result;
}

}  // namespace util
}  // namespace tile
}  // namespace vertexai
