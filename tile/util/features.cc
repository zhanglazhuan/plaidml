// Copyright 2019, Intel Corporation

#include <unordered_map>
#include <vector>
#include <set>

#include "tile/util/features.h"

namespace vertexai {
namespace tile {
namespace util {

using namespace stripe;  // NOLINT

// Generate the feature of a refinement
std::vector<size_t> RefFeatures(Block* block, const Refinement& ref) {
  size_t n_dim = ref.access.size();
  std::vector<size_t> features(n_dim * 3 + 2);
  std::vector<Extent> extents = ref.Extents(block->idxs);
  size_t tot_size = 1;
  for (size_t i = 0, count = 2; i< n_dim; ++i, count += 3) {
    features[count] = ref.interior_shape.dims[i].stride;
    features[count + 1] = extents[i].min;
    features[count + 2] = extents[i].max;
    tot_size *= (extents[i].max - extents[i].min + 1);
  }
  features[0] = tot_size;
  features[1] = block->constraints.size();
  return features;
}

// Generate the feature string of a refinement
std::string RefFeaturesStr(Block* block, const Refinement& ref) {
  std::vector<size_t> ref_features = RefFeatures(block, ref);
  if (ref_features.empty()) {
    return "";
  }
  std::string features_str = std::to_string(ref_features[0]);
  for (size_t i = 1; i < ref_features.size(); ++i) {
    features_str = features_str + " " + std::to_string(ref_features[i]);
  }
  return features_str;
}

// Extract ModelKey from a string
ModelKey ExtractModelKeyFromString(const std::string& str) {
  ModelKey result;
  std::istringstream tokenStream(str);
  std::string token;
  while (std::getline(tokenStream, token, ' ')) {
    result.push_back(std::stoi(token));
  }
  return result;
}

// Extract ModelKey from a block
ModelKey ExtractModelKeyFromBlock(Block *block) {
  ModelKey model_key;
  for (const auto& ref : block->refs) {
    if (!ref.AllZeroAccess()) {
      model_key.push_back(ref.access.size());
    }
  }
  if (model_key.size() != 1) {
    throw std::runtime_error("There must be only one refinement with non-zero accesses.");
  }
  return model_key;
}

std::vector<size_t> TiledBlockFeatures(Block *block, ModelKey* model_key) {
  std::vector<size_t> features;
  size_t n_dim = 0;
  size_t dir;
  // Get features
  for (const auto& ref : block->refs) {
    if (!ref.AllZeroAccess()) {
      if (features.empty()) {
        features = RefFeatures(block, ref);
        n_dim = ref.access.size();
        dir = static_cast<size_t>(ref.dir);
      }
      else {
        throw std::runtime_error("There must be only one non-zero access refinement in a block.");
      }
    }
  }
  if (features.empty()) {
    throw std::runtime_error("There is no non-zero access refinement in the block.");
  }
  if (model_key) {
    // If model_key is not null, generate model key
    model_key->clear();
    model_key->push_back(dir);
    model_key->push_back(n_dim);
  }
  return features;
}

std::string TiledBlockFeaturesStr(Block *block) {
  ModelKey model_key;
  std::vector<size_t> features = TiledBlockFeatures(block, &model_key);
  if (features.empty()) {
    return "";
  }
  std::string features_str = std::to_string(model_key[0]);
  for (size_t i = 1; i < model_key.size(); ++i) {
    features_str = features_str + " " + std::to_string(model_key[i]);
  }
  for (size_t i = 0; i < features.size(); ++i) {
    features_str = features_str + " " + std::to_string(features[i]);
  }
  return features_str;
}

}  // namespace util
}  // namespace tile
}  // namespace vertexai
