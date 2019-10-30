// Copyright 2019, Intel Corporation
#pragma once

#include <string>

#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace util {

typedef std::vector<size_t> ModelKey;

// Hash function of ModelKey
struct ModelKeyHash {
  size_t operator()(const ModelKey& key) const {
    size_t sum = 0;
    for (const auto& elem : key) {
      sum += elem;
    }
    return sum;
  }
};

// Generate the features of the tiled block
std::vector<size_t> TiledBlockFeatures(stripe::Block *block, ModelKey* model_key);

// Generate the feature strings of the tiled block
std::string TiledBlockFeaturesStr(stripe::Block *block);

// Extract ModelKey from a string
ModelKey ExtractModelKeyFromString(const std::string& str);

// Extract ModelKey from a block
ModelKey ExtractModelKeyFromBlock(stripe::Block *block);

}  // namespace util
}  // namespace tile
}  // namespace vertexai
