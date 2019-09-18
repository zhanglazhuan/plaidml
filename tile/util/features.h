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

// Return the ordered index in block
std::vector<std::string> OrderedIndex(stripe::Block* block);

// Return the ordered refinements in block
std::vector<std::string> OrderedRefinements(stripe::Block* block);

// Generate the features of the tiled block
std::vector<size_t> TiledBlockFeatures(stripe::Block *block, ModelKey* model_key,
                                       const std::vector<std::string>& sorted_idxs,
                                       const std::vector<std::string>& sorted_refs);
std::vector<size_t> TiledBlockFeatures(stripe::Block *block, ModelKey* model_key);

// Generate the feature strings of the tiled block
std::string TiledBlockFeaturesStr(stripe::Block *block,
                                  const std::vector<std::string>& sorted_idxs,
                                  const std::vector<std::string>& sorted_refs);
std::string TiledBlockFeaturesStr(stripe::Block *block);

// Generate a hash code for a block's features
size_t HashBlockFeatures(stripe::Block *block,
                         const std::vector<std::string>& sorted_idxs,
                         const std::vector<std::string>& sorted_refs);

// Extract ModelKey from a string
ModelKey ExtractModelKeyFromString(const std::string& str);

// Extract ModelKey from a block
ModelKey ExtractModelKeyFromBlock(stripe::Block *block);

// Number of feature dimensions according to the model key
size_t NumFeatureDim(const ModelKey& key);

// Number of feature dimensions according to the block
size_t NumFeatureDim(stripe::Block *block);

}  // namespace util
}  // namespace tile
}  // namespace vertexai
