// Copyright 2019, Intel Corporation

#include <unordered_map>
#include <vector>
#include <set>

#include "tile/util/features.h"

namespace vertexai {
namespace tile {
namespace util {

using namespace stripe;  // NOLINT

// Return the ordered index in block
// It currently does not change the order in the block
std::vector<std::string> OrderedIndex(Block* block) {
  std::vector<std::string> sorted_idx;
  // Collect information of index
  for (const auto& idx : block->idxs) {
    if (idx.affine == Affine()) {
      sorted_idx.push_back(idx.name);
    }
  }
  return sorted_idx;
}

// Return the ordered refinements in block
std::vector<std::string> OrderedRefinements(Block* block) {
  struct RefInfo {
    size_t n_idxs; // # index in the accesses
  };
  std::vector<std::string> sorted_refs;
  std::unordered_map<std::string, RefInfo> ref_info_map;
  // Collect information of refinements
  for (const auto& ref : block->refs) {
    auto access = ref.FlatAccess();
    auto acc_map = access.getMap();
    RefInfo ref_info = {acc_map.size()};
    ref_info_map.emplace(ref.into(), ref_info);
    sorted_refs.push_back(ref.into());
  }
  // Sort
  std::sort(sorted_refs.begin(), sorted_refs.end(),
    [ref_info_map](const std::string& ref0, const std::string& ref1) {
      auto& ri0 = ref_info_map.at(ref0);
      auto& ri1 = ref_info_map.at(ref1);
      return ri0.n_idxs < ri1.n_idxs;
    }
  );
  return sorted_refs;
}

// Generate the feature of a refinement
std::vector<size_t> RefFeatures(Block* block, const Refinement& ref,
                                const std::vector<std::string>& sorted_idxs) {
  std::vector<size_t> features;
  auto access = ref.FlatAccess();
  auto acc_map = access.getMap();
  for (const auto& idx_name : sorted_idxs) {
    Index* idx = block->idx_by_name(idx_name);
    if (idx->affine == Affine()) {
      const auto& it = acc_map.find(idx->name);
      if (it == acc_map.end()) {
        features.push_back(0);
      }
      else {
        features.push_back(it->second);
      }
    }
  }
  return features;
}

// Generate the feature string of a refinement
std::string RefFeaturesStr(Block* block, const Refinement& ref,
                           const std::vector<std::string>& sorted_idxs) {
  std::vector<size_t> ref_features = RefFeatures(block, ref, sorted_idxs);
  if (ref_features.empty()) {
    return "";
  }
  std::string features_str = std::to_string(ref_features[0]);
  for (size_t i = 1; i < ref_features.size(); ++i) {
    features_str = features_str + " " + std::to_string(ref_features[i]);
  }
  return features_str;
}

// Generate the feature of a block
std::vector<size_t> BlockFeatures(Block* block,
                                  const std::vector<std::string>& sorted_idxs,
                                  const std::vector<std::string>& sorted_refs) {
  std::vector<size_t> features;
  // Index features
  std::string idx_features;
  for (const auto& idx_name : sorted_idxs) {
    Index* idx = block->idx_by_name(idx_name);
    if (idx->affine == Affine()) { 
      features.push_back(idx->range);
    }
  }
  // Refinement features
  for (auto ref_into : sorted_refs) {
    auto ref = block->ref_by_into(ref_into);
    if (ref->dir == RefDir::In) {
      auto in_feature = RefFeatures(block, *ref, sorted_idxs);
      features.insert(features.end(), in_feature.begin(), in_feature.end());
    }
  }
  for (auto ref_into : sorted_refs) {
    auto ref = block->ref_by_into(ref_into);
    if (IsWriteDir(ref->dir)) {
      auto out_feature = RefFeatures(block, *ref, sorted_idxs);
      features.insert(features.end(), out_feature.begin(), out_feature.end());
    }
  }
  return features;
}

// Generate the feature string of a block
std::string BlockFeaturesStr(Block* block,
                             const std::vector<std::string>& sorted_idxs,
                             const std::vector<std::string>& sorted_refs) {
  std::string features_str;
  auto features = BlockFeatures(block, sorted_idxs, sorted_refs);
  size_t n_refs = block->refs.size();
  size_t n_idxs = 0;
  for (const auto& idx : block->idxs) {
    if (idx.affine == Affine()) {
      ++n_idxs;
    }
  }
  // First n_idxs elements are index
  for (size_t i = 0; i < n_idxs; ++i) {
    features_str = features_str + std::to_string(features[i]) + " ";
  }
  features_str[features_str.size() - 1] = ';';
  // Then refinements
  for (size_t i = 0; i < n_refs; ++i) {
    for (size_t j = 0; j < n_idxs; ++j) {
      size_t pos = (i + 1) * n_idxs + j;
      features_str = features_str + std::to_string(features[pos]) + " ";
    }
    features_str[features_str.size() - 1] = ';';
  }
  features_str[features_str.size() - 1] = '.';
  return features_str;
}

// Generate the feature of the stmts' kinds
std::vector<size_t> StmtFeatures(Block* block) {
  std::vector<size_t> stmt_features;
  for (const auto& stmt : block->stmts) {
    stmt_features.push_back(static_cast<int>(stmt->kind()));
  }
  return stmt_features;
}

// Generate the feature string of the stmts' kinds
std::string StmtFeaturesStr(Block* block) {
  std::vector<size_t> stmt_features = StmtFeatures(block);
  if (stmt_features.empty()) {
    return "";
  }
  std::string features_str = std::to_string(stmt_features[0]);
  for (size_t i = 1; i < stmt_features.size(); ++i) {
    features_str = features_str + " " + std::to_string(stmt_features[i]);
  }
  return features_str + '.';
}

// Generate a hash code for a block's features
size_t HashBlockFeatures(stripe::Block *block,
                         const std::vector<std::string>& sorted_idxs,
                         const std::vector<std::string>& sorted_refs) {
  std::string str = BlockFeaturesStr(block, sorted_idxs, sorted_refs);
  return std::hash<std::string>{}(str);
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
  ModelKey model_key = StmtFeatures(block);
  model_key.push_back(block->refs.size() + 1); // 1(#index) + #refs
  model_key.push_back(block->idxs.size());;
  return model_key;
}

// Number of feature dimensions according to the model key
size_t NumFeatureDim(const ModelKey& key) {
  size_t last = key.size() - 1;
  return key[last - 1] * key[last];
}

// Number of feature dimensions according to the block
size_t NumFeatureDim(Block *block) {
  return (block->refs.size() + 1) * block->idxs.size();
}

std::vector<size_t> TiledBlockFeatures(Block *block, ModelKey* model_key,
                                       const std::vector<std::string>& sorted_idxs,
                                       const std::vector<std::string>& sorted_refs) {
  // Generate features
  std::vector<size_t> features = BlockFeatures(block, sorted_idxs, sorted_refs);
  auto inner = block->SubBlock(0);
  std::vector<size_t> inner_features = BlockFeatures(inner.get(), sorted_idxs, sorted_refs);
  features.insert(features.end(), inner_features.begin(), inner_features.end());
  if (model_key) {
    // If model_key is not null, generate model key
    *model_key = StmtFeatures(inner.get());
    model_key->push_back(block->refs.size() + 1); // 1(#index) + #refs
    model_key->push_back(block->idxs.size());
  }
  return features;
}

std::vector<size_t> TiledBlockFeatures(Block *block, ModelKey* model_key) {
  std::vector<std::string> sorted_idxs = OrderedIndex(block);
  std::vector<std::string> sorted_refs = OrderedRefinements(block);
  return TiledBlockFeatures(block, model_key, sorted_idxs, sorted_refs);
}

std::string TiledBlockFeaturesStr(Block *block,
                                  const std::vector<std::string>& sorted_idxs,
                                  const std::vector<std::string>& sorted_refs) {
  std::string outer_features = BlockFeaturesStr(block, sorted_idxs, sorted_refs);
  auto inner = block->SubBlock(0);
  std::string inner_features = BlockFeaturesStr(inner.get(), sorted_idxs, sorted_refs);
  std::string stmt_features = StmtFeaturesStr(inner.get());
  return stmt_features + outer_features + inner_features;
}

std::string TiledBlockFeaturesStr(Block *block) {
  std::vector<std::string> sorted_idxs = OrderedIndex(block);
  std::vector<std::string> sorted_refs = OrderedRefinements(block);
  return TiledBlockFeaturesStr(block, sorted_idxs, sorted_refs);
}

}  // namespace util
}  // namespace tile
}  // namespace vertexai
