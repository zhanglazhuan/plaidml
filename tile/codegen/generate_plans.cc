// Copyright 2018, Intel Corporation

#include <filesystem>

#include "base/util/any_factory_map.h"
#include "base/util/env.h"
#include "base/util/file.h"
#include "tile/codegen/deps.h"
#include "tile/codegen/generate_plans.h"
#include "tile/codegen/tile.h"
#include "tile/math/util.h"
#include "tile/stripe/stripe.h"
#include "tile/util/features.h"

namespace vertexai {
namespace tile {
namespace codegen {

using namespace stripe;  // NOLINT
using namespace math;    // NOLINT

TensorShape MakeOddTile(const TensorShape& tile) {
  TensorShape odd_tile = tile;
  for (size_t i = 0; i < odd_tile.dims.size(); ++i) {
    if ((odd_tile.dims[i].size & 0x1) == 0) {
      ++odd_tile.dims[i].size;
    }
  }
  return odd_tile;
}

class TilePlanGenerator {

public:
  TilePlanGenerator(Block* target, const proto::GeneratePlansPass& options);
  void GeneratePlans(size_t k);
  void Dump(const std::string& fn);
  bool IsValidPlan();

private:
  Block* target_;
  const proto::GeneratePlansPass options_;
  std::set<const Index*> acc_idxs_;
  std::vector<Index*> index_;
  std::vector<uint64_t> plan_;
  std::vector<std::string> sorted_idxs_;
  std::vector<std::string> sorted_refs_;
  std::vector<std::vector<uint64_t>> plan_list_;
};

TilePlanGenerator::TilePlanGenerator(Block* target, const proto::GeneratePlansPass& options):
    target_{target}, options_{options}, acc_idxs_(target->accumulation_idxs(true)) {
  for (auto& idx : target->idxs) {
    if (idx.affine == Affine()) {
      index_.push_back(&idx);
    }
  }
  plan_.resize(index_.size());
  sorted_idxs_ = util::OrderedIndex(target);
  sorted_refs_ = util::OrderedRefinements(target);
};

bool TilePlanGenerator::IsValidPlan() {
  std::map<std::string, size_t> tile_by_name;
  for (size_t i = 0; i < index_.size(); ++i) {
    tile_by_name.emplace(index_[i]->name, plan_[i]);
  }
  // Check memory usage of the inner block
  size_t tot_bytes = 0;
  for (const auto& ref : target_->refs) {
    auto tiled = ref.ApplyTile(tile_by_name);
    int64_t bytes = options_.odd_size() ?
      Codec::Resolve(MakeOddTile(tiled))->byte_size() : Codec::Resolve(tiled)->byte_size();  
    tot_bytes += bytes;
  }
  return tot_bytes <= options_.max_mem_size();
} 

// Dump tile plans to file
void TilePlanGenerator::Dump(const std::string& fn) {
  std::ostringstream plans_str;
  for (const auto& plan : plan_list_) {
    for (const auto elem : plan) {
      plans_str << elem << " ";
    }
    plans_str << "\n";
  }
  WriteFile(fn, plans_str.str());
}

// Recursively generate tile plans
void TilePlanGenerator::GeneratePlans(size_t k) {
  if (k >= index_.size()) {
    // Now we have a new plan
    if (IsValidPlan()) {
      plan_list_.push_back(plan_);
    }
  }
  else {
    size_t range = index_[k]->range;
    if (!options_.acc_idxs() && acc_idxs_.find(index_[k]) != acc_idxs_.end()) {
      // index_[k] is accumulation index
      plan_[k] = range;
      GeneratePlans(k + 1);
    }
    else {
      for (size_t i = 1; i <= range; ++i) {
        if (options_.only_even()) {
          if (range % i == 0) {
            plan_[k] = i;
            GeneratePlans(k + 1);
          }
        }
        else if (options_.only_po2()) {
          if ((range == i) || (math::IsPo2(i) && (range > 5 || range % i == 0))) {
            plan_[k] = i;
            GeneratePlans(k + 1);
          }
        }
        else {
          plan_[k] = i;
          GeneratePlans(k + 1);
        }
      }
    }
  }
}

void GeneratePlans(const AliasMap& alias_map, Block* block, const proto::GeneratePlansPass& options) {
  if (block->name != "main") {
    return;
  }
  std::shared_ptr<Block> target = nullptr;
  // Try to find the target block that is for testing
  auto stmt_it = block->stmts.begin();
  for (; stmt_it != block->stmts.end(); ++stmt_it) {
    auto stmt = *stmt_it;
    if (ZeroBlock(stmt)) {
      continue;
    }
    auto sub = Block::Downcast(stmt);
    if (!sub) {
      continue;
    }
    // Should not have padding tags
    if (sub->has_tag("eltwise_padding")) {
      continue;
    }
    target = sub;
    break;
  }
  if (!target) {
    throw std::runtime_error("Nothing to test.");
  }
  auto generator = std::make_shared<TilePlanGenerator>(target.get(), options);
  generator->GeneratePlans(0);
  std::string out_path = env::Get("CM_TRAIN_DIR") + "/" + options.plan_file();
  generator->Dump(out_path);
}

void GeneratePlansPass::Apply(CompilerState* state) const {
  // Test if this is the first part without plan file
  std::string out_path = env::Get("CM_TRAIN_DIR") + "/" + options_.plan_file();
  if (FileExists(out_path)) {
    // We already have the plan file, do nothing
    return;
  }
  auto reqs = stripe::FromProto(options_.reqs());
  RunOnBlocks(state->entry(), reqs,
              [this](const AliasMap& alias_map, stripe::Block* block) {  //
                GeneratePlans(alias_map, block, options_);
              });
}

namespace {
[[gnu::unused]] char reg = []() -> char {
  CompilePassFactory<GeneratePlansPass, proto::GeneratePlansPass>::Register();
  return 0;
}();
}  // namespace
}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
