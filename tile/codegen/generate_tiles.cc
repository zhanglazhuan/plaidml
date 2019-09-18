// Copyright 2018, Intel Corporation

#include "base/util/any_factory_map.h"
#include "base/util/env.h"
#include "tile/codegen/deps.h"
#include "tile/codegen/generate_tiles.h"
#include "tile/codegen/tile.h"
#include "tile/math/util.h"
#include "tile/stripe/stripe.h"
#include "tile/util/features.h"
#include "tile/util/train_status.h"

namespace vertexai {
namespace tile {
namespace codegen {

using namespace stripe;  // NOLINT
using namespace math;    // NOLINT

class BlockGenerator {

public:
  BlockGenerator(Block* target, const std::string& train_dir,
                 const std::string& plan_file, const proto::GenerateTilesPass& options);
  bool GenerateBlocks();
  void GenerateBlock(const std::vector<size_t>& plan);
  StatementList& BlockList() { return block_list_; }

private:
  Block* target_;
  std::string plan_file_;
  const proto::GenerateTilesPass& options_;
  StatementList block_list_;
  std::vector<std::string> sorted_idxs_;
  std::vector<std::string> sorted_refs_;
  std::string block_hash_;
  util::TrainStatus status_;
};

BlockGenerator::BlockGenerator(Block* target, const std::string& train_dir, 
                               const std::string& plan_file, const proto::GenerateTilesPass& options):
    target_{target}, plan_file_{plan_file}, options_{options}, status_{train_dir} {
  sorted_idxs_ = util::OrderedIndex(target);
  sorted_refs_ = util::OrderedRefinements(target);
  block_hash_ = std::to_string(util::HashBlockFeatures(target, sorted_idxs_, sorted_refs_)) + " ";
}

// Generate all tiled blocks
bool BlockGenerator::GenerateBlocks() {
  int last_tested = status_.LastTestedTile();
  int last_built = status_.LastBuiltTile();
  // If last_tested == last_built, it means that last part was successful.
  // So we will generate options_.max_blocks blocks from (last_tested + 1).
  // Otherwise, last part was failed on tile (last_built + 1).
  // If last_built > last_tested + 1, we generate the block in range (last_tested, last_built).
  // Otherwise, we skip last_built.
  int first;
  int last;
  if (last_tested == last_built) {
    first = last_tested + 1;
    last = last_tested + options_.max_blocks();
  }
  else {
    status_.AddFailedTile(last_built + 1);
    if (last_built > last_tested + 1) {
      first = last_tested + 1;
      last = last_built;
    }
    else {
      first = last_tested + 2;
      last = first + options_.max_blocks() - 1;
    }
  }
  while (first <= last && status_.IsFailedTile(first)) {
    ++first;
  }
  if (first > last) {
    return false;
  }
  int count = 0;
  // Read plan file and restore plans
  std::string line;
  std::ifstream ifs(plan_file_, std::ios_base::in);
  while (std::getline(ifs, line)) {
    if (first <= count && count <= last) {
      std::stringstream ss(line);
      std::string token;
      std::vector<size_t> plan;
      while (std::getline(ss, token, ' ')) {
        plan.push_back(std::stoi(token));
      }
      GenerateBlock(plan);
    }
    ++count;
  }
  ifs.close();
  status_.SetFirstGeneratedTile(first);
  return true;
};

// Generate the tiled block according to the plan
void BlockGenerator::GenerateBlock(const std::vector<size_t>& plan) {
  // Clone a new block first
  auto new_block = CloneBlock(*target_);
  // Tile the block according to the plan
  ApplyTile(new_block.get(), plan, false, false, options_.interleave());
  new_block->add_tags(FromProto(options_.outer_set()));
  auto inner = new_block->SubBlock(0);
  inner->add_tags(FromProto(options_.inner_set()));
  // Encode features in the comments
  new_block->comments = block_hash_ + FEATURE_HEAD + util::TiledBlockFeaturesStr(new_block.get(), sorted_idxs_, sorted_refs_);
  // Add the tiled block into the program
  block_list_.push_back(new_block);
}

void GenerateTiles(const AliasMap& alias_map, Block* block, const proto::GenerateTilesPass& options) {
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

  std::string train_dir = env::Get("CM_TRAIN_DIR");
  std::string plan_path = train_dir + "/" + options.plan_file();
  auto bg = std::make_shared<BlockGenerator>(target.get(), train_dir, plan_path, options);
  if (bg->GenerateBlocks()) {
    // Insert the generated tiled blocks into the program
    block->stmts.insert(stmt_it, bg->BlockList().begin(), bg->BlockList().end());
    // Remove the target block from the program
    block->stmts.erase(stmt_it);
  }
}

void GenerateTilesPass::Apply(CompilerState* state) const {
  auto reqs = stripe::FromProto(options_.reqs());
  RunOnBlocks(state->entry(), reqs,
              [this](const AliasMap& alias_map, stripe::Block* block) {  //
                GenerateTiles(alias_map, block, options_);
              });
}

namespace {
[[gnu::unused]] char reg = []() -> char {
  CompilePassFactory<GenerateTilesPass, proto::GenerateTilesPass>::Register();
  return 0;
}();
}  // namespace
}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
