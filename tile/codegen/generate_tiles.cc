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
  bool GenerateAllBlocks();
  void GeneratePlanBlocks(const std::vector<size_t>& plan);
  StatementList& BlockList() { return block_list_; }

private:
  Block* target_;
  std::set<std::string> acc_idxs_;
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
  auto accs = target->accumulation_idxs();
  for (const auto& idx : accs) {
    acc_idxs_.insert(idx->name);
  }
}

// Generate all tiled blocks
bool BlockGenerator::GenerateAllBlocks() {
  int last_tested = status_.LastTestedTile();
  int last_built = status_.LastBuiltTile();
  // If last_tested == last_built, it means that last part was successful.
  // So we will generate options_.max_blocks blocks from (last_tested + 1).
  // Otherwise, last part was failed on tile (last_built + 1).
  // If last_built > last_tested + 1, we generate the block in range (last_tested, last_built).
  // Otherwise, we skip last_built.
  int first_block;
  int last_block;
  int blocks_per_plan = target_->refs.size();
  if (last_tested == last_built) {
    first_block = last_tested + 1;
    last_block = last_tested + options_.max_plans() * blocks_per_plan;
  }
  else {
    status_.AddFailedTile((last_built + 1) / blocks_per_plan);
    if (last_built >= last_tested + blocks_per_plan) {
      first_block = last_tested + 1;
      last_block = (last_built + 1) / blocks_per_plan * blocks_per_plan - 1;
    }
    else {
      status_.SetLastTestedTile(last_tested + blocks_per_plan);
      first_block = last_tested + blocks_per_plan + 1;
      last_block = first_block + options_.max_plans() * blocks_per_plan - 1;
    }
  }
  while (status_.IsFailedTile(first_block / blocks_per_plan)) {
    first_block += blocks_per_plan;
  }
  if (first_block > last_block) {
    return false;
  }
  int first_plan = first_block / blocks_per_plan;
  int last_plan = last_block / blocks_per_plan;
  int count = 0;
  // Read plan file and restore plans
  std::string line;
  std::ifstream ifs(plan_file_, std::ios_base::in);
  while (std::getline(ifs, line)) {
    if (first_plan <= count && count <= last_plan) {
      std::stringstream ss(line);
      std::string token;
      std::vector<size_t> plan;
      while (std::getline(ss, token, ' ')) {
        plan.push_back(std::stoi(token));
      }
      GeneratePlanBlocks(plan);
    }
    ++count;
  }
  ifs.close();
  status_.SetFirstGeneratedTile(first_block);
  return true;
};

// Generate the tiled blocks according to the plan
void BlockGenerator::GeneratePlanBlocks(const std::vector<size_t>& plan) {
  for (auto& ref : target_->refs) {
    // We generate a tiled block according to each refinement
    // Clone a new block first
    auto new_block = CloneBlock(*target_);
    // Shrink other refinements
    for (auto& other : new_block->refs) {
      if (other.into() != ref.into()) {
        // Set all zero for the access
        for (auto& acc : other.mut().access) {
          acc = Affine(0);
        }
      }
    }
    // Tile the block according to the plan
    ApplyTile(new_block.get(), plan, false, false, options_.interleave());
    new_block->add_tags(FromProto(options_.outer_set()));
    // Generate the outer plan that splits the accumulation/non-accumulation index
    std::vector<size_t> outer_plan;
    for (auto& idx : new_block->idxs) {
      bool is_acc_idx = acc_idxs_.find(idx.name) != acc_idxs_.end();
      outer_plan.push_back(is_acc_idx ? idx.range : 1);
    }
    ApplyTile(new_block.get(), outer_plan, false, false, false);
    // Reduce the workgroups of the outer block to speed up testing
    size_t workgroups = 1;
    for (auto& idx : new_block->idxs) {
//      if (workgroups * idx.range <= options_.min_workgroups()) {
        workgroups *= idx.range;
//      }
//      else {
//        idx.range = options_.min_workgroups() / workgroups + 1;
//        workgroups *= idx.range;
//      }
    }
    // Reduce the accumulation iterations
    auto middle = new_block->SubBlock(0);
    middle->add_tags(FromProto(options_.middle_set()));
    size_t iterations = 1;
    for (auto& idx : middle->idxs) {
//      if (iterations * idx.range <= options_.min_accumulation()) {
        iterations *= idx.range;
//      }
//      else {
//        idx.range = options_.min_accumulation() / iterations + 1;
//        iterations *= idx.range;
//      }
    }
    // Reduce the unused index in the inner block
    auto inner = middle->SubBlock(0);
    auto flat = ref.FlatAccess();
    std::set<std::string> used_idxs;
    for (auto& kvp : flat.getMap()) {
      if (kvp.first != "") {
        used_idxs.insert(kvp.first);
      }
    }
    for (auto& idx : inner->idxs) {
      if (used_idxs.find(idx.name) == used_idxs.end()) {
        idx.range = 1;
      }
    }
    inner->add_tags(FromProto(options_.inner_set()));
    // Encode features in the comments
    new_block->comments = std::to_string(workgroups) + " " + std::to_string(iterations)
        + " " + FEATURE_HEAD + util::TiledBlockFeaturesStr(inner.get());
    // Add the tiled block into the program
    block_list_.push_back(new_block);
  }
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
  if (bg->GenerateAllBlocks()) {
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
