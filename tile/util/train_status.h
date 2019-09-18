// Copyright 2019, Intel Corporation
#pragma once

#include <string>

#include "tile/stripe/stripe.h"
#include "tile/util/train_const.h"

namespace vertexai {
namespace tile {
namespace util {

class TrainStatus {
public:
  explicit TrainStatus(const std::string& train_dir);
  void SetLastTestedTile(int id);
  int LastTestedTile();
  void SetFirstGeneratedTile(int id);
  int FirstGeneratedTile();
  void StartBuildTiles();
  void BuiltOneTile(const std::string& feature);
  int LastBuiltTile();
  void LoadFailedTiles();
  void AddFailedTile(int id);
  bool IsFailedTile(int id);

private:
  std::string last_tested_file_;
  std::string first_generated_file_;
  std::string failed_tile_file_;
  std::string last_built_file_;
  std::unordered_set<int> failed_tiles;
};

}  // namespace util
}  // namespace tile
}  // namespace vertexai
