// Copyright 2019, Intel Corporation

#include <filesystem>
#include <unordered_map>
#include <vector>
#include <set>

#include "base/util/file.h"
#include "tile/util/train_const.h"
#include "tile/util/train_status.h"

namespace vertexai {
namespace tile {
namespace util {

using namespace stripe;  // NOLINT

TrainStatus::TrainStatus(const std::string& train_dir) {
  last_tested_file_ = train_dir + '/' + LAST_TESTED_FILE;
  first_generated_file_ = train_dir + '/' + FIRST_GENERATED_FILE;
  failed_tile_file_ = train_dir + '/' + FAILED_TILE_FILE;
  last_built_file_ = train_dir + '/' + LAST_BUILT_FILE;
}

void TrainStatus::SetLastTestedTile(int id) {
  WriteFile(last_tested_file_, std::to_string(id));
}

int TrainStatus::LastTestedTile() {
  if (FileExists(last_tested_file_)) {
    std::string content = ReadFile(last_tested_file_);
    return std::stoi(content);
  }
  return -1;
}

void TrainStatus::SetFirstGeneratedTile(int id) {
  WriteFile(first_generated_file_, std::to_string(id));
}

int TrainStatus::FirstGeneratedTile() {
  std::string content = ReadFile(first_generated_file_);
  return std::stoi(content);
}

void TrainStatus::LoadFailedTiles() {
  std::ifstream ifs(failed_tile_file_, std::ios_base::in);
  std::string line;
  while (std::getline(ifs, line)) {
    failed_tiles.insert(std::stoi(line));
  }
}

void TrainStatus::AddFailedTile(int id) {
  // Append the failed case
  WriteFile(failed_tile_file_, std::to_string(id) + "\n", false, true);
  if (failed_tiles.size() == 0) {
    LoadFailedTiles();
  }
  else {
    failed_tiles.insert(id);
  }
}

bool TrainStatus::IsFailedTile(int id) {
  if (!FileExists(failed_tile_file_)) {
    return false;
  }
  if (failed_tiles.size() == 0) {
    LoadFailedTiles();
  }
  return failed_tiles.find(id) != failed_tiles.end();
}

void TrainStatus::StartBuildTiles() {
  WriteFile(last_built_file_, std::to_string(FirstGeneratedTile() - 1));
}

void TrainStatus::BuiltOneTile(const std::string& feature) {
  size_t pos = feature.find(FEATURE_HEAD);
  if (pos != std::string::npos) {
    WriteFile(last_built_file_, std::to_string(LastBuiltTile() + 1));
  }
}

int TrainStatus::LastBuiltTile() {
  if (FileExists(last_built_file_)) {
    std::string content = ReadFile(last_built_file_);
    return std::stoi(content);
  }
  return LastTestedTile();
}

}  // namespace util
}  // namespace tile
}  // namespace vertexai
