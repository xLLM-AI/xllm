/* Copyright 2025 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "npu_rank_table_env.h"

#include <glog/logging.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>

namespace xllm {
namespace parallel_state {
namespace {

constexpr char kTorchNpuRankTableFile[] = "RANK_TABLE_FILE";

std::string normalize_rank_table_file(const std::string& rank_tablefile) {
  if (rank_tablefile.empty()) {
    return "";
  }

  const std::filesystem::path rank_table_path(rank_tablefile);
  std::error_code ec;
  CHECK(std::filesystem::exists(rank_table_path, ec) && !ec)
      << "rank_tablefile does not exist: " << rank_tablefile;
  CHECK(!std::filesystem::is_symlink(rank_table_path, ec) && !ec)
      << "rank_tablefile must not be a symlink: " << rank_tablefile;

  const std::filesystem::path absolute_path =
      std::filesystem::absolute(rank_table_path, ec);
  CHECK(!ec) << "Failed to resolve rank_tablefile absolute path: "
             << rank_tablefile << ", error: " << ec.message();
  const std::string normalized_path = absolute_path.lexically_normal().string();
  std::ifstream rank_table_stream(normalized_path);
  CHECK(rank_table_stream.good())
      << "rank_tablefile is not readable: " << normalized_path;
  return normalized_path;
}

}  // namespace

void sync_torch_npu_rank_table_file_env(const std::string& rank_tablefile) {
  if (rank_tablefile.empty()) {
    return;
  }

  static std::mutex env_mutex;
  static std::string configured_rank_tablefile;

  const std::string normalized_rank_tablefile =
      normalize_rank_table_file(rank_tablefile);

  std::lock_guard<std::mutex> lock(env_mutex);
  const char* current = std::getenv(kTorchNpuRankTableFile);
  if (current != nullptr && normalized_rank_tablefile == current &&
      configured_rank_tablefile == normalized_rank_tablefile) {
    return;
  }

  CHECK_EQ(::setenv(kTorchNpuRankTableFile,
                    normalized_rank_tablefile.c_str(),
                    /*overwrite=*/1),
           0)
      << "Failed to set " << kTorchNpuRankTableFile << " for torch_npu.";
  configured_rank_tablefile = normalized_rank_tablefile;
  LOG(INFO) << "Set " << kTorchNpuRankTableFile
            << " for torch_npu ProcessGroupHCCL: " << configured_rank_tablefile;
}

}  // namespace parallel_state
}  // namespace xllm
