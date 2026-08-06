/* Copyright 2025-2026 The xLLM Authors.

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

#pragma once

#include <glog/logging.h>

#include "core/common/macros.h"
#include "core/util/json_reader.h"
#include "mapping_npu.h"
#include "rank_generator.h"
namespace xllm {

class DiTMapping final {
 public:
  struct Options {
    // cfg size
    PROPERTY(int32_t, dit_cfg_size) = -1;
    // tp size
    PROPERTY(int32_t, dit_tp_size) = -1;
    // sp size
    PROPERTY(int32_t, dit_sp_size) = -1;
    // dp size
    PROPERTY(int32_t, dit_dp_size) = -1;
    // vae size
    PROPERTY(int32_t, dit_vae_size) = -1;
  };

  DiTMapping(const int32_t world_size,
             const int32_t rank,
             const Options& options);

  int32_t get_num_nodes();

  void parse_parallel_info();

  void validate();

  void set_group_by_type(
      ParallelInfo& parallel_info,
      const std::string& group_type,
      const std::vector<std::vector<int32_t>>& rank_per_group);

  std::tuple<int32_t, int32_t> get_current_group_id(
      const std::vector<std::vector<int>>& rank_per_group,
      int target_rank_id);

  const ParallelInfo& get_parallel_info(const std::string& group_type) const;

  nlohmann::json to_json();

 private:
  Options options_;
  int32_t num_nodes_;
  int32_t world_size_ = 0;
  int32_t rank_ = 0;
  int32_t local_world_size_ = 0;
  ParallelInfo sp_ = ParallelInfo();
  ParallelInfo tp_ = ParallelInfo();
  ParallelInfo cfg_ = ParallelInfo();
  ParallelInfo dp_ = ParallelInfo();
  ParallelInfo vae_ = ParallelInfo();
};
}  // namespace xllm
