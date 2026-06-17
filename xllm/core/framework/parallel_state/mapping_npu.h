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

#pragma once

#include "core/common/macros.h"
#include "core/util/json_reader.h"

namespace xllm {

struct ParallelInfo {
  ParallelInfo(int32_t group_size = 1,
               int32_t num_group = -1,
               std::vector<std::vector<int32_t>> rank_per_group = {},
               int32_t current_group_id = {},
               int32_t rank = 0,
               const std::string& domain = "",
               int32_t buffer_size = 128,
               const std::string& backend = "")
      : group_size_(group_size),
        num_group_(num_group),
        rank_per_group_(rank_per_group),
        current_group_id_(current_group_id),
        rank_(rank),
        domain_(domain),
        buffer_size_(buffer_size),
        backend_(backend) {};

  ParallelInfo(const ParallelInfo& other)
      : group_size_(other.group_size()),
        num_group_(other.num_group()),
        rank_per_group_(other.rank_per_group()),
        current_group_id_(other.current_group_id()),
        rank_(other.rank()),
        domain_(other.domain()),
        buffer_size_(other.buffer_size()),
        backend_(other.backend()) {};

  nlohmann::json to_json(int32_t buffer_offset = 0) {
    nlohmann::json data;
    data["group_size"] = group_size_;
    // data["num_group"] = num_group_;
    data["rankIds"] = rank_per_group_[current_group_id_];
    data["groupId"] = current_group_id_;
    data["rank"] = rank_;
    // data["domain"] = domain_;
    data["bufferSize"] = buffer_size_ + buffer_offset;
    data["backend"] = backend_;
    return data;
  }

  // group size
  PROPERTY(int32_t, group_size) = 1;

  // size of current group
  PROPERTY(int32_t, num_group) = -1;

  // id of current group
  PROPERTY(int32_t, current_group_id) = -1;

  PROPERTY(std::vector<std::vector<int32_t>>, rank_per_group) = {};

  // rank of current process
  PROPERTY(int32_t, rank) = 0;

  // domain of current process
  PROPERTY(std::string, domain);

  // buffer size
  PROPERTY(int32_t, buffer_size) = 128;

  // backend : lccl / hccl
  PROPERTY(std::string, backend);
};

class MappingNPU final {
 public:
  struct Options {
    PROPERTY(int32_t, num_lccl_comm_shards) = 1;
    PROPERTY(int32_t, lccl_comm_shard_id) = 0;
    // dp size
    PROPERTY(int32_t, dp_size) = -1;
    // tp size
    PROPERTY(int32_t, tp_size) = -1;
    // moe tp size
    PROPERTY(int32_t, moe_tp_size) = -1;
    // moe ep size
    PROPERTY(int32_t, moe_ep_size) = -1;
    // pp size (dont support now)
    PROPERTY(int32_t, pp_size) = -1;
    // sp size (dont support now)
    PROPERTY(int32_t, sp_size) = -1;
    // cp size
    PROPERTY(int32_t, cp_size) = -1;
    // kv split size: number of ranks across which KV cache is sharded.
    // -1 means "follow cp_size" (legacy: KV split width == CP size).
    // 1 means no KV split (each CP rank holds a full KV replica, ATB prefix
    // AllGather can be skipped). Other K must divide cp_size.
    PROPERTY(int32_t, kv_split_size) = -1;
  };

  MappingNPU(std::string rank_table_file,
             const int32_t world_size,
             const int32_t rank,
             const Options& options);

  int32_t get_num_nodes();

  void parse_parallel_info();

  void validate();

  void get_tp_group(ParallelInfo& parallel_info);

  void get_dp_group(ParallelInfo& parallel_info);

  void get_cp_group(ParallelInfo& parallel_info);

  // Build the KV-split parallel group. Layout reuses get_dp_group's stride
  // scheme so that when `attn_kv_split_.group_size() == attn_cp_.group_size()`
  // the resulting rank_ids match attn_cp_ exactly (and the ATB
  // ExternalCommManager de-duplicates the HCCL commDomain). When kv_split_size
  // == 1 each rank is its own group (no AllGather participants), matching the
  // "full-replica / skip prefix AllGather" mode.
  void get_kv_split_group(ParallelInfo& parallel_info);

  void get_domain(ParallelInfo& src,
                  ParallelInfo& dst,
                  const int32_t start_idx);

  std::tuple<int32_t, int32_t> get_current_group_id(
      const std::vector<std::vector<int>>& rank_per_group,
      int target_rank_id);

  nlohmann::json to_json();

 private:
  // HACK: Static counter for auto-assigning buffer_offset in multi-model
  // scenarios. Increments for each MappingNPU instance. This is a hack and
  // should be refactored later.
  static int32_t s_buffer_offset_counter_;
  int32_t buffer_offset_ = 0;
  Options options_;
  std::string rank_table_file_;
  int32_t num_nodes_;
  int32_t world_size_ = 0;
  int32_t rank_ = 0;
  int32_t local_world_size_ = 0;
  ParallelInfo word_embed_tp_ = ParallelInfo();
  ParallelInfo word_embed_dp_ = ParallelInfo();
  ParallelInfo attn_tp_ = ParallelInfo();
  ParallelInfo attn_o_proj_tp_ = ParallelInfo();
  ParallelInfo attn_dp_ = ParallelInfo();
  ParallelInfo attn_o_proj_dp_ = ParallelInfo();
  ParallelInfo mlp_tp_ = ParallelInfo();
  ParallelInfo mlp_dp_ = ParallelInfo();
  ParallelInfo moe_tp_ = ParallelInfo();
  ParallelInfo moe_ep_ = ParallelInfo();
  ParallelInfo lm_head_tp_ = ParallelInfo();
  ParallelInfo lm_head_dp_ = ParallelInfo();
  ParallelInfo attn_inner_sp_ = ParallelInfo();
  ParallelInfo attn_cp_ = ParallelInfo();
  ParallelInfo attn_kv_split_ = ParallelInfo();

  int32_t lccl_comm_domain_lower_bound_;
  int32_t lccl_comm_domain_upper_bound_;
};
}  // namespace xllm
