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

#include "dit_mapping.h"

#include <glog/logging.h>

namespace xllm {

DiTMapping::DiTMapping(const int32_t world_size,
                       const int32_t rank,
                       const Options& options)
    : rank_(rank), options_(options), world_size_(world_size) {
  tp_.backend("hccl");
  sp_.backend("hccl");
  cfg_.backend("hccl");
  dp_.backend("hccl");
  vae_.backend("hccl");
  parse_parallel_info();
  validate();
  RankGenerator rank_generator(world_size);
  std::vector<int32_t> group_ranks = {
      tp_.group_size(), sp_.group_size(), cfg_.group_size(), dp_.group_size()};
  std::vector<std::string> group_order = {"tp", "sp", "cfg", "dp"};
  auto ranks_mapping =
      rank_generator.get_ranks_mapping(group_ranks, group_order);

  set_group_by_type(tp_, "tp", ranks_mapping.at("tp"));
  set_group_by_type(sp_, "sp", ranks_mapping.at("sp"));
  set_group_by_type(cfg_, "cfg", ranks_mapping.at("cfg"));
  set_group_by_type(dp_, "dp", ranks_mapping.at("dp"));

  std::vector<int32_t> vae_group_ranks = {vae_.group_size()};
  std::vector<std::string> vae_group_order = {"vae"};
  auto ranks_mapping_vae =
      rank_generator.get_ranks_mapping(vae_group_ranks, vae_group_order);
  set_group_by_type(vae_, "vae", ranks_mapping_vae.at("vae"));
}

void DiTMapping::parse_parallel_info() {
  if (options_.dit_tp_size() != -1) {
    tp_.group_size(options_.dit_tp_size());
  }
  if (options_.dit_sp_size() != -1) {
    sp_.group_size(options_.dit_sp_size());
  }
  if (options_.dit_cfg_size() != -1) {
    cfg_.group_size(options_.dit_cfg_size());
  }
  if (options_.dit_dp_size() != -1) {
    dp_.group_size(options_.dit_dp_size());
  }
  if (options_.dit_vae_size() != -1) {
    vae_.group_size(options_.dit_vae_size());
  }
}

void DiTMapping::validate() {
  CHECK(cfg_.group_size() * tp_.group_size() * sp_.group_size() *
            dp_.group_size() ==
        world_size_)
      << "World size must equal to cfg_size * tp_size * sp_size. "
         "cfg_size is " +
             std::to_string(cfg_.group_size()) +
             ". "
             "tp_size is " +
             std::to_string(tp_.group_size()) +
             ". "
             "sp_size is " +
             std::to_string(sp_.group_size()) +
             ". "
             "dp_size is " +
             std::to_string(dp_.group_size()) +
             ". "
             "world_size is " +
             std::to_string(world_size_) +
             ". "
             "Please check `cfg`, `tp`, `sp`, `dp` and `world_size`.";

  CHECK(cfg_.group_size() <= 2 && cfg_.group_size() >= 1)
      << "cfg_size must less than 2 "
         "cfg_size is " +
             std::to_string(cfg_.group_size()) + ". Please check `cfg` .";

  CHECK(vae_.group_size() <= world_size_)
      << "vae_size could not greater than world_size. "
         "vae_size is " +
             std::to_string(vae_.group_size()) + ", world_size is " +
             std::to_string(world_size_) +
             ". Please check `vae` and 'world_size'.";

  CHECK(world_size_ % vae_.group_size() == 0)
      << "world_size could not be divided by vae_size. "
         "vae_size is " +
             std::to_string(vae_.group_size()) + ", world_size is " +
             std::to_string(world_size_) +
             ". Please check `vae` and 'world_size'.";
}

void DiTMapping::set_group_by_type(
    ParallelInfo& parallel_info,
    const std::string& group_type,
    const std::vector<std::vector<int32_t>>& rank_per_group) {
  parallel_info.rank_per_group(rank_per_group);
  auto group_size = rank_per_group[0].size();
  parallel_info.num_group(world_size_ / group_size);
  auto [current_group_id, local_rank] =
      get_current_group_id(rank_per_group, rank_);
  CHECK(current_group_id >= 0 && local_rank >= 0)
      << "Failed to get current group id : " << current_group_id
      << " local_rank " << local_rank;
  parallel_info.current_group_id(current_group_id);
  parallel_info.rank(local_rank);
}

std::tuple<int32_t, int32_t> DiTMapping::get_current_group_id(
    const std::vector<std::vector<int32_t>>& rank_per_group,
    int32_t target_rank_id) {
  for (int32_t idx = 0; idx < rank_per_group.size(); ++idx) {
    const auto& group = rank_per_group[idx];
    auto it = std::find(group.begin(), group.end(), target_rank_id);
    if (it != group.end()) {
      return std::make_tuple(idx, std::distance(group.begin(), it));
    }
  }
  return std::make_tuple(-1, -1);
}

const ParallelInfo& DiTMapping::get_parallel_info(
    const std::string& group_type) const {
  if (group_type == "tp") {
    return tp_;
  } else if (group_type == "sp") {
    return sp_;
  } else if (group_type == "cfg") {
    return cfg_;
  } else if (group_type == "dp") {
    return dp_;
  } else if (group_type == "vae") {
    return vae_;
  } else {
    LOG(FATAL) << "get unexpected group_type: " << group_type;
  }
}

nlohmann::json DiTMapping::to_json() {
  nlohmann::json data;

  data["SpSize"] = options_.dit_sp_size();
  data["TpSize"] = options_.dit_tp_size();
  data["CfgSize"] = options_.dit_cfg_size();
  data["worldSize"] = world_size_;
  data["rank"] = rank_;
  data["sp"] = sp_.to_json();
  data["tp"] = tp_.to_json();
  data["cfg"] = cfg_.to_json();
  data["dp"] = dp_.to_json();
  data["vae"] = vae_.to_json();
  return data;
}

}  // namespace xllm
