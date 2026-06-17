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

#include <torch/torch.h>

#include <string>
#include <vector>

#include "state_dict.h"

namespace xllm {

namespace weight {

void load_weight(const StateDict& state_dict,
                 const std::string& name,
                 torch::Tensor& weight,
                 bool& weight_is_loaded);

void load_sharded_weight(const StateDict& state_dict,
                         const std::string& name,
                         int64_t dim,
                         int32_t rank,
                         int32_t world_size,
                         torch::Tensor& weight,
                         bool& weight_is_loaded);

using TensorTransform = std::function<torch::Tensor(const torch::Tensor&)>;

// Lazy parameter registration/update spec used by load_state_dict paths.
struct LazyParameterSpec {
  torch::Tensor* tensor = nullptr;
  bool* is_loaded = nullptr;
  std::string name;
  std::vector<int64_t> sizes;
  torch::TensorOptions options;
};

// Ensure parameter storage exists and matches shape/dtype.
void ensure_parameter_storage(torch::nn::Module* module,
                              const LazyParameterSpec& spec);

// Batch version of ensure_parameter_storage.
void ensure_parameter_storage(torch::nn::Module* module,
                              const std::vector<LazyParameterSpec>& specs);

void load_sharded_weight(const StateDict& state_dict,
                         const std::string& name,
                         TensorTransform transform_func,
                         int64_t dim,
                         int32_t rank,
                         int32_t world_size,
                         torch::Tensor& weight,
                         bool& weight_is_loaded);

void load_fused_weight(const StateDict& state_dict,
                       const std::vector<std::string>& prefixes,
                       const std::string& name,
                       int64_t dim,
                       int32_t rank,
                       int32_t world_size,
                       std::vector<torch::Tensor>& accumulated_tensors,
                       torch::Tensor& weight,
                       bool& weight_is_loaded,
                       int32_t num_kv_head_replicas = 1);

bool load_tensor_list(const StateDict& state_dict,
                      const std::vector<std::string>& prefixes,
                      const std::string& name,
                      int64_t dim,
                      int32_t rank,
                      int32_t world_size,
                      std::vector<torch::Tensor>& accumulated_tensors,
                      int32_t num_kv_head_replicas = 1);

void load_moe_weight(const StateDict& state_dict,
                     const std::string& sub_prefix,
                     const std::string& name,
                     int64_t dim,
                     int64_t rank,
                     int64_t world_size,
                     int64_t start_expert_id,
                     int64_t num_experts_per_rank,
                     std::vector<torch::Tensor>& accumulated_tensors,
                     torch::Tensor& weight,
                     bool& weight_is_loaded);

void load_moe_weight(const StateDict& state_dict,
                     const std::string& sub_prefix,
                     const std::string& name,
                     TensorTransform transform_func,
                     int64_t dim,
                     int64_t rank,
                     int64_t world_size,
                     int64_t start_expert_id,
                     int64_t num_experts_per_rank,
                     std::vector<torch::Tensor>& accumulated_tensors,
                     torch::Tensor& weight,
                     bool& weight_is_loaded);

void load_moe_all_expert_weight(const StateDict& state_dict,
                                const std::string& sub_prefix,
                                const std::string& name,
                                int64_t dim,
                                int64_t rank,
                                int64_t world_size,
                                int64_t num_total_experts,
                                std::vector<torch::Tensor>& accumulated_tensors,
                                torch::Tensor& weight,
                                bool& weight_is_loaded);

void load_moe_fused_weight(const StateDict& state_dict,
                           const std::vector<std::string>& prefixes,
                           const std::string& name,
                           int64_t rank,
                           int64_t world_size,
                           int64_t start_expert_id,
                           int64_t num_experts_per_rank,
                           std::vector<torch::Tensor>& w1_tensors,
                           std::vector<torch::Tensor>& w3_tensors,
                           torch::Tensor& w13,
                           bool& w1_is_loaded,
                           bool& w3_is_loaded,
                           bool& w13_is_loaded);

void load_moe_fused_weight(const StateDict& state_dict,
                           const std::vector<std::string>& prefixes,
                           const std::string& name,
                           TensorTransform transform_func,
                           int64_t rank,
                           int64_t world_size,
                           int64_t start_expert_id,
                           int64_t num_experts_per_rank,
                           std::vector<torch::Tensor>& w1_tensors,
                           std::vector<torch::Tensor>& w3_tensors,
                           torch::Tensor& w13,
                           bool& w1_is_loaded,
                           bool& w3_is_loaded,
                           bool& w13_is_loaded);

// load_merged_weight: assumes all shards have the same size (uniform shard
// sizes)
void load_merged_weight(const StateDict& state_dict,
                        const std::string& name,
                        int64_t dim,
                        int32_t rank,
                        int32_t world_size,
                        int32_t shard_tensor_count,
                        int64_t shard_size,
                        torch::Tensor& weight,
                        bool& weight_is_loaded);

// load_merged_weight_v2: supports variable shard sizes (each shard can have
// different size)
void load_merged_weight_v2(const StateDict& state_dict,
                           const std::string& name,
                           int64_t dim,
                           int32_t rank,
                           int32_t world_size,
                           int32_t shard_tensor_count,
                           const std::vector<int64_t>& shard_sizes,
                           torch::Tensor& weight,
                           bool& weight_is_loaded);

}  // namespace weight

// helper macros for defining and loading weights
#define DEFINE_WEIGHT(name) \
  torch::Tensor name##_;    \
  bool name##_is_loaded_ = false;

#define DEFINE_FUSED_WEIGHT(name) \
  torch::Tensor name##_;          \
  bool name##_is_loaded_ = false; \
  std::vector<torch::Tensor> name##_list_;

#define LOAD_FUSED_WEIGHT(name, dim)      \
  weight::load_fused_weight(state_dict,   \
                            prefixes,     \
                            #name,        \
                            dim,          \
                            rank,         \
                            world_size,   \
                            name##_list_, \
                            name##_,      \
                            name##_is_loaded_);

#define LOAD_QKV_WEIGHT(name, dim, num_kv_head_replicas) \
  weight::load_fused_weight(state_dict,                  \
                            prefixes,                    \
                            #name,                       \
                            dim,                         \
                            rank,                        \
                            world_size,                  \
                            name##_list_,                \
                            name##_,                     \
                            name##_is_loaded_,           \
                            num_kv_head_replicas);

#define LOAD_SHARDED_WEIGHT(name, dim) \
  weight::load_sharded_weight(         \
      state_dict, #name, dim, rank, world_size, name##_, name##_is_loaded_);

#define LOAD_SHARDED_WEIGHT_WITH_TRANSFORM(name, dim) \
  weight::load_sharded_weight(state_dict,             \
                              #name,                  \
                              transform_func,         \
                              dim,                    \
                              rank,                   \
                              world_size,             \
                              name##_,                \
                              name##_is_loaded_);

#define LOAD_WEIGHT(name) \
  weight::load_weight(state_dict, #name, name##_, name##_is_loaded_);

#define LOAD_MOE_WEIGHT(sub_prefix, key, name, dim) \
  weight::load_moe_weight(state_dict,               \
                          sub_prefix,               \
                          key,                      \
                          dim,                      \
                          rank,                     \
                          world_size,               \
                          start_expert_id,          \
                          num_experts_per_rank,     \
                          name##_list_,             \
                          name##_,                  \
                          name##_is_loaded_);

#define LOAD_MOE_WEIGHT_TRANSFORM(sub_prefix, key, name, transform, dim) \
  weight::load_moe_weight(state_dict,                                    \
                          sub_prefix,                                    \
                          key,                                           \
                          transform,                                     \
                          dim,                                           \
                          rank,                                          \
                          world_size,                                    \
                          start_expert_id,                               \
                          num_experts_per_rank,                          \
                          name##_list_,                                  \
                          name##_,                                       \
                          name##_is_loaded_);

#define LOAD_MOE_ALL_EXPERT_WEIGHT(sub_prefix, key, name, dim) \
  weight::load_moe_all_expert_weight(state_dict,               \
                                     sub_prefix,               \
                                     key,                      \
                                     dim,                      \
                                     rank,                     \
                                     world_size,               \
                                     num_total_experts,        \
                                     name##_list_,             \
                                     name##_,                  \
                                     name##_is_loaded_);

#define LOAD_MOE_FUSED_WEIGHT(key, w1, w3, w13)       \
  weight::load_moe_fused_weight(state_dict,           \
                                prefixes,             \
                                key,                  \
                                rank,                 \
                                world_size,           \
                                start_expert_id,      \
                                num_experts_per_rank, \
                                w1##_list_,           \
                                w3##_list_,           \
                                w13##_,               \
                                w1##_is_loaded_,      \
                                w3##_is_loaded_,      \
                                w13##_is_loaded_);

#define LOAD_MOE_FUSED_WEIGHT_TRANSFORM(key, w1, w3, w13, transform) \
  weight::load_moe_fused_weight(state_dict,                          \
                                prefixes,                            \
                                key,                                 \
                                transform,                           \
                                rank,                                \
                                world_size,                          \
                                start_expert_id,                     \
                                num_experts_per_rank,                \
                                w1##_list_,                          \
                                w3##_list_,                          \
                                w13##_,                              \
                                w1##_is_loaded_,                     \
                                w3##_is_loaded_,                     \
                                w13##_is_loaded_);

#define LOAD_MERGED_WEIGHT(name, dim)            \
  weight::load_merged_weight(state_dict,         \
                             #name,              \
                             dim,                \
                             rank,               \
                             world_size,         \
                             shard_tensor_count, \
                             shard_size,         \
                             name##_,            \
                             name##_is_loaded_);

#define LOAD_MERGED_WEIGHT_V2(name, dim)            \
  weight::load_merged_weight_v2(state_dict,         \
                                #name,              \
                                dim,                \
                                rank,               \
                                world_size,         \
                                shard_tensor_count, \
                                shard_sizes,        \
                                name##_,            \
                                name##_is_loaded_);
}  // namespace xllm
