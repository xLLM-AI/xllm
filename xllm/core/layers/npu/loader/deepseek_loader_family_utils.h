/* Copyright 2025-2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <absl/strings/match.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <algorithm>
#include <cctype>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace xllm {
namespace layer {
namespace deepseek_loader_family_utils {

template <typename StateDictLike,
          typename WeightShardMap,
          typename SetKvWeightFn,
          typename ProcessExpertWeightsFn,
          typename ProcessSharedExpertWeightsFn,
          typename ProcessMlpCommonWeightsFn,
          typename ProcessGeneralWeightsFn>
inline void load_state_dict_common(
    const StateDictLike& state_dict,
    const std::unordered_map<std::string, int>& weight_mapping_w8a8,
    const WeightShardMap& weight_shard_w8a8,
    SetKvWeightFn set_kv_weight,
    ProcessExpertWeightsFn process_expert_weights,
    ProcessSharedExpertWeightsFn process_shared_expert_weights,
    ProcessMlpCommonWeightsFn process_mlp_common_weights,
    ProcessGeneralWeightsFn process_general_weights) {
  for (const auto& [name, tensor] : state_dict) {
    if (absl::EndsWith(name, "self_attn.kv_b_proj.weight")) {
      const int index = weight_mapping_w8a8.at(name);
      set_kv_weight(state_dict, name, index, weight_shard_w8a8.at(index));
      continue;
    }

    if (absl::StartsWith(name, "mlp.experts")) {
      process_expert_weights(state_dict, name, tensor);
      continue;
    }

    if (absl::StartsWith(name, "mlp.shared_experts")) {
      process_shared_expert_weights(state_dict, name, tensor);
      continue;
    }

    if (absl::StartsWith(name, "mlp") && !absl::StrContains(name, "gate.")) {
      process_mlp_common_weights(state_dict, name, tensor);
      continue;
    }

    process_general_weights(state_dict, name, tensor);
  }
}

inline int extract_expert_index(const std::string& name) {
  static const std::string prefix = "experts.";
  size_t pos = name.find(prefix);
  if (pos != std::string::npos) {
    pos += prefix.length();
    size_t end_pos = pos;
    while (end_pos < name.length() &&
           std::isdigit(static_cast<unsigned char>(name[end_pos]))) {
      ++end_pos;
    }
    if (end_pos > pos) {
      return std::stoi(name.substr(pos, end_pos - pos));
    }
  }
  return -1;
}

inline std::string extract_endswith(const std::string& input) {
  std::vector<std::string> parts;
  std::stringstream ss(input);
  std::string part;
  while (std::getline(ss, part, '.')) {
    parts.emplace_back(part);
  }
  if (parts.size() < 2) {
    return "";
  }
  return parts[parts.size() - 2] + "." + parts[parts.size() - 1];
}

template <typename Mapping>
inline int get_mapped_index(const std::string& name, const Mapping& mapping) {
  const auto it = mapping.find(name);
  if (it == mapping.end()) {
    LOG(WARNING) << "Parameter '" << name
                 << "' not found in mapping and will not be used.";
    return -1;
  }
  return it->second;
}

template <typename StateDictLike>
inline torch::Tensor get_sharded_tensor(const StateDictLike& state_dict,
                                        const std::string& name,
                                        int dim,
                                        int rank,
                                        int world_size) {
  if (world_size > 1) {
    return state_dict.get_sharded_tensor(name, dim, rank, world_size);
  }
  return state_dict.get_tensor(name);
}

template <typename WeightTensors, typename MakeTensorFn>
inline void initialize_weight_tensors(WeightTensors& weight_tensors,
                                      int weight_count,
                                      MakeTensorFn make_tensor) {
  for (int i = 0; i < weight_count; ++i) {
    weight_tensors[i] = make_tensor();
  }
}

template <typename WeightTensors>
inline void convert_offsets_to_int8(
    WeightTensors& weight_tensors,
    std::initializer_list<int> indices,
    c10::optional<at::Device> device = c10::nullopt) {
  for (int index : indices) {
    auto converted = weight_tensors[index].to(torch::kInt8);
    if (device.has_value()) {
      converted = converted.to(*device);
    }
    weight_tensors[index] = converted;
  }
}

template <typename WeightTensors>
inline void handle_device_specific_bias(WeightTensors& weight_tensors,
                                        int dp_local_tp_rank,
                                        int bias_index) {
  if (dp_local_tp_rank == 0) {
    return;
  }
  const torch::Tensor& original_tensor = weight_tensors[bias_index];
  weight_tensors[bias_index] =
      torch::zeros(original_tensor.sizes(), original_tensor.options());
}

template <typename StateDictLike,
          typename WeightShardMap,
          typename DeviceExpertList,
          typename ExpertsWeightsMap,
          typename GetShardedTensorFn,
          typename CacheEplbTensorFn>
inline void process_expert_weights_common(
    const StateDictLike& state_dict,
    const std::string& name,
    const torch::Tensor& tensor,
    const std::unordered_map<std::string, int>& weight_mapping_w8a8,
    const WeightShardMap& weight_shard_w8a8,
    const DeviceExpertList& device_expert_list,
    ExpertsWeightsMap& experts_weights,
    std::mutex& experts_mutex,
    int rank,
    int local_world_size,
    int ep_rank,
    int num_experts_per_partition,
    int ep_local_tp_rank,
    int ep_local_tp_size,
    bool enable_eplb,
    bool decode_is_bf16,
    GetShardedTensorFn get_sharded_tensor_fn,
    CacheEplbTensorFn cache_eplb_tensor_fn) {
  const int expert_index = extract_expert_index(name);
  const std::string suffix = extract_endswith(name);
  const int index = get_mapped_index(suffix, weight_mapping_w8a8);
  if (index == -1) {
    return;
  }

  const bool is_sharded = weight_shard_w8a8.count(index) > 0;
  const bool needs_eplb = enable_eplb && (rank % local_world_size ==
                                          expert_index % local_world_size);

  const int start_idx = ep_rank * num_experts_per_partition;
  const int end_idx = (ep_rank + 1) * num_experts_per_partition;
  const int safe_end =
      std::min(end_idx, static_cast<int>(device_expert_list.size()));

  auto begin_it = device_expert_list.cbegin() + start_idx;
  auto end_it = device_expert_list.cbegin() + safe_end;
  auto expert_it = std::find(begin_it, end_it, expert_index);
  const bool in_partition = expert_it != end_it;

  if (!needs_eplb && !in_partition) {
    return;
  }

  torch::Tensor processed_tensor;
  {
    std::lock_guard<std::mutex> lock(experts_mutex);
    processed_tensor = is_sharded
                           ? get_sharded_tensor_fn(state_dict,
                                                   name,
                                                   weight_shard_w8a8.at(index),
                                                   ep_local_tp_rank,
                                                   ep_local_tp_size)
                           : tensor;
    if (!decode_is_bf16) {
      if (absl::EndsWith(name, "_offset")) {
        processed_tensor = processed_tensor.to(torch::kFloat16);
      } else if (absl::EndsWith(name, "_scale")) {
        processed_tensor = processed_tensor.to(torch::kFloat32);
      }
    }
  }

  if (needs_eplb) {
    std::lock_guard<std::mutex> lock(experts_mutex);
    cache_eplb_tensor_fn(expert_index, suffix, processed_tensor);
  }

  if (!in_partition) {
    return;
  }

  std::vector<size_t> matches_pos;
  for (auto iter = expert_it; iter != end_it; ++iter) {
    if (*iter == expert_index) {
      matches_pos.emplace_back(
          std::distance(device_expert_list.cbegin(), iter) - start_idx);
    }
  }

  if (matches_pos.empty()) {
    return;
  }

  std::lock_guard<std::mutex> lock(experts_mutex);
  for (auto pos : matches_pos) {
    experts_weights[suffix][pos] = processed_tensor.clone();
  }
}

template <typename StateDictLike,
          typename WeightShardMap,
          typename WeightTensors,
          typename CorrectTensorDtypeFn,
          typename PostAssignFn>
inline void process_general_weights_common(
    const StateDictLike& state_dict,
    const std::string& name,
    const torch::Tensor& tensor,
    const std::unordered_map<std::string, int>& weight_mapping_w8a8,
    const WeightShardMap& weight_shard_w8a8,
    WeightTensors& weight_tensors,
    const at::Device& target_device,
    int dp_local_tp_rank,
    int dp_local_tp_size,
    CorrectTensorDtypeFn correct_tensor_dtype_fn,
    PostAssignFn post_assign_fn) {
  const int index = get_mapped_index(name, weight_mapping_w8a8);
  if (index == -1) {
    return;
  }

  const bool is_sharded = weight_shard_w8a8.count(index) > 0;
  torch::Tensor tmp_tensor =
      is_sharded ? get_sharded_tensor(state_dict,
                                      name,
                                      weight_shard_w8a8.at(index),
                                      dp_local_tp_rank,
                                      dp_local_tp_size)
                       .to(target_device)
                 : tensor.to(target_device);

  correct_tensor_dtype_fn(tmp_tensor, name);
  weight_tensors[index] = tmp_tensor;
  post_assign_fn(index, name, tensor, tmp_tensor);
}

template <typename StateDictLike,
          typename WeightShardMap,
          typename WeightTensors,
          typename SharedExpertsWeightsMap,
          typename GetTensorFn>
inline void process_mlp_common_weights_common(
    const StateDictLike& state_dict,
    const std::string& name,
    const torch::Tensor& tensor,
    const std::unordered_map<std::string, int>& weight_mapping_w8a8,
    const WeightShardMap& weight_shard_w8a8,
    WeightTensors& weight_tensors,
    SharedExpertsWeightsMap& shared_experts_weights,
    std::mutex& shared_experts_mutex,
    GetTensorFn get_tensor_fn) {
  const int index = get_mapped_index(name, weight_mapping_w8a8);
  if (index == -1) {
    return;
  }

  const bool is_sharded = weight_shard_w8a8.count(index) > 0;
  std::lock_guard<std::mutex> lock(shared_experts_mutex);
  torch::Tensor tmp_tensor = get_tensor_fn(index, is_sharded);

  if (absl::StrContains(name, "down_proj")) {
    weight_tensors[index] = tmp_tensor;
  } else {
    shared_experts_weights[name] = tmp_tensor;
  }
}

template <typename WeightTensors, typename SharedExpertsWeightsMap>
inline void process_shared_expert_weights_common(
    const std::string& name,
    const std::unordered_map<std::string, int>& weight_mapping_w8a8,
    WeightTensors& weight_tensors,
    SharedExpertsWeightsMap& shared_experts_weights,
    const torch::Tensor& tmp_tensor) {
  const int index = get_mapped_index(name, weight_mapping_w8a8);
  if (index == -1) {
    return;
  }

  if (absl::StrContains(name, "down_proj")) {
    weight_tensors[index] = tmp_tensor;
  } else {
    shared_experts_weights[name] = tmp_tensor;
  }
}

template <typename ClearFn>
inline torch::Tensor merge_experts_weights_common(
    std::vector<torch::Tensor>& experts,
    bool transpose,
    ClearFn clear_fn,
    c10::optional<at::Device> device = c10::nullopt) {
  torch::Tensor merged_tensor = torch::stack(experts, 0);
  if (device.has_value()) {
    merged_tensor = merged_tensor.to(*device);
  }
  if (transpose) {
    merged_tensor = merged_tensor.transpose(1, 2);
  }
  merged_tensor = merged_tensor.contiguous();
  clear_fn(experts);
  return merged_tensor;
}

template <typename ClearFn>
inline torch::Tensor merge_experts_weights_common(
    std::vector<torch::Tensor>& experts_gate,
    std::vector<torch::Tensor>& experts_up,
    bool transpose,
    ClearFn clear_fn,
    c10::optional<at::Device> device = c10::nullopt) {
  for (size_t i = 0; i < experts_up.size(); ++i) {
    experts_gate[i] = torch::cat({experts_gate[i], experts_up[i]}, 0);
  }

  torch::Tensor merged_tensor = torch::stack(experts_gate, 0);
  if (device.has_value()) {
    merged_tensor = merged_tensor.to(*device);
  }
  if (transpose) {
    merged_tensor = merged_tensor.transpose(1, 2);
  }
  merged_tensor = merged_tensor.contiguous();
  clear_fn(experts_gate);
  clear_fn(experts_up);
  return merged_tensor;
}

template <typename WeightTensors,
          typename SharedExpertsWeightsMap,
          typename MakeTensorFn>
inline void merge_shared_experts_weights_common(
    WeightTensors& weight_tensors,
    SharedExpertsWeightsMap& shared_experts_weights,
    const torch::Tensor& tensor_placeholder,
    bool use_shared_experts_prefix,
    const std::string& quantize_type,
    int gateup_weight_idx,
    int gateup_offset_idx,
    int gateup_scale_idx,
    MakeTensorFn make_tensor_fn) {
  const std::string prefix =
      use_shared_experts_prefix ? "mlp.shared_experts." : "mlp.";
  auto merge_and_clear = [&](int index,
                             const std::string& gate_suffix,
                             const std::string& up_suffix) {
    const std::string gate_name = prefix + gate_suffix;
    const std::string up_name = prefix + up_suffix;
    weight_tensors[index] = make_tensor_fn(torch::cat(
        {shared_experts_weights[gate_name], shared_experts_weights[up_name]},
        0));
    shared_experts_weights[gate_name] = tensor_placeholder;
    shared_experts_weights[up_name] = tensor_placeholder;
  };

  merge_and_clear(gateup_weight_idx, "gate_proj.weight", "up_proj.weight");
  if (quantize_type == "w8a8_dynamic") {
    merge_and_clear(
        gateup_offset_idx, "gate_proj.weight_offset", "up_proj.weight_offset");
    merge_and_clear(
        gateup_scale_idx, "gate_proj.weight_scale", "up_proj.weight_scale");
  }
}

template <typename WeightTensors>
inline void squeeze_weight_tensors(WeightTensors& weight_tensors,
                                   const std::vector<int>& squeeze_indices) {
  for (int index : squeeze_indices) {
    if (weight_tensors[index].dim() > 1) {
      weight_tensors[index] = weight_tensors[index].squeeze();
    }
  }
}

template <typename StateDictLike, typename CorrectTensorDtypeFn>
inline void set_kv_weight_common(const StateDictLike& state_dict,
                                 const std::string& tensor_name,
                                 int weight_position,
                                 int dim,
                                 std::vector<torch::Tensor>& weight_tensors,
                                 const at::Device& target_device,
                                 int parallel_world_size,
                                 int dp_local_tp_rank,
                                 int dp_local_tp_size,
                                 int num_key_value_heads,
                                 int qk_nope_head_dim,
                                 int v_head_dim,
                                 int kv_lora_rank,
                                 CorrectTensorDtypeFn correct_tensor_dtype_fn) {
  torch::Tensor mutable_tensor;
  if (parallel_world_size <= 1) {
    mutable_tensor = state_dict.get_tensor(tensor_name).to(target_device);
  } else {
    mutable_tensor =
        get_sharded_tensor(
            state_dict, tensor_name, dim, dp_local_tp_rank, dp_local_tp_size)
            .to(target_device);
  }
  correct_tensor_dtype_fn(mutable_tensor, tensor_name);

  torch::Tensor kv_b_proj_weight =
      mutable_tensor.reshape({num_key_value_heads / dp_local_tp_size,
                              qk_nope_head_dim + v_head_dim,
                              kv_lora_rank});
  torch::Tensor k_b_proj_preprocessed =
      kv_b_proj_weight.slice(1, 0, qk_nope_head_dim).contiguous();
  torch::Tensor v_b_proj_preprocessed =
      kv_b_proj_weight.slice(1, qk_nope_head_dim, qk_nope_head_dim + v_head_dim)
          .transpose(1, 2)
          .contiguous();

  weight_tensors[weight_position] = k_b_proj_preprocessed.to(target_device);
  weight_tensors[weight_position + 6] = v_b_proj_preprocessed.to(target_device);
}

template <typename WeightTensors, typename ViewTensorFn, typename TransRopeFn>
inline void preprocess_linear_for_rope_common(
    WeightTensors& weight_tensors,
    const std::vector<std::string>& linear_for_rope,
    const std::unordered_map<std::string, int>& weight_mapping_w8a8,
    const std::string& quantize_type,
    ViewTensorFn view_tensor_fn,
    TransRopeFn trans_rope_weight_fn) {
  for (const auto& name : linear_for_rope) {
    if (quantize_type.empty() && !absl::EndsWith(name, "weight")) {
      continue;
    }
    const int index = weight_mapping_w8a8.at(name);
    weight_tensors[index] = view_tensor_fn(weight_tensors[index], name, true);
    weight_tensors[index] = trans_rope_weight_fn(weight_tensors[index]);
    weight_tensors[index] =
        absl::EndsWith(name, "weight")
            ? view_tensor_fn(weight_tensors[index], name, false)
            : view_tensor_fn(weight_tensors[index], name, false).flatten();
  }
}

inline void initialize_device_expert_list(
    std::vector<int32_t>& device_expert_list,
    int num_device,
    int num_device_expert,
    bool enable_eplb,
    int redundant_experts_num) {
  int32_t num_device_route_expert = num_device_expert;
  if (enable_eplb) {
    num_device_route_expert = num_device_expert - redundant_experts_num;
  }
  for (int i = 0; i < num_device * num_device_route_expert; ++i) {
    device_expert_list.emplace_back(i);
    if (enable_eplb && (i + 1) % num_device_route_expert == 0) {
      for (int redundant_expert = 0; redundant_expert < redundant_experts_num;
           ++redundant_expert) {
        device_expert_list.emplace_back(i);
      }
    }
  }
}

template <typename ExpertsWeightsMap>
inline void reserve_experts_weights(ExpertsWeightsMap& experts_weights,
                                    std::mutex& experts_mutex,
                                    int num_of_device_experts,
                                    const std::string& quantize_type) {
  experts_weights.clear();
  std::vector<std::string> weight_names = {
      "gate_proj.weight", "up_proj.weight", "down_proj.weight"};
  if (quantize_type == "w8a8_dynamic") {
    weight_names.emplace_back("gate_proj.weight_offset");
    weight_names.emplace_back("up_proj.weight_offset");
    weight_names.emplace_back("down_proj.weight_offset");
    weight_names.emplace_back("gate_proj.weight_scale");
    weight_names.emplace_back("up_proj.weight_scale");
    weight_names.emplace_back("down_proj.weight_scale");
  }
  std::lock_guard<std::mutex> lock(experts_mutex);
  for (const auto& weight_name : weight_names) {
    experts_weights[weight_name] =
        std::vector<torch::Tensor>(num_of_device_experts);
  }
}

inline std::string get_expert_shm_key(int32_t layer_id,
                                      int32_t first_k_dense_replace,
                                      int32_t expert_index,
                                      const std::string& suffix) {
  return "layer_" + std::to_string(layer_id - first_k_dense_replace) + "_" +
         "expert_" + std::to_string(expert_index) + "_" + suffix;
}

template <typename WeightTensors>
inline void convert_descaled_weights_to_float(
    WeightTensors& weight_tensors,
    std::initializer_list<int> indices) {
  for (int index : indices) {
    weight_tensors[index] = weight_tensors[index].to(torch::kFloat32);
  }
}

inline torch::Tensor convert_fp16_to_int64(const torch::Tensor& fp16_tensor) {
  auto float_tensor = fp16_tensor.to(torch::kFloat32);
  auto int32_tensor = float_tensor.view(torch::kInt32);
  return int32_tensor.to(torch::kInt64);
}

inline torch::Tensor view_tensor(torch::Tensor weight,
                                 const std::string& name,
                                 bool pre_view,
                                 int prefill_num_attention_heads_per_rank,
                                 int qk_nope_head_dim,
                                 int prefill_qk_rope_head_dim,
                                 int kv_lora_rank) {
  if (absl::StrContains(name, "q_b_proj")) {
    if (pre_view) {
      return weight
          .view({prefill_num_attention_heads_per_rank,
                 qk_nope_head_dim + prefill_qk_rope_head_dim,
                 -1})
          .contiguous();
    }
    return weight
        .view({prefill_num_attention_heads_per_rank *
                   (qk_nope_head_dim + prefill_qk_rope_head_dim),
               -1})
        .contiguous();
  }
  if (absl::StrContains(name, "kv_a_proj_with_mqa")) {
    return weight.view({kv_lora_rank + prefill_qk_rope_head_dim, -1})
        .contiguous();
  }
  return weight;
}

inline torch::Tensor trans_rope_weight(torch::Tensor weight,
                                       int prefill_qk_rope_head_dim,
                                       bool clone_first) {
  torch::Tensor output = clone_first ? weight.clone() : weight;
  const int64_t d = weight.size(-2);
  const int64_t rope_dim = prefill_qk_rope_head_dim;
  torch::Tensor weight_1 =
      weight.slice(-2, d - rope_dim, torch::indexing::None, 2).contiguous();
  torch::Tensor weight_2 =
      weight.slice(-2, d - rope_dim + 1, torch::indexing::None, 2).contiguous();
  torch::Tensor combined = torch::cat({weight_1, weight_2}, -2);
  output.slice(-2, d - rope_dim, d).copy_(combined);
  return output.contiguous();
}

}  // namespace deepseek_loader_family_utils
}  // namespace layer
}  // namespace xllm
