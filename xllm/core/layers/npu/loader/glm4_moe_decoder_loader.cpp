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

#include "glm4_moe_decoder_loader.h"

#include <glog/logging.h>
#include <torch/torch.h>
#include <torch_npu/csrc/core/npu/NPUCachingAllocator.h>
#include <torch_npu/csrc/core/npu/NPUException.h>

#include <cctype>
#include <map>
#include <sstream>
#include <unordered_map>

#include "core/layers/npu/npu_glm4_moe_decoder_layer.h"
#include "glm_moe_loader_constants.h"

namespace xllm {
namespace layer {

using namespace glm4_moe_decoder_constants;

Glm4MoeDecoderLoader::Glm4MoeDecoderLoader(
    uint64_t weight_count,
    const ModelContext& context,
    int32_t layer_id,
    int32_t prefill_param_firstKDenseReplace,
    LoadMode mode)
    : BaseLoader(weight_count, context, mode),
      layer_id_(layer_id),
      prefill_param_firstKDenseReplace_(prefill_param_firstKDenseReplace) {
  auto model_args = context.get_model_args();
  auto parallel_args = context.get_parallel_args();
  auto options = context.get_tensor_options();

  tensor_placeholder_ = torch::zeros({1}, options.device(target_device()));
  weight_mapping_ = WEIGHT_MAPPING;
  weight_mapping_w8a8_ = WEIGHT_MAPPING_W8A8;

  if (model_args.use_qk_norm()) {
    weight_count_ = weight_count = 70;
    weight_mapping_w8a8_["self_attn.q_norm.weight"] = Q_NORM_WEIGHT;
    weight_mapping_w8a8_["self_attn.k_norm.weight"] = K_NORM_WEIGHT;
    weight_mapping_["self_attn.q_norm.weight"] = Q_NORM_WEIGHT;
    weight_mapping_["self_attn.k_norm.weight"] = K_NORM_WEIGHT;
  }

  working_tensors().resize(weight_count_);
  if (mode == LoadMode::kManual) {
    at_weight_tensors_.resize(weight_count_);
  }

  num_experts_ = model_args.num_experts();
  ep_size_ = parallel_args.ep_size();
  ep_local_tp_size_ = parallel_args.world_size() / ep_size_;
  CHECK_EQ(parallel_args.world_size(), ep_size_ * ep_local_tp_size_);
  ep_local_tp_rank_ = parallel_args.rank() % ep_local_tp_size_;
  num_experts_per_partition_ = model_args.num_experts() / ep_size_;
  ep_rank_ = parallel_args.rank() / ep_local_tp_size_;
  start_expert_id_ = ep_rank_ * num_experts_per_partition_;
  end_expert_id_ = start_expert_id_ + num_experts_per_partition_ - 1;

  dp_size_ = parallel_args.dp_size();
  dp_local_tp_size_ = parallel_args.world_size() / dp_size_;
  CHECK_EQ(parallel_args.world_size(), dp_size_ * dp_local_tp_size_);
  dp_local_tp_rank_ = parallel_args.rank() % dp_local_tp_size_;

  n_kv_heads_ = static_cast<int32_t>(model_args.n_kv_heads().value());
}

void Glm4MoeDecoderLoader::resize_experts_weights(int num_of_device_experts) {
  experts_weights_["gate_proj.weight"] =
      std::vector<torch::Tensor>(num_of_device_experts);
  experts_weights_["up_proj.weight"] =
      std::vector<torch::Tensor>(num_of_device_experts);
  experts_weights_["down_proj.weight"] =
      std::vector<torch::Tensor>(num_of_device_experts);
  if (quantize_type_.compare("w8a8_dynamic") == 0) {
    experts_weights_["gate_proj.weight_offset"] =
        std::vector<torch::Tensor>(num_of_device_experts);
    experts_weights_["up_proj.weight_offset"] =
        std::vector<torch::Tensor>(num_of_device_experts);
    experts_weights_["down_proj.weight_offset"] =
        std::vector<torch::Tensor>(num_of_device_experts);
    experts_weights_["gate_proj.weight_scale"] =
        std::vector<torch::Tensor>(num_of_device_experts);
    experts_weights_["up_proj.weight_scale"] =
        std::vector<torch::Tensor>(num_of_device_experts);
    experts_weights_["down_proj.weight_scale"] =
        std::vector<torch::Tensor>(num_of_device_experts);
  }
}

void Glm4MoeDecoderLoader::load_state_dict(const StateDict& state_dict) {
  for (const auto& [name, tensor] : state_dict) {
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

void Glm4MoeDecoderLoader::verify_loaded_weights() const {
  const auto& t = working_tensors();
  for (const auto& [name, index] : weight_mapping_) {
    if (name == "down_proj.weight" || name == "gate_proj.weight" ||
        name == "up_proj.weight" || name == "mlp.gate.weight" ||
        name == "mlp.gate.e_score_correction_bias") {
      continue;
    }
    CHECK(t[index].defined() && t[index].numel() > 0)
        << layer_id_ << "-weight is not loaded for " << name;
  }
}

void Glm4MoeDecoderLoader::merge_host_at_weights() {
  merge_shared_experts_weights();
  if (layer_id_ >= prefill_param_firstKDenseReplace_) {
    merge_experts_weights();
  }

  auto& t = working_tensors();
  auto target_options =
      torch::TensorOptions().dtype(torch::kFloat16).device(target_device());
  auto zero_fp16 = [&]() { return torch::zeros({1}, target_options); };

  t[IN_QKV_WEIGHT_0] =
      torch::cat({t[IN_QKV_WEIGHT_0], t[IN_QKV_WEIGHT_1], t[IN_QKV_WEIGHT_2]},
                 0)
          .contiguous();
  t[IN_QKV_WEIGHT_1] = zero_fp16();
  t[IN_QKV_WEIGHT_2] = zero_fp16();

  t[IN_QKV_BIAS_0] = t[IN_QKV_BIAS_0].squeeze();
  t[IN_QKV_BIAS_1] = t[IN_QKV_BIAS_1].squeeze();
  t[IN_QKV_BIAS_2] = t[IN_QKV_BIAS_2].squeeze();

  t[IN_QKV_BIAS_0] =
      torch::cat({t[IN_QKV_BIAS_0], t[IN_QKV_BIAS_1], t[IN_QKV_BIAS_2]}, 0)
          .contiguous();
  t[IN_QKV_BIAS_1] = zero_fp16();
  t[IN_QKV_BIAS_2] = zero_fp16();

  if (quantize_type_.compare("w8a8_dynamic") == 0) {
    t[IN_QKV_DESCALE_0] = t[IN_QKV_DESCALE_0].squeeze();
    t[IN_QKV_DESCALE_1] = t[IN_QKV_DESCALE_1].squeeze();
    t[IN_QKV_DESCALE_2] = t[IN_QKV_DESCALE_2].squeeze();

    t[IN_QKV_DESCALE_0] =
        torch::cat(
            {t[IN_QKV_DESCALE_0], t[IN_QKV_DESCALE_1], t[IN_QKV_DESCALE_2]}, 0)
            .contiguous();

    t[IN_QKV_DESCALE_1] = zero_fp16();
    t[IN_QKV_DESCALE_2] = zero_fp16();

    t[IN_QKV_DENSE_BIAS] = zero_fp16();
    t[IN_QKV_DENSE_DESCALE] = zero_fp16();

    t[IN_QKV_OFFSET_0] = t[IN_QKV_OFFSET_0].to(torch::kInt8);
    t[IN_QKV_OFFSET_1] = zero_fp16();
    t[IN_QKV_OFFSET_2] = zero_fp16();
    t[IN_QKV_DENSE_OFFSET] = t[IN_QKV_DENSE_OFFSET].contiguous().view(-1);

    t[IN_QKV_SCALE_1] = zero_fp16();
    t[IN_QKV_SCALE_2] = zero_fp16();
    t[IN_QKV_DENSE_SCALE] = t[IN_QKV_DENSE_SCALE].contiguous().view(-1);
  }
}

void Glm4MoeDecoderLoader::process_expert_weights(const StateDict& state_dict,
                                                  const std::string& name,
                                                  const torch::Tensor& tensor) {
  int expert_index = extract_expert_index(name);
  if (expert_index < start_expert_id_ || expert_index > end_expert_id_) {
    return;
  }

  const std::string suffix = extract_endswith(name);
  const auto& weight_mapping = (quantize_type_.compare("w8a8_dynamic") == 0)
                                   ? weight_mapping_w8a8_
                                   : weight_mapping_;
  const auto& shard_map = (quantize_type_.compare("w8a8_dynamic") == 0)
                              ? WEIGHT_SHARD_W8A8
                              : WEIGHT_SHARD;
  const int index = get_mapped_index(suffix, weight_mapping);
  if (index == -1) {
    return;
  }
  const int local_index = expert_index % num_experts_per_partition_;
  const bool is_sharded = shard_map.count(index);

  std::lock_guard<std::mutex> lock(experts_mutex_);
  torch::Tensor tmp_tensor = is_sharded
                                 ? get_sharded_tensor(state_dict,
                                                      name,
                                                      shard_map.at(index),
                                                      ep_local_tp_rank_,
                                                      ep_local_tp_size_)
                                 : tensor.to(target_device());

  experts_weights_[suffix][local_index] = tmp_tensor.clone();
}

void Glm4MoeDecoderLoader::process_shared_expert_weights(
    const StateDict& state_dict,
    const std::string& name,
    const torch::Tensor& tensor) {
  torch::Tensor tmp_tensor;
  const auto& weight_mapping = (quantize_type_.compare("w8a8_dynamic") == 0)
                                   ? weight_mapping_w8a8_
                                   : weight_mapping_;
  const auto& shard_map = (quantize_type_.compare("w8a8_dynamic") == 0)
                              ? WEIGHT_SHARD_W8A8
                              : WEIGHT_SHARD;
  std::lock_guard<std::mutex> lock(shared_experts_mutex_);
  const int index = get_mapped_index(name, weight_mapping);
  if (index == -1) {
    return;
  }

  const bool is_sharded = shard_map.count(index);
  tmp_tensor = is_sharded
                   ? get_sharded_tensor(state_dict, name, shard_map.at(index))
                   : tensor.to(target_device());

  if (absl::StrContains(name, "down_proj")) {
    working_tensors()[index] = tmp_tensor;
  } else {
    shared_experts_weights_[name] = tmp_tensor;
  }
}

void Glm4MoeDecoderLoader::process_mlp_common_weights(
    const StateDict& state_dict,
    const std::string& name,
    const torch::Tensor& tensor) {
  const auto& weight_mapping = (quantize_type_.compare("w8a8_dynamic") == 0)
                                   ? weight_mapping_w8a8_
                                   : weight_mapping_;
  const auto& shard_map = (quantize_type_.compare("w8a8_dynamic") == 0)
                              ? WEIGHT_SHARD_W8A8
                              : WEIGHT_SHARD;
  const int index = get_mapped_index(name, weight_mapping);
  if (index == -1) {
    return;
  }
  const bool is_sharded = shard_map.count(index);

  std::lock_guard<std::mutex> lock(shared_experts_mutex_);

  torch::Tensor tmp_tensor = is_sharded
                                 ? get_sharded_tensor(state_dict,
                                                      name,
                                                      shard_map.at(index),
                                                      dp_local_tp_rank_,
                                                      dp_local_tp_size_)
                                 : tensor.to(target_device());
  if (absl::StrContains(name, "down_proj")) {
    working_tensors()[index] = tmp_tensor;
  } else {
    shared_experts_weights_[name] = tmp_tensor;
  }
}

void Glm4MoeDecoderLoader::process_general_weights(
    const StateDict& state_dict,
    const std::string& name,
    const torch::Tensor& tensor) {
  const auto& weight_mapping = (quantize_type_.compare("w8a8_dynamic") == 0)
                                   ? weight_mapping_w8a8_
                                   : weight_mapping_;
  const auto& shard_map = (quantize_type_.compare("w8a8_dynamic") == 0)
                              ? WEIGHT_SHARD_W8A8
                              : WEIGHT_SHARD;

  if (weight_mapping.find(name) == weight_mapping.end()) {
    return;
  }

  const int index = get_mapped_index(name, weight_mapping);
  if (index == -1) {
    return;
  }
  const bool is_sharded = shard_map.count(index);
  torch::Tensor tmp_tensor;
  int32_t tp_rank = dp_local_tp_rank_;
  int32_t tp_size = dp_local_tp_size_;
  if (index == IN_QKV_WEIGHT_1 || index == IN_QKV_WEIGHT_2 ||
      index == IN_QKV_BIAS_1 || index == IN_QKV_BIAS_2 ||
      index == IN_QKV_DESCALE_1 || index == IN_QKV_DESCALE_2) {
    if (n_kv_heads_ < dp_local_tp_size_) {
      int32_t repeat_times = (dp_local_tp_size_ / n_kv_heads_);
      tp_rank = tp_rank / repeat_times;
      tp_size = n_kv_heads_;
    }
  }
  if (is_sharded) {
    tmp_tensor = get_sharded_tensor(
        state_dict, name, shard_map.at(index), tp_rank, tp_size);
  } else {
    tmp_tensor = tensor.to(target_device());
  }
  if (index == BLOCK_SPARSE_MOE_GATE_BIAS) {
    auto min_val = tmp_tensor.min();
    tmp_tensor = tmp_tensor - min_val;
  }
  correct_tensor_dtype(tmp_tensor, name);
  auto& t = working_tensors();
  if (quantize_type_.compare("w8a8_dynamic") == 0) {
    auto it = SPECIAL_MULTI_ASSIGN_W8A8.find(name);
    if (it != SPECIAL_MULTI_ASSIGN_W8A8.end()) {
      for (int idx : it->second) {
        t[idx] = tmp_tensor;
      }
      return;
    }
  }
  t[index] = tmp_tensor;
}

torch::Tensor Glm4MoeDecoderLoader::get_sharded_tensor(
    const StateDict& state_dict,
    const std::string& name,
    int dim) {
  if (parallel_args_.world_size() > 1) {
    return state_dict
        .get_sharded_tensor(
            name, dim, parallel_args_.rank(), parallel_args_.world_size())
        .to(target_device());
  } else {
    return state_dict.get_tensor(name).to(target_device());
  }
}

torch::Tensor Glm4MoeDecoderLoader::get_sharded_tensor(
    const StateDict& state_dict,
    const std::string& name,
    int dim,
    int local_tp_rank,
    int local_tp_size) {
  if (local_tp_size > 1) {
    return state_dict
        .get_sharded_tensor(name, dim, local_tp_rank, local_tp_size)
        .to(target_device());
  } else {
    return state_dict.get_tensor(name).to(target_device());
  }
}

std::string Glm4MoeDecoderLoader::extract_endswith(const std::string& input) {
  std::vector<std::string> parts;
  std::stringstream ss(input);
  std::string part;
  while (std::getline(ss, part, '.')) {
    parts.push_back(part);
  }
  if (parts.size() < 2) {
    return "";
  }
  std::string result = parts[parts.size() - 2] + "." + parts[parts.size() - 1];

  return result;
}

int Glm4MoeDecoderLoader::extract_expert_index(const std::string& name) {
  std::string prefix = "experts.";
  size_t pos = name.find(prefix);
  if (pos != std::string::npos) {
    pos += prefix.length();
    size_t end_pos = pos;
    while (end_pos < name.length() && std::isdigit(name[end_pos])) {
      ++end_pos;
    }
    if (end_pos > pos) {
      return std::stoi(name.substr(pos, end_pos - pos));
    }
  }

  return -1;
}

void Glm4MoeDecoderLoader::merge_shared_experts_weights() {
  auto& t = working_tensors();
  auto merge_and_clear = [this, &t](int index,
                                    torch::Tensor& shared_experts_gate,
                                    torch::Tensor& shared_experts_up) {
    t[index] = torch::cat({shared_experts_gate, shared_experts_up}, 0)
                   .to(target_device())
                   .contiguous();
    shared_experts_gate = tensor_placeholder_;
    shared_experts_up = tensor_placeholder_;
  };

  if (layer_id_ >= prefill_param_firstKDenseReplace_) {
    merge_and_clear(
        IN_MLP_GATEUP_WEIGHT_SHARED_EXPERT,
        shared_experts_weights_["mlp.shared_experts.gate_proj.weight"],
        shared_experts_weights_["mlp.shared_experts.up_proj.weight"]);
    if (quantize_type_ == "w8a8_dynamic") {
      merge_and_clear(
          IN_MLP_GATEUP_OFFSET_SHARED_EXPERT,
          shared_experts_weights_["mlp.shared_experts.gate_proj.weight_offset"],
          shared_experts_weights_["mlp.shared_experts.up_proj.weight_offset"]);
      merge_and_clear(
          IN_MLP_GATEUP_SCALE_SHARED_EXPERT,
          shared_experts_weights_["mlp.shared_experts.gate_proj.weight_scale"],
          shared_experts_weights_["mlp.shared_experts.up_proj.weight_scale"]);
      t[IN_MLP_GATEUP_OFFSET_SHARED_EXPERT] =
          t[IN_MLP_GATEUP_OFFSET_SHARED_EXPERT].squeeze();
      t[IN_MLP_GATEUP_SCALE_SHARED_EXPERT] =
          t[IN_MLP_GATEUP_SCALE_SHARED_EXPERT].squeeze();
      t[IN_MLP_DOWN_OFFSET_SHARED_EXPERT] =
          t[IN_MLP_DOWN_OFFSET_SHARED_EXPERT].squeeze();
      t[IN_MLP_DOWN_SCALE_SHARED_EXPERT] =
          t[IN_MLP_DOWN_SCALE_SHARED_EXPERT].squeeze();
    }
  } else {
    merge_and_clear(IN_MLP_GATEUP_WEIGHT_SHARED_EXPERT,
                    shared_experts_weights_["mlp.gate_proj.weight"],
                    shared_experts_weights_["mlp.up_proj.weight"]);
    if (quantize_type_ == "w8a8_dynamic") {
      merge_and_clear(IN_MLP_GATEUP_OFFSET_SHARED_EXPERT,
                      shared_experts_weights_["mlp.gate_proj.weight_offset"],
                      shared_experts_weights_["mlp.up_proj.weight_offset"]);
      merge_and_clear(IN_MLP_GATEUP_SCALE_SHARED_EXPERT,
                      shared_experts_weights_["mlp.gate_proj.weight_scale"],
                      shared_experts_weights_["mlp.up_proj.weight_scale"]);
      t[IN_MLP_GATEUP_OFFSET_SHARED_EXPERT] =
          t[IN_MLP_GATEUP_OFFSET_SHARED_EXPERT].squeeze();
      t[IN_MLP_GATEUP_SCALE_SHARED_EXPERT] =
          t[IN_MLP_GATEUP_SCALE_SHARED_EXPERT].squeeze();
    }
  }
}

void Glm4MoeDecoderLoader::merge_experts_weights() {
  auto& t = working_tensors();
  try {
    torch::Tensor mlp_gateup_weight;
    if (quantize_type_.compare("w8a8_dynamic") == 0) {
      mlp_gateup_weight =
          merge_experts_weights(experts_weights_["gate_proj.weight"],
                                experts_weights_["up_proj.weight"],
                                /*transpose=*/true);

      t[IN_MLP_GATEUP_OFFSET] =
          merge_experts_weights(experts_weights_["gate_proj.weight_offset"],
                                experts_weights_["up_proj.weight_offset"]);
      t[IN_MLP_GATEUP_SCALE] =
          merge_experts_weights(experts_weights_["gate_proj.weight_scale"],
                                experts_weights_["up_proj.weight_scale"]);
      t[IN_MLP_GATEUP_WEIGHT] =
          cast_nz(mlp_gateup_weight, IN_MLP_GATEUP_WEIGHT);
    } else {
      mlp_gateup_weight =
          merge_experts_weights(experts_weights_["gate_proj.weight"],
                                experts_weights_["up_proj.weight"],
                                /*transpose=*/false);
      if (load_to_host()) {
        // Preserve manual-mode semantics: IN_MLP_GATEUP_WEIGHT is NZ on device.
        nz_indices_.insert(IN_MLP_GATEUP_WEIGHT);
        t[IN_MLP_GATEUP_WEIGHT] = mlp_gateup_weight.contiguous();
      } else {
        t[IN_MLP_GATEUP_WEIGHT] =
            at_npu::native::npu_format_cast(mlp_gateup_weight, 2).contiguous();
      }
    }
  } catch (const std::exception& e) {
    LOG(ERROR) << "[ERROR] Exception in gateup weight processing: " << e.what();
    throw;
  }

  try {
    torch::Tensor mlp_down_weight =
        merge_experts_weights(experts_weights_["down_proj.weight"],
                              /*transpose=*/false);

    if (load_to_host()) {
      // Preserve manual-mode semantics: IN_MLP_DOWN_WEIGHT is NZ on device.
      nz_indices_.insert(IN_MLP_DOWN_WEIGHT);
      t[IN_MLP_DOWN_WEIGHT] = mlp_down_weight.contiguous();
    } else {
      t[IN_MLP_DOWN_WEIGHT] =
          at_npu::native::npu_format_cast(mlp_down_weight, 2).contiguous();
    }

    if (quantize_type_.compare("w8a8_dynamic") == 0) {
      t[IN_MLP_DOWN_OFFSET] =
          merge_experts_weights(experts_weights_["down_proj.weight_offset"]);
      t[IN_MLP_DOWN_SCALE] =
          merge_experts_weights(experts_weights_["down_proj.weight_scale"]);
    }
  } catch (const std::exception& e) {
    LOG(ERROR) << "[ERROR] Exception in down weight processing: " << e.what();
    throw;
  }
}

torch::Tensor Glm4MoeDecoderLoader::merge_experts_weights(
    std::vector<torch::Tensor>& experts,
    bool transpose) {
  torch::Tensor merged_tensor = torch::stack(experts, 0).to(target_device());
  if (transpose) {
    merged_tensor = merged_tensor.transpose(1, 2);
  }
  merged_tensor = merged_tensor.contiguous();
  for (auto& expert : experts) {
    expert = torch::Tensor();
  }

  return merged_tensor;
}

torch::Tensor Glm4MoeDecoderLoader::merge_experts_weights(
    std::vector<torch::Tensor>& experts_gate,
    std::vector<torch::Tensor>& experts_up,
    bool transpose) {
  for (size_t i = 0; i < experts_up.size(); ++i) {
    experts_gate[i] = torch::cat({experts_gate[i], experts_up[i]}, 0);
  }
  torch::Tensor merged_tensor =
      torch::stack(experts_gate, 0).to(target_device());
  if (transpose) {
    merged_tensor = merged_tensor.transpose(1, 2);
  }
  merged_tensor = merged_tensor.contiguous();
  for (auto& expert : experts_gate) {
    expert = torch::Tensor();
  }
  for (auto& expert : experts_up) {
    expert = torch::Tensor();
  }

  return merged_tensor;
}

int Glm4MoeDecoderLoader::get_mapped_index(
    const std::string& name,
    const std::unordered_map<std::string, int>& mapping) {
  const auto it = mapping.find(name);
  if (it == mapping.end()) {
    LOG(ERROR) << "Missing mapping for: " << name;
    return -1;
  }
  return it->second;
}

}  // namespace layer
}  // namespace xllm
