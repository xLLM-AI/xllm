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

#include "qwen3_moe_decoder_loader.h"

#include <glog/logging.h>
#include <torch/torch.h>
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/torch_npu.h>

#include <cctype>
#include <sstream>
#include <unordered_set>

#include "qwen_loader_constants.h"
#include "xllm_atb_layers/core/include/atb_speed/base/hosttensor_binder.h"
#include "xllm_atb_layers/core/include/atb_speed/base/model.h"
#include "xllm_atb_layers/core/include/atb_speed/log.h"
#include "xllm_atb_layers/core/include/atb_speed/utils/model_factory.h"
#include "xllm_atb_layers/models/qwen3/layer/moe_decoder_layer.h"

namespace xllm {
namespace layer {
using namespace qwen3_moe_decoder_constants;

Qwen3MoeDecoderLoader::Qwen3MoeDecoderLoader(uint64_t weight_count,
                                             const ModelContext& context,
                                             LoadMode mode)
    : BaseLoader(weight_count, context, mode) {
  auto model_args = context.get_model_args();
  auto options = context.get_tensor_options();

  auto& t = working_tensors();
  auto target_options = options.device(target_device());
  for (uint64_t i = 0; i < weight_count; ++i) {
    t[i] = torch::zeros({1}, target_options);
  }

  num_experts_ = model_args.num_experts();
  ep_size_ = parallel_args_.ep_size();
  ep_local_tp_size_ = parallel_args_.world_size() / ep_size_;
  CHECK_EQ(parallel_args_.world_size(), ep_size_ * ep_local_tp_size_);
  ep_local_tp_rank_ = parallel_args_.rank() % ep_local_tp_size_;
  num_experts_per_partition_ = model_args.num_experts() / ep_size_;
  ep_rank_ = parallel_args_.rank() / ep_local_tp_size_;
  start_expert_id_ = ep_rank_ * num_experts_per_partition_;
  end_expert_id_ = start_expert_id_ + num_experts_per_partition_ - 1;
  n_kv_heads_ = static_cast<int32_t>(model_args.n_kv_heads().value());

  dp_size_ = parallel_args_.dp_size();
  dp_local_tp_size_ = parallel_args_.world_size() / dp_size_;
  CHECK_EQ(parallel_args_.world_size(), dp_size_ * dp_local_tp_size_);
  dp_local_tp_rank_ = parallel_args_.rank() % dp_local_tp_size_;
}

void Qwen3MoeDecoderLoader::load_state_dict(const StateDict& state_dict) {
  for (const auto& [name, tensor] : state_dict) {
    if (absl::StartsWith(name, "mlp.experts")) {
      process_expert_weights(state_dict, name, tensor);
      continue;
    }

    if (absl::StartsWith(name, "mlp") && !absl::StrContains(name, "gate.")) {
      process_mlp_common_weights(state_dict, name, tensor);
      continue;
    }

    process_general_weights(state_dict, name, tensor);
  }
}

void Qwen3MoeDecoderLoader::verify_loaded_weights() const {
  if (mode() == LoadMode::kManual) {
    verify_loaded_weights("qwen3_moe");
  }
}

void Qwen3MoeDecoderLoader::verify_loaded_weights(
    const std::string& prefix) const {
  const auto& t = working_tensors();
  for (const auto& [name, index] : WEIGHT_MAPPING) {
    if (name == "down_proj.weight" || name == "gate_proj.weight" ||
        name == "up_proj.weight") {
      continue;
    }
    CHECK(t[index].sizes() != std::vector<int64_t>({1}))
        << "weight is not loaded for " << name;
  }
}

void Qwen3MoeDecoderLoader::merge_experts_weights() {
  auto& t = working_tensors();

  try {
    torch::Tensor mlp_gateup_weight;
    if (quantize_type_.compare("w8a8_dynamic") == 0) {
      mlp_gateup_weight =
          merge_experts_weights(experts_weights_["gate_proj.weight"],
                                experts_weights_["up_proj.weight"],
                                /*transpose=*/true);
      t[IN_MLP_GATEUP_OFFSET_EXPERT] =
          merge_experts_weights(experts_weights_["gate_proj.weight_offset"],
                                experts_weights_["up_proj.weight_offset"]);
      t[IN_MLP_GATEUP_SCALE_EXPERT] =
          merge_experts_weights(experts_weights_["gate_proj.weight_scale"],
                                experts_weights_["up_proj.weight_scale"]);
    } else {
      mlp_gateup_weight =
          merge_experts_weights(experts_weights_["gate_proj.weight"],
                                experts_weights_["up_proj.weight"],
                                /*transpose=*/false);
    }
    t[IN_MLP_GATEUP_WEIGHT_EXPERT] =
        cast_nz(mlp_gateup_weight, IN_MLP_GATEUP_WEIGHT_EXPERT);
  } catch (const std::exception& e) {
    LOG(ERROR) << "[ERROR] Exception in gateup weight processing: " << e.what();
    throw;
  }

  try {
    torch::Tensor mlp_down_weight =
        merge_experts_weights(experts_weights_["down_proj.weight"],
                              /*transpose=*/false);

    if (quantize_type_.compare("w8a8_dynamic") == 0) {
      t[IN_MLP_DOWN_OFFSET_EXPERT] =
          merge_experts_weights(experts_weights_["down_proj.weight_offset"]);
      t[IN_MLP_DOWN_SCALE_EXPERT] =
          merge_experts_weights(experts_weights_["down_proj.weight_scale"]);
    }
    t[IN_MLP_DOWN_WEIGHT_EXPERT] =
        cast_nz(mlp_down_weight, IN_MLP_DOWN_WEIGHT_EXPERT);
  } catch (const std::exception& e) {
    LOG(ERROR) << "[ERROR] Exception in down weight processing: " << e.what();
    throw;
  }
}

torch::Tensor Qwen3MoeDecoderLoader::merge_experts_weights(
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

std::string Qwen3MoeDecoderLoader::extract_endswith(const std::string& input) {
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

int Qwen3MoeDecoderLoader::extract_expert_index(const std::string& name) {
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

void Qwen3MoeDecoderLoader::resize_experts_weights(int num_of_device_experts) {
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

void Qwen3MoeDecoderLoader::process_expert_weights(
    const StateDict& state_dict,
    const std::string& name,
    const torch::Tensor& tensor) {
  int expert_index = extract_expert_index(name);
  if (expert_index < start_expert_id_ || expert_index > end_expert_id_) {
    return;
  }

  const std::string suffix = extract_endswith(name);
  const auto& weight_mapping = (quantize_type_.compare("w8a8_dynamic") == 0)
                                   ? WEIGHT_MAPPING_W8A8
                                   : WEIGHT_MAPPING;
  const auto& shard_map = (quantize_type_.compare("w8a8_dynamic") == 0)
                              ? WEIGHT_SHARD_W8A8
                              : WEIGHT_SHARD;
  const int index = get_mapped_index(suffix, weight_mapping);
  const int local_index = expert_index % num_experts_per_partition_;
  const bool is_sharded = shard_map.count(index);

  torch::Tensor tmp_tensor = is_sharded
                                 ? get_sharded_tensor(state_dict,
                                                      name,
                                                      shard_map.at(index),
                                                      ep_local_tp_rank_,
                                                      ep_local_tp_size_)
                                 : tensor;

  experts_weights_[suffix][local_index] = tmp_tensor.clone();
}

int Qwen3MoeDecoderLoader::get_mapped_index(
    const std::string& name,
    const std::unordered_map<std::string, int>& mapping) {
  const auto it = mapping.find(name);
  if (it == mapping.end()) {
    LOG(ERROR) << "Missing mapping for: " << name;
    return -1;
  }

  return it->second;
}

void Qwen3MoeDecoderLoader::process_mlp_common_weights(
    const StateDict& state_dict,
    const std::string& name,
    const torch::Tensor& tensor) {
  const auto& weight_mapping = (quantize_type_.compare("w8a8_dynamic") == 0)
                                   ? WEIGHT_MAPPING_W8A8
                                   : WEIGHT_MAPPING;
  const auto& shard_map = (quantize_type_.compare("w8a8_dynamic") == 0)
                              ? WEIGHT_SHARD_W8A8
                              : WEIGHT_SHARD;
  const int index = get_mapped_index(name, weight_mapping);
  const bool is_sharded = shard_map.count(index);

  torch::Tensor tmp_tensor = is_sharded
                                 ? get_sharded_tensor(state_dict,
                                                      name,
                                                      shard_map.at(index),
                                                      dp_local_tp_rank_,
                                                      dp_local_tp_size_)
                                       .to(target_device())
                                 : tensor.to(target_device());
  if (absl::StrContains(name, "down_proj")) {
    working_tensors()[index] = tmp_tensor;
  } else {
    shared_experts_weights_[name] = tmp_tensor;
  }
}

void Qwen3MoeDecoderLoader::process_general_weights(
    const StateDict& state_dict,
    const std::string& name,
    const torch::Tensor& tensor) {
  const auto& weight_mapping = (quantize_type_.compare("w8a8_dynamic") == 0)
                                   ? WEIGHT_MAPPING_W8A8
                                   : WEIGHT_MAPPING;
  const auto& shard_map = (quantize_type_.compare("w8a8_dynamic") == 0)
                              ? WEIGHT_SHARD_W8A8
                              : WEIGHT_SHARD;

  if (weight_mapping.find(name) == weight_mapping.end()) {
    return;
  }

  const int index = get_mapped_index(name, weight_mapping);
  const bool is_sharded = shard_map.count(index);
  torch::Tensor tmp_tensor;
  int32_t tp_rank = dp_local_tp_rank_;
  int32_t tp_size = dp_local_tp_size_;

  static const std::unordered_set<int> qkv_tensor_indices = {IN_QKV_WEIGHT_1,
                                                             IN_QKV_WEIGHT_2,
                                                             IN_QKV_BIAS_1,
                                                             IN_QKV_BIAS_2,
                                                             IN_QKV_DESCALE_1,
                                                             IN_QKV_DESCALE_2,
                                                             IN_QKV_OFFSET_1,
                                                             IN_QKV_OFFSET_2,
                                                             IN_QKV_SCALE_1,
                                                             IN_QKV_SCALE_2};

  if (qkv_tensor_indices.count(index) > 0) {
    if (n_kv_heads_ < dp_local_tp_size_) {
      int32_t repeat_times = (dp_local_tp_size_ / n_kv_heads_);

      tp_rank = tp_rank / repeat_times;
      tp_size = n_kv_heads_;
    }
  }
  if (is_sharded) {
    tmp_tensor = get_sharded_tensor(
                     state_dict, name, shard_map.at(index), tp_rank, tp_size)
                     .to(target_device());
  } else {
    tmp_tensor = tensor.to(target_device());
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

torch::Tensor Qwen3MoeDecoderLoader::get_sharded_tensor(
    const StateDict& state_dict,
    const std::string& name,
    int dim) {
  if (parallel_args_.world_size() > 1) {
    return state_dict.get_sharded_tensor(
        name, dim, parallel_args_.rank(), parallel_args_.world_size());
  } else {
    return state_dict.get_tensor(name);
  }
}

torch::Tensor Qwen3MoeDecoderLoader::get_sharded_tensor(
    const StateDict& state_dict,
    const std::string& name,
    int dim,
    int local_tp_rank,
    int local_tp_size) {
  if (local_tp_size > 1) {
    return state_dict.get_sharded_tensor(
        name, dim, local_tp_rank, local_tp_size);
  } else {
    return state_dict.get_tensor(name);
  }
}

torch::Tensor Qwen3MoeDecoderLoader::merge_experts_weights(
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

void Qwen3MoeDecoderLoader::merge_host_at_weights() {
  merge_experts_weights();

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

  if (quantize_type_.compare("w8a8_dynamic") == 0) {
    t[IN_QKV_BIAS_0] = zero_fp16();
    t[IN_QKV_BIAS_1] = zero_fp16();
    t[IN_QKV_BIAS_2] = zero_fp16();
    t[IN_ATTENTION_OUT_BIAS] = zero_fp16();

    t[IN_QKV_DESCALE_0] = zero_fp16();
    t[IN_QKV_DESCALE_1] = zero_fp16();
    t[IN_QKV_DESCALE_2] = zero_fp16();
    t[IN_ATTENTION_OUT_DESCALE] = zero_fp16();

    t[IN_QKV_OFFSET_0] =
        torch::cat({t[IN_QKV_OFFSET_0], t[IN_QKV_OFFSET_1], t[IN_QKV_OFFSET_2]},
                   0)
            .contiguous()
            .view(-1);
    t[IN_QKV_OFFSET_1] = zero_fp16();
    t[IN_QKV_OFFSET_2] = zero_fp16();
    t[IN_ATTENTION_OUT_OFFSET] =
        t[IN_ATTENTION_OUT_OFFSET].contiguous().view(-1);

    t[IN_QKV_SCALE_0] =
        torch::cat({t[IN_QKV_SCALE_0], t[IN_QKV_SCALE_1], t[IN_QKV_SCALE_2]}, 0)
            .contiguous()
            .view(-1);
    t[IN_QKV_SCALE_1] = zero_fp16();
    t[IN_QKV_SCALE_2] = zero_fp16();
    t[IN_ATTENTION_OUT_SCALE] = t[IN_ATTENTION_OUT_SCALE].contiguous().view(-1);
  }
}

}  // namespace layer
}  // namespace xllm
