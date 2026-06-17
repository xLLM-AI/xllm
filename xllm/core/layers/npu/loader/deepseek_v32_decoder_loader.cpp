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
#include "deepseek_v32_decoder_loader.h"

#include <torch_npu/csrc/core/npu/NPUFormat.h>

#include "core/framework/config/eplb_config.h"
#include "deepseek_decoder_loader_constants.h"

namespace xllm {
namespace layer {

using namespace deepseek_v32_decoder_constants;

DeekseekV32DecoderLoader::DeekseekV32DecoderLoader(
    uint64_t weight_count,
    const ModelContext& context,
    int32_t layer_id,
    int32_t prefill_firstKDenseReplace,
    int32_t prefill_numOfDeviceExperts,
    int32_t prefill_qkRopeHeadDim,
    int32_t prefill_numAttentionHeadsPerRank,
    int32_t decode_worldSize,
    int32_t qk_nope_head_dim,
    int32_t kv_lora_rank,
    int32_t num_key_value_heads,
    int32_t v_head_dim,
    bool prefill_isBF16,
    bool decode_isBF16,
    LoadMode mode)
    : BaseLoader(weight_count, context, mode),
      layer_id_(layer_id),
      prefill_firstKDenseReplace_(prefill_firstKDenseReplace),
      prefill_numOfDeviceExperts_(prefill_numOfDeviceExperts),
      prefill_qkRopeHeadDim_(prefill_qkRopeHeadDim),
      prefill_numAttentionHeadsPerRank_(prefill_numAttentionHeadsPerRank),
      decode_worldSize_(decode_worldSize),
      qk_nope_head_dim_(qk_nope_head_dim),
      kv_lora_rank_(kv_lora_rank),
      num_key_value_heads_(num_key_value_heads),
      v_head_dim_(v_head_dim),
      prefill_isBF16_(prefill_isBF16),
      decode_isBF16_(decode_isBF16) {
  auto model_args = context.get_model_args();
  auto options = context.get_tensor_options();

  rank_ = parallel_args_.rank();
  first_k_dense_replace_ = model_args.first_k_dense_replace();
  n_layers_ = model_args.n_layers();
  num_experts_ = model_args.n_routed_experts();
  localWorldSize_ = parallel_args_.mapping().localWorldSize();
  ep_size_ = parallel_args_.ep_size();
  ep_local_tp_size_ = parallel_args_.world_size() / ep_size_;
  CHECK_EQ(parallel_args_.world_size(), ep_size_ * ep_local_tp_size_);
  ep_local_tp_rank_ = parallel_args_.rank() % ep_local_tp_size_;
  num_experts_per_partition_ = model_args.n_routed_experts() / ep_size_;
  redundant_experts_num_ =
      ::xllm::EPLBConfig::get_instance().redundant_experts_num();
  if (::xllm::EPLBConfig::get_instance().enable_eplb()) {
    num_experts_per_partition_ += redundant_experts_num_;
  }
  ep_rank_ = parallel_args_.rank() / ep_local_tp_size_;
  start_expert_id_ = ep_rank_ * num_experts_per_partition_;
  end_expert_id_ = start_expert_id_ + num_experts_per_partition_ - 1;
  initialize_tensors(options);
  initialize_weight_tensors(options);
}

void DeekseekV32DecoderLoader::initialize_tensors(
    const torch::TensorOptions& options) {
  tensor_placeholder_ = torch::zeros({1}, options.device(target_device()));
  reserve_experts_weights(prefill_numOfDeviceExperts_);
  initialize_device_expert_list(decode_worldSize_, num_experts_per_partition_);
}

void DeekseekV32DecoderLoader::load_state_dict(const StateDict& state_dict) {
  for (const auto& [name, tensor] : state_dict) {
    if (absl::EndsWith(name, "self_attn.kv_b_proj.weight")) {
      int index = WEIGHT_MAPPING_W8A8.at(name);
      set_kv_weight(state_dict, name, index, WEIGHT_SHARD_W8A8.at(index));
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

void DeekseekV32DecoderLoader::verify_loaded_weights(
    const std::string& prefix) const {
  const auto& t = working_tensors();
  for (const auto& [index, name] : WEIGHT_MAPPING) {
    CHECK(t[index].sizes() != std::vector<int64_t>({1}))
        << "weight is not loaded for " << prefix + name;
  }
}

int DeekseekV32DecoderLoader::extract_expert_index(const std::string& name) {
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

void DeekseekV32DecoderLoader::process_expert_weights(
    const StateDict& state_dict,
    const std::string& name,
    const torch::Tensor& tensor) {
  int expert_index = extract_expert_index(name);
  const std::string suffix = extract_endswith(name);
  const int index = get_mapped_index(suffix, WEIGHT_MAPPING_W8A8);
  if (index == -1) {
    return;
  }

  const bool is_sharded = WEIGHT_SHARD_W8A8.count(index);
  const bool needs_eplb =
      ::xllm::EPLBConfig::get_instance().enable_eplb() &&
      (rank_ % localWorldSize_ == expert_index % localWorldSize_);

  const int start_idx = ep_rank_ * num_experts_per_partition_;
  const int end_idx = (ep_rank_ + 1) * num_experts_per_partition_;
  const int safe_end =
      std::min(end_idx, static_cast<int>(device_expert_list_.size()));

  auto it = std::find(device_expert_list_.cbegin() + start_idx,
                      device_expert_list_.cbegin() + safe_end,
                      expert_index);
  const bool in_partition = it != device_expert_list_.cbegin() + safe_end;

  if (!needs_eplb && !in_partition) {
    return;
  }

  torch::Tensor processed_tensor;
  {
    std::lock_guard<std::mutex> lock(experts_mutex_);
    processed_tensor = is_sharded
                           ? get_sharded_tensor(state_dict,
                                                name,
                                                WEIGHT_SHARD_W8A8.at(index),
                                                ep_local_tp_rank_,
                                                ep_local_tp_size_)
                           : tensor;

    if (!decode_isBF16_) {
      if (absl::EndsWith(name, "_offset")) {
        processed_tensor = processed_tensor.to(torch::kFloat16);
      } else if (absl::EndsWith(name, "_scale")) {
        processed_tensor = processed_tensor.to(torch::kFloat32);
      }
    }
  }

  if (needs_eplb) {
    std::lock_guard<std::mutex> lock(experts_mutex_);
    std::string shm_key = get_expert_shm_key(layer_id_, expert_index, suffix);
    shared_buffer_->add_tensor(expert_index,
                               layer_id_ - first_k_dense_replace_,
                               shm_key,
                               processed_tensor.contiguous());
  }

  if (in_partition) {
    std::vector<size_t> matches_pos;
    for (auto iter = it; iter != device_expert_list_.cbegin() + safe_end;
         ++iter) {
      if (*iter == expert_index) {
        matches_pos.emplace_back(
            std::distance(device_expert_list_.cbegin(), iter) - start_idx);
      }
    }

    if (!matches_pos.empty()) {
      std::lock_guard<std::mutex> lock(experts_mutex_);
      for (auto pos : matches_pos) {
        experts_weights_[suffix][pos] = processed_tensor.clone();
      }
    }
  }
}

void DeekseekV32DecoderLoader::initialize_weight_tensors(
    const torch::TensorOptions& options) {
  auto& t = working_tensors();
  for (uint64_t i = 0; i < weight_count_; ++i) {
    t[i] = torch::zeros({1}, options.device(target_device()));
  }

  if (::xllm::EPLBConfig::get_instance().enable_eplb()) {
    const int64_t size =
        50LL * 1024LL * 1024LL * int64_t(n_layers_ - first_k_dense_replace_);
    shared_buffer_ = std::make_unique<ExpertBufferManager>(
        num_experts_, n_layers_ - first_k_dense_replace_, size);
  }
}

void DeekseekV32DecoderLoader::convert_offsets_to_int8() {
  auto& t = working_tensors();
  auto convert_to_int8 = [this, &t](int index) {
    t[index] = t[index].to(torch::kInt8);
    if (!load_to_host()) {
      t[index] = t[index].to(target_device());
    }
  };
  convert_to_int8(IN_Q_PROJ_A_OFFSET);
  convert_to_int8(IN_Q_PROJ_B_OFFSET);
  convert_to_int8(IN_KV_PROJ_WITH_MQA_OFFSET);
  convert_to_int8(IN_ATTENTION_OUT_OFFSET);
}

void DeekseekV32DecoderLoader::handle_device_specific_bias() {
  auto& t = working_tensors();
  if (dp_local_tp_rank_ != 0) {
    torch::Tensor original_tensor = t[IN_ATTENTION_OUT_BIAS];
    t[IN_ATTENTION_OUT_BIAS] =
        torch::zeros(original_tensor.sizes(),
                     torch::TensorOptions()
                         .dtype(original_tensor.dtype())
                         .device(original_tensor.device()));
  }
}

std::string DeekseekV32DecoderLoader::extract_endswith(
    const std::string& input) {
  std::vector<std::string> parts;
  std::stringstream ss(input);
  std::string part;
  while (std::getline(ss, part, '.')) {
    parts.emplace_back(part);
  }
  if (parts.size() < 2) {
    return "";
  }
  std::string result = parts[parts.size() - 2] + "." + parts[parts.size() - 1];
  return result;
}

torch::Tensor DeekseekV32DecoderLoader::get_sharded_tensor(
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

torch::Tensor DeekseekV32DecoderLoader::get_sharded_tensor(
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

int DeekseekV32DecoderLoader::get_mapped_index(
    const std::string& name,
    const std::unordered_map<std::string, int>& mapping) {
  const auto it = mapping.find(name);
  if (it == mapping.end()) {
    LOG(WARNING) << "Parameter '" << name
                 << "' not found in mapping and will not be used.";
    return -1;
  }
  return it->second;
}

void DeekseekV32DecoderLoader::squeeze_experts_weights() {
  auto& t = working_tensors();
  for (const auto& index : SQUEEZE_WEIGHT_VEC) {
    if (t[index].dim() > 1) {
      t[index] = t[index].squeeze();
    }
  }
}

void DeekseekV32DecoderLoader::process_general_weights(
    const StateDict& state_dict,
    const std::string& name,
    const torch::Tensor& tensor) {
  const int index = get_mapped_index(name, WEIGHT_MAPPING_W8A8);
  if (index == -1) {
    return;
  }
  const bool is_sharded = WEIGHT_SHARD_W8A8.count(index);
  torch::Tensor tmp_tensor;

  tmp_tensor = is_sharded ? get_sharded_tensor(state_dict,
                                               name,
                                               WEIGHT_SHARD_W8A8.at(index),
                                               dp_local_tp_rank_,
                                               dp_local_tp_size_)
                                .to(target_device())
                          : tensor.to(target_device());

  correct_tensor_dtype(tmp_tensor, name);
  auto& t = working_tensors();
  t[index] = tmp_tensor;
  if (absl::StartsWith(name, "self_attn.q_a_proj")) {
    const int index_re = get_mapped_index(name, WEIGHT_MAPPING_W8A8_RECOMPUTE);
    torch::Tensor tmp_tensor_re = tensor.to(target_device());
    t[index_re] = tmp_tensor_re;
  }
  if (layer_id_ != n_layers_ && absl::StrContains(name, "layernorm.weight")) {
    t[index + 1] = torch::zeros_like(tmp_tensor);
  }
}

void DeekseekV32DecoderLoader::process_mlp_common_weights(
    const StateDict& state_dict,
    const std::string& name,
    const torch::Tensor& tensor) {
  const int index = get_mapped_index(name, WEIGHT_MAPPING_W8A8);
  if (index == -1) {
    return;
  }
  const bool is_sharded = WEIGHT_SHARD_W8A8.count(index);
  std::lock_guard<std::mutex> lock(shared_experts_mutex_);

  torch::Tensor tmp_tensor =
      is_sharded ? get_sharded_tensor(state_dict,
                                      name,
                                      WEIGHT_SHARD_W8A8.at(index),
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

void DeekseekV32DecoderLoader::merge_experts_weights() {
  auto& t = working_tensors();
  torch::Tensor mlp_gateup_weight =
      merge_experts_weights(experts_weights_["gate_proj.weight"],
                            experts_weights_["up_proj.weight"],
                            /*transpose=*/true);
  // IN_MLP_GATEUP_WEIGHT_EXPERT: always NZ (both modes agree).
  t[IN_MLP_GATEUP_WEIGHT_EXPERT] =
      cast_nz(mlp_gateup_weight, IN_MLP_GATEUP_WEIGHT_EXPERT);
  if (quantize_type_ == "w8a8_dynamic") {
    t[IN_MLP_GATEUP_OFFSET_EXPERT] =
        merge_experts_weights(experts_weights_["gate_proj.weight_offset"],
                              experts_weights_["up_proj.weight_offset"]);
    t[IN_MLP_GATEUP_SCALE_EXPERT] =
        merge_experts_weights(experts_weights_["gate_proj.weight_scale"],
                              experts_weights_["up_proj.weight_scale"]);
  }

  // IN_MLP_DOWN_WEIGHT_EXPERT: eager always NZ; manual NZ on A3 else ND.
  torch::Tensor mlp_down_weight = merge_experts_weights(
      experts_weights_["down_proj.weight"], /*transpose=*/false);
  bool down_is_nz;
  if (load_to_host()) {
#if defined(USE_A3)
    down_is_nz = true;
#else
    down_is_nz = false;
#endif
  } else {
    down_is_nz = true;
  }
  if (down_is_nz) {
    if (load_to_host()) {
      nz_indices_.insert(IN_MLP_DOWN_WEIGHT_EXPERT);
      t[IN_MLP_DOWN_WEIGHT_EXPERT] = mlp_down_weight.contiguous();
    } else {
      t[IN_MLP_DOWN_WEIGHT_EXPERT] = at_npu::native::npu_format_cast(
                                         mlp_down_weight, ACL_FORMAT_FRACTAL_NZ)
                                         .contiguous();
    }
  } else {
    t[IN_MLP_DOWN_WEIGHT_EXPERT] = mlp_down_weight.contiguous();
  }

  if (quantize_type_ == "w8a8_dynamic") {
    t[IN_MLP_DOWN_OFFSET_EXPERT] =
        merge_experts_weights(experts_weights_["down_proj.weight_offset"]);
    t[IN_MLP_DOWN_SCALE_EXPERT] =
        merge_experts_weights(experts_weights_["down_proj.weight_scale"]);
  }
}

torch::Tensor DeekseekV32DecoderLoader::merge_experts_weights(
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

torch::Tensor DeekseekV32DecoderLoader::merge_experts_weights(
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

void DeekseekV32DecoderLoader::process_shared_expert_weights(
    const StateDict& state_dict,
    const std::string& name,
    const torch::Tensor& tensor) {
  torch::Tensor tmp_tensor;
  std::lock_guard<std::mutex> lock(shared_experts_mutex_);
  const int index = get_mapped_index(name, WEIGHT_MAPPING_W8A8);
  if (index == -1) {
    return;
  }

  const bool is_sharded = WEIGHT_SHARD_W8A8.count(index);

  if (::xllm::EPLBConfig::get_instance().expert_parallel_degree() == 2) {
    tmp_tensor = tensor.to(target_device());
  } else if (layer_id_ < first_k_dense_replace_) {
    tmp_tensor = is_sharded ? get_sharded_tensor(state_dict,
                                                 name,
                                                 WEIGHT_SHARD_W8A8.at(index),
                                                 dp_local_tp_rank_,
                                                 dp_local_tp_size_)
                                  .to(target_device())
                            : tensor.to(target_device());
  } else {
    tmp_tensor = is_sharded ? get_sharded_tensor(
                                  state_dict, name, WEIGHT_SHARD_W8A8.at(index))
                                  .to(target_device())
                            : tensor.to(target_device());
  }

  if (absl::StrContains(name, "down_proj")) {
    working_tensors()[index] = tmp_tensor;
  } else {
    shared_experts_weights_[name] = tmp_tensor;
  }
}

void DeekseekV32DecoderLoader::set_kv_weight(const StateDict& state_dict,
                                             const std::string& tensor_name,
                                             int weight_position,
                                             int dim) {
  torch::Tensor mutable_tensor;
  if (parallel_args_.world_size() <= 1) {
    mutable_tensor = state_dict.get_tensor(tensor_name).to(target_device());
    correct_tensor_dtype(mutable_tensor, tensor_name);
  } else {
    mutable_tensor =
        get_sharded_tensor(
            state_dict, tensor_name, dim, dp_local_tp_rank_, dp_local_tp_size_)
            .to(target_device());
    correct_tensor_dtype(mutable_tensor, tensor_name);
  }

  torch::Tensor kv_b_proj_weight =
      mutable_tensor.reshape({num_key_value_heads_ / dp_local_tp_size_,
                              qk_nope_head_dim_ + v_head_dim_,
                              kv_lora_rank_});
  torch::Tensor k_b_proj_preprocessed =
      kv_b_proj_weight.slice(1, 0, qk_nope_head_dim_).contiguous();
  torch::Tensor v_b_proj_preprocessed =
      kv_b_proj_weight
          .slice(1, qk_nope_head_dim_, qk_nope_head_dim_ + v_head_dim_)
          .transpose(1, 2)
          .contiguous();
  auto& t = working_tensors();
  t[weight_position] = k_b_proj_preprocessed.to(target_device());
  t[weight_position + 6] = v_b_proj_preprocessed.to(target_device());
}

void DeekseekV32DecoderLoader::preprocess_linear_for_rope() {
  auto& t = working_tensors();
  for (const auto& name : LINEAR_FOR_ROPE) {
    if (quantize_type_ == "") {
      if (!absl::EndsWith(name, "weight")) {
        continue;
      }
    }
    int index = WEIGHT_MAPPING_W8A8.at(name);
    t[index] = view_tensor(t[index], name, true);
    t[index] = trans_rope_weight(t[index]);
    t[index] = (!absl::EndsWith(name, "weight"))
                   ? view_tensor(t[index], name, false).flatten()
                   : view_tensor(t[index], name, false);
  }
}

torch::Tensor DeekseekV32DecoderLoader::view_tensor(torch::Tensor weight,
                                                    const std::string& name,
                                                    bool pre_view) {
  if (absl::StrContains(name, "q_b_proj")) {
    if (pre_view) {
      return weight
          .view({prefill_numAttentionHeadsPerRank_,
                 qk_nope_head_dim_ + prefill_qkRopeHeadDim_,
                 -1})
          .contiguous();
    } else {
      return weight
          .view({prefill_numAttentionHeadsPerRank_ *
                     (qk_nope_head_dim_ + prefill_qkRopeHeadDim_),
                 -1})
          .contiguous();
    }
  } else if (absl::StrContains(name, "kv_a_proj_with_mqa")) {
    return weight.view({kv_lora_rank_ + prefill_qkRopeHeadDim_, -1})
        .contiguous();
  }
  return weight;
}

torch::Tensor DeekseekV32DecoderLoader::trans_rope_weight(
    torch::Tensor weight) {
  // Manual mode clones the tensor so shared host buffers are not mutated;
  // eager mode mutates in place (matching original behavior).
  if (load_to_host()) {
    auto host_weight = weight.clone();
    int64_t d = weight.size(-2);
    int64_t rope_dim = prefill_qkRopeHeadDim_;
    torch::Tensor weight_1 =
        weight.slice(-2, d - rope_dim, torch::indexing::None, 2).contiguous();
    torch::Tensor weight_2 =
        weight.slice(-2, d - rope_dim + 1, torch::indexing::None, 2)
            .contiguous();
    torch::Tensor combined = torch::cat({weight_1, weight_2}, -2);
    host_weight.slice(-2, d - rope_dim, d).copy_(combined);
    return host_weight.contiguous();
  }
  int64_t d = weight.size(-2);
  int64_t rope_dim = prefill_qkRopeHeadDim_;
  torch::Tensor weight_1 =
      weight.slice(-2, d - rope_dim, torch::indexing::None, 2).contiguous();
  torch::Tensor weight_2 =
      weight.slice(-2, d - rope_dim + 1, torch::indexing::None, 2).contiguous();
  torch::Tensor combined = torch::cat({weight_1, weight_2}, -2);
  weight.slice(-2, d - rope_dim, d).copy_(combined);
  return weight.contiguous();
}

void DeekseekV32DecoderLoader::initialize_device_expert_list(
    int num_device,
    int num_device_expert) {
  int32_t num_device_route_expert = num_device_expert;
  if (::xllm::EPLBConfig::get_instance().enable_eplb()) {
    num_device_route_expert = num_device_expert - redundant_experts_num_;
  }
  for (int i = 0; i < num_device * num_device_route_expert; ++i) {
    device_expert_list_.emplace_back(i);
    if (::xllm::EPLBConfig::get_instance().enable_eplb() &&
        (i + 1) % num_device_route_expert == 0) {
      for (int redundant_expert = 0; redundant_expert < redundant_experts_num_;
           ++redundant_expert)
        device_expert_list_.emplace_back(i);
    }
  }
}

torch::Tensor DeekseekV32DecoderLoader::convert_fp16_to_int64(
    const torch::Tensor& fp16_tensor) {
  auto float_tensor = fp16_tensor.to(torch::kFloat32);
  auto int32_tensor = float_tensor.view(torch::kInt32);
  auto int64_tensor = int32_tensor.to(torch::kInt64);
  return int64_tensor;
}

void DeekseekV32DecoderLoader::convert_descaled_weights_to_float() {
  auto& t = working_tensors();
  auto convert_to_float = [&t](int index) {
    t[index] = t[index].to(torch::kFloat32);
  };
  convert_to_float(IN_Q_PROJ_A_DESCALE);
  convert_to_float(IN_Q_PROJ_B_DESCALE);
  convert_to_float(IN_KV_PROJ_WITH_MQA_DESCALE);
  convert_to_float(IN_ATTENTION_OUT_DESCALE);
}

void DeekseekV32DecoderLoader::reserve_experts_weights(
    int num_of_device_experts) {
  experts_weights_.clear();
  std::vector<std::string> weight_names = {
      "gate_proj.weight", "up_proj.weight", "down_proj.weight"};
  if (quantize_type_ == "w8a8_dynamic") {
    weight_names.emplace_back("gate_proj.weight_offset");
    weight_names.emplace_back("up_proj.weight_offset");
    weight_names.emplace_back("down_proj.weight_offset");
    weight_names.emplace_back("gate_proj.weight_scale");
    weight_names.emplace_back("up_proj.weight_scale");
    weight_names.emplace_back("down_proj.weight_scale");
  }
  std::lock_guard<std::mutex> lock(experts_mutex_);
  for (const auto& weight_name : weight_names) {
    experts_weights_[weight_name] =
        std::vector<torch::Tensor>(num_of_device_experts);
  }
}

std::string DeekseekV32DecoderLoader::get_expert_shm_key(
    int32_t layer_id,
    int32_t expert_index,
    const std::string& suffix) {
  std::string shm_key =
      "layer_" + std::to_string(layer_id - first_k_dense_replace_) + "_" +
      "expert_" + std::to_string(expert_index) + "_" + suffix;
  return shm_key;
}

void DeekseekV32DecoderLoader::merge_shared_experts_weights() {
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

  if (layer_id_ >= prefill_firstKDenseReplace_) {
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
    }
  }
}

void DeekseekV32DecoderLoader::merge_host_at_weights() {
  auto& t = working_tensors();
  if (quantize_type_ == "w8a8_dynamic") {
    if (prefill_isBF16_) {
      convert_descaled_weights_to_float();
    }
    convert_offsets_to_int8();
    handle_device_specific_bias();
  }

  merge_shared_experts_weights();
  if (layer_id_ >= prefill_firstKDenseReplace_) {
    merge_experts_weights();
  }

  squeeze_experts_weights();

  preprocess_linear_for_rope();

  t[IN_Q_PROJ_A_WEIGHT] =
      torch::cat({t[IN_KV_PROJ_WITH_MQA_WEIGHT], t[IN_Q_PROJ_A_WEIGHT]}, 0)
          .contiguous();
  if (quantize_type_ == "w8a8_dynamic") {
    t[IN_Q_PROJ_A_BIAS] =
        torch::cat({t[IN_KV_PROJ_WITH_MQA_BIAS], t[IN_Q_PROJ_A_BIAS]}, 0)
            .contiguous();
    t[IN_Q_PROJ_A_DESCALE] =
        torch::cat({t[IN_KV_PROJ_WITH_MQA_DESCALE], t[IN_Q_PROJ_A_DESCALE]}, 0)
            .contiguous();
  }

  // IN_Q_PROJ_A_WEIGHT, IN_Q_PROJ_A_RECOMPUTE_WEIGHT, IN_Q_PROJ_B_WEIGHT:
  // always NZ on device (both modes agree).
  t[IN_Q_PROJ_A_WEIGHT] = cast_nz(t[IN_Q_PROJ_A_WEIGHT], IN_Q_PROJ_A_WEIGHT);
  t[IN_Q_PROJ_A_RECOMPUTE_WEIGHT] =
      cast_nz(t[IN_Q_PROJ_A_RECOMPUTE_WEIGHT], IN_Q_PROJ_A_RECOMPUTE_WEIGHT);
  t[IN_Q_PROJ_B_WEIGHT] = cast_nz(t[IN_Q_PROJ_B_WEIGHT], IN_Q_PROJ_B_WEIGHT);

  t[IN_KV_PROJ_WITH_MQA_WEIGHT] = tensor_placeholder_;
  t[IN_KV_PROJ_WITH_MQA_BIAS] = tensor_placeholder_;
  t[IN_KV_PROJ_WITH_MQA_DESCALE] = tensor_placeholder_;
  t[IN_KV_PROJ_WITH_MQA_OFFSET] = tensor_placeholder_;
  t[IN_KV_PROJ_WITH_MQA_SCALE] = tensor_placeholder_;
  if (::xllm::EPLBConfig::get_instance().expert_parallel_degree() != 2) {
    t[IN_BLOCK_SPARSE_MOE_GATE_WEIGHT] =
        torch::roll(t[IN_BLOCK_SPARSE_MOE_GATE_WEIGHT],
                    {-1 * ep_rank_ * num_experts_per_partition_},
                    {0})
            .contiguous();
    t[IN_BLOCK_SPARSE_MOE_GATE_BIAS] =
        torch::roll(t[IN_BLOCK_SPARSE_MOE_GATE_BIAS],
                    {-1 * ep_rank_ * num_experts_per_partition_},
                    {0})
            .to(torch::kFloat32)
            .contiguous();
  }
  t[IN_BLOCK_SPARSE_MOE_GATE_WEIGHT] =
      t[IN_BLOCK_SPARSE_MOE_GATE_WEIGHT].to(torch::kFloat32);
  t[IN_MLP_GATEUP_SCALE_EXPERT] =
      t[IN_MLP_GATEUP_SCALE_EXPERT].to(torch::kBFloat16);
  t[IN_MLP_GATEUP_SCALE_SHARED_EXPERT] =
      t[IN_MLP_GATEUP_SCALE_SHARED_EXPERT].to(torch::kBFloat16);
  t[IN_MLP_DOWN_SCALE_EXPERT] =
      t[IN_MLP_DOWN_SCALE_EXPERT].to(torch::kBFloat16);
  t[IN_MLP_DOWN_SCALE_SHARED_EXPERT] =
      t[IN_MLP_DOWN_SCALE_SHARED_EXPERT].to(torch::kBFloat16);
  if (quantize_type_ == "w8a8_dynamic") {
    if (!prefill_isBF16_) {
      t[IN_Q_PROJ_A_DESCALE] = convert_fp16_to_int64(t[IN_Q_PROJ_A_DESCALE]);
      t[IN_Q_PROJ_A_RECOMPUTE_DESCALE] =
          convert_fp16_to_int64(t[IN_Q_PROJ_A_RECOMPUTE_DESCALE]);
      t[IN_Q_PROJ_B_DESCALE] = convert_fp16_to_int64(t[IN_Q_PROJ_B_DESCALE]);
      t[IN_ATTENTION_OUT_DESCALE] =
          convert_fp16_to_int64(t[IN_ATTENTION_OUT_DESCALE]);

      t[IN_MLP_GATEUP_OFFSET_SHARED_EXPERT] =
          t[IN_MLP_GATEUP_OFFSET_SHARED_EXPERT].to(torch::kFloat16);
      t[IN_MLP_GATEUP_SCALE_SHARED_EXPERT] =
          t[IN_MLP_GATEUP_SCALE_SHARED_EXPERT].to(torch::kFloat32);
      t[IN_MLP_DOWN_SCALE_SHARED_EXPERT] =
          t[IN_MLP_DOWN_SCALE_SHARED_EXPERT].to(torch::kFloat32);
      t[IN_MLP_GATEUP_OFFSET_EXPERT] =
          t[IN_MLP_GATEUP_OFFSET_EXPERT].to(torch::kFloat16);
      t[IN_MLP_DOWN_OFFSET_EXPERT] =
          t[IN_MLP_DOWN_OFFSET_EXPERT].to(torch::kFloat16);
      t[IN_MLP_DOWN_SCALE_EXPERT] =
          t[IN_MLP_DOWN_SCALE_EXPERT].to(torch::kFloat32);
    }
  }
}

}  // namespace layer
}  // namespace xllm
