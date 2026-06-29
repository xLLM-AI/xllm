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
#include "deepseek_v32_decoder_loader.h"

#include <torch_npu/csrc/core/npu/NPUFormat.h>

#include <algorithm>
#include <optional>
#include <sstream>

#include "core/framework/config/eplb_config.h"
#include "core/kernels/ops_api.h"
#include "core/layers/npu/npu_base_layer.h"
#include "deepseek_decoder_loader_constants.h"

namespace xllm {
namespace layer {

using namespace deepseek_v32_decoder_constants;

namespace {
constexpr int32_t kQProjALinearIndex = 0;
constexpr int32_t kQProjBLinearIndex = 1;
constexpr int32_t kKvAProjLinearIndex = 2;
constexpr int32_t kOLinearIndex = 5;
constexpr int32_t kIndexerWqBLinearIndex = 6;

std::string tensor_shape_string(const torch::Tensor& tensor) {
  std::ostringstream oss;
  oss << "[";
  for (int64_t i = 0; i < tensor.dim(); ++i) {
    if (i > 0) {
      oss << ", ";
    }
    oss << tensor.size(i);
  }
  oss << "]";
  return oss.str();
}

std::string tensor_debug_string(const torch::Tensor& tensor) {
  if (!tensor.defined()) {
    return "undefined";
  }
  std::ostringstream oss;
  oss << "shape=" << tensor_shape_string(tensor)
      << ", dtype=" << tensor.scalar_type() << ", device=" << tensor.device()
      << ", contiguous=" << tensor.is_contiguous();
  return oss.str();
}

bool is_placeholder_tensor(const torch::Tensor& tensor) {
  return !tensor.defined() || (tensor.dim() == 1 && tensor.size(0) == 1);
}

bool has_any_defined_expert(const std::vector<torch::Tensor>& experts) {
  return std::any_of(experts.begin(), experts.end(), [](const auto& tensor) {
    return tensor.defined();
  });
}

void check_cat_input_tensor(const torch::Tensor& tensor,
                            const std::string& name,
                            const std::string& cat_name,
                            int32_t layer_id) {
  CHECK(tensor.defined()) << "GLM/DeepSeekV32 layer " << layer_id
                          << " missing tensor before cat " << cat_name << ": "
                          << name;
  CHECK(!is_placeholder_tensor(tensor))
      << "GLM/DeepSeekV32 layer " << layer_id
      << " placeholder tensor before cat " << cat_name << ": " << name
      << ", tensor=" << tensor_debug_string(tensor);
}

torch::Tensor cat_with_debug(const std::vector<torch::Tensor>& tensors,
                             int64_t dim,
                             const std::vector<std::string>& names,
                             const std::string& cat_name,
                             int32_t layer_id) {
  CHECK_EQ(tensors.size(), names.size())
      << "GLM/DeepSeekV32 layer " << layer_id
      << " internal cat debug size mismatch for " << cat_name;
  for (size_t i = 0; i < tensors.size(); ++i) {
    check_cat_input_tensor(tensors[i], names[i], cat_name, layer_id);
  }
  const int64_t rank = tensors[0].dim();
  const int64_t normalized_dim = dim < 0 ? dim + rank : dim;
  CHECK_GE(normalized_dim, 0)
      << "GLM/DeepSeekV32 layer " << layer_id << " invalid cat dim for "
      << cat_name << ": dim=" << dim
      << ", tensor=" << tensor_debug_string(tensors[0]);
  CHECK_LT(normalized_dim, rank)
      << "GLM/DeepSeekV32 layer " << layer_id << " invalid cat dim for "
      << cat_name << ": dim=" << dim
      << ", tensor=" << tensor_debug_string(tensors[0]);
  for (size_t i = 1; i < tensors.size(); ++i) {
    CHECK_EQ(tensors[i].dim(), rank)
        << "GLM/DeepSeekV32 layer " << layer_id
        << " tensor rank mismatch before cat " << cat_name << ": " << names[0]
        << "=" << tensor_debug_string(tensors[0]) << ", " << names[i] << "="
        << tensor_debug_string(tensors[i]);
    for (int64_t axis = 0; axis < rank; ++axis) {
      if (axis == normalized_dim) {
        continue;
      }
      CHECK_EQ(tensors[i].size(axis), tensors[0].size(axis))
          << "GLM/DeepSeekV32 layer " << layer_id
          << " tensor shape mismatch before cat " << cat_name << ": "
          << names[0] << "=" << tensor_debug_string(tensors[0]) << ", "
          << names[i] << "=" << tensor_debug_string(tensors[i])
          << ", dim=" << dim;
    }
  }
  return torch::cat(tensors, dim);
}

void check_required_w4a8_tensor(const torch::Tensor& tensor,
                                const std::string& name,
                                int32_t layer_id) {
  CHECK(tensor.defined()) << "GLM/DeepSeekV32 W4A8 layer " << layer_id
                          << " missing required tensor " << name;
  CHECK(!is_placeholder_tensor(tensor))
      << "GLM/DeepSeekV32 W4A8 layer " << layer_id
      << " required tensor still placeholder: " << name
      << ", tensor=" << tensor_debug_string(tensor);
}

void check_expert_vector(const std::vector<torch::Tensor>& experts,
                         const std::string& name,
                         int32_t layer_id) {
  CHECK(!experts.empty()) << "GLM/DeepSeekV32 W4A8 layer " << layer_id
                          << " expert vector is empty: " << name;
  size_t missing_count = 0;
  size_t first_missing = experts.size();
  size_t first_defined = experts.size();
  for (size_t i = 0; i < experts.size(); ++i) {
    if (!experts[i].defined()) {
      if (first_missing == experts.size()) {
        first_missing = i;
      }
      ++missing_count;
    } else if (first_defined == experts.size()) {
      first_defined = i;
    }
  }
  CHECK_EQ(missing_count, 0)
      << "GLM/DeepSeekV32 W4A8 layer " << layer_id
      << " has missing expert tensors for " << name
      << ": total=" << experts.size() << ", missing=" << missing_count
      << ", first_missing=" << first_missing << ", first_defined="
      << (first_defined == experts.size() ? std::string("<none>")
                                          : std::to_string(first_defined));
}

void check_expert_vector_pair(const std::vector<torch::Tensor>& experts_gate,
                              const std::vector<torch::Tensor>& experts_up,
                              const std::string& name,
                              int32_t layer_id) {
  CHECK_EQ(experts_gate.size(), experts_up.size())
      << "GLM/DeepSeekV32 W4A8 layer " << layer_id
      << " expert vector size mismatch for " << name
      << ": gate=" << experts_gate.size() << ", up=" << experts_up.size();
  check_expert_vector(experts_gate, name + ".gate", layer_id);
  check_expert_vector(experts_up, name + ".up", layer_id);
}
}  // namespace

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
    const std::vector<int32_t>& attn_linear_quant_types,
    bool skip_topk,
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
      decode_isBF16_(decode_isBF16),
      skip_topk_(skip_topk),
      indexer_rope_interleave_(false),
      attn_linear_quant_types_(attn_linear_quant_types) {
  auto model_args = context.get_model_args();
  auto quant_args = context.get_quant_args();
  auto options = context.get_tensor_options();

  rank_ = parallel_args_.rank();
  first_k_dense_replace_ = model_args.first_k_dense_replace();
  n_layers_ = model_args.n_layers();
  num_experts_ = model_args.n_routed_experts();
  index_n_heads_ = model_args.index_n_heads();
  index_head_dim_ = model_args.index_head_dim();
  indexer_rope_interleave_ = model_args.indexer_rope_interleave();
  quant_group_size_ = static_cast<int32_t>(quant_args.group_size());
  if (quantize_type_ == "w4a8_dynamic") {
    CHECK_GE(quant_group_size_, 0)
        << "W4A8_DYNAMIC group_size must be >= 0, got " << quant_group_size_;
    CHECK_EQ(quant_args.quant_version(), "1.0.0")
        << "W4A8_DYNAMIC only supports quant_version 1.0.0, got "
        << (quant_args.quant_version().empty() ? "<empty>"
                                               : quant_args.quant_version());
    CHECK(!load_to_host())
        << "W4A8_DYNAMIC MoE ATB path currently requires eager loader because "
        << "manual loader cannot preserve the routed expert W4 packed-NZ "
        << "layout during CPU staging.";
  }
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

bool DeekseekV32DecoderLoader::use_quant_weight_mapping() const {
  return quantize_type_ == "w8a8_dynamic" || quantize_type_ == "w4a8_dynamic";
}

bool DeekseekV32DecoderLoader::is_attn_dynamic_desc(int32_t index) const {
  if (index < 0) {
    return false;
  }
  const size_t linear_index = static_cast<size_t>(index);
  return linear_index < attn_linear_quant_types_.size() &&
         attn_linear_quant_types_[linear_index] ==
             static_cast<int32_t>(LinearTypeV2::W8A8_DYNAMIC);
}

bool DeekseekV32DecoderLoader::is_attn_quant_desc(int32_t index) const {
  if (index < 0) {
    return false;
  }
  const size_t linear_index = static_cast<size_t>(index);
  return linear_index < attn_linear_quant_types_.size() &&
         (attn_linear_quant_types_[linear_index] ==
              static_cast<int32_t>(LinearTypeV2::W8A8) ||
          attn_linear_quant_types_[linear_index] ==
              static_cast<int32_t>(LinearTypeV2::W8A8_DYNAMIC));
}

bool DeekseekV32DecoderLoader::should_skip_indexer_weight(
    const std::string& name) const {
  return skip_topk_ && absl::StartsWith(name, "self_attn.indexer.");
}

void DeekseekV32DecoderLoader::reset_skipped_indexer_weights() {
  if (!skip_topk_) {
    return;
  }
  auto& t = working_tensors();
  for (int32_t index = IN_INDEXER_WQ_B_WEIGHT;
       index <= IN_INDEXER_PROJ_COMPRESS_IDX;
       ++index) {
    t[index] = tensor_placeholder_;
  }
}

int DeekseekV32DecoderLoader::get_w4a8_expert_shard_dim(
    const std::string& suffix) const {
  if (absl::StartsWith(suffix, "gate_proj.") ||
      absl::StartsWith(suffix, "up_proj.")) {
    return 0;
  }
  if (absl::StartsWith(suffix, "down_proj.")) {
    return 1;
  }
  return -1;
}

void DeekseekV32DecoderLoader::process_expert_weights(
    const StateDict& state_dict,
    const std::string& name,
    const torch::Tensor& tensor) {
  int expert_index = extract_expert_index(name);
  const std::string suffix = extract_endswith(name);
  const bool is_w4a8_extra = quantize_type_ == "w4a8_dynamic" &&
                             (absl::EndsWith(suffix, "weight_scale_second") ||
                              absl::EndsWith(suffix, "weight_offset") ||
                              absl::EndsWith(suffix, "scale_bias"));
  int32_t index = -1;
  int32_t shard_dim = -1;
  if (is_w4a8_extra) {
    shard_dim = get_w4a8_expert_shard_dim(suffix);
  } else {
    index = get_mapped_index(suffix, WEIGHT_MAPPING_W8A8);
    if (index == -1) {
      return;
    }
    if (WEIGHT_SHARD_W8A8.count(index) > 0) {
      shard_dim = WEIGHT_SHARD_W8A8.at(index);
    }
  }

  const bool is_sharded = shard_dim >= 0;
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
    processed_tensor = is_sharded ? get_sharded_tensor(state_dict,
                                                       name,
                                                       shard_dim,
                                                       ep_local_tp_rank_,
                                                       ep_local_tp_size_)
                                  : tensor;

    if (quantize_type_ == "w8a8_dynamic" && !decode_isBF16_) {
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
      auto experts_it = experts_weights_.find(suffix);
      CHECK(experts_it != experts_weights_.end())
          << "GLM/DeepSeekV32 W4A8 layer " << layer_id_
          << " routed expert suffix is not reserved: " << suffix
          << ", tensor=" << tensor_debug_string(processed_tensor);
      for (auto pos : matches_pos) {
        CHECK_LT(pos, experts_it->second.size())
            << "GLM/DeepSeekV32 W4A8 layer " << layer_id_
            << " routed expert position out of range: suffix=" << suffix
            << ", pos=" << pos
            << ", reserved_size=" << experts_it->second.size();
        experts_it->second[pos] = processed_tensor.clone();
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
  auto convert_to_int8 = [this, &t](int32_t index) {
    t[index] = t[index].to(torch::kInt8);
    if (!load_to_host()) {
      t[index] = t[index].to(target_device());
    }
  };
  if (!is_attn_dynamic_desc(kQProjALinearIndex)) {
    convert_to_int8(IN_Q_PROJ_A_OFFSET);
  }
  if (!is_attn_dynamic_desc(kKvAProjLinearIndex)) {
    convert_to_int8(IN_KV_PROJ_WITH_MQA_OFFSET);
  }
  if (!is_attn_dynamic_desc(kQProjBLinearIndex)) {
    convert_to_int8(IN_Q_PROJ_B_OFFSET);
  }
  if (!is_attn_dynamic_desc(kIndexerWqBLinearIndex)) {
    convert_to_int8(IN_INDEXER_WQ_B_OFFSET);
  }
  if (!is_attn_dynamic_desc(kOLinearIndex)) {
    convert_to_int8(IN_ATTENTION_OUT_OFFSET);
  }
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
  if (should_skip_indexer_weight(name)) {
    return;
  }
  const int index = get_mapped_index(name, WEIGHT_MAPPING_W8A8);
  if (index == -1) {
    return;
  }
  const bool is_dynamic_q_b_quant_param =
      use_quant_weight_mapping() && is_attn_dynamic_desc(kQProjBLinearIndex) &&
      (index == IN_Q_PROJ_B_OFFSET || index == IN_Q_PROJ_B_SCALE) &&
      (absl::EndsWith(name, "weight_offset") ||
       absl::EndsWith(name, "weight_scale"));
  const bool is_dynamic_o_proj_quant_param =
      use_quant_weight_mapping() && is_attn_dynamic_desc(kOLinearIndex) &&
      (index == IN_ATTENTION_OUT_OFFSET || index == IN_ATTENTION_OUT_SCALE) &&
      (absl::EndsWith(name, "weight_offset") ||
       absl::EndsWith(name, "weight_scale"));
  const bool is_dynamic_indexer_wq_b_quant_param =
      use_quant_weight_mapping() &&
      is_attn_dynamic_desc(kIndexerWqBLinearIndex) &&
      (index == IN_INDEXER_WQ_B_OFFSET || index == IN_INDEXER_WQ_B_SCALE) &&
      (absl::EndsWith(name, "weight_offset") ||
       absl::EndsWith(name, "weight_scale"));
  const bool is_sharded =
      WEIGHT_SHARD_W8A8.count(index) || is_dynamic_q_b_quant_param;
  torch::Tensor tmp_tensor;

  if (is_sharded) {
    const int32_t shard_dim =
        is_dynamic_q_b_quant_param ? 0 : WEIGHT_SHARD_W8A8.at(index);
    tmp_tensor =
        get_sharded_tensor(
            state_dict, name, shard_dim, dp_local_tp_rank_, dp_local_tp_size_)
            .to(target_device());
  } else {
    tmp_tensor = tensor.to(target_device());
  }

  correct_tensor_dtype(tmp_tensor, name);
  if (use_quant_weight_mapping() && absl::StartsWith(name, "self_attn.") &&
      absl::EndsWith(name, "weight_scale")) {
    tmp_tensor = prefill_isBF16_ ? tmp_tensor.to(torch::kBFloat16)
                                 : tmp_tensor.to(torch::kFloat32);
  } else if (use_quant_weight_mapping() &&
             absl::StartsWith(name, "self_attn.") &&
             absl::EndsWith(name, "weight_offset")) {
    tmp_tensor = tmp_tensor.to(torch::kFloat16);
  }
  if (is_dynamic_o_proj_quant_param || is_dynamic_indexer_wq_b_quant_param) {
    tmp_tensor = tmp_tensor.flatten().contiguous();
  }
  auto& t = working_tensors();
  t[index] = tmp_tensor;
  if (absl::StartsWith(name, "self_attn.q_a_proj")) {
    const int index_re = get_mapped_index(name, WEIGHT_MAPPING_W8A8_RECOMPUTE);
    torch::Tensor tmp_tensor_re = tmp_tensor;
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
  const bool is_w4a8_dynamic = quantize_type_ == "w4a8_dynamic";
  auto select_w4a8_second_scale =
      [this](const std::string& scale_second_key,
             const std::string& offset_key) -> std::vector<torch::Tensor>& {
    auto scale_second_it = experts_weights_.find(scale_second_key);
    if (scale_second_it != experts_weights_.end() &&
        has_any_defined_expert(scale_second_it->second)) {
      return scale_second_it->second;
    }
    auto offset_it = experts_weights_.find(offset_key);
    CHECK(offset_it != experts_weights_.end())
        << "GLM/DeepSeekV32 W4A8 layer " << layer_id_ << " neither "
        << scale_second_key << " nor " << offset_key
        << " is reserved for second scale";
    return offset_it->second;
  };
  if (is_w4a8_dynamic) {
    check_expert_vector_pair(experts_weights_["gate_proj.weight"],
                             experts_weights_["up_proj.weight"],
                             "gateup.weight",
                             layer_id_);
  }
  torch::Tensor mlp_gateup_weight =
      merge_experts_weights(experts_weights_["gate_proj.weight"],
                            experts_weights_["up_proj.weight"],
                            /*transpose=*/!is_w4a8_dynamic);
  if (is_w4a8_dynamic) {
    t[IN_MLP_GATEUP_WEIGHT_EXPERT] = mlp_gateup_weight;
  } else {
    // IN_MLP_GATEUP_WEIGHT_EXPERT: always NZ (both modes agree).
    t[IN_MLP_GATEUP_WEIGHT_EXPERT] =
        cast_nz(mlp_gateup_weight, IN_MLP_GATEUP_WEIGHT_EXPERT);
  }
  if (quantize_type_ == "w8a8_dynamic") {
    t[IN_MLP_GATEUP_OFFSET_EXPERT] =
        merge_experts_weights(experts_weights_["gate_proj.weight_offset"],
                              experts_weights_["up_proj.weight_offset"]);
    t[IN_MLP_GATEUP_SCALE_EXPERT] =
        merge_experts_weights(experts_weights_["gate_proj.weight_scale"],
                              experts_weights_["up_proj.weight_scale"]);
  } else if (is_w4a8_dynamic) {
    check_expert_vector_pair(experts_weights_["gate_proj.weight_scale"],
                             experts_weights_["up_proj.weight_scale"],
                             "gateup.weight_scale",
                             layer_id_);
    t[IN_MLP_GATEUP_SCALE_EXPERT] =
        merge_experts_weights(experts_weights_["gate_proj.weight_scale"],
                              experts_weights_["up_proj.weight_scale"]);
    check_expert_vector_pair(experts_weights_["gate_proj.scale_bias"],
                             experts_weights_["up_proj.scale_bias"],
                             "gateup.scale_bias",
                             layer_id_);
    t[IN_MLP_GATEUP_BIAS_EXPERT] =
        merge_experts_weights(experts_weights_["gate_proj.scale_bias"],
                              experts_weights_["up_proj.scale_bias"]);
    if (quant_group_size_ > 0) {
      auto& gate_second_scale = select_w4a8_second_scale(
          "gate_proj.weight_scale_second", "gate_proj.weight_offset");
      auto& up_second_scale = select_w4a8_second_scale(
          "up_proj.weight_scale_second", "up_proj.weight_offset");
      check_expert_vector_pair(
          gate_second_scale, up_second_scale, "gateup.second_scale", layer_id_);
      t[IN_MLP_GATEUP_OFFSET_EXPERT] =
          merge_experts_weights(gate_second_scale, up_second_scale);
    }
  }

  // IN_MLP_DOWN_WEIGHT_EXPERT: eager always NZ; manual ND.
  if (is_w4a8_dynamic) {
    check_expert_vector(
        experts_weights_["down_proj.weight"], "down.weight", layer_id_);
  }
  torch::Tensor mlp_down_weight = merge_experts_weights(
      experts_weights_["down_proj.weight"], /*transpose=*/false);
  bool down_is_nz;
  if (load_to_host()) {
    down_is_nz = false;
  } else {
    down_is_nz = true;
  }
  if (is_w4a8_dynamic) {
    t[IN_MLP_DOWN_WEIGHT_EXPERT] = mlp_down_weight;
  } else if (down_is_nz) {
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
  } else if (is_w4a8_dynamic) {
    check_expert_vector(experts_weights_["down_proj.weight_scale"],
                        "down.weight_scale",
                        layer_id_);
    t[IN_MLP_DOWN_SCALE_EXPERT] =
        merge_experts_weights(experts_weights_["down_proj.weight_scale"]);
    check_expert_vector(
        experts_weights_["down_proj.scale_bias"], "down.scale_bias", layer_id_);
    t[IN_MLP_DOWN_BIAS_EXPERT] =
        merge_experts_weights(experts_weights_["down_proj.scale_bias"]);
    if (quant_group_size_ > 0) {
      auto& down_second_scale = select_w4a8_second_scale(
          "down_proj.weight_scale_second", "down_proj.weight_offset");
      check_expert_vector(down_second_scale, "down.second_scale", layer_id_);
      t[IN_MLP_DOWN_OFFSET_EXPERT] = merge_experts_weights(down_second_scale);
    }
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
    experts_gate[i] = cat_with_debug(
        {experts_gate[i], experts_up[i]},
        /*dim=*/0,
        {"gate expert " + std::to_string(i), "up expert " + std::to_string(i)},
        /*cat_name=*/"routed gateup expert",
        layer_id_);
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

void DeekseekV32DecoderLoader::preprocess_w4a8_dynamic_experts_weights() {
  if (quantize_type_ != "w4a8_dynamic" ||
      layer_id_ < prefill_firstKDenseReplace_) {
    return;
  }

  auto& t = working_tensors();
  check_required_w4a8_tensor(
      t[IN_MLP_GATEUP_WEIGHT_EXPERT], "IN_MLP_GATEUP_WEIGHT_EXPERT", layer_id_);
  check_required_w4a8_tensor(
      t[IN_MLP_DOWN_WEIGHT_EXPERT], "IN_MLP_DOWN_WEIGHT_EXPERT", layer_id_);
  check_required_w4a8_tensor(
      t[IN_MLP_GATEUP_SCALE_EXPERT], "IN_MLP_GATEUP_SCALE_EXPERT", layer_id_);
  check_required_w4a8_tensor(
      t[IN_MLP_DOWN_SCALE_EXPERT], "IN_MLP_DOWN_SCALE_EXPERT", layer_id_);
  check_required_w4a8_tensor(
      t[IN_MLP_GATEUP_BIAS_EXPERT], "IN_MLP_GATEUP_BIAS_EXPERT", layer_id_);
  check_required_w4a8_tensor(
      t[IN_MLP_DOWN_BIAS_EXPERT], "IN_MLP_DOWN_BIAS_EXPERT", layer_id_);
  if (quant_group_size_ > 0) {
    check_required_w4a8_tensor(t[IN_MLP_GATEUP_OFFSET_EXPERT],
                               "IN_MLP_GATEUP_OFFSET_EXPERT",
                               layer_id_);
    check_required_w4a8_tensor(
        t[IN_MLP_DOWN_OFFSET_EXPERT], "IN_MLP_DOWN_OFFSET_EXPERT", layer_id_);
  }

  kernel::W4A8DynamicMoePreprocessParams params;
  params.w13_weight = t[IN_MLP_GATEUP_WEIGHT_EXPERT];
  params.w2_weight = t[IN_MLP_DOWN_WEIGHT_EXPERT];
  params.w13_weight_scale = t[IN_MLP_GATEUP_SCALE_EXPERT];
  params.w2_weight_scale = t[IN_MLP_DOWN_SCALE_EXPERT];
  params.w13_weight_scale_second =
      quant_group_size_ > 0
          ? std::optional<torch::Tensor>(t[IN_MLP_GATEUP_OFFSET_EXPERT])
          : std::nullopt;
  params.w2_weight_scale_second =
      quant_group_size_ > 0
          ? std::optional<torch::Tensor>(t[IN_MLP_DOWN_OFFSET_EXPERT])
          : std::nullopt;
  params.w13_scale_bias = t[IN_MLP_GATEUP_BIAS_EXPERT];
  params.w2_scale_bias = t[IN_MLP_DOWN_BIAS_EXPERT];
  params.group_size = quant_group_size_;
  params.pack_weight_to_int32 = false;

  torch::Tensor processed_w13;
  torch::Tensor processed_w2;
  torch::Tensor processed_w13_scale;
  torch::Tensor processed_w2_scale;
  std::optional<torch::Tensor> processed_w13_scale_bias;
  std::optional<torch::Tensor> processed_w2_scale_bias;
  std::tie(processed_w13,
           processed_w2,
           processed_w13_scale,
           processed_w2_scale,
           processed_w13_scale_bias,
           processed_w2_scale_bias) =
      kernel::w4a8_dynamic_moe_preprocess(params);

  t[IN_MLP_GATEUP_WEIGHT_EXPERT] = processed_w13;
  t[IN_MLP_DOWN_WEIGHT_EXPERT] = processed_w2;
  t[IN_MLP_GATEUP_SCALE_EXPERT] = processed_w13_scale;
  t[IN_MLP_DOWN_SCALE_EXPERT] = processed_w2_scale;
  if (processed_w13_scale_bias.has_value()) {
    t[IN_MLP_GATEUP_BIAS_EXPERT] = processed_w13_scale_bias.value();
  }
  if (processed_w2_scale_bias.has_value()) {
    t[IN_MLP_DOWN_BIAS_EXPERT] = processed_w2_scale_bias.value();
  }
  t[IN_MLP_GATEUP_OFFSET_EXPERT] = tensor_placeholder_;
  t[IN_MLP_DOWN_OFFSET_EXPERT] = tensor_placeholder_;
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
  std::vector<std::string> linear_for_rope;
  linear_for_rope.reserve(6);
  linear_for_rope.emplace_back("self_attn.q_b_proj.weight");
  if (is_attn_dynamic_desc(kQProjBLinearIndex)) {
    linear_for_rope.emplace_back("self_attn.q_b_proj.weight_offset");
    linear_for_rope.emplace_back("self_attn.q_b_proj.weight_scale");
  } else {
    linear_for_rope.emplace_back("self_attn.q_b_proj.quant_bias");
    linear_for_rope.emplace_back("self_attn.q_b_proj.deq_scale");
  }
  linear_for_rope.emplace_back("self_attn.kv_a_proj_with_mqa.weight");
  if (is_attn_dynamic_desc(kKvAProjLinearIndex)) {
    linear_for_rope.emplace_back("self_attn.kv_a_proj_with_mqa.weight_offset");
    linear_for_rope.emplace_back("self_attn.kv_a_proj_with_mqa.weight_scale");
  } else {
    linear_for_rope.emplace_back("self_attn.kv_a_proj_with_mqa.quant_bias");
    linear_for_rope.emplace_back("self_attn.kv_a_proj_with_mqa.deq_scale");
  }
  for (const std::string& name : linear_for_rope) {
    if (!use_quant_weight_mapping()) {
      if (!absl::EndsWith(name, "weight")) {
        continue;
      }
    }
    int32_t index = WEIGHT_MAPPING_W8A8.at(name);
    if (is_placeholder_tensor(t[index])) {
      continue;
    }
    t[index] = view_tensor(t[index], name, true);
    t[index] = trans_rope_weight(t[index]);
    t[index] = view_tensor(t[index], name, false);
    if (absl::StrContains(name, "q_b_proj") && absl::EndsWith(name, "weight") &&
        is_attn_dynamic_desc(kQProjBLinearIndex)) {
      t[index] = t[index].transpose(0, 1).contiguous();
    }
    const bool keep_dynamic_kv_a_quant_shape =
        absl::StrContains(name, "kv_a_proj_with_mqa") &&
        is_attn_dynamic_desc(kKvAProjLinearIndex) &&
        (absl::EndsWith(name, "weight_scale") ||
         absl::EndsWith(name, "weight_offset"));
    if (!absl::EndsWith(name, "weight") && !keep_dynamic_kv_a_quant_shape) {
      t[index] = t[index].flatten();
    }
  }

  if (indexer_rope_interleave_) {
    CHECK_GT(prefill_qkRopeHeadDim_, 0)
        << "GLM indexer rope dim must be positive";
    CHECK_GE(index_head_dim_, prefill_qkRopeHeadDim_)
        << "GLM indexer head dim must be greater than or equal to rope dim";
    CHECK_GT(index_n_heads_, 0) << "GLM indexer head count must be positive";

    std::vector<std::string> indexer_linear_for_rope;
    indexer_linear_for_rope.reserve(5);
    indexer_linear_for_rope.emplace_back("self_attn.indexer.wq_b.weight");
    if (is_attn_dynamic_desc(kIndexerWqBLinearIndex)) {
      indexer_linear_for_rope.emplace_back(
          "self_attn.indexer.wq_b.weight_offset");
      indexer_linear_for_rope.emplace_back(
          "self_attn.indexer.wq_b.weight_scale");
    } else {
      indexer_linear_for_rope.emplace_back("self_attn.indexer.wq_b.quant_bias");
      indexer_linear_for_rope.emplace_back("self_attn.indexer.wq_b.deq_scale");
    }
    indexer_linear_for_rope.emplace_back("self_attn.indexer.wk.weight");

    for (const std::string& name : indexer_linear_for_rope) {
      if (!use_quant_weight_mapping() && !absl::EndsWith(name, "weight")) {
        continue;
      }
      auto index_it = WEIGHT_MAPPING_W8A8.find(name);
      if (index_it == WEIGHT_MAPPING_W8A8.end()) {
        LOG(WARNING) << "Skip unsupported GLM5 indexer rope tensor: " << name;
        continue;
      }
      int32_t index = index_it->second;
      if (is_placeholder_tensor(t[index])) {
        continue;
      }
      t[index] = view_indexer_tensor(t[index], name, true);
      t[index] = trans_front_rope_weight(t[index]);
      t[index] = view_indexer_tensor(t[index], name, false);
      if (!absl::EndsWith(name, "weight")) {
        t[index] = t[index].flatten();
      }
    }
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

torch::Tensor DeekseekV32DecoderLoader::view_indexer_tensor(
    torch::Tensor weight,
    const std::string& name,
    bool pre_view) {
  if (absl::StrContains(name, "indexer.wq_b")) {
    if (pre_view) {
      return weight.view({index_n_heads_, index_head_dim_, -1}).contiguous();
    }
    return weight.view({index_n_heads_ * index_head_dim_, -1}).contiguous();
  }
  if (absl::StrContains(name, "indexer.wk")) {
    return weight.view({index_head_dim_, -1}).contiguous();
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

torch::Tensor DeekseekV32DecoderLoader::trans_front_rope_weight(
    torch::Tensor weight) {
  int64_t rope_dim = prefill_qkRopeHeadDim_;
  torch::Tensor output = load_to_host() ? weight.clone() : weight;
  torch::Tensor weight_1 = weight.slice(-2, 0, rope_dim, 2).contiguous();
  torch::Tensor weight_2 = weight.slice(-2, 1, rope_dim, 2).contiguous();
  torch::Tensor combined = torch::cat({weight_1, weight_2}, -2);
  output.slice(-2, 0, rope_dim).copy_(combined);
  return output.contiguous();
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
  auto convert_to_float = [&t](int32_t index) {
    t[index] = t[index].to(torch::kFloat32);
  };
  if (!is_attn_dynamic_desc(kQProjALinearIndex)) {
    convert_to_float(IN_Q_PROJ_A_DESCALE);
  }
  if (!is_attn_dynamic_desc(kKvAProjLinearIndex)) {
    convert_to_float(IN_KV_PROJ_WITH_MQA_DESCALE);
  }
  if (!is_attn_dynamic_desc(kQProjBLinearIndex)) {
    convert_to_float(IN_Q_PROJ_B_DESCALE);
  }
  if (!is_attn_dynamic_desc(kIndexerWqBLinearIndex)) {
    convert_to_float(IN_INDEXER_WQ_B_DESCALE);
  }
  if (!is_attn_dynamic_desc(kOLinearIndex)) {
    convert_to_float(IN_ATTENTION_OUT_DESCALE);
  }
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
  } else if (quantize_type_ == "w4a8_dynamic") {
    weight_names.emplace_back("gate_proj.weight_scale");
    weight_names.emplace_back("up_proj.weight_scale");
    weight_names.emplace_back("down_proj.weight_scale");
    weight_names.emplace_back("gate_proj.weight_scale_second");
    weight_names.emplace_back("up_proj.weight_scale_second");
    weight_names.emplace_back("down_proj.weight_scale_second");
    weight_names.emplace_back("gate_proj.weight_offset");
    weight_names.emplace_back("up_proj.weight_offset");
    weight_names.emplace_back("down_proj.weight_offset");
    weight_names.emplace_back("gate_proj.scale_bias");
    weight_names.emplace_back("up_proj.scale_bias");
    weight_names.emplace_back("down_proj.scale_bias");
  }
  std::lock_guard<std::mutex> lock(experts_mutex_);
  for (const std::string& weight_name : weight_names) {
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
  auto merge_and_clear = [this, &t](int32_t index,
                                    const std::string& gate_name,
                                    torch::Tensor& shared_experts_gate,
                                    const std::string& up_name,
                                    torch::Tensor& shared_experts_up) {
    CHECK(shared_experts_gate.defined())
        << "GLM/DeepSeekV32 layer " << layer_id_
        << " missing shared expert tensor: " << gate_name;
    CHECK(shared_experts_up.defined())
        << "GLM/DeepSeekV32 layer " << layer_id_
        << " missing shared expert tensor: " << up_name;
    t[index] = cat_with_debug({shared_experts_gate, shared_experts_up},
                              /*dim=*/0,
                              {gate_name, up_name},
                              /*cat_name=*/"shared gateup",
                              layer_id_)
                   .to(target_device())
                   .contiguous();
    shared_experts_gate = tensor_placeholder_;
    shared_experts_up = tensor_placeholder_;
  };
  auto maybe_merge_and_clear = [&](int32_t index,
                                   const std::string& gate_name,
                                   torch::Tensor& shared_experts_gate,
                                   const std::string& up_name,
                                   torch::Tensor& shared_experts_up) {
    const bool gate_defined = shared_experts_gate.defined();
    const bool up_defined = shared_experts_up.defined();
    if (!gate_defined && !up_defined) {
      return;
    }
    CHECK(gate_defined && up_defined)
        << "GLM/DeepSeekV32 layer " << layer_id_
        << " partially loaded shared expert tensors: " << gate_name << "="
        << tensor_debug_string(shared_experts_gate) << ", " << up_name << "="
        << tensor_debug_string(shared_experts_up);
    merge_and_clear(
        index, gate_name, shared_experts_gate, up_name, shared_experts_up);
  };

  if (layer_id_ >= prefill_firstKDenseReplace_) {
    merge_and_clear(
        IN_MLP_GATEUP_WEIGHT_SHARED_EXPERT,
        "mlp.shared_experts.gate_proj.weight",
        shared_experts_weights_["mlp.shared_experts.gate_proj.weight"],
        "mlp.shared_experts.up_proj.weight",
        shared_experts_weights_["mlp.shared_experts.up_proj.weight"]);
    if (quantize_type_ == "w8a8_dynamic" ||
        (quantize_type_ == "w4a8_dynamic" && quant_group_size_ > 0)) {
      merge_and_clear(
          IN_MLP_GATEUP_OFFSET_SHARED_EXPERT,
          "mlp.shared_experts.gate_proj.weight_offset",
          shared_experts_weights_["mlp.shared_experts.gate_proj.weight_offset"],
          "mlp.shared_experts.up_proj.weight_offset",
          shared_experts_weights_["mlp.shared_experts.up_proj.weight_offset"]);
    }
    if (use_quant_weight_mapping()) {
      merge_and_clear(
          IN_MLP_GATEUP_SCALE_SHARED_EXPERT,
          "mlp.shared_experts.gate_proj.weight_scale",
          shared_experts_weights_["mlp.shared_experts.gate_proj.weight_scale"],
          "mlp.shared_experts.up_proj.weight_scale",
          shared_experts_weights_["mlp.shared_experts.up_proj.weight_scale"]);
    }
  } else {
    merge_and_clear(IN_MLP_GATEUP_WEIGHT_SHARED_EXPERT,
                    "mlp.gate_proj.weight",
                    shared_experts_weights_["mlp.gate_proj.weight"],
                    "mlp.up_proj.weight",
                    shared_experts_weights_["mlp.up_proj.weight"]);
    if (quantize_type_ == "w8a8_dynamic" ||
        (quantize_type_ == "w4a8_dynamic" && quant_group_size_ > 0)) {
      merge_and_clear(IN_MLP_GATEUP_OFFSET_SHARED_EXPERT,
                      "mlp.gate_proj.weight_offset",
                      shared_experts_weights_["mlp.gate_proj.weight_offset"],
                      "mlp.up_proj.weight_offset",
                      shared_experts_weights_["mlp.up_proj.weight_offset"]);
    }
    if (use_quant_weight_mapping()) {
      merge_and_clear(IN_MLP_GATEUP_SCALE_SHARED_EXPERT,
                      "mlp.gate_proj.weight_scale",
                      shared_experts_weights_["mlp.gate_proj.weight_scale"],
                      "mlp.up_proj.weight_scale",
                      shared_experts_weights_["mlp.up_proj.weight_scale"]);
    }
  }
}

void DeekseekV32DecoderLoader::merge_host_at_weights() {
  auto& t = working_tensors();
  if (use_quant_weight_mapping()) {
    if (prefill_isBF16_) {
      convert_descaled_weights_to_float();
    }
    convert_offsets_to_int8();
    handle_device_specific_bias();
  }

  merge_shared_experts_weights();
  if (layer_id_ >= prefill_firstKDenseReplace_) {
    merge_experts_weights();
    preprocess_w4a8_dynamic_experts_weights();
  }

  squeeze_experts_weights();

  reset_skipped_indexer_weights();

  preprocess_linear_for_rope();

  if (use_quant_weight_mapping()) {
    CHECK_EQ(is_attn_dynamic_desc(kQProjALinearIndex),
             is_attn_dynamic_desc(kKvAProjLinearIndex))
        << "DeepSeek V32 currently requires self_attn.q_a_proj and "
        << "self_attn.kv_a_proj_with_mqa to share the same quant_desc.";
  }
  t[IN_Q_PROJ_A_WEIGHT] =
      cat_with_debug(
          {t[IN_KV_PROJ_WITH_MQA_WEIGHT], t[IN_Q_PROJ_A_WEIGHT]},
          /*dim=*/0,
          {"self_attn.kv_a_proj_with_mqa.weight", "self_attn.q_a_proj.weight"},
          /*cat_name=*/"q_a_kv_a.weight",
          layer_id_)
          .contiguous();
  if (use_quant_weight_mapping()) {
    if (is_attn_dynamic_desc(kQProjALinearIndex)) {
      t[IN_Q_PROJ_A_SCALE] =
          cat_with_debug({t[IN_KV_PROJ_WITH_MQA_SCALE], t[IN_Q_PROJ_A_SCALE]},
                         /*dim=*/0,
                         {"self_attn.kv_a_proj_with_mqa.weight_scale",
                          "self_attn.q_a_proj.weight_scale"},
                         /*cat_name=*/"q_a_kv_a.scale",
                         layer_id_)
              .flatten()
              .contiguous();
      if (t[IN_KV_PROJ_WITH_MQA_OFFSET].sizes() != std::vector<int64_t>({1}) &&
          t[IN_Q_PROJ_A_OFFSET].sizes() != std::vector<int64_t>({1})) {
        t[IN_Q_PROJ_A_OFFSET] =
            cat_with_debug(
                {t[IN_KV_PROJ_WITH_MQA_OFFSET], t[IN_Q_PROJ_A_OFFSET]},
                /*dim=*/0,
                {"self_attn.kv_a_proj_with_mqa.weight_offset",
                 "self_attn.q_a_proj.weight_offset"},
                /*cat_name=*/"q_a_kv_a.offset",
                layer_id_)
                .flatten()
                .contiguous();
      }
    } else {
      t[IN_Q_PROJ_A_BIAS] =
          cat_with_debug({t[IN_KV_PROJ_WITH_MQA_BIAS], t[IN_Q_PROJ_A_BIAS]},
                         /*dim=*/0,
                         {"self_attn.kv_a_proj_with_mqa.quant_bias",
                          "self_attn.q_a_proj.quant_bias"},
                         /*cat_name=*/"q_a_kv_a.bias",
                         layer_id_)
              .contiguous();
      t[IN_Q_PROJ_A_DESCALE] =
          cat_with_debug(
              {t[IN_KV_PROJ_WITH_MQA_DESCALE], t[IN_Q_PROJ_A_DESCALE]},
              /*dim=*/0,
              {"self_attn.kv_a_proj_with_mqa.deq_scale",
               "self_attn.q_a_proj.deq_scale"},
              /*cat_name=*/"q_a_kv_a.descale",
              layer_id_)
              .contiguous();
    }
  }

  if (is_attn_dynamic_desc(kQProjBLinearIndex)) {
    const int64_t q_b_scale_size = t[IN_Q_PROJ_B_SCALE].numel();
    CHECK(t[IN_Q_PROJ_B_WEIGHT].dim() != 2 ||
          t[IN_Q_PROJ_B_WEIGHT].size(1) == q_b_scale_size)
        << "GLM/DeepSeekV32 layer " << layer_id_
        << " dynamic q_b weight/scale mismatch before ATB: q_b_weight="
        << tensor_debug_string(t[IN_Q_PROJ_B_WEIGHT])
        << ", q_b_scale=" << tensor_debug_string(t[IN_Q_PROJ_B_SCALE]);
    if (!is_placeholder_tensor(t[IN_Q_PROJ_B_OFFSET])) {
      CHECK_EQ(t[IN_Q_PROJ_B_OFFSET].numel(), q_b_scale_size)
          << "GLM/DeepSeekV32 layer " << layer_id_
          << " dynamic q_b offset/scale mismatch before ATB: q_b_offset="
          << tensor_debug_string(t[IN_Q_PROJ_B_OFFSET])
          << ", q_b_scale=" << tensor_debug_string(t[IN_Q_PROJ_B_SCALE]);
    }
  }
  if (is_attn_dynamic_desc(kIndexerWqBLinearIndex) &&
      !is_placeholder_tensor(t[IN_INDEXER_WQ_B_WEIGHT])) {
    const int64_t indexer_wq_b_scale_size = t[IN_INDEXER_WQ_B_SCALE].numel();
    CHECK(t[IN_INDEXER_WQ_B_WEIGHT].dim() != 2 ||
          t[IN_INDEXER_WQ_B_WEIGHT].size(0) == indexer_wq_b_scale_size)
        << "GLM/DeepSeekV32 layer " << layer_id_
        << " dynamic indexer wq_b weight/scale mismatch before ATB: "
        << "indexer_wq_b_weight="
        << tensor_debug_string(t[IN_INDEXER_WQ_B_WEIGHT])
        << ", indexer_wq_b_scale="
        << tensor_debug_string(t[IN_INDEXER_WQ_B_SCALE]);
    if (!is_placeholder_tensor(t[IN_INDEXER_WQ_B_OFFSET])) {
      CHECK_EQ(t[IN_INDEXER_WQ_B_OFFSET].numel(), indexer_wq_b_scale_size)
          << "GLM/DeepSeekV32 layer " << layer_id_
          << " dynamic indexer wq_b offset/scale mismatch before ATB: "
          << "indexer_wq_b_offset="
          << tensor_debug_string(t[IN_INDEXER_WQ_B_OFFSET])
          << ", indexer_wq_b_scale="
          << tensor_debug_string(t[IN_INDEXER_WQ_B_SCALE]);
    }
  }

  auto cast_attn_weight =
      [this](torch::Tensor tensor, int32_t weight_index, int32_t linear_index) {
        if (is_attn_dynamic_desc(linear_index)) {
          return tensor.contiguous();
        }
        return cast_nz(tensor, weight_index);
      };
  t[IN_Q_PROJ_A_WEIGHT] = cast_attn_weight(
      t[IN_Q_PROJ_A_WEIGHT], IN_Q_PROJ_A_WEIGHT, kQProjALinearIndex);
  t[IN_Q_PROJ_A_RECOMPUTE_WEIGHT] =
      cast_attn_weight(t[IN_Q_PROJ_A_RECOMPUTE_WEIGHT],
                       IN_Q_PROJ_A_RECOMPUTE_WEIGHT,
                       kQProjALinearIndex);
  t[IN_Q_PROJ_B_WEIGHT] = cast_attn_weight(
      t[IN_Q_PROJ_B_WEIGHT], IN_Q_PROJ_B_WEIGHT, kQProjBLinearIndex);
  if (is_attn_quant_desc(kIndexerWqBLinearIndex) &&
      !is_placeholder_tensor(t[IN_INDEXER_WQ_B_WEIGHT])) {
    t[IN_INDEXER_WQ_B_WEIGHT] = cast_attn_weight(t[IN_INDEXER_WQ_B_WEIGHT],
                                                 IN_INDEXER_WQ_B_WEIGHT,
                                                 kIndexerWqBLinearIndex);
  }
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
  t[IN_MLP_GATEUP_SCALE_SHARED_EXPERT] =
      t[IN_MLP_GATEUP_SCALE_SHARED_EXPERT].to(torch::kBFloat16);
  t[IN_MLP_DOWN_SCALE_SHARED_EXPERT] =
      t[IN_MLP_DOWN_SCALE_SHARED_EXPERT].to(torch::kBFloat16);
  if (quantize_type_ != "w4a8_dynamic") {
    t[IN_MLP_GATEUP_SCALE_EXPERT] =
        t[IN_MLP_GATEUP_SCALE_EXPERT].to(torch::kBFloat16);
    t[IN_MLP_DOWN_SCALE_EXPERT] =
        t[IN_MLP_DOWN_SCALE_EXPERT].to(torch::kBFloat16);
  }
  if (use_quant_weight_mapping()) {
    if (!prefill_isBF16_) {
      if (!is_attn_dynamic_desc(kQProjALinearIndex)) {
        t[IN_Q_PROJ_A_DESCALE] = convert_fp16_to_int64(t[IN_Q_PROJ_A_DESCALE]);
        t[IN_Q_PROJ_A_RECOMPUTE_DESCALE] =
            convert_fp16_to_int64(t[IN_Q_PROJ_A_RECOMPUTE_DESCALE]);
      }
      if (!is_attn_dynamic_desc(kQProjBLinearIndex)) {
        t[IN_Q_PROJ_B_DESCALE] = convert_fp16_to_int64(t[IN_Q_PROJ_B_DESCALE]);
      }
      if (!is_attn_dynamic_desc(kIndexerWqBLinearIndex) &&
          !is_placeholder_tensor(t[IN_INDEXER_WQ_B_DESCALE])) {
        t[IN_INDEXER_WQ_B_DESCALE] =
            convert_fp16_to_int64(t[IN_INDEXER_WQ_B_DESCALE]);
      }
      if (!is_attn_dynamic_desc(kOLinearIndex)) {
        t[IN_ATTENTION_OUT_DESCALE] =
            convert_fp16_to_int64(t[IN_ATTENTION_OUT_DESCALE]);
      }

      t[IN_MLP_GATEUP_OFFSET_SHARED_EXPERT] =
          t[IN_MLP_GATEUP_OFFSET_SHARED_EXPERT].to(torch::kFloat16);
      t[IN_MLP_GATEUP_SCALE_SHARED_EXPERT] =
          t[IN_MLP_GATEUP_SCALE_SHARED_EXPERT].to(torch::kFloat32);
      t[IN_MLP_DOWN_SCALE_SHARED_EXPERT] =
          t[IN_MLP_DOWN_SCALE_SHARED_EXPERT].to(torch::kFloat32);
      if (quantize_type_ == "w8a8_dynamic") {
        t[IN_MLP_GATEUP_OFFSET_EXPERT] =
            t[IN_MLP_GATEUP_OFFSET_EXPERT].to(torch::kFloat16);
        t[IN_MLP_DOWN_OFFSET_EXPERT] =
            t[IN_MLP_DOWN_OFFSET_EXPERT].to(torch::kFloat16);
        t[IN_MLP_DOWN_SCALE_EXPERT] =
            t[IN_MLP_DOWN_SCALE_EXPERT].to(torch::kFloat32);
      }
    }
  }
}

}  // namespace layer
}  // namespace xllm
