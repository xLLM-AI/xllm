/* Copyright 2025 The xLLM Authors. All Rights Reserved.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include "utils.h"

#include <glog/logging.h>

namespace xllm {

namespace weight {

namespace {
bool should_relax_quant_vector_shape_check(const std::string& name) {
  return name == "weight_scale" || name == "weight_offset";
}

torch::Tensor reshape_quant_vector_if_compatible(const std::string& name,
                                                 const torch::Tensor& expected,
                                                 torch::Tensor loaded) {
  // Dynamic W8A8 checkpoints may store per-channel scale/offset as [N, 1],
  // while runtime tensors are allocated as [N]. Treat them as equivalent.
  if (expected.sizes() == loaded.sizes()) {
    return loaded;
  }
  if (should_relax_quant_vector_shape_check(name) &&
      expected.numel() == loaded.numel()) {
    return loaded.reshape(expected.sizes());
  }
  return loaded;
}

torch::Tensor materialize_cpu_tensor_for_copy(const torch::Tensor& tensor) {
  if (!tensor.defined() || tensor.device().is_cuda() ||
      tensor.device().type() != torch::kCPU) {
    return tensor;
  }
  if (tensor.is_contiguous()) {
    return tensor.clone();
  }
  return tensor.contiguous().clone();
}

void copy_weight_tensor(const std::string& name,
                        torch::Tensor& target,
                        const torch::Tensor& source) {
  auto materialized = materialize_cpu_tensor_for_copy(source);
  if (target.device().type() != torch::kCPU &&
      materialized.device().type() == torch::kCPU) {
    VLOG(1) << "Materialized CPU tensor before weight copy: name=" << name
            << ", target_device=" << target.device()
            << ", sizes=" << materialized.sizes();
  }
  target.copy_(materialized);
}

std::string full_weight_name(std::string_view prefix,
                             const std::string& name) {
  std::string full_name(prefix);
  full_name += name;
  return full_name;
}

std::string full_weight_name(std::string_view prefix,
                             const std::string& sub_prefix,
                             const std::string& name) {
  std::string full_name(prefix);
  full_name += sub_prefix;
  full_name += name;
  return full_name;
}
}  // namespace

void load_weight(const StateDict& state_dict,
                 const std::string& name,
                 torch::Tensor& weight,
                 bool& weight_is_loaded) {
  torch::NoGradGuard no_grad;
  const auto tensor = state_dict.get_tensor(name);
  if (tensor.defined()) {
    const auto reshaped_tensor =
        reshape_quant_vector_if_compatible(name, weight, tensor);
    CHECK(!weight_is_loaded)
        << "weight already loaded, name: " << state_dict.prefix() << name;
    CHECK_EQ(weight.sizes(), reshaped_tensor.sizes())
        << "weight size mismatch for " << state_dict.prefix() << name;
    copy_weight_tensor(
        full_weight_name(state_dict.prefix(), name), weight, reshaped_tensor);
    weight_is_loaded = true;
  }
}

void ensure_parameter_storage(torch::nn::Module* module,
                              const LazyParameterSpec& spec) {
  CHECK(module != nullptr) << "module must not be null";
  CHECK(spec.tensor != nullptr) << "spec.tensor must not be null";
  CHECK(spec.is_loaded != nullptr) << "spec.is_loaded must not be null";

  torch::ScalarType expected_dtype =
      c10::typeMetaToScalarType(spec.options.dtype());
  torch::Tensor& tensor = *spec.tensor;
  bool& tensor_is_loaded = *spec.is_loaded;
  if (!tensor.defined()) {
    tensor = module->register_parameter(spec.name,
                                        torch::empty(spec.sizes, spec.options),
                                        /*requires_grad=*/false);
    tensor_is_loaded = false;
    return;
  }
  if (tensor.sizes().vec() != spec.sizes ||
      tensor.scalar_type() != expected_dtype) {
    tensor.set_data(torch::empty(spec.sizes, spec.options));
    tensor_is_loaded = false;
  }
}

void ensure_parameter_storage(torch::nn::Module* module,
                              const std::vector<LazyParameterSpec>& specs) {
  for (const LazyParameterSpec& spec : specs) {
    ensure_parameter_storage(module, spec);
  }
}

void load_sharded_weight(const StateDict& state_dict,
                         const std::string& name,
                         int64_t dim,
                         int32_t rank,
                         int32_t world_size,
                         torch::Tensor& weight,
                         bool& weight_is_loaded) {
  const auto tensor =
      state_dict.get_sharded_tensor(name, dim, rank, world_size);
  if (tensor.defined()) {
    const auto reshaped_tensor =
        reshape_quant_vector_if_compatible(name, weight, tensor);
    CHECK_EQ(weight.sizes(), reshaped_tensor.sizes())
        << "weight size mismatch for " << state_dict.prefix() << name;
    copy_weight_tensor(
        full_weight_name(state_dict.prefix(), name), weight, reshaped_tensor);
    weight_is_loaded = true;
  }
}

void load_sharded_weight(const StateDict& state_dict,
                         const std::string& name,
                         TensorTransform transform_func,
                         int64_t dim,
                         int32_t rank,
                         int32_t world_size,
                         torch::Tensor& weight,
                         bool& weight_is_loaded) {
  auto tensor = state_dict.get_sharded_tensor(name, dim, rank, world_size);
  if (tensor.defined()) {
    tensor = transform_func(tensor);
    const auto reshaped_tensor =
        reshape_quant_vector_if_compatible(name, weight, tensor);
    CHECK(!weight_is_loaded)
        << "weight already loaded, name: " << state_dict.prefix() << name;
    CHECK_EQ(weight.sizes(), reshaped_tensor.sizes())
        << "weight size mismatch for " << state_dict.prefix() << name;
    copy_weight_tensor(
        full_weight_name(state_dict.prefix(), name), weight, reshaped_tensor);
    weight_is_loaded = true;
  }
}

void load_fused_weight(const StateDict& state_dict,
                       const std::vector<std::string>& prefixes,
                       const std::string& name,
                       int64_t dim,
                       int32_t rank,
                       int32_t world_size,
                       std::vector<torch::Tensor>& accumulated_tensors,
                       torch::Tensor& weight,
                       bool& weight_is_loaded,
                       int32_t num_kv_head_replicas) {
  // return if the weight is already loaded
  if (weight_is_loaded) {
    return;
  }

  weight_is_loaded = load_tensor_list(state_dict,
                                      prefixes,
                                      name,
                                      dim,
                                      rank,
                                      world_size,
                                      accumulated_tensors,
                                      num_kv_head_replicas);

  if (weight_is_loaded) {
    const auto merged_weight = torch::cat(accumulated_tensors, /*dim=*/dim);
    const auto reshaped_weight =
        reshape_quant_vector_if_compatible(name, weight, merged_weight);
    CHECK_EQ(weight.sizes(), reshaped_weight.sizes())
        << "weight size mismatch for " << state_dict.prefix() << name;
    copy_weight_tensor(
        full_weight_name(state_dict.prefix(), name), weight, reshaped_weight);
    // release the memory for weight_list
    accumulated_tensors.clear();
  }
}

bool load_tensor_list(const StateDict& state_dict,
                      const std::vector<std::string>& prefixes,
                      const std::string& name,
                      int64_t dim,
                      int32_t rank,
                      int32_t world_size,
                      std::vector<torch::Tensor>& tensors,
                      int32_t num_kv_head_replicas) {
  // resize the accumulated weight list if needed
  if (tensors.size() < prefixes.size()) {
    tensors.resize(prefixes.size());
  }

  // load the weights from the state_dict
  for (size_t i = 0; i < prefixes.size(); ++i) {
    if (tensors[i].defined()) {
      continue;
    }

    // When the number of key/value heads is smaller than the number of query
    // heads (e.g., multi-query/grouped-query attention), the key/value head may
    // be replicated while the query heads are partitioned.
    if (i == 1 && num_kv_head_replicas > 1) {
      rank = rank / num_kv_head_replicas;
      world_size = world_size / num_kv_head_replicas;
    }

    const std::string tensor_name = prefixes[i] + name;
    torch::Tensor tensor;
    if (dim < 0) {
      tensor = state_dict.get_tensor(tensor_name);
    } else {
      tensor =
          state_dict.get_sharded_tensor(tensor_name, dim, rank, world_size);
    }
    if (tensor.defined()) {
      tensors[i] = tensor;
    }
  }

  return std::all_of(tensors.begin(),
                     tensors.end(),
                     [](const torch::Tensor& t) { return t.defined(); });
}

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
                     bool& weight_is_loaded) {
  load_moe_weight(state_dict,
                  sub_prefix,
                  name,
                  nullptr,
                  dim,
                  rank,
                  world_size,
                  start_expert_id,
                  num_experts_per_rank,
                  accumulated_tensors,
                  weight,
                  weight_is_loaded);
}

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
                     bool& weight_is_loaded) {
  // return if the weight is already loaded
  if (weight_is_loaded) {
    return;
  }
  std::vector<std::string> prefixes;
  for (size_t idx = 0; idx < num_experts_per_rank; idx++) {
    std::string expert_id_str = std::to_string(start_expert_id + idx) + ".";
    prefixes.emplace_back(expert_id_str + sub_prefix);
  }

  weight_is_loaded = load_tensor_list(
      state_dict, prefixes, name, dim, rank, world_size, accumulated_tensors);

  if (weight_is_loaded) {
    const auto merged_weight = torch::stack(accumulated_tensors);
    const auto reshaped_weight =
        reshape_quant_vector_if_compatible(name, weight, merged_weight);
    CHECK_EQ(weight.sizes(), reshaped_weight.sizes())
        << "weight size mismatch for " << state_dict.prefix() << "["
        << start_expert_id << ":" << (start_expert_id + num_experts_per_rank)
        << "]." << sub_prefix << name;
    if (transform_func) {
      weight.set_data(transform_func(reshaped_weight));
    } else {
      copy_weight_tensor(full_weight_name(state_dict.prefix(), sub_prefix, name),
                         weight,
                         reshaped_weight);
    }
    // release the memory for weight_list
    accumulated_tensors.clear();
  }
}

void load_moe_all_expert_weight(const StateDict& state_dict,
                                const std::string& sub_prefix,
                                const std::string& name,
                                int64_t dim,
                                int64_t rank,
                                int64_t world_size,
                                int64_t num_total_experts,
                                std::vector<torch::Tensor>& accumulated_tensors,
                                torch::Tensor& weight,
                                bool& weight_is_loaded) {
  // return if the weight is already loaded
  if (weight_is_loaded) {
    return;
  }
  // load all expert weight from state_dict
  std::vector<std::string> prefixes;
  for (size_t idx = 0; idx < num_total_experts; idx++) {
    std::string expert_id_str = std::to_string(idx) + ".";
    prefixes.emplace_back(expert_id_str + sub_prefix);
  }

  weight_is_loaded = load_tensor_list(
      state_dict, prefixes, name, dim, rank, world_size, accumulated_tensors);

  if (weight_is_loaded) {
    const auto merged_weight = torch::stack(accumulated_tensors);
    const auto reshaped_weight =
        reshape_quant_vector_if_compatible(name, weight, merged_weight);
    CHECK_EQ(weight.sizes(), reshaped_weight.sizes())
        << "weight size mismatch for " << state_dict.prefix() << "[" << 0 << ":"
        << (0 + num_total_experts) << "]." << sub_prefix << name;
    copy_weight_tensor(full_weight_name(state_dict.prefix(), sub_prefix, name),
                       weight,
                       reshaped_weight);
    // release the memory for weight_list
    accumulated_tensors.clear();
  }
}

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
                           bool& w13_is_loaded) {
  load_moe_fused_weight(state_dict,
                        prefixes,
                        name,
                        nullptr,
                        rank,
                        world_size,
                        start_expert_id,
                        num_experts_per_rank,
                        w1_tensors,
                        w3_tensors,
                        w13,
                        w1_is_loaded,
                        w3_is_loaded,
                        w13_is_loaded);
}

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
                           bool& w13_is_loaded) {
  // return if the weight is already loaded
  if (w13_is_loaded) {
    return;
  }
  CHECK_EQ(prefixes.size(), 2) << "only support load moe gate_proj and up_proj";

  std::vector<std::string> w1_prefixes, w3_prefixes;
  for (size_t idx = 0; idx < num_experts_per_rank; idx++) {
    std::string expert_id_str = std::to_string(start_expert_id + idx) + ".";
    w1_prefixes.emplace_back(expert_id_str + prefixes[0]);
    w3_prefixes.emplace_back(expert_id_str + prefixes[1]);
  }

  const int64_t dim = 0;
  if (!w1_is_loaded) {
    w1_is_loaded = load_tensor_list(
        state_dict, w1_prefixes, name, dim, rank, world_size, w1_tensors);
  }
  if (!w3_is_loaded) {
    w3_is_loaded = load_tensor_list(
        state_dict, w3_prefixes, name, dim, rank, world_size, w3_tensors);
  }
  w13_is_loaded = w1_is_loaded && w3_is_loaded;

  if (w13_is_loaded) {
    std::vector<torch::Tensor> w13_vec(num_experts_per_rank);
    for (size_t idx = 0; idx < num_experts_per_rank; idx++) {
      w13_vec[idx] = torch::cat({w1_tensors[idx], w3_tensors[idx]});
    }
    const auto merged_weight = torch::stack(w13_vec);
    const auto reshaped_weight =
        reshape_quant_vector_if_compatible(name, w13, merged_weight);
    CHECK_EQ(w13.sizes(), reshaped_weight.sizes())
        << "weight size mismatch for " << state_dict.prefix() << "["
        << start_expert_id << ":" << (start_expert_id + num_experts_per_rank)
        << "].{" << prefixes[0] << ", " << prefixes[1] << "}." << name;
    if (transform_func) {
      w13.set_data(transform_func(reshaped_weight));
    } else {
      copy_weight_tensor(
          full_weight_name(state_dict.prefix(), name), w13, reshaped_weight);
    }

    // release the memory for weight_list
    w1_tensors.clear();
    w3_tensors.clear();
  }
}

void load_merged_weight(const StateDict& state_dict,
                        const std::string& name,
                        int64_t dim,
                        int32_t rank,
                        int32_t world_size,
                        int32_t shard_tensor_count,
                        int64_t shard_size,
                        torch::Tensor& weight,
                        bool& weight_is_loaded) {
  if (weight_is_loaded) {
    return;
  }
  const auto& tensor = state_dict.get_tensor(name);
  if (!tensor.defined()) {
    return;
  }
  CHECK_EQ(tensor.size(dim), shard_tensor_count * shard_size * world_size)
      << name << "[" << dim << "] size mismatch for " << state_dict.prefix()
      << name;
  std::vector<torch::Tensor> shard_tensors;
  for (size_t shard_id = 0; shard_id < shard_tensor_count; shard_id++) {
    int64_t shard_offset =
        shard_id * shard_size * world_size + rank * shard_size;
    shard_tensors.push_back(
        tensor.slice(dim, shard_offset, shard_offset + shard_size));
  }
  auto merged_weight = torch::cat(shard_tensors, dim);
  const auto reshaped_weight =
      reshape_quant_vector_if_compatible(name, weight, merged_weight);
  CHECK_EQ(weight.sizes(), reshaped_weight.sizes())
      << "weight size mismatch for " << state_dict.prefix() << name;
  copy_weight_tensor(
      full_weight_name(state_dict.prefix(), name), weight, reshaped_weight);
  weight_is_loaded = true;
}

void load_merged_weight_v2(const StateDict& state_dict,
                           const std::string& name,
                           int64_t dim,
                           int32_t rank,
                           int32_t world_size,
                           int32_t shard_tensor_count,
                           const std::vector<int64_t>& shard_sizes,
                           torch::Tensor& weight,
                           bool& weight_is_loaded) {
  if (weight_is_loaded) {
    return;
  }
  const auto& tensor = state_dict.get_tensor(name);
  if (!tensor.defined()) {
    return;
  }

  // Check that shard_tensor_count matches the size of shard_sizes vector
  CHECK_EQ(shard_tensor_count, static_cast<int32_t>(shard_sizes.size()))
      << "shard_tensor_count does not match shard_sizes vector size for "
      << state_dict.prefix() << name;

  // Calculate total expected size
  int64_t total_expected_size = 0;
  for (int64_t size : shard_sizes) {
    total_expected_size += size * world_size;
  }
  CHECK_EQ(tensor.size(dim), total_expected_size)
      << name << "[" << dim << "] size mismatch for " << state_dict.prefix()
      << name;

  std::vector<torch::Tensor> shard_tensors;

  for (size_t shard_id = 0; shard_id < shard_tensor_count; shard_id++) {
    int64_t shard_size = shard_sizes[shard_id];

    // Calculate the offset for this shard and rank
    // First, skip all the data from previous shards for all ranks
    int64_t shard_offset = 0;
    for (int32_t prev_shard = 0; prev_shard < shard_id; ++prev_shard) {
      shard_offset += shard_sizes[prev_shard] * world_size;
    }

    // Then add the offset within this shard for the current rank
    shard_offset += rank * shard_size;

    shard_tensors.push_back(
        tensor.slice(dim, shard_offset, shard_offset + shard_size));
  }

  auto merged_weight = torch::cat(shard_tensors, dim);
  const auto reshaped_weight =
      reshape_quant_vector_if_compatible(name, weight, merged_weight);
  CHECK_EQ(weight.sizes(), reshaped_weight.sizes())
      << "weight size mismatch for " << state_dict.prefix() << name;
  copy_weight_tensor(
      full_weight_name(state_dict.prefix(), name), weight, reshaped_weight);
  weight_is_loaded = true;
}

}  // namespace weight

}  // namespace xllm
