/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include <torch/torch.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/framework/model_loader/dit_model_loader.h"
#include "core/framework/state_dict/state_dict.h"
#include "models/dit/transformers/minimax_h3_qualified_source_schema.h"

namespace xllm {

inline torch::Tensor h3_qualified_group_qkv(const torch::Tensor& query,
                                            const torch::Tensor& key,
                                            const torch::Tensor& value,
                                            int64_t heads,
                                            int64_t head_dim) {
  if (heads <= 0 || head_dim <= 0 || query.dim() != 2 ||
      query.size(0) != heads * head_dim || key.sizes() != query.sizes() ||
      value.sizes() != query.sizes() ||
      query.scalar_type() != key.scalar_type() ||
      query.scalar_type() != value.scalar_type() ||
      query.device() != key.device() || query.device() != value.device()) {
    throw std::invalid_argument("MiniMax-H3 QKV source metadata mismatch");
  }
  const int64_t input = query.size(1);
  return torch::stack({query.reshape({heads, head_dim, input}),
                       key.reshape({heads, head_dim, input}),
                       value.reshape({heads, head_dim, input})},
                      1)
      .reshape({3 * heads * head_dim, input})
      .contiguous();
}

inline torch::Tensor h3_qualified_up_gate_to_gate_up(
    const torch::Tensor& weight) {
  if (weight.dim() != 2 || weight.size(0) <= 0 || weight.size(0) % 2 != 0) {
    throw std::invalid_argument(
        "MiniMax-H3 FFN source must contain two equal halves");
  }
  const auto halves = weight.chunk(2, 0);
  return torch::cat({halves[1], halves[0]}, 0).contiguous();
}

inline torch::Tensor h3_qualified_rope_inv_freq(
    const H3QualifiedShapeConfig& shape) {
  if (shape.rope_freq_dim <= 0 || !std::isfinite(shape.rope_theta) ||
      shape.rope_theta <= 0.0) {
    throw std::invalid_argument("MiniMax-H3 RoPE configuration is invalid");
  }
  const torch::Tensor frequencies = torch::arange(
      0, 2 * shape.rope_freq_dim, 2, torch::dtype(torch::kFloat32));
  return 1.0 /
         torch::pow(shape.rope_theta,
                    frequencies / static_cast<double>(2 * shape.rope_freq_dim));
}

class H3QualifiedWeightSource final {
 public:
  H3QualifiedWeightSource(std::unique_ptr<DiTFolderLoader> loader,
                          H3QualifiedShapeConfig shape)
      : loader_(std::move(loader)), shape_(std::move(shape)) {
    if (!loader_) {
      throw std::invalid_argument("MiniMax-H3 qualified loader is null");
    }
    index_and_validate(loader_->get_state_dicts());
  }

  // Also supports owned safetensors shards without a component-folder loader.
  H3QualifiedWeightSource(std::vector<std::unique_ptr<StateDict>> shards,
                          H3QualifiedShapeConfig shape)
      : owned_shards_(std::move(shards)), shape_(std::move(shape)) {
    index_and_validate(owned_shards_);
  }

  H3QualifiedWeightSource(const H3QualifiedWeightSource&) = delete;
  H3QualifiedWeightSource& operator=(const H3QualifiedWeightSource&) = delete;

  StateDict fixed_weights() const {
    TensorMap fixed;
    fixed.reserve(19 + 8 * shape_.refiner_layers);
    copy_mappings(h3_qualified_fixed_mappings(shape_), fixed);
    fixed.emplace("rope.inv_freq", h3_qualified_rope_inv_freq(shape_));
    for (int64_t index = 0; index < shape_.refiner_layers; ++index) {
      append_block(
          "token_refiner.refiner_blocks." + std::to_string(index) + ".",
          "token_refiner.blocks." + std::to_string(index) + ".",
          false,
          fixed);
    }
    return StateDict(std::move(fixed));
  }

  StateDict block_weights(int64_t index) const {
    if (index < 0 || index >= shape_.layers) {
      throw std::invalid_argument(
          "MiniMax-H3 qualified block index is out of range");
    }
    TensorMap block;
    block.reserve(10);
    append_block(
        "transformer_blocks." + std::to_string(index) + ".", "", true, block);
    return StateDict(std::move(block));
  }

  const H3QualifiedShapeConfig& shape() const { return shape_; }

 private:
  using TensorMap = std::unordered_map<std::string, torch::Tensor>;

  void index_and_validate(
      const std::vector<std::unique_ptr<StateDict>>& shards) {
    if (shape_.hidden_size <= 0 || shape_.heads <= 0 || shape_.head_dim <= 0 ||
        shape_.ffn_dim <= 0 || shape_.layers <= 0 ||
        shape_.refiner_layers < 0) {
      throw std::invalid_argument(
          "MiniMax-H3 qualified weight geometry is invalid");
    }
    const auto inventory = h3_qualified_source_inventory(shape_);
    index_.reserve(inventory.size());
    for (const auto& shard : shards) {
      if (!shard) {
        throw std::invalid_argument(
            "MiniMax-H3 qualified source contains a null shard");
      }
      for (const auto& [name, tensor] : *shard) {
        if (!index_.emplace(name, &tensor).second) {
          throw std::invalid_argument("MiniMax-H3 duplicate source tensor: " +
                                      name);
        }
      }
    }
    for (const auto& spec : inventory) {
      const auto iterator = index_.find(spec.source);
      if (iterator == index_.end()) {
        throw std::invalid_argument("MiniMax-H3 missing source tensor: " +
                                    spec.source);
      }
      const torch::Tensor& tensor = *iterator->second;
      if (!tensor.defined() || tensor.sizes().vec() != spec.shape ||
          tensor.scalar_type() != spec.dtype || !tensor.device().is_cpu()) {
        throw std::invalid_argument(
            "MiniMax-H3 source shape/dtype/device mismatch: " + spec.source);
      }
    }
  }

  void copy_mappings(const std::vector<H3QualifiedTensorMapping>& mappings,
                     TensorMap& target) const {
    for (const auto& mapping : mappings) {
      target.emplace(mapping.target, *index_.at(mapping.source));
    }
  }

  void append_block(const std::string& source_prefix,
                    const std::string& target_prefix,
                    bool with_adaln,
                    TensorMap& target) const {
    copy_mappings(h3_qualified_block_mappings(
                      shape_, source_prefix, target_prefix, with_adaln),
                  target);
    target.emplace(
        target_prefix + "attn.qkv_proj.weight",
        h3_qualified_group_qkv(*index_.at(source_prefix + "attn.to_q.weight"),
                               *index_.at(source_prefix + "attn.to_k.weight"),
                               *index_.at(source_prefix + "attn.to_v.weight"),
                               shape_.heads,
                               shape_.head_dim));
    target.emplace(target_prefix + "mlp.fc1.weight",
                   h3_qualified_up_gate_to_gate_up(
                       *index_.at(source_prefix + "ff.net.0.proj.weight")));
  }

  std::unique_ptr<DiTFolderLoader> loader_;
  std::vector<std::unique_ptr<StateDict>> owned_shards_;
  H3QualifiedShapeConfig shape_;
  // Borrowed from immutable shards owned by one of the members above.
  std::unordered_map<std::string, const torch::Tensor*> index_;
};

}  // namespace xllm
