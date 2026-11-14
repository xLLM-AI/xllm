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

#include <glog/logging.h>
#include <torch/torch.h>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "models/dit/transformers/minimax_h3_blocks.h"
#include "models/dit/transformers/minimax_h3_qualified_source_schema.h"
#include "models/dit/transformers/minimax_h3_qualified_weights.h"
#include "models/dit/utils/minimax_h3_qualified_input.h"

namespace xllm {

struct MiniMaxH3DenoiserOutput {
  torch::Tensor condition_projection;
  torch::Tensor refined_condition;
  torch::Tensor video_embedding;
  torch::Tensor audio_embedding;
  torch::Tensor packed_hidden;
  torch::Tensor time_embedding;
  torch::Tensor rope_frequencies;
  torch::Tensor combined_indices;
  // Populated only by explicit model-boundary diagnostics.
  torch::Tensor transformer_hidden;
  torch::Tensor final_activation;
  torch::Tensor video_velocity;
  torch::Tensor audio_velocity;
  int64_t executed_layers = 0;
};

using MiniMaxH3LayerObserver =
    std::function<void(int64_t step,
                       int64_t layer,
                       const MiniMaxH3ResidualBranchTrace& used_trace)>;
class MiniMaxH3StreamingDenoiserImpl final : public torch::nn::Module {
 public:
  static constexpr size_t kFixedTensorCount = 35;
  static constexpr size_t kBlockTensorCount = 10;

  explicit MiniMaxH3StreamingDenoiserImpl(const torch::TensorOptions& options) {
    const MiniMaxH3C4Config config;
    const torch::TensorOptions fp32 = options.dtype(torch::kFloat32);
    const torch::TensorOptions bf16 = options.dtype(torch::kBFloat16);
    video_patch_proj_ = register_module("video_patch_proj",
                                        MiniMaxH3Dense(config.video_patch_dim,
                                                       config.hidden_size,
                                                       /*with_bias=*/true,
                                                       fp32));
    audio_patch_proj_ = register_module("audio_patch_proj",
                                        MiniMaxH3Dense(config.audio_dim,
                                                       config.hidden_size,
                                                       /*with_bias=*/true,
                                                       fp32));
    condition_proj_ = register_module("condition_proj",
                                      MiniMaxH3Dense(config.text_dim,
                                                     config.hidden_size,
                                                     /*with_bias=*/true,
                                                     bf16));
    time_embedder_ = register_module("time_embedder",
                                     MiniMaxH3TimeEmbedder(config, options));
    rope_ = register_module("rope",
                            MiniMaxH3Rope(config.rope_inv_freq_len, options));
    token_refiner_ = register_module("token_refiner",
                                     MiniMaxH3TokenRefiner(config, options));
    streaming_block_ =
        register_module("streaming_block", MiniMaxH3DiTBlock(config, options));
    final_layer_ =
        register_module("final_layer", MiniMaxH3FinalLayer(config, options));
  }

  void load_fixed_weights(const StateDict& source) {
    if (fixed_loaded_) {
      throw std::logic_error(
          "MiniMax-H3 streaming denoiser fixed weights are already loaded");
    }
    auto parameters = named_parameters(/*recurse=*/true);
    auto buffers = named_buffers(/*recurse=*/true);
    torch::NoGradGuard no_grad;
    for (auto& parameter : parameters) {
      if (!parameter.key().starts_with("streaming_block.")) {
        copy_fixed_tensor(parameter.key(), parameter.value(), source);
      }
    }
    for (auto& buffer : buffers) {
      if (!buffer.key().starts_with("streaming_block.")) {
        copy_fixed_tensor(buffer.key(), buffer.value(), source);
      }
    }
    if (loaded_fixed_names_.size() != kFixedTensorCount) {
      throw std::logic_error(
          "MiniMax-H3 streaming denoiser fixed tensor count mismatch");
    }
    fixed_loaded_ = true;
  }

  MiniMaxH3DenoiserOutput forward(
      const H3QualifiedWeightSource& source,
      const H3PackedLayout& layout,
      const torch::Tensor& video_rows,
      const torch::Tensor& audio_rows,
      const torch::Tensor& timesteps,
      const torch::Tensor& inverse_indices,
      int64_t step,
      const MiniMaxH3LayerObserver& observer = nullptr,
      bool retain_preparation = false,
      bool retain_model_boundary = false,
      bool verify_intermediates = false) {
    verify_fixed_weights();
    if (step == 0) {
      validate_inputs(
          layout, video_rows, audio_rows, timesteps, inverse_indices);
    }
    const torch::Device device = condition_proj_->weight().device();
    const MiniMaxH3C4Config config;

    torch::Tensor condition = layout.condition_hidden.squeeze(0).to(device);
    torch::Tensor projected_condition = condition_proj_->forward(condition);
    torch::Tensor refiner_cu = torch::tensor(
        std::vector<int64_t>{0, condition.size(0)},
        torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
    torch::Tensor refined_condition =
        token_refiner_->forward_output_only(projected_condition, refiner_cu);
    torch::Tensor video_embedding =
        video_patch_proj_->forward(video_rows.to(device));
    torch::Tensor audio_embedding =
        audio_patch_proj_->forward(audio_rows.to(device));
    torch::Tensor hidden =
        torch::zeros({layout.aligned_length, config.hidden_size},
                     projected_condition.options());
    hidden.index_copy_(0, layout.text_pos.to(device), refined_condition);
    hidden.index_copy_(
        0, layout.img_pos.to(device), video_embedding.to(torch::kBFloat16));
    hidden.index_copy_(
        0, layout.audio_pos.to(device), audio_embedding.to(torch::kBFloat16));
    torch::Tensor time_embedding =
        time_embedder_->forward(timesteps.to(device));
    torch::Tensor rope_frequencies =
        rope_->forward(layout.position_ids.to(torch::kFloat32).to(device));
    torch::Tensor device_inverse = inverse_indices.to(device);
    torch::Tensor combined_indices = minimax_h3_combined_adaln_indices(
        device_inverse, layout.token_tags.to(device));
    MiniMaxH3DenoiserOutput output;
    if (observer || retain_preparation) {
      output.condition_projection = projected_condition;
      output.refined_condition = refined_condition;
      output.video_embedding = video_embedding;
      output.audio_embedding = audio_embedding;
      output.packed_hidden = hidden.slice(0, 0, layout.used_length);
      output.time_embedding = time_embedding;
      output.rope_frequencies =
          rope_frequencies.slice(0, 0, layout.used_length);
      output.combined_indices =
          combined_indices.slice(0, 0, layout.used_length);
    }
    condition = torch::Tensor();
    projected_condition = torch::Tensor();
    refined_condition = torch::Tensor();
    video_embedding = torch::Tensor();
    audio_embedding = torch::Tensor();
    hidden = hidden.slice(0, 0, layout.used_length);
    combined_indices = combined_indices.slice(0, 0, layout.used_length);
    rope_frequencies = rope_frequencies.slice(0, 0, layout.used_length);
    device_inverse = device_inverse.slice(0, 0, layout.used_length);
    const torch::Tensor used_cu_seqlens = torch::tensor(
        {0, static_cast<int32_t>(layout.used_length)}, torch::kInt32);
    for (int64_t layer = 0; layer < MiniMaxH3TransformerConfig::kNumLayers;
         ++layer) {
      StateDict block = source.block_weights(layer);
      load_block_weights(block, layer);
      if (observer) {
        MiniMaxH3ResidualBranchTrace block_trace =
            streaming_block_->forward(hidden,
                                      time_embedding,
                                      combined_indices,
                                      rope_frequencies,
                                      used_cu_seqlens);
        hidden = block_trace.output;
        if (verify_intermediates &&
            !torch::isfinite(hidden).all().item<bool>()) {
          LOG(FATAL) << "MiniMax-H3 block " << layer << " produced NaN or Inf";
        }
        observer(
            step,
            layer,
            {.adaln_parameters = block_trace.adaln_parameters,
             .shift_msa = block_trace.shift_msa.slice(0, 0, layout.used_length),
             .scale_msa = block_trace.scale_msa.slice(0, 0, layout.used_length),
             .gate_msa = block_trace.gate_msa.slice(0, 0, layout.used_length),
             .shift_mlp = block_trace.shift_mlp.slice(0, 0, layout.used_length),
             .scale_mlp = block_trace.scale_mlp.slice(0, 0, layout.used_length),
             .gate_mlp = block_trace.gate_mlp.slice(0, 0, layout.used_length),
             .norm1_output =
                 block_trace.norm1_output.slice(0, 0, layout.used_length),
             .attention_input =
                 block_trace.attention_input.slice(0, 0, layout.used_length),
             .attention_output =
                 block_trace.attention_output.slice(0, 0, layout.used_length),
             .attention_delta =
                 block_trace.attention_delta.slice(0, 0, layout.used_length),
             .norm2_output =
                 block_trace.norm2_output.slice(0, 0, layout.used_length),
             .mlp_input = block_trace.mlp_input.slice(0, 0, layout.used_length),
             .mlp_output =
                 block_trace.mlp_output.slice(0, 0, layout.used_length),
             .mlp_delta = block_trace.mlp_delta.slice(0, 0, layout.used_length),
             .output = hidden.slice(0, 0, layout.used_length)});
      } else {
        hidden = streaming_block_->forward_output_only(hidden,
                                                       time_embedding,
                                                       combined_indices,
                                                       rope_frequencies,
                                                       used_cu_seqlens);
        if (verify_intermediates &&
            !torch::isfinite(hidden).all().item<bool>()) {
          LOG(FATAL) << "MiniMax-H3 block " << layer << " produced NaN or Inf";
        }
      }
    }
    if (retain_model_boundary) {
      output.transformer_hidden = hidden;
    }
    MiniMaxH3FinalOutput final =
        final_layer_->forward(hidden, time_embedding, device_inverse, layout);
    if (observer || retain_preparation || retain_model_boundary) {
      output.final_activation =
          final.activation.slice(0, 0, layout.used_length);
    }
    output.video_velocity = final.raw_selected_video_logits;
    output.audio_velocity = final.raw_selected_audio_logits;
    output.executed_layers = MiniMaxH3TransformerConfig::kNumLayers;
    return output;
  }

  size_t loaded_fixed_tensor_count() const {
    return loaded_fixed_names_.size();
  }

  int64_t loaded_block_index() const { return loaded_block_index_; }

 private:
  void copy_fixed_tensor(const std::string& name,
                         torch::Tensor& target,
                         const StateDict& source) {
    torch::Tensor value = source.get_tensor(name);
    if (name.ends_with(".attn.qkv_proj.weight")) {
      value = minimax_h3_reorder_grouped_qkv(
          value,
          MiniMaxH3TransformerConfig::kNumAttentionHeads,
          MiniMaxH3TransformerConfig::kAttentionHeadDim);
    } else if (name.ends_with(".mlp.fc1.weight")) {
      value = minimax_h3_reorder_gate_up_to_up_gate(value);
    }
    if (target.sizes() != value.sizes() ||
        target.scalar_type() != value.scalar_type()) {
      throw std::invalid_argument(
          "MiniMax-H3 streaming fixed tensor metadata mismatch for `" + name +
          "`");
    }
    target.copy_(value.to(target.device()));
    loaded_fixed_names_.emplace(name);
  }

  void load_block_weights(const StateDict& source, int64_t layer) {
    if (layer < 0 || layer >= MiniMaxH3TransformerConfig::kNumLayers) {
      throw std::invalid_argument(
          "MiniMax-H3 streaming block index is out of range");
    }
    auto targets = streaming_block_->named_parameters(/*recurse=*/true);
    if (targets.size() != kBlockTensorCount) {
      throw std::logic_error(
          "MiniMax-H3 streaming block tensor count mismatch");
    }
    torch::NoGradGuard no_grad;
    for (auto& target_entry : targets) {
      const std::string source_name = target_entry.key();
      torch::Tensor value = source.get_tensor(source_name);
      if (target_entry.key() == "attn.qkv_proj.weight") {
        value = minimax_h3_reorder_grouped_qkv(
            value,
            MiniMaxH3TransformerConfig::kNumAttentionHeads,
            MiniMaxH3TransformerConfig::kAttentionHeadDim);
      } else if (target_entry.key() == "mlp.fc1.weight") {
        value = minimax_h3_reorder_gate_up_to_up_gate(value);
      }
      torch::Tensor& target = target_entry.value();
      if (target.sizes() != value.sizes() ||
          target.scalar_type() != value.scalar_type()) {
        throw std::invalid_argument(
            "MiniMax-H3 streaming block tensor metadata mismatch for `" +
            source_name + "`");
      }
      target.copy_(value.to(target.device()));
    }
    loaded_block_index_ = layer;
  }

  void verify_fixed_weights() const {
    if (!fixed_loaded_ || loaded_fixed_names_.size() != kFixedTensorCount) {
      throw std::logic_error(
          "MiniMax-H3 streaming denoiser fixed weights are incomplete");
    }
  }

  static void validate_latent_rows(const H3PackedLayout& layout,
                                   const torch::Tensor& video_rows,
                                   const torch::Tensor& audio_rows) {
    if (!video_rows.defined() || video_rows.dim() != 2 ||
        video_rows.size(0) != layout.img_pos.numel() ||
        video_rows.size(1) != 96 ||
        video_rows.scalar_type() != torch::kFloat32 || !audio_rows.defined() ||
        audio_rows.dim() != 2 ||
        audio_rows.size(0) != layout.audio_pos.numel() ||
        audio_rows.size(1) != 32 ||
        audio_rows.scalar_type() != torch::kFloat32) {
      throw std::invalid_argument(
          "MiniMax-H3 denoise rows must be FP32 in packed position order");
    }
  }

  static void validate_inputs(const H3PackedLayout& layout,
                              const torch::Tensor& video_rows,
                              const torch::Tensor& audio_rows,
                              const torch::Tensor& timesteps,
                              const torch::Tensor& inverse_indices) {
    validate_latent_rows(layout, video_rows, audio_rows);
    if (layout.condition_hidden.sizes().vec() !=
            std::vector<int64_t>({1,
                                  layout.text_pos.numel(),
                                  MiniMaxH3TransformerConfig::kTextDim}) ||
        layout.condition_hidden.scalar_type() != torch::kBFloat16 ||
        layout.position_ids.sizes().vec() !=
            std::vector<int64_t>({layout.aligned_length, 3}) ||
        layout.position_ids.scalar_type() != torch::kFloat64 ||
        layout.token_tags.sizes().vec() !=
            std::vector<int64_t>({layout.aligned_length}) ||
        layout.token_tags.scalar_type() != torch::kInt64 ||
        layout.cu_seqlens.scalar_type() != torch::kInt32 ||
        !torch::equal(
            layout.cu_seqlens,
            torch::tensor({0,
                           static_cast<int32_t>(layout.used_length),
                           static_cast<int32_t>(layout.aligned_length)},
                          torch::kInt32))) {
      throw std::invalid_argument(
          "MiniMax-H3 denoiser packed layout metadata is invalid");
    }
    if (!timesteps.defined() || timesteps.dim() != 1 ||
        timesteps.numel() <= 0 || timesteps.scalar_type() != torch::kFloat32 ||
        !torch::isfinite(timesteps).all().item<bool>() ||
        timesteps.min().item<float>() < 0.0F ||
        timesteps.max().item<float>() > 1.0F || !inverse_indices.defined() ||
        inverse_indices.dim() != 1 ||
        inverse_indices.numel() != layout.aligned_length ||
        inverse_indices.scalar_type() != torch::kInt64 ||
        inverse_indices.min().item<int64_t>() < 0 ||
        inverse_indices.max().item<int64_t>() >= timesteps.numel()) {
      throw std::invalid_argument(
          "MiniMax-H3 denoiser timestep metadata is invalid");
    }

    std::vector<bool> occupied(static_cast<size_t>(layout.used_length), false);
    const auto claim = [&occupied, &layout](const torch::Tensor& positions,
                                            const char* name) {
      if (positions.scalar_type() != torch::kInt64 || positions.dim() != 1) {
        throw std::invalid_argument(std::string("MiniMax-H3 ") + name +
                                    " positions must be int64 [N]");
      }
      const torch::Tensor cpu = positions.to(torch::kCPU).contiguous();
      const int64_t* values = cpu.const_data_ptr<int64_t>();
      for (int64_t index = 0; index < cpu.numel(); ++index) {
        const int64_t position = values[index];
        if (position < 0 || position >= layout.used_length ||
            occupied[static_cast<size_t>(position)]) {
          throw std::invalid_argument(
              "MiniMax-H3 packed positions overlap or are out of range");
        }
        occupied[static_cast<size_t>(position)] = true;
      }
    };
    claim(layout.text_pos, "text");
    claim(layout.img_pos, "image");
    claim(layout.audio_pos, "audio");
    if (std::find(occupied.begin(), occupied.end(), false) != occupied.end() ||
        layout.update_mask.scalar_type() != torch::kBool ||
        layout.update_mask.numel() != layout.img_pos.numel() ||
        layout.audio_update_mask.scalar_type() != torch::kBool ||
        layout.audio_update_mask.numel() != layout.audio_pos.numel()) {
      throw std::invalid_argument(
          "MiniMax-H3 packed positions or update masks are incomplete");
    }
  }

  MiniMaxH3Dense video_patch_proj_{nullptr};
  MiniMaxH3Dense audio_patch_proj_{nullptr};
  MiniMaxH3Dense condition_proj_{nullptr};
  MiniMaxH3TimeEmbedder time_embedder_{nullptr};
  MiniMaxH3Rope rope_{nullptr};
  MiniMaxH3TokenRefiner token_refiner_{nullptr};
  MiniMaxH3DiTBlock streaming_block_{nullptr};
  MiniMaxH3FinalLayer final_layer_{nullptr};
  std::unordered_set<std::string> loaded_fixed_names_;
  int64_t loaded_block_index_ = -1;
  bool fixed_loaded_ = false;
};
TORCH_MODULE(MiniMaxH3StreamingDenoiser);

}  // namespace xllm
