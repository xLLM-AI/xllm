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

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "core/framework/state_dict/state_dict.h"
#include "models/dit/transformers/minimax_h3_denoiser.h"
#include "models/dit/transformers/minimax_h3_tp_uaa_block.h"
#include "models/dit/utils/dit_parallel_linear.h"

namespace xllm {

// ── MiniMax-H3 resident TP2 x U8 transformer ───────────────────────────────
//
// All 50 blocks stay on device for the whole request; every rank materializes
// only its own column-parallel TP shard plus replicated output weights, so
// there is no per-layer CPU weight streaming like the Commit 05 dense reference
// path does. One MiniMaxH3UaaWorkspace is allocated by this owner and handed to
// all 50 blocks for all 49 forwards.
class MiniMaxH3TpUaaResidentTransformerImpl final : public torch::nn::Module {
 public:
  static constexpr int64_t kResidentBlockCount =
      MiniMaxH3TransformerConfig::kNumLayers;
  static constexpr size_t kBlockTensorCount = 10;

  MiniMaxH3TpUaaResidentTransformerImpl(const MiniMaxH3C4Config& config,
                                        ProcessGroup* tp_group,
                                        ProcessGroup* sp_group,
                                        const torch::TensorOptions& options)
      : config_(config), workspace_(std::make_unique<MiniMaxH3UaaWorkspace>()) {
    blocks_ = register_module("blocks", torch::nn::ModuleList());
    block_layers_.reserve(static_cast<size_t>(kResidentBlockCount));
    for (int64_t layer = 0; layer < kResidentBlockCount; ++layer) {
      MiniMaxH3TpUaaBlock block(config, tp_group, sp_group, options);
      blocks_->push_back(block);
      block_layers_.emplace_back(std::move(block));
    }
    tp_group_ = block_layers_.front()->attention()->tp_group();
    sp_group_ = block_layers_.front()->attention()->sp_group();
  }

  // Loads this rank's shard of all 50 blocks once. The full source inventory is
  // validated first so a partial or renamed checkpoint fails closed.
  void load_source_weights(const H3QualifiedWeightSource& source) {
    if (loaded_block_count_ != 0) {
      throw std::logic_error(
          "MiniMax-H3 resident transformer weights are already loaded");
    }
    for (int64_t layer = 0; layer < kResidentBlockCount; ++layer) {
      StateDict block = source.block_weights(layer);
      block_layers_[static_cast<size_t>(layer)]->load_state_dict(block);
      block_layers_[static_cast<size_t>(layer)]->verify_loaded_weights();
      ++loaded_block_count_;
    }
  }

  // hidden / rope / combined_indices are this rank's row shard; cu_seqlens
  // describes the FULL packed sequence, which is what every rank sees after the
  // Ulysses transpose.
  torch::Tensor forward(const torch::Tensor& hidden,
                        const torch::Tensor& time_embedding,
                        const torch::Tensor& combined_indices,
                        const torch::Tensor& rope_frequencies,
                        const torch::Tensor& cu_seqlens,
                        const MiniMaxH3LayerObserver& observer = nullptr,
                        int64_t step = 0,
                        bool verify_intermediates = false) {
    if (loaded_block_count_ != kResidentBlockCount) {
      throw std::logic_error(
          "MiniMax-H3 resident transformer does not hold all 50 blocks");
    }
    validate_forward_shapes(
        hidden, time_embedding, combined_indices, rope_frequencies, cu_seqlens);
    workspace_->reserve(hidden.size(0), hidden.options());
    torch::Tensor current = hidden;
    for (int64_t layer = 0; layer < kResidentBlockCount; ++layer) {
      MiniMaxH3TpUaaBlock& block = block_layers_[static_cast<size_t>(layer)];
      if (observer) {
        MiniMaxH3ResidualBranchTrace trace =
            block->forward(current,
                           time_embedding,
                           combined_indices,
                           rope_frequencies,
                           cu_seqlens,
                           workspace_.get(),
                           verify_intermediates);
        current = trace.output;
        observer(step, layer, trace);
      } else {
        current = block->forward_output_only(current,
                                             time_embedding,
                                             combined_indices,
                                             rope_frequencies,
                                             cu_seqlens,
                                             workspace_.get(),
                                             verify_intermediates);
      }
      if (verify_intermediates &&
          !torch::isfinite(current).all().item<bool>()) {
        LOG(FATAL) << "MiniMax-H3 resident block " << layer
                   << " produced NaN or Inf";
      }
      ++block_forwards_;
    }
    return current;
  }

  int64_t loaded_block_count() const { return loaded_block_count_; }
  int64_t block_forwards() const { return block_forwards_; }
  int64_t workspace_reallocations() const {
    return workspace_->reallocations();
  }
  int64_t workspace_local_rows() const { return workspace_->local_rows(); }
  void reset_counters() { block_forwards_ = 0; }
  ProcessGroup* tp_group() const { return tp_group_; }
  ProcessGroup* sp_group() const { return sp_group_; }
  MiniMaxH3TpUaaBlock block_at(int64_t layer) const {
    if (layer < 0 || layer >= kResidentBlockCount) {
      throw std::out_of_range(
          "MiniMax-H3 resident block index is out of range");
    }
    return block_layers_[static_cast<size_t>(layer)];
  }

  int64_t resident_module_bytes() const {
    int64_t bytes = 0;
    const auto accumulate = [&bytes](const auto& item) {
      bytes += item.value().numel() *
               static_cast<int64_t>(item.value().element_size());
    };
    for (const auto& item : named_parameters(/*recurse=*/true)) {
      accumulate(item);
    }
    for (const auto& item : named_buffers(/*recurse=*/true)) {
      accumulate(item);
    }
    return bytes;
  }

 private:
  void validate_forward_shapes(const torch::Tensor& hidden,
                               const torch::Tensor& time_embedding,
                               const torch::Tensor& combined_indices,
                               const torch::Tensor& rope_frequencies,
                               const torch::Tensor& cu_seqlens) const {
    if (!hidden.defined() || hidden.dim() != 2 || hidden.size(0) <= 0 ||
        hidden.size(1) != config_.hidden_size ||
        hidden.scalar_type() != torch::kBFloat16) {
      throw std::invalid_argument(
          "MiniMax-H3 resident transformer hidden must be BF16 [S,hidden]");
    }
    if (!time_embedding.defined() || time_embedding.dim() != 2 ||
        time_embedding.size(1) != config_.time_embed_dim ||
        time_embedding.scalar_type() != torch::kFloat32) {
      throw std::invalid_argument(
          "MiniMax-H3 resident transformer time embedding must be FP32 [M,D]");
    }
    if (!combined_indices.defined() || combined_indices.dim() != 1 ||
        combined_indices.size(0) != hidden.size(0) ||
        combined_indices.scalar_type() != torch::kInt64) {
      throw std::invalid_argument(
          "MiniMax-H3 resident transformer AdaLN indices mismatch");
    }
    if (!rope_frequencies.defined() || rope_frequencies.dim() != 2 ||
        rope_frequencies.size(0) != hidden.size(0)) {
      throw std::invalid_argument(
          "MiniMax-H3 resident transformer RoPE row mismatch");
    }
    if (!cu_seqlens.defined() || cu_seqlens.dim() != 1 ||
        cu_seqlens.numel() < 2 ||
        (cu_seqlens.scalar_type() != torch::kInt32 &&
         cu_seqlens.scalar_type() != torch::kInt64)) {
      throw std::invalid_argument(
          "MiniMax-H3 resident transformer cu_seqlens mismatch");
    }
    const torch::Tensor bounds =
        cu_seqlens.to(torch::kCPU).to(torch::kInt64).contiguous();
    const int64_t* values = bounds.const_data_ptr<int64_t>();
    if (values[0] != 0 || values[bounds.numel() - 1] !=
                              hidden.size(0) * sp_group_->world_size()) {
      throw std::invalid_argument(
          "MiniMax-H3 resident transformer cu_seqlens must cover the full "
          "packed sequence");
    }
  }

  MiniMaxH3C4Config config_;
  std::unique_ptr<MiniMaxH3UaaWorkspace> workspace_;
  torch::nn::ModuleList blocks_{nullptr};
  std::vector<MiniMaxH3TpUaaBlock> block_layers_;
  ProcessGroup* tp_group_ = nullptr;
  ProcessGroup* sp_group_ = nullptr;
  int64_t loaded_block_count_ = 0;
  int64_t block_forwards_ = 0;
};
TORCH_MODULE(MiniMaxH3TpUaaResidentTransformer);

// ── MiniMax-H3 resident TP2 x U8 denoiser ──────────────────────────────────
//
// Mirrors MiniMaxH3StreamingDenoiserImpl::forward (video + audio rows, row
// timesteps, step) so the Commit 05 pipeline can drive 49 forwards unchanged,
// but the 50 blocks are resident and sharded instead of streamed from CPU.
// The fixed stage (patch/condition projections, time embedder, RoPE, token
// refiner, final layer) is replicated exactly as in the dense reference path.
class MiniMaxH3TpUaaResidentDenoiserImpl final : public torch::nn::Module {
 public:
  static constexpr size_t kFixedTensorCount = 35;

  MiniMaxH3TpUaaResidentDenoiserImpl(ProcessGroup* tp_group,
                                     ProcessGroup* sp_group,
                                     const torch::TensorOptions& options) {
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
    final_layer_ =
        register_module("final_layer", MiniMaxH3FinalLayer(config, options));
    transformer_ = register_module(
        "transformer",
        MiniMaxH3TpUaaResidentTransformer(config, tp_group, sp_group, options));
    sp_group_ = transformer_->sp_group();
  }

  void load_weights(const H3QualifiedWeightSource& source) {
    if (fixed_loaded_) {
      throw std::logic_error(
          "MiniMax-H3 resident denoiser fixed weights are already loaded");
    }
    StateDict fixed = source.fixed_weights();
    torch::NoGradGuard no_grad;
    for (auto& parameter : named_parameters(/*recurse=*/true)) {
      if (!parameter.key().starts_with("transformer.")) {
        copy_fixed_tensor(parameter.key(), parameter.value(), fixed);
      }
    }
    for (auto& buffer : named_buffers(/*recurse=*/true)) {
      if (!buffer.key().starts_with("transformer.")) {
        copy_fixed_tensor(buffer.key(), buffer.value(), fixed);
      }
    }
    if (loaded_fixed_names_.size() != kFixedTensorCount) {
      throw std::logic_error(
          "MiniMax-H3 resident denoiser fixed tensor count mismatch");
    }
    transformer_->load_source_weights(source);
    fixed_loaded_ = true;
  }

  MiniMaxH3DenoiserOutput forward(
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
    const MiniMaxH3C4Config config;
    const torch::Device device = condition_proj_->weight().device();

    // ── replicated fixed stage, identical math to the dense reference ──
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

    // ── U8 row shard -> 50 resident blocks -> row gather ──
    // Blocks run over the FULL aligned sequence (aligned % sp == 0) with
    // layout.cu_seqlens = {0, used, aligned}; the padding rows form their own
    // attention segment, so used-row numerics are untouched.
    torch::Tensor local_hidden =
        dit::sp_split_sequence(hidden, /*dim=*/0, sp_group_).contiguous();
    torch::Tensor local_rope =
        dit::sp_split_sequence(rope_frequencies, /*dim=*/0, sp_group_)
            .contiguous();
    torch::Tensor local_indices =
        dit::sp_split_sequence(combined_indices, /*dim=*/0, sp_group_)
            .contiguous();
    torch::Tensor local_hidden_out =
        transformer_->forward(local_hidden,
                              time_embedding,
                              local_indices,
                              local_rope,
                              layout.cu_seqlens,
                              observer,
                              step,
                              verify_intermediates);
    hidden = parallel_state::gather(
        local_hidden_out.contiguous(), sp_group_, /*dim=*/0);
    if (retain_model_boundary) {
      // Lightweight model-boundary capture (Commit-05 plan 11.1): keep only the
      // used rows of the gathered 50-block output; every block still ran the
      // trace-free forward_output_only path, so there is no per-layer trace.
      output.transformer_hidden = hidden.slice(0, 0, layout.used_length);
    }

    MiniMaxH3FinalOutput final_output =
        final_layer_->forward(hidden, time_embedding, device_inverse, layout);
    if (observer || retain_preparation || retain_model_boundary) {
      output.final_activation =
          final_output.activation.slice(0, 0, layout.used_length);
    }
    output.video_velocity = final_output.raw_selected_video_logits;
    output.audio_velocity = final_output.raw_selected_audio_logits;
    output.executed_layers = MiniMaxH3TransformerConfig::kNumLayers;
    return output;
  }

  size_t loaded_fixed_tensor_count() const {
    return loaded_fixed_names_.size();
  }
  int64_t loaded_block_count() const {
    return transformer_->loaded_block_count();
  }
  int64_t block_forwards() const { return transformer_->block_forwards(); }
  int64_t workspace_reallocations() const {
    return transformer_->workspace_reallocations();
  }
  void reset_counters() { transformer_->reset_counters(); }
  MiniMaxH3TpUaaResidentTransformer transformer() const { return transformer_; }
  ProcessGroup* sp_group() const { return sp_group_; }

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
          "MiniMax-H3 resident denoiser fixed tensor metadata mismatch for `" +
          name + "`");
    }
    target.copy_(value.to(target.device()));
    loaded_fixed_names_.emplace(name);
  }

  void verify_fixed_weights() const {
    if (!fixed_loaded_ || loaded_fixed_names_.size() != kFixedTensorCount ||
        transformer_->loaded_block_count() !=
            MiniMaxH3TpUaaResidentTransformerImpl::kResidentBlockCount) {
      throw std::logic_error(
          "MiniMax-H3 resident denoiser weights are incomplete");
    }
  }

  static void validate_inputs(const H3PackedLayout& layout,
                              const torch::Tensor& video_rows,
                              const torch::Tensor& audio_rows,
                              const torch::Tensor& timesteps,
                              const torch::Tensor& inverse_indices) {
    if (!video_rows.defined() || video_rows.dim() != 2 ||
        video_rows.size(0) != layout.img_pos.numel() ||
        video_rows.size(1) != 96 ||
        video_rows.scalar_type() != torch::kFloat32 || !audio_rows.defined() ||
        audio_rows.dim() != 2 ||
        audio_rows.size(0) != layout.audio_pos.numel() ||
        audio_rows.size(1) != 32 ||
        audio_rows.scalar_type() != torch::kFloat32) {
      throw std::invalid_argument(
          "MiniMax-H3 resident denoise rows must be FP32 in packed order");
    }
    if (layout.aligned_length <= 0 || layout.used_length <= 0 ||
        layout.aligned_length < layout.used_length ||
        layout.aligned_length % kMiniMaxH3SpSize != 0) {
      throw std::invalid_argument(
          "MiniMax-H3 resident denoiser requires an sp-divisible aligned "
          "sequence");
    }
    if (!timesteps.defined() || timesteps.dim() != 1 ||
        timesteps.numel() <= 0 || timesteps.scalar_type() != torch::kFloat32 ||
        !inverse_indices.defined() || inverse_indices.dim() != 1 ||
        inverse_indices.numel() != layout.aligned_length ||
        inverse_indices.scalar_type() != torch::kInt64) {
      throw std::invalid_argument(
          "MiniMax-H3 resident denoiser timestep metadata is invalid");
    }
    if (!layout.cu_seqlens.defined() ||
        !torch::equal(
            layout.cu_seqlens,
            torch::tensor({0,
                           static_cast<int32_t>(layout.used_length),
                           static_cast<int32_t>(layout.aligned_length)},
                          torch::kInt32))) {
      throw std::invalid_argument(
          "MiniMax-H3 resident denoiser packed cu_seqlens is invalid");
    }
  }

  ProcessGroup* sp_group_ = nullptr;
  MiniMaxH3Dense video_patch_proj_{nullptr};
  MiniMaxH3Dense audio_patch_proj_{nullptr};
  MiniMaxH3Dense condition_proj_{nullptr};
  MiniMaxH3TimeEmbedder time_embedder_{nullptr};
  MiniMaxH3Rope rope_{nullptr};
  MiniMaxH3TokenRefiner token_refiner_{nullptr};
  MiniMaxH3FinalLayer final_layer_{nullptr};
  MiniMaxH3TpUaaResidentTransformer transformer_{nullptr};
  std::unordered_set<std::string> loaded_fixed_names_;
  bool fixed_loaded_ = false;
};
TORCH_MODULE(MiniMaxH3TpUaaResidentDenoiser);

}  // namespace xllm
