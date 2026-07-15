/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include <glog/logging.h>
#include <torch/torch.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "core/kernels/ops_api.h"
#include "core/layers/npu/npu_column_parallel_linear_impl.h"
#include "core/layers/npu/npu_rms_norm_impl.h"
#include "core/layers/npu/rotary_embedding.h"
#include "framework/model_loader.h"
#include "models/llm/npu/qwen3.h"
#include "models/model_registry.h"

namespace xllm::npu::model {

class DFlashQwen3ModelImpl final : public QWen3ModelImpl {
 public:
  explicit DFlashQwen3ModelImpl(const ModelContext& context)
      : QWen3ModelImpl(context) {
    const ModelArgs& model_args = context.get_model_args();
    const ParallelArgs& parallel_args = context.get_parallel_args();
    const int32_t dp_size = parallel_args.dp_size();
    const int32_t cp_size = parallel_args.cp_size();
    CHECK_GT(dp_size, 0) << "DFlash dp_size must be positive.";
    CHECK_GT(cp_size, 0) << "DFlash cp_size must be positive.";
    CHECK_EQ(parallel_args.world_size() % (dp_size * cp_size), 0)
        << "DFlash world_size must be divisible by dp_size * cp_size.";
    tp_size_ = parallel_args.world_size() / (dp_size * cp_size);
    CHECK_GT(tp_size_, 0) << "DFlash tp_size must be positive.";
    tp_rank_ = parallel_args.rank() % tp_size_;

    tensor_options_ = context.get_tensor_options();
    head_dim_ = model_args.head_dim();
    rms_norm_eps_ = model_args.rms_norm_eps();
    CHECK_GT(head_dim_, 0) << "DFlash head_dim must be positive.";
    CHECK_GT(model_args.layers_to_capture().size(), 0u)
        << "DFlash requires dflash_config.target_layer_ids.";

    fc_ = register_module("fc", layer::NpuColumnParallelLinear(context));
    hidden_norm_ = register_module("hidden_norm", layer::NpuRMSNorm(context));
    rotary_embedding_ = std::make_shared<xllm::RotaryEmbeddingGeneric>(
        head_dim_,
        model_args.max_position_embeddings(),
        layer::rotary::compute_inv_freq(
            head_dim_, model_args.rope_theta(), tensor_options_),
        // DFlash draft is fixed Qwen3-dense: NeoX-style, non-interleaved rope.
        /*interleaved=*/false,
        tensor_options_);
  }

  void load_state_dict(const StateDict& state_dict) override {
    fc_->load_state_dict(state_dict.get_dict_with_prefix("fc."));
    hidden_norm_->load_state_dict(
        state_dict.get_dict_with_prefix("hidden_norm."));
    load_context_kv_weights(state_dict);
    for (int32_t i = 0; i < static_cast<int32_t>(layers_.size()); ++i) {
      layers_[i]->load_state_dict(
          state_dict.get_dict_with_prefix("layers." + std::to_string(i) + "."));
    }
    norm_->load_state_dict(state_dict.get_dict_with_prefix("norm."));
  }

  void verify_loaded_weights(const std::string& prefix) const override {
    fc_->verify_loaded_weights(prefix + "fc.");
    hidden_norm_->verify_loaded_weights(prefix + "hidden_norm.");
    verify_context_kv_weights();
    for (int32_t i = 0; i < static_cast<int32_t>(layers_.size()); ++i) {
      layers_[i]->verify_loaded_weights(prefix + "layers." + std::to_string(i) +
                                        ".");
    }
    norm_->verify_loaded_weights(prefix + "norm.");
  }

  void merge_loaded_weights() override {
    fc_->merge_loaded_weights();
    hidden_norm_->merge_loaded_weights();
    build_fused_context_kv_weights();
    for (QWen3DecoderLayer& layer : layers_) {
      layer->merge_loaded_weights();
    }
    norm_->merge_loaded_weights();
  }

 protected:
  torch::Tensor gen_append_attn_mask(int32_t q_len,
                                     int32_t kv_len,
                                     int32_t max_kv_len,
                                     torch::Dtype dtype,
                                     torch::Device device) override {
    // Block-diffusion draft attends the full context non-causally: every draft
    // position sees the whole block, so all q_len rows share one kv-only mask.
    // Do not restore a causal (per-row diagonal) mask here.
    torch::Tensor non_causal_mask = attn_mask_.gen_append_mask(
        /*q_len=*/1, kv_len, max_kv_len, dtype, device);
    return non_causal_mask.repeat({q_len, 1});
  }

 public:
  // Projects the target's captured hidden states into the draft's dense KV
  // cache: fc -> hidden_norm -> fused per-layer K/V linear -> per-layer k_norm
  // + RoPE -> ATB ReshapeAndCache. The projection/norm/rope run here in the
  // model layer (as Eagle3 keeps its fc in the model); only the KV scatter is
  // delegated to the ATB operator wrapper. Sits outside forward() because it
  // has no attention and its shape doesn't match the decode graph.
  ModelOutput write_context_kv(const torch::Tensor& target_hidden,
                               const torch::Tensor& positions,
                               const torch::Tensor& device_cache_slots,
                               std::vector<KVCache>& kv_caches,
                               const ModelInputParams& input_params) {
    const int64_t num_layers = static_cast<int64_t>(layers_.size());
    CHECK_EQ(static_cast<int64_t>(kv_caches.size()), num_layers);
    CHECK(device_cache_slots.defined())
        << "DFlash context K/V requires device new_cache_slots.";
    CHECK_EQ(device_cache_slots.numel(), target_hidden.size(0))
        << "DFlash device cache slot count mismatch.";

    torch::Tensor projected_hidden = fc_(target_hidden, 0);
    projected_hidden = hidden_norm_(projected_hidden, 0);
    CHECK(fused_kv_weight_.defined())
        << "DFlash fused K/V weight is not initialized.";
    CHECK_GT(local_kv_heads_, 0) << "DFlash local KV heads is invalid.";

    const int64_t num_context = projected_hidden.size(0);
    torch::Tensor all_kv =
        torch::nn::functional::linear(projected_hidden, fused_kv_weight_);
    // fused output: [num_context, num_layers, 2(k/v), local_kv_heads, head_dim]
    // -> permute to [2(k/v), num_layers, num_context, local_kv_heads, head_dim]
    // so the selects below peel k/v first, then per-layer. The single
    // contiguous() copy is intentional: the cache writer needs a contiguous
    // per-layer k/v view, and doing it once beats re-materializing per layer.
    all_kv =
        all_kv.view({num_context, num_layers, 2, local_kv_heads_, head_dim_})
            .permute({2, 1, 0, 3, 4})
            .contiguous();

    torch::Tensor all_key = all_kv.select(/*dim=*/0, /*index=*/0);
    torch::Tensor all_value = all_kv.select(/*dim=*/0, /*index=*/1);

    // k_norm has a distinct weight per draft layer, but the normalization
    // itself (fp32 RMSNorm over the per-head [head_dim] vector) is uniform, so
    // apply it to all layers at once: stack the per-layer weights to broadcast
    // over [num_layers, num_context, local_kv_heads, head_dim] and normalize in
    // one shot instead of a per-layer loop. Mirrors minimax_rms_norm's batched
    // fp32 RMSNorm.
    torch::Tensor all_key_normed = apply_k_norm(all_key, k_norm_weight_);

    // RoPE depends only on positions and head_dim, not the layer index, so all
    // layers share the same rotation. Flatten [num_layers, num_context, ...] to
    // [num_layers * num_context, ...] and repeat positions per layer to apply
    // RoPE in a single fused call instead of one per layer.
    torch::Tensor flat_key = all_key_normed.reshape(
        {num_layers * num_context, local_kv_heads_, head_dim_});
    torch::Tensor repeated_positions = positions.repeat({num_layers});
    flat_key = apply_rope(flat_key, repeated_positions);
    all_key_normed =
        flat_key.view({num_layers, num_context, local_kv_heads_, head_dim_});

    // Cache write is inherently per-layer (each layer has its own KV cache).
#if defined(USE_NPU)
    const int32_t device_index = all_key_normed.device().index();
#endif
    for (int64_t i = 0; i < num_layers; ++i) {
      kernel::ReshapePagedCacheParams scatter_params;
      scatter_params.key = all_key_normed[i];
      scatter_params.value = all_value[i];
      scatter_params.k_cache = kv_caches[i].get_k_cache();
      scatter_params.v_cache = kv_caches[i].get_v_cache();
      scatter_params.slot_mapping = device_cache_slots;
      kernel::reshape_paged_cache(scatter_params);
#if defined(USE_NPU)
      // The standard attention layer op records this event from inside ATB;
      // this custom scatter path must record it explicitly so a
      // PD-disaggregated PUSH transfer's per-layer synchronizer does not stall
      // waiting on it.
      if (input_params.parallel.layer_synchronizer != nullptr &&
          !input_params.parallel.layer_synchronizer->record_event(
              i, device_index)) {
        return ModelOutput();
      }
#endif
    }
    return ModelOutput(projected_hidden);
  }

 private:
  // Batched fp32 RMSNorm over the trailing head_dim. `weight` broadcasts over
  // the leading dims (per-layer weights stacked to [num_layers,1,1,head_dim]).
  torch::Tensor apply_k_norm(const torch::Tensor& key,
                             const torch::Tensor& weight) const {
    torch::Tensor key_fp32 = key.to(torch::kFloat32);
    torch::Tensor variance = key_fp32.pow(2).mean(/*dim=*/-1, /*keepdim=*/true);
    torch::Tensor normed_key =
        key_fp32 * torch::rsqrt(variance + rms_norm_eps_);
    // Qwen3 k_norm scales by the raw weight (not gemma-style 1 + weight).
    return (normed_key * weight).to(key.scalar_type());
  }

  torch::Tensor apply_rope(const torch::Tensor& key,
                           const torch::Tensor& positions) const {
    CHECK(rotary_embedding_ != nullptr)
        << "DFlash rotary embedding is not initialized.";
    return std::get<1>(rotary_embedding_->forward(key, key, positions));
  }

  // Collect per-layer k/v projections and concatenate them layer-major
  // ([l0_k, l0_v, l1_k, l1_v, ...]) into one fused weight, so a single matmul
  // projects all draft layers. Built directly here; there is no merge step.
  // Accumulates per-layer k/v/norm weights from one state dict. A sharded HF
  // checkpoint delivers draft layers across multiple state dicts, so only the
  // layers present in this shard are filled; missing ones stay undefined until
  // a later shard. build_fused_context_kv_weights() cat/stacks them once all
  // shards are loaded (called from merge_loaded_weights).
  void load_context_kv_weights(const StateDict& state_dict) {
    const int32_t num_layers = static_cast<int32_t>(layers_.size());
    if (per_layer_k_proj_.empty()) {
      per_layer_k_proj_.resize(num_layers);
      per_layer_v_proj_.resize(num_layers);
      per_layer_k_norm_.resize(num_layers);
    }
    for (int32_t i = 0; i < num_layers; ++i) {
      StateDict layer_dict =
          state_dict.get_dict_with_prefix("layers." + std::to_string(i) + ".");
      torch::Tensor k_proj =
          layer_dict.get_sharded_tensor("self_attn.k_proj.weight",
                                        /*dim=*/0,
                                        tp_rank_,
                                        tp_size_);
      torch::Tensor v_proj =
          layer_dict.get_sharded_tensor("self_attn.v_proj.weight",
                                        /*dim=*/0,
                                        tp_rank_,
                                        tp_size_);
      torch::Tensor k_norm = layer_dict.get_tensor("self_attn.k_norm.weight");
      if (!k_proj.defined() && !v_proj.defined() && !k_norm.defined()) {
        continue;  // this draft layer lives in another shard
      }
      CHECK(k_proj.defined()) << "Failed to find DFlash draft layers." << i
                              << ".self_attn.k_proj.weight.";
      CHECK(v_proj.defined()) << "Failed to find DFlash draft layers." << i
                              << ".self_attn.v_proj.weight.";
      CHECK(k_norm.defined()) << "Failed to find DFlash draft layers." << i
                              << ".self_attn.k_norm.weight.";
      CHECK_EQ(k_proj.dim(), 2) << "DFlash k_proj weight must be 2D.";
      CHECK_EQ(v_proj.dim(), 2) << "DFlash v_proj weight must be 2D.";
      CHECK_EQ(k_proj.size(0), v_proj.size(0))
          << "DFlash k/v projection output size mismatch.";
      CHECK_EQ(k_proj.size(1), v_proj.size(1))
          << "DFlash k/v projection weight shape mismatch.";
      CHECK_EQ(k_proj.size(0) % head_dim_, 0)
          << "DFlash k_proj output size must align to head_dim.";
      const int64_t layer_local_kv_heads = k_proj.size(0) / head_dim_;
      if (local_kv_heads_ == 0) {
        local_kv_heads_ = layer_local_kv_heads;
      } else {
        CHECK_EQ(local_kv_heads_, layer_local_kv_heads)
            << "DFlash local KV heads mismatch.";
      }
      per_layer_k_proj_[i] = k_proj.to(tensor_options_);
      per_layer_v_proj_[i] = v_proj.to(tensor_options_);
      // k_norm is applied per draft layer (one weight per captured layer). A
      // plain fp32 RMSNorm over the per-head [head_dim] vector matches HF Qwen3
      // semantics.
      per_layer_k_norm_[i] = k_norm.to(tensor_options_).to(torch::kFloat32);
    }
  }

  // Fuses the accumulated per-layer weights into the layer-major fused k/v
  // weight ([l0_k, l0_v, l1_k, l1_v, ...]) and stacked k_norm. Called once all
  // shards are loaded; every draft layer must be present by now.
  void build_fused_context_kv_weights() {
    const int32_t num_layers = static_cast<int32_t>(layers_.size());
    std::vector<torch::Tensor> kv_weights;
    kv_weights.reserve(num_layers * 2);
    std::vector<torch::Tensor> k_norm_weights;
    k_norm_weights.reserve(num_layers);
    for (int32_t i = 0; i < num_layers; ++i) {
      CHECK(per_layer_k_proj_[i].defined())
          << "Missing DFlash draft layers." << i << ".self_attn.k_proj.weight.";
      CHECK(per_layer_v_proj_[i].defined())
          << "Missing DFlash draft layers." << i << ".self_attn.v_proj.weight.";
      CHECK(per_layer_k_norm_[i].defined())
          << "Missing DFlash draft layers." << i << ".self_attn.k_norm.weight.";
      kv_weights.emplace_back(per_layer_k_proj_[i]);
      kv_weights.emplace_back(per_layer_v_proj_[i]);
      k_norm_weights.emplace_back(per_layer_k_norm_[i]);
    }
    fused_kv_weight_ = torch::cat(kv_weights, /*dim=*/0).contiguous();
    // Stack per-layer k_norm weights to [num_layers, 1, 1, head_dim] so a
    // single batched RMSNorm broadcasts over [num_layers, num_context,
    // kv_heads, head_dim] at once instead of a per-layer loop.
    k_norm_weight_ =
        torch::stack(k_norm_weights, /*dim=*/0).view({num_layers, 1, 1, -1});
    per_layer_k_proj_.clear();
    per_layer_v_proj_.clear();
    per_layer_k_norm_.clear();
  }

  // Verifies every draft layer's context-K/V weights arrived across all shards
  // before they are fused. Runs before merge_loaded_weights(), so it checks the
  // accumulated per-layer tensors rather than the fused result.
  void verify_context_kv_weights() const {
    const int32_t num_layers = static_cast<int32_t>(layers_.size());
    CHECK_EQ(static_cast<int32_t>(per_layer_k_proj_.size()), num_layers)
        << "DFlash context K/V weights were not accumulated.";
    CHECK_GT(local_kv_heads_, 0) << "DFlash local KV heads is invalid.";
    for (int32_t i = 0; i < num_layers; ++i) {
      CHECK(per_layer_k_proj_[i].defined())
          << "Missing DFlash draft layers." << i << ".self_attn.k_proj.weight.";
      CHECK(per_layer_v_proj_[i].defined())
          << "Missing DFlash draft layers." << i << ".self_attn.v_proj.weight.";
      CHECK(per_layer_k_norm_[i].defined())
          << "Missing DFlash draft layers." << i << ".self_attn.k_norm.weight.";
    }
  }

  layer::NpuColumnParallelLinear fc_{nullptr};
  layer::NpuRMSNorm hidden_norm_{nullptr};
  std::shared_ptr<xllm::NpuRotaryEmbedding> rotary_embedding_;

  // Context-K/V write weights. load_context_kv_weights accumulates per-layer
  // tensors across shards; build_fused_context_kv_weights fuses them into the
  // layer-major fused k/v projection and the stacked per-layer k_norm.
  std::vector<torch::Tensor> per_layer_k_proj_;
  std::vector<torch::Tensor> per_layer_v_proj_;
  std::vector<torch::Tensor> per_layer_k_norm_;
  torch::Tensor fused_kv_weight_;
  torch::Tensor k_norm_weight_;
  torch::TensorOptions tensor_options_;
  int64_t head_dim_ = 0;
  double rms_norm_eps_ = 1e-6;
  int32_t tp_rank_ = 0;
  int32_t tp_size_ = 1;
  int64_t local_kv_heads_ = 0;
};
TORCH_MODULE(DFlashQwen3Model);

class DFlashQwen3ForCausalLMImpl final
    : public LlmForCausalLMImplBase<DFlashQwen3Model> {
 public:
  explicit DFlashQwen3ForCausalLMImpl(const ModelContext& context)
      : LlmForCausalLMImplBase<DFlashQwen3Model>(context) {}

  void load_model(std::unique_ptr<ModelLoader> loader,
                  std::string prefix = "model.") override {
    for (const std::unique_ptr<StateDict>& state_dict :
         loader->get_state_dicts()) {
      StateDict sub_dict = state_dict->get_dict_with_prefix(prefix);
      if (sub_dict.size() == 0) {
        sub_dict = state_dict->get_dict_with_prefix("");
      }
      model_->load_state_dict(sub_dict);
    }
    model_->verify_loaded_weights("");
    model_->merge_loaded_weights();
  }

  ModelOutput write_context_kv(const torch::Tensor& target_hidden,
                               const torch::Tensor& positions,
                               const torch::Tensor& device_cache_slots,
                               std::vector<KVCache>& kv_caches,
                               const ModelInputParams& input_params) {
    return model_->write_context_kv(
        target_hidden, positions, device_cache_slots, kv_caches, input_params);
  }
};
TORCH_MODULE(DFlashQwen3ForCausalLM);

// Draft config carries model_type="qwen3" and loads via the qwen3_atb args
// loader; worker_impl then overwrites args.model_type to "DFlashDraftModel"
// so this factory is picked to build the draft body. No dedicated DFlash args
// loader is registered.
REGISTER_CAUSAL_MODEL_WITH_VARNAME(dflash_draft_model,
                                   DFlashDraftModel,
                                   DFlashQwen3ForCausalLM);

}  // namespace xllm::npu::model
