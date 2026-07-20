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

#pragma once

#include "core/framework/config/kernel_config.h"
#include "core/framework/config/scheduler_config.h"
#include "core/framework/model/model_output.h"
#include "core/framework/model_context.h"
#include "core/framework/parallel_state/npu_dp_ep_padding.h"
#include "core/layers/npu/npu_qwen3_moe_decoder_layer_impl.h"
#include "llm_model_base.h"

namespace xllm::npu::model {

using torch::indexing::None;
using ISlice = torch::indexing::Slice;

class Qwen3MoeDecoderLayerImpl : public torch::nn::Module {
 public:
  Qwen3MoeDecoderLayerImpl(const ModelContext& context, const int32_t i) {
    // register submodules
    decoder_layer_ = register_module(
        "decoder_layer", layer::NpuQwen3MoeDecoderLayer(context, i));
  }

  torch::Tensor forward(torch::Tensor x,
                        std::optional<torch::Tensor>& residual,
                        torch::Tensor cos_pos,
                        torch::Tensor sin_pos,
                        torch::Tensor attn_mask,
                        KVCache& kv_cache,
                        const ModelInputParams& input_params,
                        aclrtEvent* event = nullptr,
                        std::atomic<bool>* event_flag = nullptr) {
    return decoder_layer_(x,
                          residual,
                          cos_pos,
                          sin_pos,
                          attn_mask,
                          kv_cache,
                          input_params,
                          event,
                          event_flag);
  }

  void load_state_dict(const StateDict& state_dict) {
    auto experts_state_dict = state_dict.get_dict_with_prefix("mlp.experts.");
    auto fused_gate_up = experts_state_dict.get_tensor("gate_up_proj");
    auto fused_down = experts_state_dict.get_tensor("down_proj");
    bool is_fused = fused_gate_up.defined() && fused_down.defined();

    if (is_fused) {
      torch::Tensor expert_gate_up = fused_gate_up;
      torch::Tensor expert_down = fused_down;

      const int num_experts = expert_gate_up.size(0);

      auto chunks = expert_gate_up.chunk(2, /*dim=*/-1);
      auto expert_gate = chunks[0].contiguous();
      auto expert_up = chunks[1].contiguous();

      std::unordered_map<std::string, torch::Tensor> out_state_dict;
      for (const auto& [name, tensor] : state_dict) {
        if (name.find("self_attn.") == 0 || name.find("mlp.gate.") == 0 ||
            name.find("input_layernorm.") == 0 ||
            name.find("post_attention_layernorm.") == 0) {
          out_state_dict.emplace(name, tensor);
        }
      }

      for (int i = 0; i < num_experts; ++i) {
        auto gate_i = expert_gate[i].transpose(0, 1);
        auto up_i = expert_up[i].transpose(0, 1);
        auto down_i = expert_down[i].transpose(0, 1);

        const std::string base = "mlp.experts." + std::to_string(i) + ".";
        out_state_dict.emplace(base + "gate_proj.weight", gate_i);
        out_state_dict.emplace(base + "up_proj.weight", up_i);
        out_state_dict.emplace(base + "down_proj.weight", down_i);
      }
      decoder_layer_->load_state_dict(StateDict(std::move(out_state_dict)));
    } else {
      decoder_layer_->load_state_dict(state_dict);
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    decoder_layer_->verify_loaded_weights(prefix);
  }

  void merge_loaded_weights() { decoder_layer_->merge_loaded_weights(); }
  void merge_and_move_pinned_host() {
    decoder_layer_->merge_and_move_pinned_host();
  }
  void free_weights() { decoder_layer_->free_weights(); }
  void reload_weights() { decoder_layer_->reload_weights(); }
  void reload_weights_from_device() {
    decoder_layer_->reload_weights_from_device();
  }

  layer::BaseLoader* get_loader() { return decoder_layer_->get_loader(); }

  void refresh_rolling_weights() { decoder_layer_->refresh_rolling_weights(); }

 private:
  layer::NpuQwen3MoeDecoderLayer decoder_layer_{nullptr};
};
TORCH_MODULE(Qwen3MoeDecoderLayer);

class Qwen3MoeModelImpl : public torch::nn::Module {
 public:
  Qwen3MoeModelImpl(const ModelContext& context)
      : device_(context.get_tensor_options().device()) {
    auto options = context.get_tensor_options();
    auto model_args = context.get_model_args();
    auto parallel_args = context.get_parallel_args();
    mrope_section_ = model_args.rope_scaling_mrope_section();
    blocks_ = register_module("layers", torch::nn::ModuleList());
    layers_.reserve(model_args.n_layers());
    // register submodules
    device_ = options.device();
    dtype_ = options.dtype().toScalarType();
    num_speculative_tokens_ = model_args.num_speculative_tokens();
    npu_embed_tokens_ =
        register_module("npu_embed_tokens", layer::NpuWordEmbedding(context));
    cos_sin_ = layer::rotary::get_concat_rotary_embedding(
        128,
        model_args.max_position_embeddings(),
        model_args.rope_theta(),
        options);

    atb_pos_emb_ = layer::NpuPosEmbedding(context);
    int32_t mask_value =
        ::xllm::SchedulerConfig::get_instance().enable_chunked_prefill() ? -9984
                                                                         : 1;
    attn_mask_ = layer::AttentionMask(options.device(),
                                      options.dtype().toScalarType(),
                                      /*mask_value=*/mask_value);
    norm_ = register_module("norm", layer::NpuRMSNorm(context));
    mapping_data_ = parallel_args.mapping_data();

    for (int32_t i = 0; i < model_args.n_layers(); ++i) {
      auto block = Qwen3MoeDecoderLayer(context, i);
      layers_.push_back(block);
      blocks_->push_back(block);
    }

    dp_size_ = parallel_args.dp_size();
    std::vector<int64_t> indices;
    dp_local_tp_size_ = parallel_args.world_size() / dp_size_;
    dp_rank_ = parallel_args.rank() / dp_local_tp_size_;
    rank_ = parallel_args.rank();
    num_experts_per_tok_ = model_args.num_experts_per_tok();
    for (int i = 0; i < parallel_args.world_size(); i += dp_local_tp_size_) {
      indices.push_back(i);
    }

    // Eagle3/DFlash target captures intermediate-layer aux hidden states to
    // drive the draft. The worker fills layers_to_capture (from config.json for
    // Eagle3, or the draft config for DFlash) before construction, so a
    // non-empty list is the capture signal. The DFlash draft model itself never
    // captures.
    const bool is_dflash_draft_model =
        model_args.model_type() == "DFlashDraftModel";
    const auto& layer_ids_from_config = model_args.layers_to_capture();
    const bool enable_aux_hidden_capture =
        !is_dflash_draft_model && !layer_ids_from_config.empty();
    if (enable_aux_hidden_capture) {
      set_aux_hidden_capture_layers(layer_ids_from_config);
      // Pre-allocate aux output buffer [max_tokens_per_batch, hidden_size *
      // num_captured]
      const size_t num_captured = layers_to_capture_set_.size();
      const int64_t aux_dim =
          model_args.hidden_size() * static_cast<int64_t>(num_captured);
      aux_output_buffer_ = torch::empty(
          {::xllm::SchedulerConfig::get_instance().max_tokens_per_batch(),
           aux_dim},
          options);
    }
  }

  void set_aux_hidden_capture_layers(const std::vector<int32_t>& layer_ids) {
    capture_aux_hidden_states_ = true;
    layers_to_capture_set_.clear();
    // Config uses 0-based layer indices.
    for (int32_t val : layer_ids) {
      layers_to_capture_set_.insert(val);
    }
    LOG(INFO) << "layers_to_capture_set_ size: "
              << layers_to_capture_set_.size();
  }

  // tokens: [num_tokens]
  // positions: [num_tokens] token pos in the sequence
  ModelOutput forward(torch::Tensor tokens,
                      torch::Tensor positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelInputParams& input_params) {
    if (dp_size_ > 1) {
      if (tokens.sizes() == 0) {
        tokens = torch::tensor({1}).to(torch::kInt32).to(device_);
        positions = torch::tensor({0}).to(torch::kInt32).to(device_);
      }
    }
    auto inputs_embeds = input_params.embedding.input_embedding;
    torch::Tensor h;
    if (inputs_embeds.defined()) {
      h = inputs_embeds;
    } else {
      h = npu_embed_tokens_(tokens, 0);
    }

    auto target_cos_sin = atb_pos_emb_(cos_sin_, positions, 0);
    auto target_cos_sin_chunks = target_cos_sin.chunk(/*chunks=*/2, /*dim=*/-1);
    auto cos_pos = target_cos_sin_chunks[0].contiguous();
    auto sin_pos = target_cos_sin_chunks[1].contiguous();
    if (positions.dim() == 2) {  // mrope
      auto apply = [this](torch::Tensor x) {
        // auto sections = mrope_section_;
        auto freqs_t = x[0].clone();
        // mrop_length == freqs_length == head_dim / 2
        int64_t mrop_length = freqs_t.size(-1) / 2;

        for (int dim_idx = 1; dim_idx <= 2; ++dim_idx) {
          int64_t offset = dim_idx;  // H -> offset=1, W -> offset=2
          int64_t section_len = mrope_section_[dim_idx];
          int64_t length = section_len * 3;

          // Since the last dim of freqs is repeated to 2*mrop_length
          // idx_first_half: [offset, offset+3, offset+6, ... < mrop_length]
          // idx_second_half: [mrop_length+offset, mrop_length+offset+3,
          //     mrop_length+offset+6, ... < 2*mrop_length]
          torch::TensorOptions options =
              torch::TensorOptions().dtype(torch::kLong).device(x.device());
          auto idx_first_half = torch::arange(offset, length, 3, options);
          auto idx_second_half = torch::arange(
              offset + mrop_length, length + mrop_length, 3, options);
          auto idx_tensor = torch::cat({idx_first_half, idx_second_half}, 0);
          // freqs_t[..., idx] = freqs[dim_idx][..., idx]
          auto src = x[dim_idx].index_select(-1, idx_tensor);
          freqs_t.index_copy_(-1, idx_tensor, src);
        }
        return freqs_t;
      };
      cos_pos = apply(cos_pos.reshape(
          {positions.sizes().front(), -1, cos_pos.sizes().back()}));
      sin_pos = apply(sin_pos.reshape(
          {positions.sizes().front(), -1, sin_pos.sizes().back()}));
    }

    torch::Tensor attn_mask;
    // for chunked prefill, generate the attn mask.
    if (!input_params.meta.batch_forward_type.is_decode()) {
      max_seq_len_ =
          ::xllm::SchedulerConfig::get_instance().enable_chunked_prefill()
              ? std::max(input_params.meta.kv_max_seq_len, max_seq_len_)
              : 128;
      if (::xllm::SchedulerConfig::get_instance().enable_chunked_prefill()) {
        attn_mask = attn_mask_.get_attn_mask(
            max_seq_len_, cos_pos.dtype().toScalarType(), cos_pos.device());

        int batch_size = input_params.attention.host.q_seq_lens.size();
        if (batch_size > 0) {
          std::vector<torch::Tensor> req_mask_vec;
          req_mask_vec.reserve(batch_size);

          for (int j = 0; j < batch_size; j++) {
            int start = input_params.attention.host.kv_seq_lens[j] -
                        input_params.attention.host.q_seq_lens[j];
            int end = input_params.attention.host.kv_seq_lens[j];

            auto req_mask_slice = attn_mask.slice(0, start, end);
            req_mask_vec.emplace_back(req_mask_slice);
          }
          attn_mask = torch::cat(req_mask_vec, 0);
        }
      } else if (input_params.meta.batch_forward_type.is_prefill()) {
        attn_mask = attn_mask_.get_attn_mask(max_seq_len_, dtype_, device_);
      }
    }
    auto deep_stacks = input_params.multimodal.deep_stacks;
    int deep_stack_size = deep_stacks.size();

    int64_t input_length = h.size(0);
    torch::Tensor expert_array = torch::arange(
        0,
        input_length * num_experts_per_tok_,
        torch::TensorOptions().dtype(torch::kInt32).device(tokens.device()));
    ModelInputParams& input_params_new =
        const_cast<ModelInputParams&>(input_params);
    input_params_new.expert.expert_array = expert_array;

    RollingLayerGuard rolling_guard(rolling_mgr_);

    std::optional<torch::Tensor> residual;
    if (::xllm::KernelConfig::get_instance().enable_intralayer_addnorm()) {
      residual = torch::zeros_like(h);
    }
    const int64_t num_tokens = h.size(0);
    const int64_t hidden_size = h.size(-1);
    size_t capture_idx = 0;
    for (size_t i = 0; i < layers_.size(); i++) {
      aclrtEvent* event = nullptr;
      std::atomic<bool>* event_flag = nullptr;
      if (input_params.parallel.layer_synchronizer != nullptr) {
        event = input_params.parallel.layer_synchronizer->get_event(i);
        event_flag =
            input_params.parallel.layer_synchronizer->get_event_flag(i);
      }
      if (!input_params.synchronize_layer(i)) {
        return ModelOutput();
      }

      if (capture_aux_hidden_states_ &&
          layers_to_capture_set_.count(static_cast<int32_t>(i)) != 0) {
        // auto aux_h = h;
        auto aux_h =
            (::xllm::KernelConfig::get_instance().enable_intralayer_addnorm())
                ? h + residual.value()
                : h;
        aux_output_buffer_.slice(0, 0, num_tokens)
            .slice(1,
                   static_cast<int64_t>(capture_idx) * hidden_size,
                   static_cast<int64_t>(capture_idx + 1) * hidden_size)
            .copy_(aux_h.reshape({num_tokens, hidden_size}));
        capture_idx++;
      }

      auto& layer = layers_[i];
      const int32_t layer_index = i;
      rolling_guard.before_layer(layer_index);
      layer(h,
            residual,
            cos_pos,
            sin_pos,
            attn_mask,
            kv_caches[i],
            input_params,
            event,
            event_flag);
      rolling_guard.after_layer(layer_index);

      if (deep_stack_size && i < deep_stack_size) {
        h = h + deep_stacks[i];
      }
    }

    if (::xllm::KernelConfig::get_instance().enable_intralayer_addnorm())
      h = h + residual.value();
    auto hidden_states = norm_(h, 0);
    if (capture_aux_hidden_states_) {
      CHECK_EQ(capture_idx, layers_to_capture_set_.size())
          << "Captured aux hidden layer count mismatch.";
      torch::Tensor aux_hidden_states =
          aux_output_buffer_.slice(0, 0, num_tokens);
      return ModelOutput(hidden_states, torch::Tensor(), aux_hidden_states);
    }
    return ModelOutput(hidden_states);
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    npu_embed_tokens_->load_state_dict(
        state_dict.get_dict_with_prefix("embed_tokens."));
    // call each layer's load_state_dict function
    for (int i = 0; i < layers_.size(); i++) {
      layers_[i]->load_state_dict(
          state_dict.get_dict_with_prefix("layers." + std::to_string(i) + "."));
    }
    norm_->load_state_dict(state_dict.get_dict_with_prefix("norm."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    npu_embed_tokens_->verify_loaded_weights(prefix + "embed_tokens.");
    for (int i = 0; i < layers_.size(); i++) {
      layers_[i]->verify_loaded_weights(prefix + "layers." + std::to_string(i) +
                                        ".");
    }
    norm_->verify_loaded_weights(prefix + "norm.");
  }

  void merge_loaded_weights() {
    npu_embed_tokens_->merge_loaded_weights();
    for (int i = 0; i < layers_.size(); i++) {
      layers_[i]->merge_loaded_weights();
    }
    norm_->merge_loaded_weights();
  }

  void merge_and_move_pinned_host() {
    npu_embed_tokens_->merge_and_move_pinned_host();
    for (size_t i = 0; i < layers_.size(); i++) {
      layers_[i]->merge_and_move_pinned_host();
    }
    norm_->merge_and_move_pinned_host();
  }

  void free_weights() {
    npu_embed_tokens_->free_weights();
    for (size_t i = 0; i < layers_.size(); i++) {
      layers_[i]->free_weights();
    }
    norm_->free_weights();
  }

  void reload_weights() {
    npu_embed_tokens_->reload_weights();
    for (size_t i = 0; i < layers_.size(); i++) {
      layers_[i]->reload_weights();
    }
    norm_->reload_weights();
  }

  void reload_non_decoder_weights() {
    npu_embed_tokens_->reload_weights();
    norm_->reload_weights();
  }

  void reload_weights_from_device() {
    npu_embed_tokens_->reload_weights_from_device();
    for (size_t i = 0; i < layers_.size(); i++) {
      layers_[i]->reload_weights_from_device();
    }
    norm_->reload_weights_from_device();
  }

  void refresh_rolling_weights() {
    for (auto& layer : layers_) {
      layer->refresh_rolling_weights();
    }
  }

  std::vector<layer::BaseLoader*> get_decoder_loaders() {
    std::vector<layer::BaseLoader*> loaders;
    loaders.reserve(layers_.size());
    for (auto& layer : layers_) {
      loaders.push_back(layer->get_loader());
    }
    return loaders;
  }

  void set_rolling_load_manager(RollingLoadManager* mgr) { rolling_mgr_ = mgr; }

  layer::NpuWordEmbedding get_npu_word_embedding() { return npu_embed_tokens_; }

  void set_npu_word_embedding(layer::NpuWordEmbedding& npu_word_embedding) {
    npu_embed_tokens_ = npu_word_embedding;
  }
  torch::Tensor get_input_embeddings(torch::Tensor input_ids) {
    return npu_embed_tokens_(input_ids, 0);
  }

 private:
  torch::nn::ModuleList blocks_{nullptr};
  std::vector<Qwen3MoeDecoderLayer> layers_;
  int32_t max_seq_len_ = 0;
  int32_t dp_rank_;
  int32_t rank_;
  int32_t dp_size_;
  int32_t dp_local_tp_size_;
  nlohmann::json mapping_data_;
  int32_t num_experts_per_tok_;
  int32_t num_speculative_tokens_ = 0;
  at::Device device_;
  torch::Dtype dtype_;
  layer::NpuWordEmbedding npu_embed_tokens_{nullptr};
  layer::AttentionMask attn_mask_;
  layer::NpuRMSNorm norm_{nullptr};
  torch::Tensor cos_sin_;
  layer::NpuPosEmbedding atb_pos_emb_{nullptr};
  std::vector<int64_t> mrope_section_;
  RollingLoadManager* rolling_mgr_ = nullptr;

  // egale3
  std::unordered_set<int32_t> layers_to_capture_set_;
  bool capture_aux_hidden_states_ = false;
  torch::Tensor aux_output_buffer_;
};
TORCH_MODULE(Qwen3MoeModel);

class Qwen3MoeForCausalLMImpl
    : public xllm::npu::model::LlmForCausalLMImplBase<Qwen3MoeModel> {
 public:
  Qwen3MoeForCausalLMImpl(const ModelContext& context)
      : xllm::npu::model::LlmForCausalLMImplBase<Qwen3MoeModel>(context) {}
};
TORCH_MODULE(Qwen3MoeForCausalLM);

// register the causal model
REGISTER_CAUSAL_MODEL_WITH_VARNAME(qwen3_moe_atb,
                                   qwen3_moe_atb,
                                   Qwen3MoeForCausalLM);

// register the model args
// example config:
// https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/main/config.json
// https://huggingface.co/Qwen/Qwen3-235B-A22B/blob/main/config.json
REGISTER_MODEL_ARGS_WITH_VARNAME(qwen3_moe_atb, qwen3_moe_atb, [&] {
  LOAD_ARG_OR(model_type, "model_type", "qwen3_moe");
  LOAD_ARG_OR(dtype, "torch_dtype", "");
  LOAD_ARG_OR(attention_bias, "attention_bias", false);
  LOAD_ARG_OR(attention_dropout, "attention_dropout", 0.0f);
  LOAD_ARG_OR(bos_token_id, "bos_token_id", 151643);
  LOAD_ARG_OR(decoder_sparse_step, "decoder_sparse_step", 1);
  LOAD_ARG_OR(eos_token_id, "eos_token_id", 151645);
  LOAD_ARG_OR(head_dim, "head_dim", 128);
  LOAD_ARG_OR(hidden_act, "hidden_act", "silu");
  LOAD_ARG_OR(hidden_size, "hidden_size", 2048);
  LOAD_ARG_OR(initializer_range, "initializer_range", 0.02f);
  LOAD_ARG_OR(intermediate_size, "intermediate_size", 6144);
  LOAD_ARG_OR(max_position_embeddings, "max_position_embeddings", 40960);
  LOAD_ARG_OR(max_window_layers, "max_window_layers", 48);
  LOAD_ARG_OR(moe_intermediate_size, "moe_intermediate_size", 768);
  LOAD_ARG_OR(norm_topk_prob, "norm_topk_prob", true);
  LOAD_ARG_OR(n_heads, "num_attention_heads", 32);
  LOAD_ARG_OR(num_experts, "num_experts", 128);
  LOAD_ARG_OR(num_experts_per_tok, "num_experts_per_tok", 8);
  LOAD_ARG_OR(n_layers, "num_hidden_layers", 48);
  LOAD_ARG_OR(n_kv_heads, "num_key_value_heads", 4);
  LOAD_ARG_OR(output_router_logits, "output_router_logits", false);
  LOAD_ARG_OR(rms_norm_eps, "rms_norm_eps", 1e-6);
  LOAD_ARG_OR(rope_theta, "rope_theta", 1000000.0f);
  LOAD_ARG_OR(router_aux_loss_coef, "router_aux_loss_coef", 0.001f);
  LOAD_ARG_OR(use_sliding_window, "use_sliding_window", false);
  LOAD_ARG_OR(tie_word_embeddings, "tie_word_embeddings", false);
  LOAD_ARG_OR(vocab_size, "vocab_size", 151936);
  LOAD_ARG_OR(mlp_only_layers, "mlp_only_layers", std::vector<int>());

  // Eagle3: layer ids (0-based) to capture from config, e.g.
  // "layers_to_capture": [2, 14, 25]; defaults to empty if missing
  LOAD_ARG_OR(layers_to_capture, "layers_to_capture", std::vector<int32_t>{});

  SET_ARG(stop_token_ids, std::unordered_set<int32_t>({args->eos_token_id()}));
});
}  // namespace xllm::npu::model
