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

#include <c10/core/DeviceGuard.h>
#include <torch/nn/functional/normalization.h>

#include <filesystem>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

#include "core/common/global_flags.h"
#include "core/framework/config/kernel_config.h"
#include "core/framework/config/scheduler_config.h"
#include "core/framework/model/aux_hidden_capture.h"
#include "core/framework/model/model_output.h"
#include "core/framework/state_dict/state_dict.h"
#include "core/layers/npu/npu_qwen3_decoder_layer_impl.h"
#include "llm_model_base.h"

namespace xllm::npu::model {

class QWen3DecoderLayerImpl
    : public LlmDecoderLayerImplBase<layer::NpuQwen3DecoderLayer> {
 public:
  QWen3DecoderLayerImpl(const ModelContext& context, const int32_t layer_id)
      : LlmDecoderLayerImplBase<layer::NpuQwen3DecoderLayer>(context,
                                                             layer_id) {}
};
TORCH_MODULE(QWen3DecoderLayer);

class QWen3ModelImpl : public LlmModelImplBase<QWen3DecoderLayer> {
 public:
  QWen3ModelImpl(const ModelContext& context)
      : LlmModelImplBase<QWen3DecoderLayer>("qwen3", context.get_model_args()),
        aux_capture_(
            context.get_model_args(),
            context.get_tensor_options(),
            ::xllm::SchedulerConfig::get_instance().max_tokens_per_batch()) {
    // register submodules
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();
    auto parallel_args = context.get_parallel_args();
    auto dp_local_tp_size =
        parallel_args.world_size() / parallel_args.dp_size();
    dp_rank_ = parallel_args.rank() / dp_local_tp_size;

    blocks_ = register_module("layers", torch::nn::ModuleList());
    layers_.reserve(model_args.n_layers());
    norm_ = register_module("norm", layer::NpuRMSNorm(context));
    npu_embed_tokens_ =
        register_module("npu_embed_tokens", layer::NpuWordEmbedding(context));
    restored_embed_tokens_ = register_module("restored_embed_tokens",
                                             layer::NpuWordEmbedding(context));
    atb_pos_emb_ = layer::NpuPosEmbedding(context);
    cos_sin_ = layer::rotary::get_concat_rotary_embedding(
        128,
        model_args.max_position_embeddings(),
        model_args.rope_theta(),
        options);
    int32_t mask_value =
        ::xllm::SchedulerConfig::get_instance().enable_chunked_prefill() ? -9984
                                                                         : 1;
    // encode_attn_mask_ =
    //   layer::AttentionMask(options.device(),
    //   options.dtype()).get_attn_mask(2048, options.device(),
    //   options.dtype());
    attn_mask_ = layer::AttentionMask(options.device(),
                                      options.dtype().toScalarType(),
                                      /*mask_value=*/mask_value);

    for (int32_t i = 0; i < model_args.n_layers(); i++) {
      auto block = QWen3DecoderLayer(context, i);
      layers_.push_back(block);
      blocks_->push_back(block);
    }
  }

  void load_restored_embed_tokens(const StateDict& state_dict,
                                  torch::Tensor global_rotation,
                                  const torch::Device& device) {
    auto embed_weight = state_dict.get_tensor("embed_tokens.weight");
    if (!embed_weight.defined()) {
      return;
    }

    CHECK_EQ(embed_weight.dim(), 2) << "Embedding weight must be a 2D tensor";
    CHECK_EQ(global_rotation.dim(), 2)
        << "QuaRot global_rotation must be a 2D tensor";
    CHECK_EQ(global_rotation.size(0), global_rotation.size(1))
        << "QuaRot global_rotation must be square";
    CHECK_EQ(embed_weight.size(1), global_rotation.size(0))
        << "QuaRot global_rotation hidden size mismatch, expected "
        << embed_weight.size(1) << ", got " << global_rotation.size(0);

    torch::Tensor restored;
    {
      torch::DeviceGuard device_guard(device);
      auto npu_options =
          torch::TensorOptions().dtype(torch::kFloat32).device(device);
      auto cpu_options = torch::TensorOptions()
                             .dtype(embed_weight.scalar_type())
                             .device(torch::kCPU);
      auto embed_weight_npu =
          embed_weight.to(npu_options, /*non_blocking=*/false, /*copy=*/true)
              .contiguous();
      auto rotation_t_npu =
          global_rotation.to(npu_options, /*non_blocking=*/false, /*copy=*/true)
              .transpose(0, 1)
              .contiguous();
      restored = torch::matmul(embed_weight_npu, rotation_t_npu)
                     .to(cpu_options, /*non_blocking=*/false, /*copy=*/true)
                     .contiguous();
    }
    StateDict embed_state_dict({{"weight", restored}});
    restored_embed_tokens_->load_state_dict(embed_state_dict);
    has_restored_embed_tokens_ = true;
  }

  bool has_restored_embed_tokens() const { return has_restored_embed_tokens_; }

  bool has_restored_npu_word_embedding() const {
    return has_restored_embed_tokens_;
  }

  void verify_restored_embed_tokens(const std::string& prefix) const {
    if (has_restored_embed_tokens_) {
      restored_embed_tokens_->verify_loaded_weights(prefix + "embed_tokens.");
    }
  }

  void merge_restored_embed_tokens() {
    if (has_restored_embed_tokens_) {
      restored_embed_tokens_->merge_loaded_weights();
    }
  }

  void merge_and_move_restored_embed_tokens() {
    if (has_restored_embed_tokens_) {
      restored_embed_tokens_->merge_and_move_pinned_host();
    }
  }

  layer::NpuWordEmbedding get_npu_word_embedding() override {
    if (has_restored_embed_tokens_) {
      return restored_embed_tokens_;
    }
    return LlmModelImplBase<QWen3DecoderLayer>::get_npu_word_embedding();
  }

  virtual ModelOutput forward(torch::Tensor tokens,
                              torch::Tensor positions,
                              std::vector<KVCache>& kv_caches,
                              const ModelInputParams& input_params) {
    bool use_deepstack = input_params.multimodal.deep_stacks.size() > 0;
    std::vector<torch::Tensor> deep_stacks;

    if (tokens.numel() == 0) {
      tokens = torch::tensor({1}).to(torch::kInt32).to(tokens.device());
      positions = torch::tensor({0}).to(torch::kInt32).to(tokens.device());
    }
    auto inputs_embeds = input_params.embedding.input_embedding;
    torch::Tensor h;
    if (inputs_embeds.defined()) {
      h = inputs_embeds;
    } else {
      h = npu_embed_tokens_(tokens, 0);
    }

    // This residual tensor would be shared by all the layers, as the
    // current layer would use the output residual from previous layer,
    // the layer could use the residual through local variable
    // without passing the residual tensor to the layer.
    torch::Tensor residual;
    if (::xllm::KernelConfig::get_instance().enable_interlayer_addnorm()) {
      residual = torch::zeros_like(h, h.options());
      set_residual(residual);
    }

    if (use_deepstack) {
      deep_stacks =
          input_params.multimodal.deep_stacks;  // [num_deepstack, hidden_size]
    }
    auto target_cos_sin = atb_pos_emb_(cos_sin_, positions, 0);
    auto target_cos_sin_chunks = target_cos_sin.chunk(/*chunks=*/2, /*dim=*/-1);
    auto cos_pos = target_cos_sin_chunks[0].contiguous();
    auto sin_pos = target_cos_sin_chunks[1].contiguous();
    if (positions.dim() == 2) {  // mrope
      auto apply = [this](torch::Tensor x) {
        auto freqs_t = x[0].clone();
        // mrop_length == freqs_length == head_dim / 2
        int64_t mrop_length = freqs_t.size(-1) / 2;

        for (int dim_idx = 1; dim_idx <= 2; ++dim_idx) {
          int64_t offset = dim_idx;
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
      if (::xllm::SchedulerConfig::get_instance().enable_chunked_prefill()) {
        int32_t max_kv_seq = input_params.meta.kv_max_seq_len;
        int32_t num_sequences = input_params.meta.num_sequences;
        if (num_sequences > 0) {
          std::vector<torch::Tensor> req_mask_vec;
          req_mask_vec.reserve(num_sequences);

          for (int32_t j = 0; j < num_sequences; j++) {
            torch::Tensor mask =
                gen_append_attn_mask(input_params.attention.host.q_seq_lens[j],
                                     input_params.attention.host.kv_seq_lens[j],
                                     max_kv_seq,
                                     cos_pos.dtype().toScalarType(),
                                     cos_pos.device());
            req_mask_vec.emplace_back(mask);
          }
          attn_mask = torch::cat(req_mask_vec, 0);
        }
      } else {
        attn_mask = attn_mask_.get_attn_mask(
            128, cos_pos.dtype().toScalarType(), cos_pos.device());
      }
    }

    ModelInputParams& input_params_new =
        const_cast<ModelInputParams&>(input_params);
    RollingLayerGuard rolling_guard(rolling_mgr_);
    for (size_t i = 0; i < layers_.size(); i++) {
      aclrtEvent* event{nullptr};
      std::atomic<bool>* event_flag{nullptr};

      if (input_params.parallel.layer_synchronizer != nullptr) {
        event = input_params.parallel.layer_synchronizer->get_event(i);
        event_flag =
            input_params.parallel.layer_synchronizer->get_event_flag(i);
      }
      if (!input_params.synchronize_layer(i)) {
        return ModelOutput();
      }

      auto& layer = layers_[i];
      const int32_t layer_index = i;
      // ATB keeps `h` as the full residual stream, so no separate residual.
      aux_capture_.capture_layer(layer_index, h, std::nullopt);

      if (layer_forward_interrupted_) {
        LOG(INFO) << "Forward interrupted at layer: " << i;
        return ModelOutput();
      }
      rolling_guard.before_layer(layer_index);

      layer(h,
            cos_pos,
            sin_pos,
            attn_mask,
            kv_caches[i],
            input_params_new,
            event,
            event_flag);

      rolling_guard.after_layer(layer_index);
      if (use_deepstack) {
        if (deep_stacks.size() > 0 && i < deep_stacks.size()) {
          h = h + deep_stacks[i];
        }
      }
    }
    auto hidden_states = norm_(h, 0);
    return aux_capture_.finalize(hidden_states);
  }

 protected:
  virtual torch::Tensor gen_append_attn_mask(int32_t q_len,
                                             int32_t kv_len,
                                             int32_t max_kv_len,
                                             torch::Dtype dtype,
                                             torch::Device device) {
    return attn_mask_.gen_append_mask(q_len, kv_len, max_kv_len, dtype, device);
  }

 private:
  torch::Tensor viusal_pos_mask_;
  AuxHiddenCapture aux_capture_;
  bool has_restored_embed_tokens_ = false;
  layer::NpuWordEmbedding restored_embed_tokens_{nullptr};
};
TORCH_MODULE(QWen3Model);

class QWen3ForCausalLMImpl : public LlmForCausalLMImplBase<QWen3Model> {
 public:
  QWen3ForCausalLMImpl(const ModelContext& context)
      : LlmForCausalLMImplBase<QWen3Model>(context),
        device_(context.get_tensor_options().device()) {}

  torch::Tensor pooler(const torch::Tensor& hidden_states,
                       const torch::Tensor& seleted_idxes) {
    auto h = hidden_states;
    if (seleted_idxes.defined()) {
      h = h.index_select(/*dim=*/0, seleted_idxes);
    }
    return torch::nn::functional::normalize(
        h, torch::nn::functional::NormalizeFuncOptions().p(2).dim(1));
  }

  bool has_restored_npu_word_embedding() const {
    return model_->has_restored_npu_word_embedding();
  }

  void load_model(std::unique_ptr<ModelLoader> loader,
                  std::string prefix = "model.") override {
    torch::Tensor global_rotation =
        load_optional_quarot_rotation(loader->model_weights_path());
    for (const auto& state_dict : loader->get_state_dicts()) {
      auto model_state_dict = state_dict->get_dict_with_prefix(
          std::vector<std::string>{"model.language_model.",
                                   "language_model.model.",
                                   prefix,
                                   "model.",
                                   ""});
      model_->load_state_dict(model_state_dict);
      if (global_rotation.defined()) {
        model_->load_restored_embed_tokens(
            model_state_dict, global_rotation, device_);
      }
      if (!embedding_mode_) {
        if (tie_word_embeddings) {
          npu_lm_head_->load_state_dict(
              state_dict->get_dict_with_prefix(std::vector<std::string>{
                  prefix + "embed_tokens.", "embed_tokens."}));
        } else {
          npu_lm_head_->load_state_dict(
              state_dict->get_dict_with_prefix("lm_head."));
        }
      }
    }

    model_->verify_loaded_weights(prefix);
    model_->verify_restored_embed_tokens(prefix);
    if (!embedding_mode_) {
      if (tie_word_embeddings) {
        npu_lm_head_->verify_loaded_weights("embed_tokens.");
      } else {
        npu_lm_head_->verify_loaded_weights("lm_head.");
      }
    }

    model_->merge_loaded_weights();
    model_->merge_restored_embed_tokens();
    if (!embedding_mode_) {
      npu_lm_head_->merge_loaded_weights();
    }
  }

  void lazy_load_model(std::unique_ptr<ModelLoader> loader,
                       std::string prefix = "model.") override {
    if (keep_host_weights) {
      LOG(INFO) << "Model weights are already kept on host.";
      return;
    }
    torch::Tensor global_rotation =
        load_optional_quarot_rotation(loader->model_weights_path());
    for (const auto& state_dict : loader->get_state_dicts()) {
      auto model_state_dict = state_dict->get_dict_with_prefix(
          std::vector<std::string>{"model.language_model.",
                                   "language_model.model.",
                                   prefix,
                                   "model.",
                                   ""});
      model_->load_state_dict(model_state_dict);
      if (global_rotation.defined()) {
        model_->load_restored_embed_tokens(
            model_state_dict, global_rotation, device_);
      }
      if (!embedding_mode_) {
        if (tie_word_embeddings) {
          npu_lm_head_->load_state_dict(
              state_dict->get_dict_with_prefix(prefix + "embed_tokens."));
        } else {
          npu_lm_head_->load_state_dict(
              state_dict->get_dict_with_prefix("lm_head."));
        }
      }
    }

    model_->verify_loaded_weights(prefix);
    model_->verify_restored_embed_tokens(prefix);
    if (!embedding_mode_) {
      if (tie_word_embeddings) {
        npu_lm_head_->verify_loaded_weights(prefix + "embed_tokens.");
      } else {
        npu_lm_head_->verify_loaded_weights("lm_head.");
      }
    }

    model_->merge_and_move_pinned_host();
    model_->merge_and_move_restored_embed_tokens();
    if (!embedding_mode_) {
      npu_lm_head_->merge_and_move_pinned_host();
    }

    keep_host_weights = true;
  }

 private:
  torch::Tensor load_optional_quarot_rotation(
      const std::filesystem::path& model_path) {
    const std::filesystem::path quarot_path =
        model_path / "optional" / "quarot.safetensors";
    if (!std::filesystem::exists(quarot_path)) {
      return torch::Tensor();
    }

    auto state_dict = StateDictFromSafeTensor::load(quarot_path.string());
    torch::Tensor global_rotation = state_dict->get_tensor("global_rotation");
    if (!global_rotation.defined()) {
      LOG(WARNING) << "Optional QuaRot file exists but global_rotation is "
                      "missing: "
                   << quarot_path.string();
      return torch::Tensor();
    }
    CHECK_EQ(global_rotation.dim(), 2)
        << "QuaRot global_rotation must be a 2D tensor";
    CHECK_EQ(global_rotation.size(0), global_rotation.size(1))
        << "QuaRot global_rotation must be square";

    auto global_rotation_cpu = global_rotation
                                   .to(torch::TensorOptions()
                                           .dtype(torch::kFloat32)
                                           .device(torch::kCPU),
                                       /*non_blocking=*/false,
                                       /*copy=*/true)
                                   .contiguous();
    LOG(INFO) << "Loaded optional QuaRot global_rotation from "
              << quarot_path.string()
              << ", shape=" << global_rotation_cpu.sizes();
    return global_rotation_cpu;
  }

  torch::Device device_;
};
TORCH_MODULE(QWen3ForCausalLM);

// register the causal model
REGISTER_CAUSAL_MODEL_WITH_VARNAME(qwen3_atb, qwen3_atb, QWen3ForCausalLM);

// register the model args
REGISTER_MODEL_ARGS_WITH_VARNAME(qwen3_atb, qwen3_atb, [&] {
  LOAD_ARG_OR(model_type, "model_type", "qwen3");
  LOAD_ARG_OR(dtype, "torch_dtype", "");
  LOAD_ARG_OR(vocab_size, "vocab_size", 152064);
  LOAD_ARG_OR(hidden_size, "hidden_size", 3584);
  LOAD_ARG_OR(hidden_act, "hidden_act", "silu");
  LOAD_ARG_OR(n_layers, "num_hidden_layers", 28);
  LOAD_ARG_OR(n_heads, "num_attention_heads", 28);
  LOAD_ARG(n_kv_heads, "num_key_value_heads");
  // LOAD_ARG_OR(no_bias, "no_bias", true);
  LOAD_ARG_OR(intermediate_size, "intermediate_size", 18944);
  LOAD_ARG_OR(max_position_embeddings, "max_position_embeddings", 32768);
  LOAD_ARG_OR(rms_norm_eps, "rms_norm_eps", 1e-6);
  LOAD_ARG_OR(eos_token_id, "eos_token_id", 151643);
  LOAD_ARG_OR(rope_theta, "rope_theta", 1000000.0f);

  // For qwen3/2.5 model < 7B,  tie_word_embeddings = true
  LOAD_ARG_OR(tie_word_embeddings, "tie_word_embeddings", false);

  LOAD_ARG_OR(use_sliding_window, "use_sliding_window", false);
  LOAD_ARG_OR(max_window_layers, "max_window_layers", 28);

  // DSpark: low-rank dim of the Markov head (top-level config key, like vLLM's
  // config.markov_rank). 0 = disabled (plain DFlash / non-DSpark models).
  LOAD_ARG_OR(markov_rank, "markov_rank", 0);
  LOAD_ARG_OR(enable_confidence_head, "enable_confidence_head", false);
  LOAD_ARG_OR(
      confidence_head_with_markov, "confidence_head_with_markov", false);

  LOAD_ARG_OR_FUNC(head_dim, "head_dim", [&] {
    return args->hidden_size() / args->n_heads();
  });

  SET_ARG(stop_token_ids, std::unordered_set<int32_t>({args->eos_token_id()}));
});

}  // namespace xllm::npu::model
