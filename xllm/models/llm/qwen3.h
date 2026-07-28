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

#include <torch/nn/functional/normalization.h>

#include "core/framework/model/model_output.h"
#include "core/util/rec_model_utils.h"
#if defined(USE_NPU)
#include "core/layers/common/attention_mask.h"
#endif
#include "core/layers/qwen3_decoder_layer.h"
#include "llm_model_base.h"

namespace xllm {

class QWen3ModelImpl : public LlmModelImplBase<layer::Qwen3DecoderLayer> {
 public:
  QWen3ModelImpl(const ModelContext& context)
      : LlmModelImplBase<layer::Qwen3DecoderLayer>("qwen3",
                                                   context.get_model_args()) {
    // register submodules
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();

    if (!mrope_section_.empty()) {
      cos_sin_ = layer::rotary::get_concat_rotary_embedding(
          128,
          model_args.max_position_embeddings(),
          model_args.rope_theta(),
          options);
    }

    layers_.reserve(model_args.n_layers());
    norm_ = register_module("norm", layer::RMSNorm(context));
    embed_tokens_ =
        register_module("embed_tokens", layer::WordEmbedding(context));
#if defined(USE_NPU)
    attn_mask_ = layer::AttentionMask(
        options.device(), options.dtype().toScalarType(), /*mask_value=*/-9984);
    dense_attn_mask_ = layer::AttentionMask(
        options.device(), options.dtype().toScalarType(), /*mask_value=*/1);
#endif
    for (int32_t i = 0; i < model_args.n_layers(); i++) {
      auto layer = layer::Qwen3DecoderLayer(context, i);
      layers_.push_back(layer);
    }
  }

  std::pair<torch::Tensor, torch::Tensor> apply_mrope(
      const torch::Tensor positions) override {
    auto target_cos_sin = cos_sin_.index({positions});
    auto target_cos_sin_chunks = target_cos_sin.chunk(/*chunks=*/2, /*dim=*/-1);
    auto cos_pos = target_cos_sin_chunks[0].contiguous();
    auto sin_pos = target_cos_sin_chunks[1].contiguous();
    auto apply = [this](torch::Tensor x) {
      auto freqs_t = x[0].clone();
      // mrop_length == freqs_length == head_dim / 2
      int64_t mrop_length = static_cast<int64_t>(freqs_t.size(-1) / 2);

      for (int dim_idx = 1; dim_idx <= 2; ++dim_idx) {
        int64_t offset = dim_idx;
        int64_t section_len = mrope_section_[dim_idx];
        int64_t length = section_len * 3;

        // Since the last dim of freqs is repeated to 2*mrop_length
        // idx_first_half: [offset, offset+3, offset+6, ... < mrop_length]
        // idx_second_half: [mrop_length+offset, mrop_length+offset+3,
        //     mrop_length+offset+6, ... < 2*mrop_length]
        auto idx_first_half = torch::arange(offset, length, 3, torch::kLong);
        auto idx_second_half = torch::arange(
            offset + mrop_length, length + mrop_length, 3, torch::kLong);

        auto idx_tensor =
            torch::cat({idx_first_half, idx_second_half}, 0).to(x.device());
        // freqs_t[..., idx] = freqs[dim_idx][..., idx]
        auto src = x[dim_idx].index_select(-1, idx_tensor);
        freqs_t.index_copy_(-1, idx_tensor, src);
      }
      return freqs_t;
    };
    cos_pos = apply(cos_pos.reshape({positions.size(0), -1, cos_pos.size(-1)}));
    sin_pos = apply(sin_pos.reshape({positions.size(0), -1, sin_pos.size(-1)}));
    return std::make_pair(cos_pos, sin_pos);
  }

  virtual ModelOutput forward(torch::Tensor tokens,
                              torch::Tensor positions,
                              std::vector<KVCache>& kv_caches,
                              const ModelInputParams& input_params) {
    bool use_deepstack = input_params.multimodal.deep_stacks.size() > 0;
    ModelInputParams& input_params_new =
        const_cast<ModelInputParams&>(input_params);
    std::vector<torch::Tensor> deep_stacks;

    if (tokens.numel() == 0) {
      tokens = torch::tensor({1}).to(torch::kInt32).to(tokens.device());
      positions = torch::tensor({1}).to(torch::kInt32).to(tokens.device());
    }
    auto inputs_embeds = input_params.embedding.input_embedding;
    torch::Tensor h;
    if (inputs_embeds.defined()) {
      h = inputs_embeds;
    } else {
      h = embed_tokens_(tokens);
    }
    if (use_deepstack) {
      deep_stacks =
          input_params.multimodal.deep_stacks;  // [num_deepstack, hidden_size]
    }

    auto& dp_token_nums = input_params_new.parallel.dp_global_token_nums;
    std::replace(dp_token_nums.begin(), dp_token_nums.end(), 0, 1);
    if (!input_params_new.attn_metadata) {
      input_params_new.attn_metadata =
          std::make_shared<layer::AttentionMetadata>(
              get_attention_metadata(input_params_new, h));
    }

    auto& attn_metadata = *(input_params_new.attn_metadata);
    bool only_prefill =
        (attn_metadata.is_prefill || attn_metadata.is_chunked_prefill);
    if (positions.dim() == 2 && only_prefill && !mrope_section_.empty()) {
      std::tie(attn_metadata.mrope_cos, attn_metadata.mrope_sin) =
          apply_mrope(positions);
    }

    std::optional<torch::Tensor> residual;
    for (size_t i = 0; i < layers_.size(); i++) {
      if (!input_params_new.synchronize_layer(static_cast<uint32_t>(i))) {
        return ModelOutput();
      }
      if (is_rec_multi_round_mode() && input_params_new.has_llmrec_params()) {
        const auto& llmrec_params = input_params_new.llmrec_params();
        attn_metadata.full_k_cache = llmrec_params->full_k_caches[i];
        attn_metadata.full_v_cache = llmrec_params->full_v_caches[i];
        attn_metadata.unshared_k_cache = llmrec_params->unshared_k_caches[i];
        attn_metadata.unshared_v_cache = llmrec_params->unshared_v_caches[i];
      }
#if defined(USE_CUDA)
      attn_metadata.plan_info->layer_id = i;
      if (attn_metadata.shared_plan_info != nullptr) {
        attn_metadata.shared_plan_info->layer_id = i;
      }
      if (attn_metadata.unshared_plan_info != nullptr) {
        attn_metadata.unshared_plan_info->layer_id = i;
      }
#endif
      auto& layer = layers_[i];
      h = layer(h,
                residual,
                positions,
                attn_metadata,
                kv_caches[i],
                input_params_new);
      if (!input_params_new.record_layer(static_cast<uint32_t>(i),
                                         h.device())) {
        return ModelOutput();
      }

      if (use_deepstack) {
        if (deep_stacks.size() > 0 && i < deep_stacks.size()) {
          h = h + deep_stacks[i];
        }
      }
    }
    auto [hidden_states, residual_out] = norm_(h, residual);
    return ModelOutput(hidden_states, residual_out);
  }

 protected:
  layer::AttentionMetadata get_attention_metadata(
      const ModelInputParams& params,
      const torch::Tensor& h) {
#if defined(USE_NPU)
    if (params.graph.attn_mask.defined()) {
      return layer::AttentionMetadataBuilder::build(
          params, model_args_.enable_mla(), params.graph.attn_mask);
    }
    max_seq_len_ = std::max(params.meta.kv_max_seq_len, max_seq_len_);
    const bool use_append_mask =
        params.is_spec_verify || params.meta.batch_forward_type.is_mixed() ||
        params.meta.batch_forward_type.is_chunked_prefill();
    if (!use_append_mask) {
      return layer::AttentionMetadataBuilder::build(
          params,
          model_args_.enable_mla(),
          dense_attn_mask_.get_attn_mask(
              max_seq_len_, h.dtype().toScalarType(), h.device()));
    }
    const int32_t num_sequences = params.meta.num_sequences;
    if (num_sequences <= 0) {
      return layer::AttentionMetadataBuilder::build(
          params,
          model_args_.enable_mla(),
          dense_attn_mask_.get_attn_mask(
              max_seq_len_, h.dtype().toScalarType(), h.device()));
    }

    std::vector<torch::Tensor> req_mask_vec;
    req_mask_vec.reserve(num_sequences);
    const int32_t max_kv_seq = params.meta.kv_max_seq_len;
    for (int32_t j = 0; j < num_sequences; ++j) {
      const int32_t q_len = params.attention.host.q_seq_lens[j];
      const int32_t kv_len = params.attention.host.kv_seq_lens[j];
      req_mask_vec.emplace_back(attn_mask_.gen_append_mask(
          q_len, kv_len, max_kv_seq, h.dtype().toScalarType(), h.device()));
    }
    return layer::AttentionMetadataBuilder::build(
        params, model_args_.enable_mla(), torch::cat(req_mask_vec, 0));
#else
    return layer::AttentionMetadataBuilder::build(params,
                                                  model_args_.enable_mla());
#endif
  }

 private:
#if defined(USE_NPU)
  layer::AttentionMask attn_mask_;
  layer::AttentionMask dense_attn_mask_;
#endif
};
TORCH_MODULE(QWen3Model);

class QWen3ForCausalLMImpl : public LlmForCausalLMImplBase<QWen3Model> {
 public:
  QWen3ForCausalLMImpl(const ModelContext& context)
      : LlmForCausalLMImplBase<QWen3Model>(context) {}

  torch::Tensor pooler(const torch::Tensor& hidden_states,
                       const torch::Tensor& seleted_idxes) {
    auto h = hidden_states;
    if (seleted_idxes.defined()) {
      h = h.index_select(/*dim=*/0, seleted_idxes);
    }
    namespace F = torch::nn::functional;
    return F::normalize(h, F::NormalizeFuncOptions().p(2).dim(1));
  }
};
TORCH_MODULE(QWen3ForCausalLM);

// register the causal model
REGISTER_CAUSAL_MODEL(qwen3, QWen3ForCausalLM);

// register the model args
REGISTER_MODEL_ARGS(qwen3, [&] {
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

  LOAD_ARG_OR_FUNC(head_dim, "head_dim", [&] {
    return args->hidden_size() / args->n_heads();
  });

  SET_ARG(stop_token_ids, std::unordered_set<int32_t>({args->eos_token_id()}));
});

}  // namespace xllm
