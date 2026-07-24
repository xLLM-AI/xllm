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

#include "core/framework/model/model_output.h"
#if defined(USE_NPU)
#include "core/common/global_flags.h"
#include "core/layers/common/attention_mask.h"
#endif
#include "core/layers/qwen3_moe_decoder_layer.h"
#include "core/util/rec_model_utils.h"
#include "llm_model_base.h"

namespace xllm {

class Qwen3MoeModelImpl : public LlmModelImplBase<layer::Qwen3MoeDecoderLayer> {
 public:
  Qwen3MoeModelImpl(const ModelContext& context)
      : LlmModelImplBase<layer::Qwen3MoeDecoderLayer>(
            "qwen3_moe",
            context.get_model_args()) {
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();
    layers_.reserve(model_args.n_layers());
    if (!mrope_section_.empty()) {
      cos_sin_ = layer::rotary::get_concat_rotary_embedding(
          128,
          model_args.max_position_embeddings(),
          model_args.rope_theta(),
          options);
    }

    // register submodules
    embed_tokens_ =
        register_module("embed_tokens", layer::WordEmbedding(context));
    norm_ = register_module("norm", layer::RMSNorm(context));
#if defined(USE_NPU)
    int32_t mask_value =
        ::xllm::SchedulerConfig::get_instance().enable_chunked_prefill() ? -9984
                                                                         : 1;
    attn_mask_ = layer::AttentionMask(
        options.device(), options.dtype().toScalarType(), mask_value);
#endif
    for (int32_t i = 0; i < model_args.n_layers(); ++i) {
      auto layer = layer::Qwen3MoeDecoderLayer(context, i);
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
        int64_t offset = dim_idx;  // H -> offset=1, W -> offset=2
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

  // tokens: [num_tokens]
  // positions: [num_tokens] token pos in the sequence
  ModelOutput forward(torch::Tensor tokens,
                      torch::Tensor positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelInputParams& input_params) override {
    ModelInputParams modified_input_params = input_params;
    if (tokens.numel() == 0) {
      tokens = torch::tensor({1}).to(torch::kInt32).to(tokens.device());
      positions = torch::tensor({1}).to(torch::kInt32).to(tokens.device());
    }
    auto& dp_token_nums = modified_input_params.parallel.dp_global_token_nums;
    std::replace(dp_token_nums.begin(), dp_token_nums.end(), 0, 1);
    auto inputs_embeds = input_params.embedding.input_embedding;
    torch::Tensor h;
    if (inputs_embeds.defined()) {
      h = inputs_embeds;
    } else {
      h = embed_tokens_(tokens);
    }

    auto deep_stacks = input_params.multimodal.deep_stacks;
    int deep_stack_size = deep_stacks.size();
    if (!modified_input_params.attn_metadata) {
      modified_input_params.attn_metadata =
          std::make_shared<layer::AttentionMetadata>(
              get_attention_metadata(modified_input_params, h));
    }
    auto& attn_metadata = *(modified_input_params.attn_metadata);
    bool only_prefill =
        (attn_metadata.is_prefill || attn_metadata.is_chunked_prefill);
    if (positions.dim() == 2 && only_prefill && !mrope_section_.empty()) {
      std::tie(attn_metadata.mrope_cos, attn_metadata.mrope_sin) =
          apply_mrope(positions);
    }

    const LlmRecMultiRoundParams* llmrec_params = nullptr;
    if (is_rec_multi_round_mode() &&
        modified_input_params.has_llmrec_params()) {
      llmrec_params = modified_input_params.llmrec_params();
      CHECK_EQ(llmrec_params->full_k_caches.size(), layers_.size())
          << "Rec multi-round mode requires full_k_caches per layer.";
      CHECK_EQ(llmrec_params->full_v_caches.size(), layers_.size())
          << "Rec multi-round mode requires full_v_caches per layer.";
      CHECK_EQ(llmrec_params->unshared_k_caches.size(), layers_.size())
          << "Rec multi-round mode requires unshared_k_caches per layer.";
      CHECK_EQ(llmrec_params->unshared_v_caches.size(), layers_.size())
          << "Rec multi-round mode requires unshared_v_caches per layer.";
    }

    std::optional<torch::Tensor> residual;
    for (size_t i = 0; i < layers_.size(); i++) {
      if (!modified_input_params.synchronize_layer(static_cast<uint32_t>(i))) {
        return ModelOutput();
      }
      if (llmrec_params != nullptr) {
        attn_metadata.full_k_cache = llmrec_params->full_k_caches[i];
        attn_metadata.full_v_cache = llmrec_params->full_v_caches[i];
        attn_metadata.unshared_k_cache = llmrec_params->unshared_k_caches[i];
        attn_metadata.unshared_v_cache = llmrec_params->unshared_v_caches[i];
      }
#if defined(USE_CUDA) || defined(USE_MUSA)
      attn_metadata.plan_info->layer_id = i;
#endif
#if defined(USE_CUDA)
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
                modified_input_params);
      if (!modified_input_params.record_layer(static_cast<uint32_t>(i),
                                              h.device())) {
        return ModelOutput();
      }

      if (deep_stack_size && i < deep_stack_size) {
        h = h + deep_stacks[i];
      }
    }
    auto [hidden_states, residual_out] = norm_(h, residual);
    return ModelOutput(hidden_states, residual_out);
  }

  StateDict update_expert_dict(const StateDict& state_dict) {
    auto dict = std::unordered_map<std::string, torch::Tensor>(
        state_dict.begin(), state_dict.end());
    std::unordered_map<std::string, torch::Tensor> expert_dict;
    for (auto& [name, expert_gate_up] : dict) {
      if (name.find(".mlp.experts.gate_up_proj") == std::string::npos) {
        continue;
      }

      const std::string prefix = name.substr(0, name.find("gate_up_proj"));
      const std::string down_name = prefix + "down_proj";
      auto expert_down = state_dict.get_tensor(down_name);
      CHECK(expert_down.defined()) << "not find down_proj: " << down_name;

      const int32_t num_experts = expert_gate_up.size(0);
      auto chunks = expert_gate_up.chunk(2, -1);
      torch::Tensor expert_gate = chunks[0].permute({0, 2, 1});
      torch::Tensor expert_up = chunks[1].permute({0, 2, 1});
      expert_down = expert_down.permute({0, 2, 1});

      for (int j = 0; j < num_experts; ++j) {
        const std::string expert_key = prefix + std::to_string(j) + ".";
        expert_dict[expert_key + "gate_proj.weight"] = expert_gate[j];
        expert_dict[expert_key + "up_proj.weight"] = expert_up[j];
        expert_dict[expert_key + "down_proj.weight"] = expert_down[j];
      }
    }
    if (expert_dict.empty()) {
      return state_dict;
    }
    dict.merge(expert_dict);
    return StateDict(std::move(dict));
  }

  void load_state_dict(const StateDict& state_dict) override {
    embed_tokens_->load_state_dict(
        state_dict.get_dict_with_prefix("embed_tokens."));
    auto new_state_dict = update_expert_dict(state_dict);
    for (size_t i = 0; i < layers_.size(); ++i) {
      layers_[i]->load_state_dict(new_state_dict.get_dict_with_prefix(
          "layers." + std::to_string(i) + "."));
    }
    norm_->load_state_dict(state_dict.get_dict_with_prefix("norm."));
  }

 private:
  layer::AttentionMetadata get_attention_metadata(
      const ModelInputParams& params,
      const torch::Tensor& h) {
#if defined(USE_NPU)
    max_seq_len_ = std::max(params.meta.kv_max_seq_len, max_seq_len_);
    torch::Tensor attn_mask;
    if (::xllm::SchedulerConfig::get_instance().enable_chunked_prefill()) {
      const int32_t max_kv_seq = params.meta.kv_max_seq_len;
      const int32_t num_sequences = params.meta.num_sequences;
      if (num_sequences > 0) {
        std::vector<torch::Tensor> req_mask_vec;
        req_mask_vec.reserve(num_sequences);

        for (int32_t j = 0; j < num_sequences; ++j) {
          auto mask =
              attn_mask_.gen_append_mask(params.attention.host.q_seq_lens[j],
                                         params.attention.host.kv_seq_lens[j],
                                         max_kv_seq,
                                         h.dtype().toScalarType(),
                                         h.device());
          req_mask_vec.emplace_back(mask);
        }
        attn_mask = torch::cat(req_mask_vec, 0);
      } else {
        attn_mask = attn_mask_.get_attn_mask(
            max_seq_len_, h.dtype().toScalarType(), h.device());
      }
    } else {
      attn_mask = attn_mask_.get_attn_mask(
          max_seq_len_, h.dtype().toScalarType(), h.device());
    }
    return layer::AttentionMetadataBuilder::build(
        params, model_args_.enable_mla(), attn_mask);
#else
    return layer::AttentionMetadataBuilder::build(params,
                                                  model_args_.enable_mla());
#endif
  }

#if defined(USE_NPU)
  layer::AttentionMask attn_mask_;
#endif
};
TORCH_MODULE(Qwen3MoeModel);

class Qwen3MoeForCausalLMImpl : public LlmForCausalLMImplBase<Qwen3MoeModel> {
 public:
  Qwen3MoeForCausalLMImpl(const ModelContext& context)
      : LlmForCausalLMImplBase<Qwen3MoeModel>(context) {}
};
TORCH_MODULE(Qwen3MoeForCausalLM);

// register the causal model
REGISTER_CAUSAL_MODEL(qwen3_moe, Qwen3MoeForCausalLM);

// register the model args
// example config:
// https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/main/config.json
// https://huggingface.co/Qwen/Qwen3-235B-A22B/blob/main/config.json
REGISTER_MODEL_ARGS(qwen3_moe, [&] {
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

  SET_ARG(stop_token_ids, std::unordered_set<int32_t>({args->eos_token_id()}));

  // arguments to be compatible with other fused moe models
  LOAD_ARG_OR(n_routed_experts, "num_experts", 128);
  SET_ARG(n_shared_experts, 0);
  SET_ARG(scoring_func, "softmax");
  SET_ARG(topk_method, "");
  SET_ARG(n_group, -1);
  SET_ARG(topk_group, 0);
  SET_ARG(routed_scaling_factor, 1.0);
});
}  // namespace xllm
