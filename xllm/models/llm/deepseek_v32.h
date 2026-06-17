/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

#include <optional>
#include <string>

#include "core/common/global_flags.h"
#include "core/framework/config/parallel_config.h"
#include "deepseek_v2.h"
#include "layers/common/attention_metadata_builder.h"
#include "layers/mlu/deepseek_v32_sp_context.h"

namespace xllm {

inline std::optional<std::string> validate_deepseek_v32_sp_flags(
    const ParallelArgs& parallel_args) {
  if (!::xllm::ParallelConfig::get_instance().enable_prefill_sp()) {
    return std::nullopt;
  }
  if (parallel_args.dp_size() != 1) {
    return "enable_prefill_sp requires dp_size == 1 for now.";
  }
  if (parallel_args.sp_group_ == nullptr ||
      parallel_args.sp_group_->world_size() <= 1) {
    return "enable_prefill_sp requires sequence parallel world_size > 1.";
  }

  return std::nullopt;
}

class DeepseekV32ModelImpl : public DeepseekV2ModelImpl {
 public:
  explicit DeepseekV32ModelImpl(const ModelContext& context)
      : DeepseekV2ModelImpl(context),
        device_(context.get_tensor_options().device()),
        sequence_parallel_group_(context.get_parallel_args().sp_group_),
        parallel_world_size_(context.get_parallel_args().world_size()) {
    const auto sp_config_error =
        validate_deepseek_v32_sp_flags(context.get_parallel_args());
    CHECK(!sp_config_error.has_value()) << sp_config_error.value();
  }

  ModelOutput forward(const torch::Tensor& tokens,
                      const torch::Tensor& positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelInputParams& input_params) {
    ModelInputParams modified_input_params = input_params;
    if (!modified_input_params.attn_metadata) {
      modified_input_params.attn_metadata =
          std::make_shared<layer::AttentionMetadata>(
              layer::AttentionMetadataBuilder::build(modified_input_params,
                                                     model_args_.enable_mla(),
                                                     /*compute_dtype=*/"half",
                                                     /*attn_mask=*/std::nullopt,
                                                     /*device=*/device_));
    }
    auto& attn_metadata = *modified_input_params.attn_metadata;
    std::optional<layer::v32_sp::DeepseekV32SPContext> sp_ctx;
    const bool requested_sequence_parallel =
        ::xllm::ParallelConfig::get_instance().enable_prefill_sp() &&
        input_params.meta.batch_forward_type.no_decode();
    if (requested_sequence_parallel) {
      if (sequence_parallel_group_ == nullptr) {
        CHECK_EQ(parallel_world_size_, 1)
            << "deepseek_v32 sequence parallel requires sp_group_.";
      } else if (sequence_parallel_group_->world_size() > 1) {
        sp_ctx = layer::v32_sp::build_deepseek_v32_sp_context(
            attn_metadata,
            input_params.meta.batch_forward_type,
            tokens,
            sequence_parallel_group_,
            sequence_parallel_group_->rank(),
            sequence_parallel_group_->world_size());
      }
    }
    if (!sp_ctx.has_value()) {
      // Fallback to the normal TP path when SP is disabled or the current
      // prefill batch cannot be split across all SP ranks.
      active_sequence_parallel_context_ = nullptr;
      return DeepseekV2ModelImpl::forward(
          tokens, positions, kv_caches, modified_input_params);
    }

    active_sequence_parallel_context_ = &sp_ctx.value();
    torch::Tensor hidden_states = embed_mod()(tokens);
    hidden_states =
        layer::v32_sp::reorder_to_local_shard(hidden_states, sp_ctx.value());
    torch::Tensor positions_local =
        layer::v32_sp::reorder_to_local_shard(positions, sp_ctx.value());
    std::optional<torch::Tensor> residual;
    for (size_t i = 0; i < layers_ref().size(); ++i) {
#if defined(USE_CUDA) || defined(USE_MUSA)
      attn_metadata.plan_info->layer_id = i;
#endif
      auto& layer = layers_ref()[i];
      prepare_decoder_layer_for_forward(i, layer, attn_metadata);
      hidden_states = layer(hidden_states,
                            residual,
                            positions_local,
                            attn_metadata,
                            kv_caches[i],
                            modified_input_params);
      if (!modified_input_params.record_layer(static_cast<uint32_t>(i),
                                              hidden_states.device())) {
        active_sequence_parallel_context_ = nullptr;
        return ModelOutput();
      }
    }
    hidden_states =
        layer::v32_sp::gather_and_restore_global(hidden_states, sp_ctx.value());
    auto [h, res] = norm_mod()(hidden_states, residual);
    active_sequence_parallel_context_ = nullptr;
    return ModelOutput(h, res);
  }

 protected:
  void prepare_decoder_layer_for_forward(
      size_t /*layer_id*/,
      layer::DeepseekV2DecoderLayer& layer,
      const layer::AttentionMetadata& /*attn_metadata*/) override {
#if defined(USE_MLU)
    layer->set_sequence_parallel_context(active_sequence_parallel_context_);
#endif
  }

 private:
  torch::Device device_;
  ProcessGroup* sequence_parallel_group_ = nullptr;
  int32_t parallel_world_size_ = 1;
  const layer::v32_sp::DeepseekV32SPContext* active_sequence_parallel_context_ =
      nullptr;
};
TORCH_MODULE(DeepseekV32Model);

class DeepseekV32ForCausalLMImpl
    : public LlmForCausalLMImplBase<DeepseekV32Model> {
 public:
  DeepseekV32ForCausalLMImpl(const ModelContext& context)
      : LlmForCausalLMImplBase<DeepseekV32Model>(context) {}
};
TORCH_MODULE(DeepseekV32ForCausalLM);

// register the causal model
REGISTER_CAUSAL_MODEL(deepseek_v32, DeepseekV32ForCausalLM);
// register the model args
// example config:
// https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json
REGISTER_MODEL_ARGS(deepseek_v32, [&] {
  LOAD_ARG_OR(model_type, "model_type", "deepseek_v32");
  LOAD_ARG_OR(dtype, "torch_dtype", "");
  LOAD_ARG_OR(vocab_size, "vocab_size", 129280);
  LOAD_ARG_OR(hidden_size, "hidden_size", 7168);
  LOAD_ARG_OR(n_layers, "num_hidden_layers", 61);
  LOAD_ARG_OR(n_heads, "num_attention_heads", 128);
  LOAD_ARG_OR(n_kv_heads, "num_key_value_heads", 128);
  LOAD_ARG_OR(intermediate_size, "intermediate_size", 18432);
  LOAD_ARG_OR(max_position_embeddings, "max_position_embeddings", 163840);
  LOAD_ARG_OR(rms_norm_eps, "rms_norm_eps", 1e-6);
  LOAD_ARG_OR(eos_token_id, "eos_token_id", 1);
  LOAD_ARG_OR(bos_token_id, "bos_token_id", 0);
  LOAD_ARG_OR(rope_theta, "rope_theta", 10000.0f);
  LOAD_ARG_OR(use_sliding_window, "use_sliding_window", false);
  LOAD_ARG_OR(sliding_window, "sliding_window", 4096);
  LOAD_ARG_OR(max_window_layers, "max_window_layers", 61);

  LOAD_ARG_OR(first_k_dense_replace, "first_k_dense_replace", 3);
  LOAD_ARG_OR(hidden_act, "hidden_act", "silu");
  LOAD_ARG_OR(moe_layer_freq, "moe_layer_freq", 1);
  LOAD_ARG_OR(topk_method, "topk_method", "noaux_tc");
  LOAD_ARG_OR(n_routed_experts, "n_routed_experts", 256);
  LOAD_ARG_OR(n_shared_experts, "n_shared_experts", 1);
  LOAD_ARG_OR(num_experts_per_tok, "num_experts_per_tok", 8);
  LOAD_ARG_OR(moe_intermediate_size, "moe_intermediate_size", 2048);
  LOAD_ARG_OR(routed_scaling_factor, "routed_scaling_factor", 2.5f);
  LOAD_ARG_OR(norm_topk_prob, "norm_topk_prob", true);
  LOAD_ARG_OR(n_group, "n_group", 8);
  LOAD_ARG_OR(topk_group, "topk_group", 4);
  LOAD_ARG_OR(scoring_func, "scoring_func", "sigmoid");
  LOAD_ARG_OR(qk_nope_head_dim, "qk_nope_head_dim", 128);
  LOAD_ARG_OR(qk_rope_head_dim, "qk_rope_head_dim", 64);
  LOAD_ARG_OR(v_head_dim, "v_head_dim", 128);
  LOAD_ARG_OR(q_lora_rank, "q_lora_rank", 1536);
  LOAD_ARG_OR(kv_lora_rank, "kv_lora_rank", 512);
  LOAD_ARG_OR(num_nextn_predict_layers, "num_nextn_predict_layers", 1);

  LOAD_ARG_OR_FUNC(head_dim, "head_dim", [&] {
    return 256;  // args->qk_nope_head_dim() + args->qk_rope_head_dim();
  });
  LOAD_ARG_OR_FUNC(
      rotary_dim, "rotary_dim", [&] { return args->qk_rope_head_dim(); });

  SET_ARG(rope_scaling_rope_type, "deepseek_yarn");
  LOAD_ARG(rope_scaling_beta_fast, "rope_scaling.beta_fast");
  LOAD_ARG(rope_scaling_beta_slow, "rope_scaling.beta_slow");
  LOAD_ARG(rope_scaling_factor, "rope_scaling.factor");
  LOAD_ARG_OR(
      rope_extrapolation_factor, "rope_scaling.extrapolation_factor", 1.0f);
  LOAD_ARG(rope_scaling_mscale, "rope_scaling.mscale");
  LOAD_ARG(rope_scaling_mscale_all_dim, "rope_scaling.mscale_all_dim");
  LOAD_ARG(rope_scaling_original_max_position_embeddings,
           "rope_scaling.original_max_position_embeddings");
  LOAD_ARG_OR(rope_scaling_attn_factor, "rope_scaling.attn_factor", 1.0f);

  SET_ARG(stop_token_ids, std::unordered_set<int32_t>({1}));

  // extra parameters for DeepSeek-V3.2-Exp
  // example config:
  // https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/config.json
  // set default value to 0 so as to distinguish from DeepSeek-V3.
  LOAD_ARG_OR(index_head_dim, "index_head_dim", 128);
  LOAD_ARG_OR(index_n_heads, "index_n_heads", 64);
  LOAD_ARG_OR(index_topk, "index_topk", 2048);

  // extra parameters to adopt with other models
  SET_ARG(indexer_rope_interleave, false);
});
}  // namespace xllm
