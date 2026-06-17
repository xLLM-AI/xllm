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

#include "core/framework/config/scheduler_config.h"
#include "core/layers/common/rotary_embedding_util.h"
#include "glm4_moe.h"
#include "mtp_model_base.h"

namespace xllm::npu::model {

class Glm4MoeMtpModelImpl
    : public xllm::npu::model::MtpModelImplBase<Glm4MoeDecoderLayer> {
 public:
  Glm4MoeMtpModelImpl(const ModelContext& context)
      : xllm::npu::model::MtpModelImplBase<Glm4MoeDecoderLayer>("glm4_moe_mtp",
                                                                context) {
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();

    int32_t mask_value =
        ::xllm::SchedulerConfig::get_instance().enable_chunked_prefill() ? -9984
                                                                         : 1;
    attn_mask_ = layer::AttentionMask(options.device(),
                                      options.dtype().toScalarType(),
                                      /*mask_value=*/mask_value);

    cos_sin_ = layer::rotary::get_concat_rotary_embedding(
        64,
        model_args.max_position_embeddings(),
        model_args.rope_theta(),
        options);
  }
};
TORCH_MODULE(Glm4MoeMtpModel);

class Glm4MoeMtpForCausalLMImpl
    : public xllm::npu::model::MtpForCausalLMImplBase<Glm4MoeMtpModel> {
 public:
  Glm4MoeMtpForCausalLMImpl(const ModelContext& context)
      : xllm::npu::model::MtpForCausalLMImplBase<Glm4MoeMtpModel>(context) {}
};
TORCH_MODULE(Glm4MoeMtpForCausalLM);

// register the causal model
REGISTER_CAUSAL_MODEL(glm4_moe_mtp, Glm4MoeMtpForCausalLM);

// example config:
// https://huggingface.co/zai-org/GLM-4.5-Air/blob/main/config.json
REGISTER_MODEL_ARGS(glm4_moe_mtp, [&] {
  LOAD_ARG_OR(model_type, "model_type", "glm4_moe_mtp");
  LOAD_ARG_OR(dtype, "torch_dtype", "");
  LOAD_ARG_OR(attention_bias, "attention_bias", false);
  LOAD_ARG_OR(attention_dropout, "attention_dropout", 0.0f);
  LOAD_ARG_OR(decoder_sparse_step, "decoder_sparse_step", 1);
  LOAD_ARG_OR(eos_token_id_vec, "eos_token_id", std::vector<int>{151329});
  LOAD_ARG_OR(head_dim, "head_dim", 128);
  LOAD_ARG_OR(hidden_act, "hidden_act", "silu");
  LOAD_ARG_OR(hidden_size, "hidden_size", 2048);
  LOAD_ARG_OR(initializer_range, "initializer_range", 0.02f);
  LOAD_ARG_OR(intermediate_size, "intermediate_size", 6144);
  LOAD_ARG_OR(max_position_embeddings, "max_position_embeddings", 40960);
  LOAD_ARG_OR(moe_intermediate_size, "moe_intermediate_size", 1536);
  LOAD_ARG_OR(routed_scaling_factor, "routed_scaling_factor", 2.5);
  LOAD_ARG_OR(norm_topk_prob, "norm_topk_prob", true);
  LOAD_ARG_OR(n_shared_experts, "n_shared_experts", 1);
  LOAD_ARG_OR(n_heads, "num_attention_heads", 96);
  LOAD_ARG_OR(num_experts, "n_routed_experts", 160);
  LOAD_ARG_OR(n_group, "n_group", 1);
  LOAD_ARG_OR(topk_group, "topk_group", 1);
  LOAD_ARG_OR(num_experts_per_tok, "num_experts_per_tok", 8);
  LOAD_ARG_OR(n_layers, "num_hidden_layers", 48);
  LOAD_ARG_OR(n_kv_heads, "num_key_value_heads", 4);
  LOAD_ARG_OR(use_qk_norm, "use_qk_norm", true);
  LOAD_ARG_OR(rms_norm_eps, "rms_norm_eps", 1e-6);
  LOAD_ARG_OR(rope_theta, "rope_theta", 1000000.0f);
  LOAD_ARG_OR(tie_word_embeddings, "tie_word_embeddings", false);
  LOAD_ARG_OR(vocab_size, "vocab_size", 151552);
  LOAD_ARG_OR(first_k_dense_replace, "first_k_dense_replace", 1);

  SET_ARG(stop_token_ids,
          std::unordered_set<int32_t>(args->eos_token_id_vec().begin(),
                                      args->eos_token_id_vec().end()));
});
}  // namespace xllm::npu::model
