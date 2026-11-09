/* Copyright 2026 The xLLM Authors.

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

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>

#include "core/framework/config/kv_cache_config.h"
#include "core/util/json_reader.h"
#include "models/model_registry.h"

namespace xllm {

inline void process_deepseek_v41_args(ModelArgs* args) {
  SET_ARG(n_activated_experts, args->num_experts_per_tok());

  auto& kv_sources = args->kv_source_layer_ids();
  std::sort(kv_sources.begin(), kv_sources.end());
  kv_sources.erase(std::unique(kv_sources.begin(), kv_sources.end()),
                   kv_sources.end());

  SET_ARG(stop_token_ids, std::unordered_set<int32_t>({args->eos_token_id()}));
}

inline void validate_deepseek_v41_args(const ModelArgs& args) {
  CHECK_GT(args.n_layers(), 0)
      << "deepseek_v41 config num_hidden_layers must be > 0, got "
      << args.n_layers();
  CHECK_GE(static_cast<int64_t>(args.compress_ratios().size()), args.n_layers())
      << "deepseek_v41 config compress_ratios must have one entry per layer "
         "(backbone + DSpark tail), got "
      << args.compress_ratios().size() << " vs " << args.n_layers();
  for (size_t i = 0; i < args.compress_ratios().size(); ++i) {
    const int32_t ratio = args.compress_ratios()[i];
    CHECK(ratio == 0 || ratio == 1 || ratio == 2 || ratio == 4)
        << "deepseek_v41 config compress_ratios[" << i
        << "] must be one of 0, 1, 2, 4, got " << ratio;
  }
  for (const int32_t layer_id : args.kv_source_layer_ids()) {
    CHECK_GE(layer_id, 0)
        << "deepseek_v41 config kv_source_layer_ids entry must be >= 0, got "
        << layer_id;
    CHECK_LT(layer_id, static_cast<int32_t>(args.compress_ratios().size()))
        << "deepseek_v41 config kv_source_layer_ids entry must index into "
           "compress_ratios, got "
        << layer_id << " vs " << args.compress_ratios().size();
    const int32_t ratio = args.compress_ratios()[static_cast<size_t>(layer_id)];
    CHECK(ratio == 1 || ratio == 2)
        << "deepseek_v41 config kv_source_layer_ids[" << layer_id
        << "] must reference a ratio 1/2 layer (only those own a V4.1 TOKEN "
           "manager), got ratio "
        << ratio;
  }
  // Native SWA tensors and Python ring metadata must use the same width.
  const int64_t block_size = KVCacheConfig::get_instance().block_size();
  CHECK_EQ(block_size, 128)
      << "DeepSeek V4.1 currently only supports block_size=128, got "
      << block_size;
  CHECK_EQ(args.window_size(), block_size)
      << "DeepSeek V4.1 window_size must match block_size, got "
      << args.window_size() << " vs " << block_size;
  CHECK_GT(args.n_routed_experts(), 0)
      << "deepseek_v41 config n_routed_experts must be > 0, got "
      << args.n_routed_experts();
  CHECK_GT(args.n_activated_experts(), 0)
      << "deepseek_v41 config num_experts_per_tok must be > 0, got "
      << args.n_activated_experts();
  CHECK_LE(args.n_activated_experts(), args.n_routed_experts())
      << "deepseek_v41 config num_experts_per_tok must be <= n_routed_experts"
      << ", got " << args.n_activated_experts() << " vs "
      << args.n_routed_experts();
  CHECK_GT(args.routed_scaling_factor(), 0.0f)
      << "deepseek_v41 config routed_scaling_factor must be > 0, got "
      << args.routed_scaling_factor();
  CHECK_GT(args.swiglu_limit(), 0.0f)
      << "deepseek_v41 config swiglu_limit must be > 0, got "
      << args.swiglu_limit();
  {
    std::string score_func = args.scoring_func();
    std::transform(
        score_func.begin(),
        score_func.end(),
        score_func.begin(),
        [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    CHECK(score_func == "softmax" || score_func == "sigmoid" ||
          score_func == "sqrtsoftplus")
        << "deepseek_v41 config scoring_func must be softmax, sigmoid or "
           "sqrtsoftplus, got "
        << args.scoring_func();
  }
  CHECK_GT(args.index_head_dim(), 0)
      << "deepseek_v41 config index_head_dim must be > 0, got "
      << args.index_head_dim();
  CHECK_GT(args.index_n_heads(), 0)
      << "deepseek_v41 config index_n_heads must be > 0, got "
      << args.index_n_heads();
  CHECK_GT(args.index_topk(), 0)
      << "deepseek_v41 config index_topk must be > 0, got "
      << args.index_topk();
  CHECK_GT(args.hc_mult(), 0)
      << "deepseek_v41 config hc_mult must be > 0, got " << args.hc_mult();
  CHECK_GE(args.hc_sinkhorn_iters(), 0)
      << "deepseek_v41 config hc_sinkhorn_iters must be >= 0, got "
      << args.hc_sinkhorn_iters();
  CHECK_GT(args.hc_eps(), 0.0f)
      << "deepseek_v41 config hc_eps must be > 0, got " << args.hc_eps();
  CHECK_GT(args.factor(), 0.0f)
      << "deepseek_v41 requires positive rope_scaling.factor, got "
      << args.factor();
  CHECK_GT(args.rope_scaling_attn_factor(), 0.0f)
      << "deepseek_v41 requires positive rope_scaling.attn_factor, got "
      << args.rope_scaling_attn_factor();
  CHECK_GT(args.rope_theta(), 0.0f)
      << "deepseek_v41 requires positive rope_theta, got " << args.rope_theta();
  CHECK_GT(args.compress_rope_theta(), 0.0f)
      << "deepseek_v41 requires positive compress_rope_theta, got "
      << args.compress_rope_theta();
}

inline bool load_deepseek_v41_model_args(const JsonReader& json,
                                         ModelArgs* args) {
  LOAD_ARG_OR(model_type, "model_type", "deepseek_v41");
  LOAD_ARG_OR(dtype, "torch_dtype", "bfloat16");

  LOAD_ARG_OR(hidden_size, "text_config.hidden_size", 5120);
  LOAD_ARG_OR(n_layers, "text_config.num_hidden_layers", 40);
  LOAD_ARG_OR(n_heads, "text_config.num_attention_heads", 64);
  LOAD_ARG_OR(n_kv_heads, "text_config.num_key_value_heads", 1);
  // V4.1 head_dim is independent of hidden_size / n_heads.
  LOAD_ARG_OR(head_dim, "text_config.head_dim", 512);
  LOAD_ARG_OR(hidden_act, "text_config.hidden_act", "silu");

  LOAD_ARG_OR(q_lora_rank, "text_config.q_lora_rank", 1280);
  LOAD_ARG_OR(o_lora_rank, "text_config.o_lora_rank", 1024);
  LOAD_ARG_OR(o_groups, "text_config.o_groups", 8);
  LOAD_ARG_OR(qk_rope_head_dim, "text_config.qk_rope_head_dim", 64);
  LOAD_ARG_OR(rope_head_dim, "text_config.qk_rope_head_dim", 64);

  LOAD_ARG_OR(n_routed_experts, "text_config.n_routed_experts", 384);
  LOAD_ARG_OR(n_activated_experts, "text_config.n_activated_experts", 6);
  LOAD_ARG_OR(num_experts_per_tok,
              "text_config.num_experts_per_tok",
              args->n_activated_experts());
  LOAD_ARG_OR(n_shared_experts, "text_config.n_shared_experts", 1);
  LOAD_ARG_OR(moe_intermediate_size, "text_config.moe_intermediate_size", 2304);
  LOAD_ARG_OR(routed_scaling_factor, "text_config.routed_scaling_factor", 1.5f);
  LOAD_ARG_OR(scoring_func, "text_config.scoring_func", "sqrtsoftplus");
  LOAD_ARG_OR(swiglu_limit, "text_config.swiglu_limit", 10.0f);
  LOAD_ARG_OR(norm_topk_prob, "text_config.norm_topk_prob", true);
  LOAD_ARG_OR(topk_method, "text_config.topk_method", "noaux_tc");

  LOAD_ARG_OR_FUNC(intermediate_size, "text_config.intermediate_size", [&] {
    return resolve_dense_intermediate_size(*args);
  });

  LOAD_ARG_OR(rms_norm_eps, "text_config.rms_norm_eps", 1e-20f);
  LOAD_ARG_OR(rope_theta, "text_config.rope_theta", 10000.0f);
  LOAD_ARG_OR(factor, "text_config.rope_scaling.factor", 16.0f);
  LOAD_ARG_OR(rope_scaling_factor, "text_config.rope_scaling.factor", 16.0f);
  LOAD_ARG_OR(beta_fast, "text_config.rope_scaling.beta_fast", 32.0f);
  LOAD_ARG_OR(beta_slow, "text_config.rope_scaling.beta_slow", 1.0f);
  LOAD_ARG_OR(
      rope_scaling_attn_factor, "text_config.rope_scaling.attn_factor", 1.0f);
  LOAD_ARG_OR(
      rope_scaling_rope_type, "text_config.rope_scaling.rope_type", "yarn");
  LOAD_ARG_OR(rope_scaling_original_max_position_embeddings,
              "text_config.rope_scaling.original_max_position_embeddings",
              65536);

  LOAD_ARG(compress_ratios, "text_config.compress_ratios");
  LOAD_ARG_OR(
      compress_rope_theta, "text_config.compress_rope_theta", 160000.0f);
  LOAD_ARG_OR_FUNC(window_size, "text_config.window_size", [&] {
    return json.value_or<int32_t>("text_config.sliding_window", 128);
  });

  LOAD_ARG_OR(index_head_dim, "text_config.index_head_dim", 128);
  LOAD_ARG_OR(index_n_heads, "text_config.index_n_heads", 32);
  LOAD_ARG_OR(index_topk, "text_config.index_topk", 512);

  LOAD_ARG_OR(hc_mult, "text_config.hc_mult", 4);
  LOAD_ARG_OR(hc_sinkhorn_iters, "text_config.hc_sinkhorn_iters", 20);
  LOAD_ARG_OR(hc_eps, "text_config.hc_eps", 1e-6f);
  LOAD_ARG_OR(scale_fmt, "scale_fmt", "ue8m0");
  LOAD_ARG_OR(scale_fmt, "quantization_config.scale_fmt", args->scale_fmt());

  LOAD_ARG(kv_source_layer_ids, "text_config.kv_source_layer_ids");
  LOAD_ARG(index_source_layer_ids, "text_config.index_source_layer_ids");
  LOAD_ARG_OR(
      candidate_source_layer_id, "text_config.candidate_source_layer_id", -1);
  LOAD_ARG_OR(candidate_topk_blocks, "text_config.candidate_topk_blocks", 2048);
  LOAD_ARG_OR(candidate_block_size, "text_config.candidate_block_size", 8);

  LOAD_ARG(engram_layer_ids, "text_config.engram_layer_ids");
  LOAD_ARG(engram_num_embeddings, "text_config.engram_num_embeddings");
  LOAD_ARG_OR(engram_max_ngram_size, "text_config.engram_max_ngram_size", 4);
  LOAD_ARG_OR(engram_vocab_size, "text_config.engram_vocab_size", 16000000);
  LOAD_ARG_OR(engram_n_heads, "text_config.engram_n_heads", 8);
  LOAD_ARG_OR(engram_head_dim, "text_config.engram_head_dim", 256);
  // -1 distinguishes an omitted Engram pad token from a configured token ID.
  LOAD_ARG_OR(engram_pad_token_id, "text_config.engram_pad_token_id", -1);
  LOAD_ARG_OR(engram_compressed_vocab_size,
              "text_config.engram_compressed_vocab_size",
              99092);

  LOAD_ARG_OR_FUNC(
      max_batch_size, "max_batch_size", [&] { return args->max_batch_size(); });
  LOAD_ARG_OR_FUNC(
      max_seq_len, "max_seq_len", [&] { return args->max_seq_len(); });

  LOAD_ARG_OR(vocab_size, "text_config.vocab_size", 129280);
  LOAD_ARG_OR(
      max_position_embeddings, "text_config.max_position_embeddings", 1048576);

  LOAD_ARG_OR(bos_token_id, "bos_token_id", 0);
  LOAD_ARG_OR(bos_token_id, "text_config.bos_token_id", args->bos_token_id());
  LOAD_ARG_OR(eos_token_id, "eos_token_id", 1);
  LOAD_ARG_OR(eos_token_id, "text_config.eos_token_id", args->eos_token_id());
  LOAD_ARG_OR(pad_token_id, "pad_token_id", 2);
  LOAD_ARG_OR(pad_token_id, "text_config.pad_token_id", args->pad_token_id());
  LOAD_ARG_OR(image_token_id, "image_token_id", -1);

  process_deepseek_v41_args(args);
  validate_deepseek_v41_args(*args);
  return true;
}

REGISTER_MODEL_BACKEND(deepseek_v41, "llm");

REGISTER_MODEL_ARGS_LOADER(deepseek_v41, &load_deepseek_v41_model_args);

}  // namespace xllm
