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

#include <xlite/xlite.h>

#include <algorithm>
#include <cstdint>

#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/scheduler_config.h"
#include "core/framework/model/model_args.h"
#include "core/framework/model_context.h"
#include "core/framework/parallel_state/parallel_args.h"
#include "core/framework/quant_args.h"               // QuantArgs (IsW8A8)
#include "core/layers/common/dsa_topk_share_plan.h"  // DsaTopkSharePlan
#include "core/layers/xlite/xlite_init_utils.h"      // XliteTpSize

namespace xllm::xlite {

class XliteConfigBuilder {
 private:
  // W8A8 quantized model (quant_method non-empty, from
  // quant_model_description.json). Hardcoding true flips shared expert layout
  // for BF16 models.
  static bool IsW8A8(const ModelContext& context) {
    const auto& q = context.get_quant_args();
    return !q.quant_method().empty();
  }

 public:
  // Qwen3 dense (MHA); also the base for Qwen3-MoE (nDenseLayers overridden).
  static XModelConfig FromQwen3(const ModelContext& context) {
    const ModelArgs& a = context.get_model_args();
    const ParallelArgs& p = context.get_parallel_args();
    XModelConfig c{};

    c.vocabSize = static_cast<uint32_t>(a.vocab_size());
    c.hiddenSize = static_cast<uint32_t>(a.hidden_size());
    c.nLayers = static_cast<uint32_t>(a.n_layers());

    c.nHeads = static_cast<uint32_t>(a.n_heads());
    c.nKvHeads = a.n_kv_heads().has_value()
                     ? static_cast<uint32_t>(a.n_kv_heads().value())
                     : c.nHeads;  // GQA -> MHA fallback

    // head_dim: explicit if set, else hidden_size / n_heads.
    uint32_t head_dim = a.head_dim() > 0 ? static_cast<uint32_t>(a.head_dim())
                                         : c.hiddenSize / c.nHeads;
    c.headDim = head_dim;
    c.ropeHeadDim = head_dim;
    c.nopeHeadDim = 0;  // MLA-only
    c.vHeadDim = 0;     // MLA-only
    c.qLoraRank = 0;
    c.kvLoraRank = 0;

    c.attnType = XMODEL_ATTN_MHA;
    c.ropeType = XMODEL_ROPE_NEOX;
    c.addBias = a.qkv_bias();
    c.qkNorm = a.use_qk_norm();

    c.normEps = a.rms_norm_eps();
    c.ropeTheta = a.rope_theta();
    c.softmaxScale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // dense: all layers dense (no MoE).
    c.nDenseLayers = c.nLayers;
    c.nRoutedExperts = 0;
    c.nSharedExperts = 0;
    c.nActExperts = 0;
    c.intermediateSize = static_cast<uint32_t>(a.intermediate_size());
    c.moeIntermediateSize = 0;
    c.scoringFunc = XMODEL_SCORING_FUNC_SOFTMAX;
    c.normTopKProb = false;

    // defTpSize overridden by XliteCausalLMBase ctor; best-effort default here.
    c.defTpSize = XliteTpSize(p);
    c.defDpSize = static_cast<uint32_t>(p.dp_size());
    c.moeEpSize = static_cast<uint32_t>(p.ep_size());
    c.moeTPSize = 1;

    c.maxSeqLen = static_cast<uint64_t>(a.max_position_embeddings());
    // Runtime fields from singletons (ModelContext has no runtime::Options).
    c.maxBatchedTokens = static_cast<uint64_t>(
        SchedulerConfig::get_instance().max_tokens_per_batch());
    c.maxBatch = static_cast<uint64_t>(
        SchedulerConfig::get_instance().max_seqs_per_batch());
    uint32_t bs =
        static_cast<uint32_t>(KVCacheConfig::get_instance().block_size());
    c.blockSize = bs;
    c.blockSizes = {bs};
    c.deepstackNumLevel = 0;
    c.weightNZ = false;
    return c;
  }

  // Qwen3-MoE: overrides MoE fields on top of FromQwen3.
  static XModelConfig FromQwen3Moe(const ModelContext& context) {
    XModelConfig c = FromQwen3(context);
    const ModelArgs& a = context.get_model_args();
    const ParallelArgs& p = context.get_parallel_args();

    c.nDenseLayers = static_cast<uint32_t>(a.first_k_dense_replace());
    c.nRoutedExperts = static_cast<uint32_t>(a.num_experts());
    c.nSharedExperts = 0;
    c.nActExperts = static_cast<uint32_t>(a.num_experts_per_tok());
    c.moeIntermediateSize = static_cast<uint32_t>(a.moe_intermediate_size());
    c.scoringFunc = XMODEL_SCORING_FUNC_SOFTMAX;
    c.normTopKProb = a.norm_topk_prob();
    // HF weights are [out, in]; LoadMoEExperts transposes to [in, out].
    c.expertsWeightTrans = true;

    c.moeEpSize = static_cast<uint32_t>(p.ep_size());
    c.moeTPSize = p.ep_size() > 1
                      ? static_cast<uint32_t>(p.world_size() / p.ep_size())
                      : XliteTpSize(p);
    return c;
  }

  // DeepSeek-V3/R1 (MLA + sigmoid MoE + shared expert, no DSA).
  static XModelConfig FromDeepseekV3(const ModelContext& context) {
    XModelConfig c = FromQwen3(context);
    const ModelArgs& a = context.get_model_args();
    const ParallelArgs& p = context.get_parallel_args();

    // MLA dims (V3: nope=128/rope=64/v=128/q_lora=1536/kv_lora=512).
    c.attnType = XMODEL_ATTN_MLA;
    c.qLoraRank = static_cast<uint32_t>(a.q_lora_rank());
    c.kvLoraRank = static_cast<uint32_t>(a.kv_lora_rank());
    c.nopeHeadDim = static_cast<uint32_t>(a.qk_nope_head_dim());
    c.ropeHeadDim = static_cast<uint32_t>(a.qk_rope_head_dim());
    c.vHeadDim = static_cast<uint32_t>(a.v_head_dim());
    // MLA MQA: nKvHeads=1 (kv_lora_rank shared across heads); matches xlite ref
    // + framework KVCacheShape.
    c.nKvHeads = 1;
    // softmaxScale = 1/sqrt(nope+rope) + YaRN mscale (applied when maxSeqLen >
    // origSeqLen).
    float qkHeadDim = static_cast<float>(c.nopeHeadDim + c.ropeHeadDim);
    c.softmaxScale = 1.0f / std::sqrt(qkHeadDim);
    int64_t origSeqLen = a.rope_scaling_original_max_position_embeddings();
    if (origSeqLen > 0 && static_cast<int64_t>(c.maxSeqLen) > origSeqLen) {
      float mscale =
          0.1f * a.rope_scaling_mscale() * std::log(a.rope_scaling_factor()) +
          1.0f;
      c.softmaxScale *= mscale * mscale;
    }

    // MoE: sigmoid + shared + group-limited (V3: 3 dense / 256 expert / 8 act /
    // 1 shared).
    c.nDenseLayers = static_cast<uint32_t>(a.first_k_dense_replace());
    c.nRoutedExperts = static_cast<uint32_t>(a.num_experts());
    c.nSharedExperts = static_cast<uint32_t>(a.n_shared_experts());
    c.nActExperts = static_cast<uint32_t>(a.num_experts_per_tok());
    c.moeIntermediateSize = static_cast<uint32_t>(a.moe_intermediate_size());
    c.scoringFunc = XMODEL_SCORING_FUNC_SIGMOID;
    c.normTopKProb = a.norm_topk_prob();
    c.routeScale = a.routed_scaling_factor();
    c.nExpertGroups = static_cast<uint32_t>(a.n_group());
    c.nLimitedGroups = static_cast<uint32_t>(a.topk_group());
    c.expertsWeightTrans = true;  // HF [out,in] -> [in,out]

    c.moeEpSize = static_cast<uint32_t>(p.ep_size());
    c.moeTPSize = p.ep_size() > 1
                      ? static_cast<uint32_t>(p.world_size() / p.ep_size())
                      : XliteTpSize(p);
    return c;
  }

  // GLM-5/5.1 (MLA + DSA indexer + sigmoid MoE + shared expert, default rope).
  // GLM5 ~= DeepSeek V3.2 + DSA + different MLA dims; no rope_scaling ->
  // YaRN/mscale skipped.
  static XModelConfig FromGlm5(const ModelContext& context) {
    XModelConfig c = FromDeepseekV3(context);
    const ModelArgs& a = context.get_model_args();

    // DSA: attnType=DSA triggers ForwardAttnIndexer -> topkIndices for sparse
    // MLA attention.
    c.attnType = XMODEL_ATTN_DSA;
    c.indexHeadDim = static_cast<uint32_t>(a.index_head_dim());
    c.indexNHeads = static_cast<uint32_t>(a.index_n_heads());
    c.indexTopK = static_cast<uint32_t>(a.index_topk());
    c.indexRopeInterleaved = a.indexer_rope_interleave();
    // csrc computes 1/sqrt(n*head_dim) internally; set ref value for
    // completeness.
    c.indexSoftmaxScale = 1.0f / std::sqrt(static_cast<float>(c.indexHeadDim));

    // DSA top-k sharing (GLM-5.2): shared layers indexer, reuse prev full
    // layer's topkIndices. Resolve per-layer via DsaTopkSharePlan; csrc
    // reads indexFullMask. Empty = all layers run indexer (GLM-5.1 behavior).
    xllm::layer::DsaTopkSharePlan sharePlan(a);
    c.indexFullMask.resize(c.nLayers);
    for (uint32_t i = 0; i < c.nLayers; ++i) {
      c.indexFullMask[i] = !sharePlan.decision_for(i).reuse_topk;
    }

    // W8A8 flags conditional on QuantArgs (IsW8A8; BF16 mis-set flips shared
    // expert layout).
    if (IsW8A8(context)) {
      c.quantAttnWeightTrans = true;
      c.quantAttnWeightNz = true;
      c.expertsWeightNZ = true;
    }
    return c;
  }

  // GLM-4 MoE (MHA + partial rotary + sigmoid MoE + shared expert, W8A8).
  // Builds on FromQwen3Moe (MHA + MoE baseline), overrides GLM4-specific
  // fields:
  //   ropeHeadDim = headDim * partial_rotary_factor (GLM4 partial rotary, not
  //   full rope) nSharedExperts = n_shared_experts (GLM4 has shared, Qwen3-MoE
  //   does not) scoringFunc = SIGMOID + routeScale + gateCaptured=false
  //   (sigmoid routing, gate at runtime)
  // W8A8: quantAttnWeightTrans=true; NO_QUANT path (o_proj/down_proj BF16)
  // unaffected.
  static XModelConfig FromGlm4Moe(const ModelContext& context) {
    XModelConfig c = FromQwen3Moe(context);
    const ModelArgs& a = context.get_model_args();

    // partial rotary: ropeHeadDim = headDim * partial_rotary_factor (front dims
    // get RoPE).
    if (a.partial_rotary_factor() > 0.0f) {
      c.ropeHeadDim = static_cast<uint32_t>(static_cast<float>(c.headDim) *
                                            a.partial_rotary_factor());
    }

    // GLM4 has shared expert (FromQwen3Moe sets nSharedExperts=0).
    c.nSharedExperts = static_cast<uint32_t>(a.n_shared_experts());

    // sigmoid routing (Qwen3-MoE is softmax). gateCaptured=false: gate computed
    // at runtime.
    c.scoringFunc = XMODEL_SCORING_FUNC_SIGMOID;
    c.routeScale = a.routed_scaling_factor();
    c.gateCaptured = false;
    c.nExpertGroups = static_cast<uint32_t>(a.n_group());
    c.nLimitedGroups = static_cast<uint32_t>(a.topk_group());

    // W8A8 flags conditional on QuantArgs (IsW8A8; hardcoded true leaks to BF16
    // GLM-4.7).
    if (IsW8A8(context)) {
      c.quantAttnWeightTrans = true;
      c.quantAttnWeightNz = true;
      c.expertsWeightNZ = true;
    }
    // expertsWeightTrans already true from FromQwen3Moe (routed expert
    // gate_up/down transpose).
    return c;
  }
};

}  // namespace xllm::xlite