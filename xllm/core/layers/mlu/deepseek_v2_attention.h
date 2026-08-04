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

#include <torch/torch.h>

#include <cstdint>
#include <memory>
#include <optional>

#include "attention.h"
#include "core/layers/mlu/dsa_topk_relay.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_args.h"
#include "framework/parallel_state/parallel_args.h"
#include "framework/parallel_state/process_group.h"
#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"
#include "layers/common/linear.h"
#include "layers/common/rms_norm.h"
#include "layers/common/rotary_embedding.h"
#include "layers/mlu/dcp_decode_context.h"
#include "layers/mlu/deepseek_v32_cp_context.h"
#include "layers/mlu/indexer.h"
#include "platform/stream.h"

namespace xllm {
namespace layer {

class DeepseekV2AttentionImpl : public torch::nn::Module {
 public:
  enum class PostAttnLayout {
    kTpShard,
    kReplicated,
    kPackedLocal,
  };

  struct ForwardResult {
    torch::Tensor output;
    PostAttnLayout layout = PostAttnLayout::kTpShard;
  };

  DeepseekV2AttentionImpl() = default;
  DeepseekV2AttentionImpl(const ModelArgs& args,
                          const QuantArgs& quant_args,
                          const ParallelArgs& parallel_args,
                          const torch::TensorOptions& options,
                          const OptimizationConfig& optimization_config,
                          bool enable_indexer = true);

  ForwardResult forward(const torch::Tensor& positions,
                        const torch::Tensor& hidden_states,
                        const AttentionMetadata& attn_metadata,
                        KVCache& kv_cache,
                        const v32_cp::DeepseekV32CPContext* sp_ctx = nullptr,
                        DsaTopkTransfer* topk_transfer = nullptr);

  bool use_replicated_attn_weights() const {
    return use_full_replicated_attention_weights_;
  }

  void load_state_dict(const StateDict& state_dict);

 private:
  struct HeadInfo {
    int64_t attn = 1;
    int64_t proj = 1;

    int64_t proj_width(int64_t dim) const { return proj * dim; }
  };

  struct QueryPrep {
    torch::Tensor q;
    torch::Tensor q_norm;
  };

  struct MlaInputs {
    torch::Tensor q_norm;
    torch::Tensor q_input;
    torch::Tensor k_input;
    torch::Tensor v_input;
  };

  torch::Tensor forward_normal_tp(const torch::Tensor& positions,
                                  const torch::Tensor& hidden_states,
                                  const AttentionMetadata& attn_metadata,
                                  KVCache& kv_cache,
                                  bool is_prefill_or_chunked_prefill,
                                  DsaTopkTransfer* topk_transfer);

  torch::Tensor forward_dcp(const torch::Tensor& positions,
                            const torch::Tensor& hidden_states,
                            const AttentionMetadata& attn_metadata,
                            KVCache& kv_cache,
                            bool is_prefill_or_chunked_prefill,
                            DsaTopkTransfer* topk_transfer);

  // ===== sequence parallel related =====
  torch::Tensor forward_sp(const torch::Tensor& positions,
                           const torch::Tensor& hidden_states,
                           const AttentionMetadata& attn_metadata,
                           const v32_cp::DeepseekV32CPContext& sp_ctx,
                           KVCache& kv_cache,
                           bool is_prefill_or_chunked_prefill,
                           DsaTopkTransfer* topk_transfer);
  QueryPrep prep_query(const torch::Tensor& hidden_states,
                       const HeadInfo& heads);
  void fill_q_input(torch::Tensor& q_input,
                    const torch::Tensor& q,
                    const torch::Tensor& positions,
                    const AttentionMetadata& attn_metadata,
                    bool use_prompt_rope);
  MlaInputs build_sp_mla_inputs(const torch::Tensor& hidden_states,
                                const torch::Tensor& positions,
                                const QueryPrep& query_prep,
                                const v32_cp::DeepseekV32CPContext& sp_ctx);
  v32_cp::PaddedGatherHandle sp_mla_comm(
      const torch::Tensor& k_input,
      const v32_cp::DeepseekV32CPContext& sp_ctx) const;
  void finish_sp_k_gather(MlaInputs& mla_inputs,
                          const v32_cp::PaddedGatherHandle& k_handle,
                          const v32_cp::DeepseekV32CPContext& sp_ctx) const;
  void decode_kv_pre_base(torch::Tensor& latent_cache,
                          const torch::Tensor& positions,
                          const AttentionMetadata& attn_metadata,
                          bool use_prompt_rope);
  void decode_qkv_pre_fused(torch::Tensor& q,
                            torch::Tensor& q_norm,
                            torch::Tensor& q_input,
                            torch::Tensor& latent_cache,
                            torch::Tensor& kv_cache,
                            std::optional<torch::Tensor> k_cache_scale,
                            const torch::Tensor& positions,
                            const AttentionMetadata& attn_metadata,
                            bool use_prompt_rope);

  void prepare_mla_inputs(torch::Tensor& q,
                          torch::Tensor& q_norm,
                          torch::Tensor& q_input,
                          torch::Tensor& latent_cache,
                          const torch::Tensor& hidden_states,
                          torch::Tensor& k_cache,
                          std::optional<torch::Tensor> k_cache_scale,
                          const torch::Tensor& positions,
                          const AttentionMetadata& attn_metadata,
                          bool enable_fused_qkv,
                          bool use_prompt_rope);

  void update_mla_k_cache(
      const torch::Tensor& k_input,
      const AttentionMetadata& attn_metadata,
      KVCache& kv_cache,
      std::optional<torch::Tensor> k_cache_scale,
      bool is_prefill_phase,
      const std::optional<torch::Tensor>& slot_mapping = std::nullopt) const;

  std::optional<DsaTopkState> resolve_dsa_topk_state(
      const torch::Tensor& positions,
      const torch::Tensor& hidden_states,
      const torch::Tensor& q_norm,
      const AttentionMetadata& attn_metadata,
      KVCache& kv_cache,
      bool is_prefill_phase,
      const DsaTopkState* external_topk = nullptr);

  AttentionMetadata build_mla_attention_metadata(
      const AttentionMetadata& attn_metadata,
      const std::optional<DsaTopkState>& topk_state) const;

  DcpAttentionResult run_dcp_paged_attention(
      const torch::Tensor& q_input,
      const DsaTopkState& global_topk,
      KVCache& kv_cache,
      const AttentionMetadata& base_metadata);

  DcpAttentionResult run_dcp_chunked_prefill_attention(
      const torch::Tensor& q_input,
      const DsaTopkState& global_topk,
      KVCache& kv_cache,
      const AttentionMetadata& base_metadata);

  torch::Tensor project_output(const torch::Tensor& attn_output,
                               const HeadInfo& heads);

  bool can_use_sp(const DsaTopkTransfer* topk_transfer) const {
    const bool reuses_topk =
        topk_transfer != nullptr && topk_transfer->input() != nullptr;
    return use_replicated_attn_weights() && (has_indexer_ || reuses_topk);
  }

  const HeadInfo& tp_heads() const { return tp_heads_; }
  const HeadInfo& full_heads() const { return full_heads_; }
  const HeadInfo& active_heads() const {
    return use_replicated_attn_weights() ? full_heads_ : tp_heads_;
  }

 private:
  bool use_full_replicated_attention_weights_ = false;
  bool use_fused_mla_qkv_ = false;
  bool enable_lighting_indexer_ = false;
  bool dcp_spans_tp_ = false;
  bool has_indexer_ = false;
  bool has_trans_ = false;
  bool interleaved_ = false;
  double eps_;
  int64_t qk_head_dim_;
  int64_t v_head_dim_;
  int64_t q_lora_rank_;
  int64_t kv_lora_rank_;
  int64_t qk_nope_head_dim_;
  int64_t qk_rope_head_dim_;
  int64_t index_topk_;
  int32_t kv_split_size_ = 1;
  int32_t kv_split_rank_ = 0;
  int32_t tp_rank_ = 0;
  int32_t block_size_ = 1;
  bool enable_mla_cache_sharding_ = false;
  ProcessGroup* tp_group_ = nullptr;
  std::unique_ptr<DcpDecodeContext> dcp_decode_context_;
  HeadInfo tp_heads_;
  HeadInfo full_heads_;
  torch::Tensor w_kc_;
  torch::Tensor w_vc_;
  torch::Tensor weight_c_;

  ReplicatedLinear q_a_proj_{nullptr};
  ColumnParallelLinear q_b_proj_{nullptr};
  ColumnParallelLinear q_proj_{nullptr};
  RMSNorm q_a_layernorm_{nullptr};

  ReplicatedLinear kv_a_proj_with_mqa_{nullptr};
  RMSNorm kv_a_layernorm_{nullptr};

  ColumnParallelLinear kv_b_proj_{nullptr};
  RowParallelLinear o_proj_{nullptr};

  Attention attn_{nullptr};
  std::shared_ptr<RotaryEmbeddingBase> rotary_emb_;
  std::shared_ptr<RotaryEmbeddingBase> indexer_rotary_emb_;
  Indexer indexer_{nullptr};
  std::unique_ptr<Stream> sp_comm_stream_;
  Attention dcp_full_head_attn_{nullptr};
};
TORCH_MODULE(DeepseekV2Attention);

}  // namespace layer
}  // namespace xllm
