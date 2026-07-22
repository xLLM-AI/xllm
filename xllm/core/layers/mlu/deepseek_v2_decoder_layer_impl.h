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

#include <optional>

#include "attention.h"
#include "core/layers/mlu/dsa_topk_relay.h"
#include "deepseek_v2_attention.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_args.h"
#include "framework/model/model_input_params.h"
#include "framework/model_context.h"
#include "framework/parallel_state/parallel_args.h"
#include "framework/parallel_state/parallel_state.h"
#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"
#include "layers/common/dense_mlp.h"
#include "layers/common/dp_utils.h"
#include "layers/common/rms_norm.h"
#include "layers/mlu/deepseek_v2_sparse_moe_block.h"
#include "layers/mlu/deepseek_v32_cp_context.h"

namespace xllm {
namespace layer {

class DeepseekV2DecoderLayerTestPeer;

class DeepseekV2DecoderLayerImpl : public torch::nn::Module {
 public:
  explicit DeepseekV2DecoderLayerImpl(const ModelContext& context,
                                      int32_t layer_id);

  DeepseekV2DecoderLayerImpl(const ModelContext& context,
                             int32_t layer_id,
                             const DsaTopkSharePlan& topk_share_plan);

  ~DeepseekV2DecoderLayerImpl() override = default;

  void load_state_dict(const StateDict& state_dict);
  void verify_loaded_weights() const;

  void set_context_parallel_context(
      const v32_cp::DeepseekV32CPContext* cp_ctx) {
    context_parallel_context_ = cp_ctx;
  }

  torch::Tensor forward(
      torch::Tensor& x,
      std::optional<torch::Tensor>& residual,
      torch::Tensor& positions,
      const AttentionMetadata& attn_metadata,
      KVCache& kv_cache,
      const ModelInputParams& input_params,
      const std::optional<torch::Tensor>& input_ids = std::nullopt,
      DsaTopkRelay* topk_relay = nullptr);

  torch::Tensor forward_mtp(torch::Tensor& x,
                            std::optional<torch::Tensor>& residual,
                            torch::Tensor& positions,
                            const AttentionMetadata& attn_metadata,
                            KVCache& kv_cache,
                            const ModelInputParams& input_params,
                            const std::optional<torch::Tensor>& input_ids,
                            const std::optional<DsaTopkState>& topk_input,
                            std::optional<DsaTopkState>& topk_output);

 private:
  DeepseekV2DecoderLayerImpl(const ModelContext& context,
                             int32_t layer_id,
                             const DsaTopkShareDecision& topk_share_decision);

  enum class PostAttnMode {
    kReplicated,
    kPackedLocal,
  };

  struct PostAttnCarrier {
    torch::Tensor ffn_in;
    torch::Tensor skip_local;
    PaddingInfo pad_info;
    PostAttnMode mode = PostAttnMode::kReplicated;
  };

  struct MoeInputPrepResult {
    torch::Tensor ffn_in;
    std::optional<PostAttnCarrier> carrier;
    std::optional<DeepseekV2SparseMoEBlockImpl::PrepOut> moe_prep;
    std::optional<DeepseekV2SparseMoEBlockImpl::ExecCfg> exec_cfg;
    bool use_sp_moe_overlap = false;
  };

  PostAttnCarrier build_post_attn_carrier(
      torch::Tensor x,
      const torch::Tensor& residual,
      DeepseekV2AttentionImpl::PostAttnLayout attn_layout);
  PostAttnCarrier build_post_attn_local(torch::Tensor x,
                                        const torch::Tensor& residual);
  MoeInputPrepResult prepare_moe_inputs(
      torch::Tensor x,
      const torch::Tensor& residual,
      const ModelInputParams& input_params,
      DeepseekV2AttentionImpl::PostAttnLayout attn_layout);

  bool can_keep_local_output(const PostAttnCarrier& carrier,
                             ProcessGroup* pg) const;
  torch::Tensor comm_out(torch::Tensor x,
                         const PostAttnCarrier& carrier,
                         ProcessGroup* pg) const;
  torch::Tensor restore_ffn_output(torch::Tensor x,
                                   const PostAttnCarrier& carrier);
  torch::Tensor reduce_out(torch::Tensor x, ProcessGroup* pg) const;
  torch::Tensor forward_impl(torch::Tensor& x,
                             std::optional<torch::Tensor>& residual,
                             torch::Tensor& positions,
                             const AttentionMetadata& attn_metadata,
                             KVCache& kv_cache,
                             const ModelInputParams& input_params,
                             const std::optional<torch::Tensor>& input_ids,
                             DsaTopkRelay* topk_relay,
                             const std::optional<DsaTopkState>* mtp_topk_input,
                             std::optional<DsaTopkState>* mtp_topk_output);

  friend class DeepseekV2DecoderLayerTestPeer;

  // parallel args
  ParallelArgs parallel_args_;
  int32_t layer_id_;
  bool is_moe_layer_;
  DsaTopkShareDecision topk_share_decision_;
  // MTP draft layer reuses the first draft step's top-k across the remaining
  // steps (cross-step sharing), isolated from the cross-layer DsaTopkRelay
  // relay used by the main model.
  bool mtp_topk_reuse_ = false;

  DeepseekV2Attention attention_{nullptr};
  DenseMLP mlp_{nullptr};
  DeepseekV2SparseMoEBlock sparse_moe_{nullptr};
  RMSNorm input_norm_{nullptr};
  RMSNorm post_norm_{nullptr};
  const v32_cp::DeepseekV32CPContext* context_parallel_context_ = nullptr;
};

TORCH_MODULE(DeepseekV2DecoderLayer);
}  // namespace layer
}  // namespace xllm
