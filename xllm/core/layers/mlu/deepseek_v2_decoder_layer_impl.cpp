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

#include "deepseek_v2_decoder_layer_impl.h"

#include <utility>

#include "layers/common/dsa_topk_share_plan.h"

namespace xllm {
namespace layer {

namespace {

bool same_sp_topology(ProcessGroup* pg, ProcessGroup* sp_pg) {
  return pg != nullptr && sp_pg != nullptr &&
         pg->world_size() == sp_pg->world_size() && pg->rank() == sp_pg->rank();
}

bool is_sp_alias_pg(ProcessGroup* pg, const ParallelArgs& parallel_args) {
  return pg == parallel_args.process_group_ ||
         pg == parallel_args.moe_ep_group_;
}

bool is_mtp_layer(const ModelArgs& model_args) {
  return model_args.model_type().ends_with("_mtp");
}

DsaTopkShareDecision resolve_topk_share_decision(
    const ModelContext& context,
    int32_t layer_id,
    const DsaTopkSharePlan& topk_share_plan) {
  if (is_mtp_layer(context.get_model_args())) {
    return DsaTopkShareDecision();
  }
  return topk_share_plan.decision_for(layer_id);
}
}  // namespace

DeepseekV2DecoderLayerImpl::DeepseekV2DecoderLayerImpl(
    const ModelContext& context,
    int32_t layer_id)
    : DeepseekV2DecoderLayerImpl(context, layer_id, DsaTopkShareDecision()) {}

DeepseekV2DecoderLayerImpl::DeepseekV2DecoderLayerImpl(
    const ModelContext& context,
    int32_t layer_id,
    const DsaTopkSharePlan& topk_share_plan)
    : DeepseekV2DecoderLayerImpl(
          context,
          layer_id,
          resolve_topk_share_decision(context, layer_id, topk_share_plan)) {}

DeepseekV2DecoderLayerImpl::DeepseekV2DecoderLayerImpl(
    const ModelContext& context,
    int32_t layer_id,
    const DsaTopkShareDecision& topk_share_decision)
    : parallel_args_(context.get_parallel_args()),
      layer_id_(layer_id),
      topk_share_decision_(topk_share_decision) {
  const auto& model_args = context.get_model_args();
  const auto& quant_args = context.get_quant_args();
  const auto& options = context.get_tensor_options();
  const bool mtp_layer = is_mtp_layer(model_args);
  is_moe_layer_ = mtp_layer ? model_args.mtp_mlp_type() != "dense"
                            : layer_id >= model_args.first_k_dense_replace();

  mtp_topk_reuse_ = is_mtp_dsa_topk_reuse_enabled(model_args);

  // DeepSeek MoE only support ep == world_size when expert parallel is on
  if (parallel_args_.ep_size() > 1) {
    CHECK(parallel_args_.ep_size() == parallel_args_.world_size())
        << "DeepSeek MoE only supports ep_size equal to world size";
  }

  // Initialize attention layers
  OptimizationConfig optimization_config = context.get_optimization_config();
  attention_ = register_module(
      "self_attn",
      DeepseekV2Attention(model_args,
                          quant_args,
                          parallel_args_,
                          options,
                          optimization_config,
                          /*enable_indexer=*/
                          mtp_layer || !topk_share_decision_.reuse_topk));

  // Initialize norm layers
  input_norm_ = register_module(
      "input_layernorm",
      RMSNorm(model_args.hidden_size(), model_args.rms_norm_eps(), options));

  post_norm_ = register_module(
      "post_attention_layernorm",
      RMSNorm(model_args.hidden_size(), model_args.rms_norm_eps(), options));

  // Initialize mlp
  if (is_moe_layer_) {
    sparse_moe_ =
        register_module("mlp",
                        DeepseekV2SparseMoEBlock(
                            model_args, quant_args, parallel_args_, options));
  } else {
    mlp_ = register_module("mlp",
                           DenseMLP(model_args.hidden_size(),
                                    model_args.intermediate_size(),
                                    /*is_gated=*/true,
                                    /*has_bias=*/false,
                                    model_args.hidden_act(),
                                    /*enable_result_reduction=*/false,
                                    quant_args,
                                    parallel_args_.tp_group_,
                                    options));
  }
}

void DeepseekV2DecoderLayerImpl::load_state_dict(const StateDict& state_dict) {
  attention_->load_state_dict(state_dict.get_dict_with_prefix("self_attn."));
  input_norm_->load_state_dict(
      state_dict.get_dict_with_prefix("input_layernorm."));
  post_norm_->load_state_dict(
      state_dict.get_dict_with_prefix("post_attention_layernorm."));
  if (sparse_moe_) {
    sparse_moe_->load_state_dict(state_dict.get_dict_with_prefix("mlp."));
  } else {
    mlp_->load_state_dict(state_dict.get_dict_with_prefix("mlp."));
  }
}

void DeepseekV2DecoderLayerImpl::verify_loaded_weights() const {
  if (sparse_moe_) {
    sparse_moe_->verify_loaded_weights();
  }
}

DeepseekV2DecoderLayerImpl::PostAttnCarrier
DeepseekV2DecoderLayerImpl::build_post_attn_local(
    torch::Tensor x,
    const torch::Tensor& residual) {
  CHECK(context_parallel_context_ != nullptr)
      << "SP carrier requires CP context";
  auto [ffn_in, skip_local] = post_norm_->forward(x, residual);

  PostAttnCarrier carrier;
  carrier.ffn_in = ffn_in;
  carrier.skip_local = skip_local.value();
  carrier.mode = PostAttnMode::kPackedLocal;
  return carrier;
}

DeepseekV2DecoderLayerImpl::PostAttnCarrier
DeepseekV2DecoderLayerImpl::build_post_attn_carrier(
    torch::Tensor x,
    const torch::Tensor& residual,
    DeepseekV2AttentionImpl::PostAttnLayout attn_layout) {
  PostAttnCarrier carrier;
  if (attn_layout == DeepseekV2AttentionImpl::PostAttnLayout::kPackedLocal) {
    carrier = build_post_attn_local(x, residual);
    carrier.ffn_in = v32_cp::all_gather_across_ranks(
        carrier.ffn_in, *context_parallel_context_);
    return carrier;
  }

  if (attn_layout == DeepseekV2AttentionImpl::PostAttnLayout::kTpShard) {
    x = xllm::parallel_state::reduce(x, parallel_args_.tp_group_);
  }
  x = x + residual;

  carrier.ffn_in = x;
  carrier.skip_local = x;
  return carrier;
}

DeepseekV2DecoderLayerImpl::MoeInputPrepResult
DeepseekV2DecoderLayerImpl::prepare_moe_inputs(
    torch::Tensor x,
    const torch::Tensor& residual,
    const ModelInputParams& input_params,
    DeepseekV2AttentionImpl::PostAttnLayout attn_layout) {
  MoeInputPrepResult result;
  if (!sparse_moe_) {
    result.carrier =
        build_post_attn_carrier(std::move(x), residual, attn_layout);
    result.ffn_in = result.carrier->ffn_in;
    return result;
  }

  result.exec_cfg = sparse_moe_->plan_exec(input_params);
  result.use_sp_moe_overlap =
      attn_layout == DeepseekV2AttentionImpl::PostAttnLayout::kPackedLocal &&
      !result.exec_cfg->enable_all2all && !result.exec_cfg->need_dp_gather &&
      sparse_moe_->has_shared();
  if (result.use_sp_moe_overlap) {
    result.carrier = build_post_attn_local(std::move(x), residual);
    result.ffn_in = result.carrier->ffn_in;
    return result;
  }

  if (attn_layout == DeepseekV2AttentionImpl::PostAttnLayout::kPackedLocal) {
    result.carrier =
        build_post_attn_carrier(std::move(x), residual, attn_layout);
    result.ffn_in = result.carrier->ffn_in;
    result.moe_prep = DeepseekV2SparseMoEBlockImpl::PrepOut{
        .ffn_in = result.carrier->ffn_in,
        .skip_local = result.carrier->skip_local,
    };
    return result;
  }

  if (result.exec_cfg->enable_all2all || result.exec_cfg->need_dp_gather) {
    result.moe_prep = sparse_moe_->prep_in(std::move(x),
                                           residual,
                                           input_params,
                                           result.exec_cfg.value(),
                                           attn_layout);
    result.ffn_in = result.moe_prep->ffn_in;
    return result;
  }

  result.carrier = build_post_attn_carrier(std::move(x), residual, attn_layout);
  result.ffn_in = result.carrier->ffn_in;
  result.moe_prep = DeepseekV2SparseMoEBlockImpl::PrepOut{
      .ffn_in = result.carrier->ffn_in,
      .skip_local = result.carrier->skip_local,
  };
  return result;
}

bool DeepseekV2DecoderLayerImpl::can_keep_local_output(
    const PostAttnCarrier& carrier,
    ProcessGroup* pg) const {
  const bool can_use_sp_fast = carrier.mode == PostAttnMode::kPackedLocal &&
                               context_parallel_context_ != nullptr &&
                               context_parallel_context_->comm_plan.ffn_can_rs;
  if (!can_use_sp_fast) {
    return false;
  }

  ProcessGroup* const sp_pg = context_parallel_context_->process_group;
  if (!pg || pg->world_size() <= 1 || pg == sp_pg) {
    return true;
  }

  if (parallel_args_.dp_size() != 1 || !is_sp_alias_pg(pg, parallel_args_)) {
    return false;
  }

  return same_sp_topology(pg, sp_pg);
}

torch::Tensor DeepseekV2DecoderLayerImpl::comm_out(
    torch::Tensor x,
    const PostAttnCarrier& carrier,
    ProcessGroup* pg) const {
  if (!can_keep_local_output(carrier, pg)) {
    return reduce_out(x, pg);
  }

  if (pg && pg->world_size() > 1) {
    return parallel_state::reduce_scatter(x, pg);
  }

  CHECK(context_parallel_context_ != nullptr)
      << "SP fast path requires CP context";
  return v32_cp::slice_local_packed(x, *context_parallel_context_);
}

torch::Tensor DeepseekV2DecoderLayerImpl::restore_ffn_output(
    torch::Tensor x,
    const PostAttnCarrier& carrier) {
  if (carrier.mode == PostAttnMode::kPackedLocal) {
    CHECK(context_parallel_context_ != nullptr)
        << "packed restore requires CP context";
    x = v32_cp::slice_local_packed(x, *context_parallel_context_);
    return x + carrier.skip_local;
  }
  return x + carrier.skip_local;
}

torch::Tensor DeepseekV2DecoderLayerImpl::reduce_out(torch::Tensor x,
                                                     ProcessGroup* pg) const {
  if (!pg || pg->world_size() <= 1) {
    return x;
  }
  return parallel_state::reduce(x, pg);
}

torch::Tensor DeepseekV2DecoderLayerImpl::forward(
    torch::Tensor& x,
    std::optional<torch::Tensor>& residual,
    torch::Tensor& positions,
    const AttentionMetadata& attn_metadata,
    KVCache& kv_cache,
    const ModelInputParams& input_params,
    const std::optional<torch::Tensor>& input_ids,
    DsaTopkRelay* topk_relay) {
  return forward_impl(x,
                      residual,
                      positions,
                      attn_metadata,
                      kv_cache,
                      input_params,
                      input_ids,
                      topk_relay,
                      /*mtp_topk_input=*/nullptr,
                      /*mtp_topk_output=*/nullptr);
}

torch::Tensor DeepseekV2DecoderLayerImpl::forward_mtp(
    torch::Tensor& x,
    std::optional<torch::Tensor>& residual,
    torch::Tensor& positions,
    const AttentionMetadata& attn_metadata,
    KVCache& kv_cache,
    const ModelInputParams& input_params,
    const std::optional<torch::Tensor>& input_ids,
    const std::optional<DsaTopkState>& topk_input,
    std::optional<DsaTopkState>& topk_output) {
  return forward_impl(x,
                      residual,
                      positions,
                      attn_metadata,
                      kv_cache,
                      input_params,
                      input_ids,
                      /*topk_relay=*/nullptr,
                      &topk_input,
                      &topk_output);
}

torch::Tensor DeepseekV2DecoderLayerImpl::forward_impl(
    torch::Tensor& x,
    std::optional<torch::Tensor>& residual,
    torch::Tensor& positions,
    const AttentionMetadata& attn_metadata,
    KVCache& kv_cache,
    const ModelInputParams& input_params,
    const std::optional<torch::Tensor>&,
    DsaTopkRelay* topk_relay,
    const std::optional<DsaTopkState>* mtp_topk_input,
    std::optional<DsaTopkState>* mtp_topk_output) {
  // Pre-attention norm
  residual = x;
  x = std::get<0>(input_norm_->forward(x));

  std::optional<DsaTopkTransfer> cross_layer_transfer;
  if (topk_relay != nullptr && !attn_metadata.is_dummy) {
    cross_layer_transfer = topk_relay->prepare_layer(topk_share_decision_);
  }

  // MTP cross-step top-k reuse bridges typed per-layer state into the same
  // atomic transfer used by attention. It only engages through forward_mtp,
  // when no cross-layer transfer is active and this is a real forward.
  std::optional<DsaTopkTransfer> mtp_transfer;
  DsaTopkTransfer* effective_transfer = cross_layer_transfer.has_value()
                                            ? &cross_layer_transfer.value()
                                            : nullptr;
  const bool use_mtp_topk_bridge =
      mtp_topk_reuse_ && effective_transfer == nullptr &&
      mtp_topk_input != nullptr && mtp_topk_output != nullptr &&
      !attn_metadata.is_dummy;
  if (use_mtp_topk_bridge) {
    mtp_transfer =
        DsaTopkTransfer::prepare_mtp_step(*mtp_topk_input, x.device());
    effective_transfer = &mtp_transfer.value();
  }

  // Attention
  x = attention_->forward(positions,
                          x,
                          attn_metadata,
                          kv_cache,
                          context_parallel_context_,
                          effective_transfer);

  if (cross_layer_transfer.has_value()) {
    topk_relay->finish_layer(topk_share_decision_,
                             cross_layer_transfer.value());
  }
  if (use_mtp_topk_bridge) {
    *mtp_topk_output = mtp_transfer->mtp_output_state();
  }
  const bool use_sp_output =
      context_parallel_context_ != nullptr && attention_->can_use_sp();
  const auto attn_layout = attention_->post_attn_layout(use_sp_output);
  auto prep = prepare_moe_inputs(
      std::move(x), residual.value(), input_params, attn_layout);
  auto& carrier = prep.carrier;
  auto& moe_prep = prep.moe_prep;
  auto& exec_cfg = prep.exec_cfg;
  x = prep.ffn_in;

  if (!carrier.has_value() || carrier->mode != PostAttnMode::kPackedLocal) {
    x = std::get<0>(post_norm_->forward(x));
    if (moe_prep.has_value()) {
      moe_prep->ffn_in = x;
    }
    if (carrier.has_value()) {
      carrier->ffn_in = x;
    }
  }
  if (moe_prep.has_value()) {
    x = sparse_moe_->gather_in(*moe_prep, input_params);
  }

  // MLP forward
  bool keep_local_output = false;
  if (sparse_moe_) {
    auto can_keep_local = [&](ProcessGroup* pg) {
      return carrier.has_value() && can_keep_local_output(*carrier, pg);
    };
    auto comm = [&](torch::Tensor y, ProcessGroup* pg) {
      CHECK(carrier.has_value()) << "local comm path requires decoder carrier";
      return comm_out(std::move(y), *carrier, pg);
    };
    auto moe_comm_fns = DeepseekV2SparseMoEBlockImpl::CommFns{
        .can_keep_local = std::move(can_keep_local),
        .comm = std::move(comm),
        .reduce =
            [&](torch::Tensor y, ProcessGroup* pg) {
              return reduce_out(std::move(y), pg);
            },
        .launch_reduce =
            [&](torch::Tensor y, ProcessGroup* pg) {
              return parallel_state::launch_reduce(std::move(y), pg);
            },
        .finish_reduce =
            [&](parallel_state::ReduceAsyncCtx ctx) {
              return parallel_state::finish_reduce(std::move(ctx));
            },
    };
    auto moe_result =
        prep.use_sp_moe_overlap
            ? sparse_moe_->forward_sp(
                  x, *context_parallel_context_, moe_comm_fns)
            : sparse_moe_->forward(x, exec_cfg->enable_all2all, moe_comm_fns);
    x = std::move(moe_result.output);
    keep_local_output = moe_result.keep_local_output;
  } else {
    keep_local_output =
        can_keep_local_output(*carrier, parallel_args_.tp_group_);
    x = mlp_(std::move(x));
    x = keep_local_output ? comm_out(x, *carrier, parallel_args_.tp_group_)
                          : reduce_out(x, parallel_args_.tp_group_);
  }
  if (keep_local_output) {
    CHECK(moe_prep.has_value() || carrier.has_value())
        << "skip local add requires prepared output state";
    const auto& skip_src =
        moe_prep.has_value() ? moe_prep->skip_local : carrier->skip_local;
    x = x + skip_src;
  } else if (moe_prep.has_value() &&
             (moe_prep->need_dp_gather || moe_prep->need_tp_pad)) {
    x = sparse_moe_->merge_out(x, *moe_prep, input_params);
  } else {
    x = restore_ffn_output(x, *carrier);
  }

  residual = std::nullopt;
  return x;
}

}  // namespace layer
}  // namespace xllm
