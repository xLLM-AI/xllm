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

#include <glog/logging.h>

#include <tuple>
#include <utility>
#include <vector>

#include "core/layers/mlu/dcp_batch_metadata.h"
#include "core/layers/mlu/deepseek_v2_attention.h"
#include "framework/parallel_state/parallel_state.h"
#include "kernels/ops_api.h"
#include "util/utils.h"

namespace xllm {
namespace layer {

torch::Tensor DeepseekV2AttentionImpl::run_dcp_chunked_prefill_attention(
    const torch::Tensor& q_input,
    const DsaTopkState& local_topk,
    std::optional<parallel_state::GatherAsyncCtx>& query_gather,
    KVCache& kv_cache,
    const AttentionMetadata& local_cache_metadata,
    const torch::Tensor& query_slot_mapping) {
  CHECK(dcp_decode_context_ != nullptr);
  AttentionMetadata local_metadata =
      build_mla_attention_metadata(local_cache_metadata, local_topk);
  torch::Tensor query = q_input;
  const bool gathers_tp_heads = dcp_spans_tp_;
  if (gathers_tp_heads) {
    CHECK(query_gather.has_value())
        << "DCP chunked prefill requires the early-launched TP query gather";
    torch::Tensor gathered_query;
    {
      auto ctx = std::move(query_gather.value());
      if (ctx.work.defined()) {
        ctx.work->wait();
      }
      gathered_query =
          torch::cat(ctx.stacked.unbind(0), /*dim=*/1).contiguous();
    }
    query = std::move(gathered_query);
    query_gather.reset();
  }
  torch::Tensor unused_key;
  torch::Tensor unused_value;
  Attention& local_attention = gathers_tp_heads ? dcp_full_head_attn_ : attn_;
  auto [local_output, local_lse] = local_attention(local_metadata,
                                                   query,
                                                   unused_key,
                                                   unused_value,
                                                   kv_cache,
                                                   /*return_lse=*/true);
  // Global slot addresses in query row order distinguish padding from valid
  // queries whose KV slots belong to another DCP rank.
  const HeadInfo& heads = tp_heads_;
  const int64_t attention_heads = dcp_spans_tp_ ? full_heads_.attn : heads.attn;
  // Head reduce-scatter is only valid when the DCP group is exactly the TP
  // group (no PCP replication); otherwise heads are merged with an all-reduce
  // and redistributed by tp_rank_ below.
  const bool head_sharded =
      dcp_spans_tp_ &&
      tp_group_->world_size() == dcp_decode_context_->world_size();
  torch::Tensor merged_output = dcp_decode_context_->merge(
      local_output.view({q_input.size(0), 1, attention_heads, kv_lora_rank_}),
      local_lse.value(),
      query_slot_mapping,
      /*head_sharded=*/head_sharded);
  if (dcp_spans_tp_ && !head_sharded) {
    const int64_t first_head = tp_rank_ * heads.attn;
    const int64_t last_head = first_head + heads.attn;
    merged_output =
        merged_output.slice(/*dim=*/2, first_head, last_head).contiguous();
  }
  return merged_output;
}

torch::Tensor DeepseekV2AttentionImpl::forward_dcp(
    const torch::Tensor& positions,
    const torch::Tensor& hidden_states,
    const AttentionMetadata& attn_metadata,
    KVCache& kv_cache,
    bool is_prefill_or_chunked_prefill,
    DsaTopkTransfer* topk_transfer) {
  CHECK(enable_mla_cache_sharding_);
  CHECK(!attn_metadata.is_dummy);
  CHECK(dcp_decode_context_ != nullptr);

  if (attn_metadata.is_prefill && hidden_states.size(0) == 0) {
    CHECK_EQ(attn_metadata.slot_mapping.numel(), 0)
        << "Empty prefill cannot supply K slots";
    return torch::empty_like(hidden_states);
  }

  const HeadInfo& heads = tp_heads_;
  const bool enable_dcp_decode = !is_prefill_or_chunked_prefill;
  const bool enable_dcp_chunked_prefill = attn_metadata.is_chunked_prefill;
  const bool enable_dcp_paged_attention =
      enable_dcp_decode || enable_dcp_chunked_prefill;
  const bool enable_dcp_prefill_cache_write = attn_metadata.is_prefill;
  AttentionMetadata cache_metadata = attn_metadata;
  const std::shared_ptr<const KVShardBatchMetadata>& shard_metadata =
      attn_metadata.kv_shard_batch_metadata;
  if (shard_metadata != nullptr) {
    CHECK(shard_metadata->local_slot_mapping.defined())
        << "cache-shard batch metadata requires localized slot mapping";
    cache_metadata.slot_mapping = shard_metadata->local_slot_mapping;
  } else {
    cache_metadata.slot_mapping =
        dcp_decode_context_->localize_slots(attn_metadata.slot_mapping);
  }

  torch::Tensor q;
  torch::Tensor q_norm;
  torch::Tensor q_input = torch::empty(
      {hidden_states.size(0), heads.attn, kv_lora_rank_ + qk_rope_head_dim_},
      hidden_states.options());
  torch::Tensor latent_cache;
  torch::Tensor k_cache = kv_cache.get_k_cache();
  std::optional<torch::Tensor> k_cache_scale = kv_cache.get_k_cache_scale();
  const bool enable_fused_qkv =
      use_fused_mla_qkv_ && !is_prefill_or_chunked_prefill;
  const bool use_prompt_rope = attn_metadata.is_prefill;
  if (!enable_fused_qkv || q_lora_rank_ <= 0) {
    auto query_prep = prep_query(hidden_states);
    q = query_prep.q;
    q_norm = query_prep.q_norm;
    fill_q_input(q_input, q, positions, cache_metadata, use_prompt_rope);
  } else {
    CHECK_GT(qk_rope_head_dim_, 0)
        << "Fused MLA QKV does not implement the NoPE path.";

    // fused_mla_q: q_a_layernorm + q_b_proj + split + bmm + rotary_emb
    q = q_a_proj_(hidden_states);
    q_norm = torch::empty_like(q);
    if (q.dim() == 2) {
      q = q.unsqueeze(1);
    }
    q_input = q_input.view(
        {q.size(0), q.size(1), q_input.size(-2), q_input.size(-1)});
    kernel::FusedMlaQParams fused_mla_q_params;
    fused_mla_q_params.q = q;
    fused_mla_q_params.output = q_input;
    fused_mla_q_params.output_norm = q_norm.view(q.sizes());
    fused_mla_q_params.gamma = q_a_layernorm_->weight();
    fused_mla_q_params.smooth_quant_scale = q_b_proj_->smooth();
    fused_mla_q_params.weight_b = q_b_proj_->weight();
    fused_mla_q_params.weight_b_scale = q_b_proj_->per_channel_scale();
    fused_mla_q_params.weight_c = weight_c_;
    fused_mla_q_params.sin = rotary_emb_->get_sin_cache();
    fused_mla_q_params.cos = rotary_emb_->get_cos_cache();
    fused_mla_q_params.position_id = positions;
    fused_mla_q_params.quant_mode = "none";
    fused_mla_q_params.eps = eps_;
    fused_mla_q_params.interleaved = interleaved_;
    kernel::fused_mla_q(fused_mla_q_params);
  }
  q_input = q_input.view({q_input.size(0), -1});

  std::optional<parallel_state::GatherAsyncCtx> query_gather;
  if (enable_dcp_paged_attention && dcp_spans_tp_) {
    parallel_state::GatherAsyncCtx& gather_ctx = query_gather.emplace();
    gather_ctx.input = q_input;
    std::vector<int64_t> stacked_shape = q_input.sizes().vec();
    stacked_shape.insert(stacked_shape.begin(), tp_group_->world_size());
    gather_ctx.stacked = torch::empty(stacked_shape, q_input.options());
    gather_ctx.work =
        tp_group_->allgather_base_async(q_input, gather_ctx.stacked);
  }

  const DsaTopkState* external_topk =
      topk_transfer != nullptr ? topk_transfer->input() : nullptr;

  std::optional<DcpIndexerGatherAsyncCtx> candidate_gather;
  int64_t candidate_topk = 0;
  std::optional<DsaTopkState> topk_state;
  if (external_topk != nullptr) {
    topk_state = *external_topk;
  } else if (has_indexer_) {
    AttentionMetadata local_indexer_metadata = attn_metadata;
    local_indexer_metadata.slot_mapping = cache_metadata.slot_mapping;
    torch::Tensor index_cache = kv_cache.get_index_cache();
    const std::optional<torch::Tensor> index_cache_scale =
        kv_cache.get_indexer_cache_scale();
    DcpIndexerLocalCandidates candidates;
    if (enable_dcp_decode) {
      if (shard_metadata != nullptr &&
          shard_metadata->local_indexer_context_lens.defined()) {
        local_indexer_metadata.kv_seq_lens =
            shard_metadata->local_indexer_context_lens;
      } else {
        local_indexer_metadata.kv_seq_lens = localize_kv_shard_context_lens(
            attn_metadata.kv_seq_lens, dcp_decode_context_->layout());
      }
      candidates = indexer_->forward_dcp_local(hidden_states,
                                               q_norm,
                                               positions,
                                               index_cache,
                                               local_indexer_metadata,
                                               index_cache_scale);
    } else {
      const auto* shard = local_indexer_metadata.kv_shard_batch_metadata.get();
      const KVShardCausalSelectorMetadata causal_selector =
          shard != nullptr && shard->causal_selector.block_table.defined()
              ? shard->causal_selector
              : build_kv_shard_causal_selector_metadata(
                    local_indexer_metadata, dcp_decode_context_->layout());
      AttentionMetadata selector_metadata = local_indexer_metadata;
      selector_metadata.q_cu_seq_lens = causal_selector.q_cu_seq_lens;
      selector_metadata.kv_cu_seq_lens = causal_selector.q_cu_seq_lens;
      selector_metadata.kv_seq_lens = causal_selector.local_context_lens;
      selector_metadata.block_table = causal_selector.block_table;
      selector_metadata.max_query_len = 1;
      selector_metadata.max_seq_len = index_topk_;
      // The decode-view selector does not consume total_kv_len. Keeping this
      // host metadata scalar avoids a device-to-host synchronization here.
      selector_metadata.total_kv_len = 0;
      selector_metadata.is_prefill = false;
      selector_metadata.is_chunked_prefill = false;
      candidates = indexer_->forward_dcp_local_prefill(hidden_states,
                                                       q_norm,
                                                       positions,
                                                       index_cache,
                                                       local_indexer_metadata,
                                                       selector_metadata,
                                                       index_cache_scale);
    }
    candidate_gather = dcp_decode_context_->launch_indexer_candidate_gather(
        candidates.scores, candidates.global_slots);
    candidate_topk = candidates.global_slots.size(1);
  } else {
    CHECK(!enable_lighting_indexer_)
        << "Shared DSA layer requires externally supplied top-k metadata.";
  }
  latent_cache = kv_a_proj_with_mqa_(hidden_states);
  if (!enable_fused_qkv) {
    decode_kv_pre_base(
        latent_cache, positions, cache_metadata, use_prompt_rope);
  } else {
    CHECK_GT(qk_rope_head_dim_, 0)
        << "Fused MLA KV does not implement the NoPE path.";
    // fused_mla_kv: kv_a_layernorm + rotary_emb + reshape_paged_cache
    const int64_t batch = latent_cache.size(0);
    const int64_t seq = latent_cache.dim() == 3 ? latent_cache.size(1) : 1;
    latent_cache = latent_cache.view({batch, seq, 1, latent_cache.size(-1)});
    kernel::FusedMlaKVParams fused_mla_kv_params;
    fused_mla_kv_params.input_kv = latent_cache;
    fused_mla_kv_params.sin = rotary_emb_->get_sin_cache();
    fused_mla_kv_params.cos = rotary_emb_->get_cos_cache();
    fused_mla_kv_params.position_id = positions;
    fused_mla_kv_params.gamma = kv_a_layernorm_->weight();
    fused_mla_kv_params.kv_cache = k_cache;
    fused_mla_kv_params.kv_cache_scale = k_cache_scale;
    fused_mla_kv_params.slot_mapping =
        cache_metadata.slot_mapping.view({batch, seq});
    fused_mla_kv_params.cache_bs_id = std::nullopt;
    fused_mla_kv_params.cache_seq_offset = std::nullopt;
    fused_mla_kv_params.quant_mode =
        k_cache_scale.has_value() ? "dynamic_per_token" : "none";
    fused_mla_kv_params.eps = eps_;
    fused_mla_kv_params.interleaved = interleaved_;
    kernel::fused_mla_kv(fused_mla_kv_params);
  }
  torch::Tensor v_input = latent_cache.slice(-1, 0, kv_lora_rank_);
  torch::Tensor k_input = latent_cache;
  k_input = k_input.view({k_input.size(0), -1});
  v_input = v_input.view({v_input.size(0), -1});
  if (!enable_dcp_prefill_cache_write) {
    update_mla_k_cache(k_input,
                       attn_metadata,
                       kv_cache,
                       k_cache_scale,
                       is_prefill_or_chunked_prefill ||
                           (enable_dcp_paged_attention && !enable_fused_qkv),
                       cache_metadata.slot_mapping);
  }
  if (candidate_gather.has_value()) {
    topk_state = dcp_decode_context_->finish_indexer_candidate_merge(
        std::move(candidate_gather.value()),
        candidate_topk,
        attn_metadata.slot_mapping);
  }

  std::optional<DsaTopkState> local_topk;
  if (topk_state.has_value() && enable_dcp_paged_attention) {
    const DsaTopkState* relayed_view =
        topk_transfer != nullptr ? topk_transfer->localized_input() : nullptr;
    if (relayed_view != nullptr) {
      local_topk = *relayed_view;
    } else {
      CHECK(dcp_decode_context_ != nullptr);
      local_topk = dcp_decode_context_->localize_topk(topk_state.value());
    }
  }
  if (topk_transfer != nullptr) {
    topk_transfer->complete(topk_state, local_topk);
  }
  AttentionMetadata kernel_metadata =
      build_mla_attention_metadata(attn_metadata, topk_state);
  if (enable_dcp_paged_attention) {
    CHECK(topk_state.has_value())
        << "GLM-5.2 DCP paged attention requires global DSA top-k metadata";
  }

  torch::Tensor attn_output;
  if (enable_dcp_decode) {
    CHECK(local_topk.has_value())
        << "DCP local decode attention requires the rank-local top-k view";
    const int64_t token_count = q_input.size(0);
    CHECK_EQ(local_topk.value().block_tables().size(0), token_count);

    AttentionMetadata local_metadata =
        build_mla_attention_metadata(attn_metadata, local_topk.value());
    local_metadata.is_prefill = false;
    local_metadata.is_chunked_prefill = false;

    torch::Tensor query = q_input;
    const bool gathers_tp_heads = dcp_spans_tp_;
    if (gathers_tp_heads) {
      CHECK(query_gather.has_value())
          << "DCP local decode attention requires the early-launched TP query "
             "gather";
      torch::Tensor gathered_query;
      {
        auto ctx = std::move(query_gather.value());
        if (ctx.work.defined()) {
          ctx.work->wait();
        }
        gathered_query =
            torch::cat(ctx.stacked.unbind(0), /*dim=*/1).contiguous();
      }
      query = std::move(gathered_query);
      query_gather.reset();
    }
    Attention& local_attention = gathers_tp_heads ? dcp_full_head_attn_ : attn_;
    const int64_t attention_heads =
        gathers_tp_heads ? full_heads_.attn : heads.attn;

    torch::Tensor output = torch::empty(
        {token_count, attention_heads * kv_lora_rank_}, q_input.options());
    std::optional<torch::Tensor> output_lse = std::nullopt;
    local_attention->decoder_forward(query,
                                     output,
                                     output_lse,
                                     kv_cache.get_k_cache(),
                                     std::nullopt,
                                     local_metadata,
                                     kv_cache.get_k_cache_scale(),
                                     std::nullopt,
                                     /*return_lse=*/true);
    CHECK(output_lse.has_value())
        << "DCP local decode attention requires local sparse LSE";
    CHECK_EQ(output_lse->dim(), 3)
        << "DCP local decode LSE must be [query, heads, 1]";
    CHECK_EQ(output_lse->size(0), token_count);
    CHECK_EQ(output_lse->size(1), attention_heads);
    CHECK_EQ(output_lse->size(2), 1);

    // Head reduce-scatter is only valid when the DCP group is exactly the TP
    // group (no PCP replication); otherwise heads are merged with an all-reduce
    // and redistributed by tp_rank_ below.
    const bool head_sharded =
        dcp_spans_tp_ &&
        tp_group_->world_size() == dcp_decode_context_->world_size();
    attn_output = dcp_decode_context_->merge(
        output.view({token_count, 1, attention_heads, kv_lora_rank_}),
        output_lse.value(),
        attn_metadata.slot_mapping,
        /*head_sharded=*/head_sharded);
    if (dcp_spans_tp_ && !head_sharded) {
      const int64_t first_head = tp_rank_ * heads.attn;
      const int64_t last_head = first_head + heads.attn;
      attn_output =
          attn_output.slice(/*dim=*/2, first_head, last_head).contiguous();
    }
    attn_output =
        attn_output.view({q_input.size(0), heads.attn * kv_lora_rank_});
  } else if (enable_dcp_chunked_prefill) {
    attn_output = run_dcp_chunked_prefill_attention(q_input,
                                                    local_topk.value(),
                                                    query_gather,
                                                    kv_cache,
                                                    cache_metadata,
                                                    attn_metadata.slot_mapping)
                      .view({q_input.size(0), heads.attn * kv_lora_rank_});
  } else {
    torch::Tensor sorted_slots;
    torch::Tensor sorted_rows;
    if (shard_metadata != nullptr &&
        shard_metadata->prefill_sorted_slots.defined()) {
      sorted_slots = shard_metadata->prefill_sorted_slots;
      sorted_rows = shard_metadata->prefill_sorted_rows;
    } else {
      std::tie(sorted_slots, sorted_rows) =
          build_prefill_slot_order(attn_metadata.slot_mapping);
      sorted_slots = sorted_slots.to(torch::kInt64);
    }
    AttentionMetadata compact_metadata;
    torch::Tensor scratch;
    {
      const AttentionMetadata& metadata = kernel_metadata;
      const int64_t token_count = k_input.size(0);
      CHECK_GT(token_count, 0) << "Empty prefill must bypass attention";
      CHECK_EQ(k_input.dim(), 2);
      CHECK_EQ(k_cache.dim(), 4);
      CHECK_EQ(k_cache.size(1), 1);
      CHECK_EQ(k_cache.size(3), k_input.size(1));
      CHECK_EQ(sorted_slots.numel(), token_count);
      CHECK_EQ(sorted_rows.numel(), token_count);
      CHECK_EQ(sorted_slots.scalar_type(), torch::kInt64);
      CHECK_EQ(sorted_rows.scalar_type(), torch::kInt64);
      CHECK_EQ(metadata.block_table.dim(), 2);
      CHECK_EQ(metadata.block_table.size(0), metadata.kv_seq_lens.numel());
      const int64_t block_size = k_cache.size(2);
      CHECK_GT(block_size, 0);

      compact_metadata = metadata;
      const torch::Tensor& topk = metadata.block_table;
      torch::Tensor columns = torch::arange(topk.size(1), topk.options());
      torch::Tensor valid =
          (columns < metadata.kv_seq_lens.unsqueeze(1)) & (topk >= 0);
      torch::Tensor search_slots = topk.to(torch::kInt64).contiguous();
      torch::Tensor cache_positions =
          torch::searchsorted(sorted_slots, search_slots);
      torch::Tensor safe_positions = cache_positions.clamp_max(token_count - 1);
      torch::Tensor matches =
          (cache_positions < token_count) &
          (sorted_slots.index_select(0, safe_positions.flatten())
               .view_as(topk) == search_slots);
      // Missing effective slots deliberately reach index_select's device bounds
      // check. Clamping them to row zero would silently discard historical K.
      torch::Tensor checked_positions = torch::where(
          valid, torch::where(matches, cache_positions, token_count), 0);
      torch::Tensor scratch_rows =
          sorted_rows.index_select(0, checked_positions.flatten())
              .view_as(topk);
      compact_metadata.block_table =
          torch::where(valid, scratch_rows, torch::where(topk < 0, -1, 0))
              .to(topk.scalar_type());

      scratch = torch::zeros({util::ceil_div(token_count, block_size),
                              1,
                              block_size,
                              k_input.size(1)},
                             k_input.options());
      xllm::kernel::ReshapePagedCacheParams write_params;
      write_params.key = k_input.unsqueeze(1);
      write_params.k_cache = scratch;
      write_params.slot_mapping =
          torch::arange(token_count, metadata.slot_mapping.options());
      xllm::kernel::reshape_paged_cache(write_params);
    }
    KVCache attention_cache(
        KVCacheTensors{std::move(scratch), torch::Tensor()});
    std::tie(attn_output, std::ignore) =
        attn_(compact_metadata, q_input, k_input, v_input, attention_cache);
  }

  if (enable_dcp_prefill_cache_write) {
    update_mla_k_cache(k_input,
                       attn_metadata,
                       kv_cache,
                       k_cache_scale,
                       /*is_prefill_phase=*/true,
                       cache_metadata.slot_mapping);
  }
  return project_output(attn_output);
}

}  // namespace layer
}  // namespace xllm
