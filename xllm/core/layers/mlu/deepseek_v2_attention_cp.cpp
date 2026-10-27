/* Copyright 2025-2026 The xLLM Authors.

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

#include <tuple>
#include <utility>

#include "core/layers/mlu/dcp_batch_metadata.h"
#include "framework/parallel_state/parallel_state.h"
#include "kernels/ops_api.h"
#include "layers/mlu/deepseek_v2_attention.h"
#include "platform/device.h"
#include "util/utils.h"

namespace xllm {
namespace layer {

torch::Tensor DeepseekV2AttentionImpl::forward_cp(
    const torch::Tensor& positions,
    const torch::Tensor& hidden_states,
    const AttentionMetadata& attn_metadata,
    const v32_cp::DeepseekV32CPContext& cp_ctx,
    KVCache& kv_cache,
    bool is_prefill_or_chunked_prefill,
    DsaTopkTransfer* topk_transfer) {
  CHECK(!use_kpool_indexer_ &&
        (has_indexer_ ||
         (topk_transfer != nullptr && topk_transfer->input() != nullptr)))
      << "deepseek_v32 context parallel requires either a lighting indexer "
         "or reused top-k state.";
  CHECK(is_prefill_or_chunked_prefill)
      << "deepseek_v32 context parallel only supports prefill batches.";
  const bool use_dcp = enable_mla_cache_sharding_ && !attn_metadata.is_dummy;
  std::optional<torch::Tensor> k_cache_scale = kv_cache.get_k_cache_scale();
  QueryPrep query_prep = prep_query(hidden_states);

  std::optional<DsaTopkState> topk_state;
  std::optional<DsaTopkState> local_topk;
  v32_cp::PaddedGatherHandle mla_handle;
  IndexerCPPreOut index_pre;
  v32_cp::PaddedGatherHandle index_handle;
  std::optional<DcpIndexerGatherAsyncCtx> candidate_gather;
  int64_t candidate_topk = 0;
  const DsaTopkState* reused_topk =
      topk_transfer != nullptr ? topk_transfer->input() : nullptr;
  const bool compute_topk = !attn_metadata.is_dummy && reused_topk == nullptr;

  Device device(hidden_states.device());
  CHECK(cp_comm_stream_ != nullptr)
      << "context-parallel attention requires a model-scoped communication "
         "stream";
  if (compute_topk) {
    index_pre = indexer_->cp_pre(hidden_states,
                                 query_prep.q_norm,
                                 positions,
                                 cp_ctx.local_attn_metadata);
    auto compute_stream = device.current_stream();
    cp_comm_stream_->wait_stream(*compute_stream);
    {
      torch::StreamGuard stream_guard = cp_comm_stream_->set_stream_guard();
      index_handle =
          v32_cp::launch_all_gather_across_ranks(index_pre.k_local, cp_ctx);
    }
  }

  torch::Tensor q_input = torch::empty({hidden_states.size(0),
                                        tp_heads_.attn,
                                        kv_lora_rank_ + qk_rope_head_dim_},
                                       hidden_states.options());
  torch::Tensor k_input = kv_a_proj_with_mqa_(hidden_states);
  fill_q_input(q_input,
               query_prep.q,
               positions,
               cp_ctx.local_attn_metadata,
               /*use_prompt_rope=*/false);
  decode_kv_pre_base(k_input,
                     positions,
                     cp_ctx.local_attn_metadata,
                     /*use_prompt_rope=*/false);
  q_input = q_input.view({q_input.size(0), -1});
  k_input = k_input.view({k_input.size(0), -1});

  torch::Tensor k_gathered;
  if (compute_topk) {
    k_gathered = parallel_state::finish_gather(index_handle);
  }
  auto compute_stream = device.current_stream();
  cp_comm_stream_->wait_stream(*compute_stream);
  {
    torch::StreamGuard stream_guard = cp_comm_stream_->set_stream_guard();
    mla_handle = v32_cp::launch_all_gather_across_ranks(k_input, cp_ctx);
  }
  if (compute_topk) {
    torch::Tensor index_cache = kv_cache.get_index_cache();
    std::optional<torch::Tensor> index_cache_scale =
        kv_cache.get_indexer_cache_scale();
    if (use_dcp) {
      CHECK(dcp_decode_context_ != nullptr);
      IndexerCPPreOut global_pre;
      global_pre.q = v32_cp::gather_and_restore_global(index_pre.q, cp_ctx);
      global_pre.k_local =
          v32_cp::restore_gathered_to_global_order(k_gathered, cp_ctx);
      global_pre.weights =
          v32_cp::gather_and_restore_global(index_pre.weights, cp_ctx);

      AttentionMetadata local_indexer_metadata = attn_metadata;
      const std::shared_ptr<const KVShardBatchMetadata>& shard_metadata =
          attn_metadata.kv_shard_batch_metadata;
      if (shard_metadata != nullptr) {
        CHECK(shard_metadata->local_slot_mapping.defined())
            << "cache-sharded CP requires localized indexer slots";
        local_indexer_metadata.slot_mapping =
            shard_metadata->local_slot_mapping;
      } else {
        local_indexer_metadata.slot_mapping =
            dcp_decode_context_->localize_slots(attn_metadata.slot_mapping);
      }
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
      DcpIndexerLocalCandidates candidates =
          indexer_->forward_dcp_local_prefill_from_cp(global_pre,
                                                      index_cache,
                                                      local_indexer_metadata,
                                                      selector_metadata,
                                                      index_cache_scale);
      candidate_gather = dcp_decode_context_->launch_indexer_candidate_gather(
          candidates.scores, candidates.global_slots);
      candidate_topk = candidates.global_slots.size(1);
    } else {
      auto index_out = indexer_->cp_post(index_pre,
                                         k_gathered,
                                         index_cache,
                                         attn_metadata,
                                         cp_ctx.gathered_slot_mapping,
                                         cp_ctx,
                                         index_cache_scale);
      topk_state.emplace(std::get<0>(index_out), std::get<1>(index_out));
    }
  }
  k_input = parallel_state::finish_gather(mla_handle);
  torch::Tensor v_input = k_input.slice(-1, 0, kv_lora_rank_);

  torch::Tensor mla_slot_mapping = cp_ctx.gathered_slot_mapping;
  if (use_dcp) {
    CHECK(cp_ctx.local_dcp_gathered_slot_mapping.defined())
        << "cache-sharded CP requires localized MLA slots";
    mla_slot_mapping = cp_ctx.local_dcp_gathered_slot_mapping;
  }
  update_mla_k_cache(k_input,
                     attn_metadata,
                     kv_cache,
                     k_cache_scale,
                     is_prefill_or_chunked_prefill,
                     mla_slot_mapping);

  if (candidate_gather.has_value()) {
    DsaTopkState global_topk =
        dcp_decode_context_->finish_indexer_candidate_merge(
            std::move(candidate_gather.value()),
            candidate_topk,
            attn_metadata.slot_mapping);
    topk_state.emplace(
        v32_cp::reorder_to_local_shard(global_topk.block_tables(), cp_ctx),
        v32_cp::reorder_to_local_shard(global_topk.context_lens(), cp_ctx));
  }

  if (attn_metadata.is_dummy) {
    topk_state.reset();
  } else {
    if (reused_topk != nullptr) {
      topk_state = *reused_topk;
    }
    CHECK(topk_state.has_value())
        << "DSA context-parallel attention requires top-k state.";
  }
  if (use_dcp && topk_state.has_value() && attn_metadata.is_chunked_prefill &&
      topk_transfer != nullptr && topk_transfer->captures_output()) {
    const DsaTopkState* relayed_view = topk_transfer->localized_input();
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
  kernel_metadata.q_cu_seq_lens = cp_ctx.local_attn_metadata.q_cu_seq_lens;
  kernel_metadata.max_query_len = cp_ctx.local_attn_metadata.max_query_len;

  torch::Tensor attn_output_local;
  if (use_dcp && attn_metadata.is_chunked_prefill) {
    torch::Tensor packed_q = v32_cp::all_gather_across_ranks(q_input, cp_ctx);
    DsaTopkState gathered_topk(
        v32_cp::all_gather_across_ranks(topk_state->block_tables(), cp_ctx),
        v32_cp::all_gather_across_ranks(topk_state->context_lens(), cp_ctx));
    DsaTopkState gathered_local_topk =
        dcp_decode_context_->localize_topk(gathered_topk);
    std::optional<parallel_state::GatherAsyncCtx> query_gather;
    if (dcp_spans_tp_) {
      parallel_state::GatherAsyncCtx& gather_ctx = query_gather.emplace();
      gather_ctx.input = packed_q;
      std::vector<int64_t> stacked_shape = packed_q.sizes().vec();
      stacked_shape.insert(stacked_shape.begin(), tp_group_->world_size());
      gather_ctx.stacked = torch::empty(stacked_shape, packed_q.options());
      gather_ctx.work =
          tp_group_->allgather_base_async(packed_q, gather_ctx.stacked);
    }
    // Query rows, top-k rows and merge padding slots share the CP rank order.
    torch::Tensor packed_output =
        run_dcp_chunked_prefill_attention(packed_q,
                                          gathered_local_topk,
                                          query_gather,
                                          kv_cache,
                                          attn_metadata,
                                          cp_ctx.gathered_slot_mapping)
            .view({cp_ctx.total_tokens, tp_heads_.attn * kv_lora_rank_});
    attn_output_local = v32_cp::slice_local_packed(packed_output, cp_ctx);
  } else if (use_dcp) {
    torch::Tensor k_cache = kv_cache.get_k_cache();
    AttentionMetadata compact_metadata;
    torch::Tensor scratch;
    {
      const AttentionMetadata& metadata = kernel_metadata;
      const torch::Tensor& sorted_slots =
          cp_ctx.sorted_gathered_slot_mapping_int64;
      const torch::Tensor& sorted_rows = cp_ctx.sorted_gathered_slot_rows;
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
    std::tie(attn_output_local, std::ignore) =
        attn_(compact_metadata, q_input, k_input, v_input, attention_cache);
  } else {
    std::tie(attn_output_local, std::ignore) =
        attn_(kernel_metadata, q_input, k_input, v_input, kv_cache);
  }
  torch::Tensor output = project_output(attn_output_local);
  output = parallel_state::reduce(output, tp_group_);
  return output;
}

}  // namespace layer
}  // namespace xllm
