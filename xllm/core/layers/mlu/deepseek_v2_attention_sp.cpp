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

#include "deepseek_v2_attention.h"
#include "platform/device.h"

namespace xllm {
namespace layer {

torch::Tensor DeepseekV2AttentionImpl::forward_sp(
    const torch::Tensor& positions,
    const torch::Tensor& hidden_states,
    const AttentionMetadata& attn_metadata,
    const v32_cp::DeepseekV32CPContext& sp_ctx,
    KVCache& kv_cache,
    bool is_prefill_or_chunked_prefill,
    DsaTopkTransfer* topk_transfer) {
  CHECK(can_use_sp(topk_transfer))
      << "deepseek_v32 sequence parallel requires either a lighting indexer "
         "or reused top-k state.";
  CHECK(is_prefill_or_chunked_prefill)
      << "deepseek_v32 sequence parallel only supports prefill batches.";
  auto k_cache_scale = kv_cache.get_k_cache_scale();
  auto query_prep = prep_query(hidden_states, active_heads());

  std::optional<DsaTopkState> topk_state;
  v32_cp::PaddedGatherHandle mla_handle;
  IndexerSPPreOut index_pre;
  v32_cp::PaddedGatherHandle index_handle;
  const DsaTopkState* reused_topk =
      topk_transfer != nullptr ? topk_transfer->input() : nullptr;
  const bool compute_topk = !attn_metadata.is_dummy && reused_topk == nullptr;

  Device device(hidden_states.device());
  CHECK(sp_comm_stream_ != nullptr)
      << "sequence-parallel attention requires a model-scoped communication "
         "stream";
  if (compute_topk) {
    index_pre = indexer_->sp_pre(hidden_states,
                                 query_prep.q_norm,
                                 positions,
                                 sp_ctx.local_attn_metadata,
                                 sp_ctx,
                                 /*quantize_output=*/false);
    auto compute_stream = device.current_stream();
    sp_comm_stream_->wait_stream(*compute_stream);
    {
      torch::StreamGuard stream_guard = sp_comm_stream_->set_stream_guard();
      index_handle = indexer_->sp_comm(index_pre.k_local, sp_ctx);
    }
  }

  auto mla_inputs =
      build_sp_mla_inputs(hidden_states, positions, query_prep, sp_ctx);

  torch::Tensor k_gathered;
  if (compute_topk) {
    k_gathered = indexer_->sp_wait_k(index_pre.k_local, index_handle, sp_ctx);
  }
  auto compute_stream = device.current_stream();
  sp_comm_stream_->wait_stream(*compute_stream);
  {
    torch::StreamGuard stream_guard = sp_comm_stream_->set_stream_guard();
    mla_handle = sp_mla_comm(mla_inputs.k_input, sp_ctx);
  }
  if (compute_topk) {
    torch::Tensor index_cache = kv_cache.get_index_cache();
    auto index_cache_scale = kv_cache.get_indexer_cache_scale();
    AttentionMetadata indexer_metadata = attn_metadata;
    if (enable_mla_cache_sharding_ && attn_metadata.is_chunked_prefill) {
      CHECK(dcp_decode_context_ != nullptr);
      const std::shared_ptr<const KVShardBatchMetadata>& shard_metadata =
          attn_metadata.kv_shard_batch_metadata;
      if (shard_metadata != nullptr) {
        CHECK(shard_metadata->expanded_indexer_block_table.defined())
            << "cache-shard batch metadata requires expanded indexer blocks";
        indexer_metadata.block_table =
            shard_metadata->expanded_indexer_block_table;
      } else {
        indexer_metadata.block_table =
            dcp_decode_context_->expand_indexer_block_table(
                attn_metadata.block_table);
      }
    }
    auto index_out = indexer_->sp_post(index_pre,
                                       k_gathered,
                                       index_cache,
                                       indexer_metadata,
                                       sp_ctx.gathered_slot_mapping,
                                       sp_ctx,
                                       index_cache_scale);
    topk_state.emplace(std::get<0>(index_out), std::get<1>(index_out));
  }
  finish_sp_k_gather(mla_inputs, mla_handle, sp_ctx);

  if (attn_metadata.is_dummy) {
    topk_state.reset();
  } else {
    if (reused_topk != nullptr) {
      topk_state = *reused_topk;
    }
    CHECK(topk_state.has_value())
        << "DSA sequence-parallel attention requires top-k state.";
  }

  torch::Tensor mla_slot_mapping = sp_ctx.gathered_slot_mapping;
  if (enable_mla_cache_sharding_) {
    CHECK(sp_ctx.local_dcp_gathered_slot_mapping.defined())
        << "cache-sharded CP requires localized gathered slot mapping";
    mla_slot_mapping = sp_ctx.local_dcp_gathered_slot_mapping;
  }
  update_mla_k_cache(mla_inputs.k_input,
                     attn_metadata,
                     kv_cache,
                     k_cache_scale,
                     is_prefill_or_chunked_prefill,
                     mla_slot_mapping);
  if (topk_transfer != nullptr) {
    topk_transfer->complete(topk_state);
  }

  AttentionMetadata kernel_metadata =
      build_mla_attention_metadata(attn_metadata, topk_state);
  kernel_metadata.q_cu_seq_lens = sp_ctx.local_attn_metadata.q_cu_seq_lens;
  kernel_metadata.max_query_len = sp_ctx.local_attn_metadata.max_query_len;
  if (enable_mla_cache_sharding_ && attn_metadata.is_chunked_prefill) {
    torch::Tensor gathered_q = v32_cp::restore_gathered_to_global_order(
        v32_cp::all_gather_across_ranks(mla_inputs.q_input, sp_ctx), sp_ctx);
    DsaTopkState gathered_topk(
        v32_cp::restore_gathered_to_global_order(
            v32_cp::all_gather_across_ranks(topk_state->block_tables(), sp_ctx),
            sp_ctx),
        v32_cp::restore_gathered_to_global_order(
            v32_cp::all_gather_across_ranks(topk_state->context_lens(), sp_ctx),
            sp_ctx));
    const DcpAttentionResult merged = run_dcp_chunked_prefill_attention(
        gathered_q, gathered_topk, kv_cache, attn_metadata);
    torch::Tensor global_output = merged.output.reshape(
        {sp_ctx.total_tokens, full_heads().attn * kv_lora_rank_});
    torch::Tensor local_output =
        v32_cp::reorder_to_local_shard(global_output, sp_ctx);
    return project_output(local_output, full_heads());
  }

  KVCache* attention_cache = &kv_cache;
  std::optional<KVCache> scratch_cache;
  if (enable_mla_cache_sharding_) {
    CHECK(sp_ctx.sorted_gathered_slot_mapping_int64.defined());
    CHECK(sp_ctx.sorted_gathered_slot_rows.defined());
    auto [scratch_metadata, built_scratch_cache] =
        build_sharded_prefill_attention_cache(
            mla_inputs.k_input,
            sp_ctx.gathered_slot_mapping,
            sp_ctx.sorted_gathered_slot_mapping_int64,
            sp_ctx.sorted_gathered_slot_rows,
            kernel_metadata);
    kernel_metadata = std::move(scratch_metadata);
    scratch_cache.emplace(std::move(built_scratch_cache));
    attention_cache = &scratch_cache.value();
  }
  auto [attn_output_local, output_lse] = attn_(kernel_metadata,
                                               mla_inputs.q_input,
                                               mla_inputs.k_input,
                                               mla_inputs.v_input,
                                               *attention_cache);
  torch::Tensor output = project_output(attn_output_local, active_heads());
  if (!use_replicated_attn_weights()) {
    // TP-sharded sequence-parallel attention: each rank computes its head
    // shard on the local sequence shard, so merge the head shards across the
    // TP group to recover the full hidden state before returning the packed
    // local layout. No-op when tp_size == 1 (replicated path never enters).
    CHECK(!enable_mla_cache_sharding_)
        << "TP-sharded sequence-parallel attention requires kv_split_size == 1";
    output = parallel_state::reduce(output, tp_group_);
  }
  return output;
}

DeepseekV2AttentionImpl::MlaInputs DeepseekV2AttentionImpl::build_sp_mla_inputs(
    const torch::Tensor& hidden_states,
    const torch::Tensor& positions,
    const QueryPrep& query_prep,
    const v32_cp::DeepseekV32CPContext& sp_ctx) {
  MlaInputs out;
  out.q_input = torch::empty({hidden_states.size(0),
                              active_heads().attn,
                              kv_lora_rank_ + qk_rope_head_dim_},
                             hidden_states.options());
  out.q_norm = query_prep.q_norm;
  torch::Tensor latent_cache = kv_a_proj_with_mqa_(hidden_states);
  fill_q_input(out.q_input,
               query_prep.q,
               positions,
               sp_ctx.local_attn_metadata,
               /*use_prompt_rope=*/false);
  decode_kv_pre_base(latent_cache,
                     positions,
                     sp_ctx.local_attn_metadata,
                     /*use_prompt_rope=*/false);
  out.v_input = latent_cache.slice(-1, 0, kv_lora_rank_);
  out.k_input = latent_cache;
  out.q_input = out.q_input.view({out.q_input.size(0), -1});
  out.k_input = out.k_input.view({out.k_input.size(0), -1});
  out.v_input = out.v_input.view({out.v_input.size(0), -1});
  return out;
}

v32_cp::PaddedGatherHandle DeepseekV2AttentionImpl::sp_mla_comm(
    const torch::Tensor& k_input,
    const v32_cp::DeepseekV32CPContext& sp_ctx) const {
  return parallel_state::launch_gather(
      k_input, sp_ctx.process_group, sp_ctx.comm_plan.tokens_per_rank);
}

void DeepseekV2AttentionImpl::finish_sp_k_gather(
    MlaInputs& mla_inputs,
    const v32_cp::PaddedGatherHandle& k_handle,
    const v32_cp::DeepseekV32CPContext& sp_ctx) const {
  (void)sp_ctx;
  mla_inputs.k_input = parallel_state::finish_gather(k_handle);
  mla_inputs.v_input = mla_inputs.k_input.slice(-1, 0, kv_lora_rank_);
}

}  // namespace layer
}  // namespace xllm
