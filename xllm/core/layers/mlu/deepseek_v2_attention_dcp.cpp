/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include <limits>
#include <tuple>
#include <utility>
#include <vector>

#include "core/layers/mlu/deepseek_v2_attention.h"
#include "framework/parallel_state/parallel_state.h"
#include "kernels/ops_api.h"
#include "util/utils.h"

namespace xllm {
namespace layer {

namespace {

torch::Tensor dcp_padding_rows(const AttentionMetadata& attn_metadata) {
  CHECK(attn_metadata.slot_mapping.defined())
      << "DCP graph padding requires slot mapping";
  CHECK_EQ(attn_metadata.slot_mapping.dim(), 1)
      << "DCP graph padding slot mapping must be one-dimensional";
  return attn_metadata.slot_mapping < 0;
}

DsaTopkState mask_dcp_padding_topk(const DsaTopkState& topk_state,
                                   const torch::Tensor& padding_rows) {
  const torch::Tensor& block_tables = topk_state.block_tables();
  CHECK_EQ(block_tables.dim(), 2)
      << "DCP decode top-k block tables must be two-dimensional";
  CHECK_EQ(block_tables.size(0), padding_rows.numel())
      << "DCP decode top-k rows must match slot mappings";
  CHECK_EQ(topk_state.context_lens().numel(), padding_rows.numel())
      << "DCP decode top-k context rows must match slot mappings";
  torch::Tensor masked_block_tables =
      torch::where(padding_rows.unsqueeze(1),
                   torch::full_like(block_tables, -1),
                   block_tables);
  torch::Tensor masked_context_lens =
      torch::where(padding_rows,
                   torch::zeros_like(topk_state.context_lens()),
                   topk_state.context_lens());
  return DsaTopkState(masked_block_tables.contiguous(),
                      masked_context_lens.contiguous());
}

DcpAttentionResult neutralize_dcp_padding_rows(
    DcpAttentionResult result,
    const torch::Tensor& padding_rows) {
  CHECK_EQ(result.output.size(0), padding_rows.numel())
      << "DCP attention output rows must match slot mappings";
  CHECK_EQ(result.lse.size(0), padding_rows.numel())
      << "DCP attention LSE rows must match slot mappings";
  const float negative_infinity = -std::numeric_limits<float>::infinity();
  result.output = torch::where(padding_rows.view({-1, 1, 1, 1}),
                               torch::zeros_like(result.output),
                               result.output);
  result.lse = torch::where(padding_rows.view({-1, 1, 1}),
                            torch::full_like(result.lse, negative_infinity),
                            result.lse);
  return result;
}

}  // namespace

DcpAttentionResult DeepseekV2AttentionImpl::run_dcp_paged_attention(
    const torch::Tensor& q_input,
    const DsaTopkState& global_topk,
    KVCache& kv_cache,
    const AttentionMetadata& base_metadata) {
  CHECK(dcp_decode_context_ != nullptr);
  const HeadInfo& heads = active_heads();
  const int64_t token_count = q_input.size(0);
  CHECK_EQ(global_topk.block_tables().size(0), token_count);
  const int64_t topk_width = global_topk.block_tables().size(1);
  torch::Tensor topk_columns = torch::arange(
      topk_width, global_topk.block_tables().options().dtype(torch::kLong));
  torch::Tensor valid_topk =
      topk_columns.unsqueeze(0) < global_topk.context_lens().unsqueeze(1);
  torch::Tensor valid_global_slots =
      torch::where(valid_topk,
                   global_topk.block_tables(),
                   torch::full_like(global_topk.block_tables(),
                                    KVShardLayout::kInvalidSlot));
  torch::Tensor gathered_cache = dcp_decode_context_->gather_topk_cache(
      valid_global_slots, kv_cache.get_k_cache());
  const int64_t selected_count =
      gathered_cache.size(0) * gathered_cache.size(1);
  const int64_t scratch_blocks =
      util::ceil_div(selected_count, static_cast<int64_t>(block_size_));
  torch::Tensor scratch_k_cache =
      torch::zeros({scratch_blocks, 1, block_size_, gathered_cache.size(2)},
                   gathered_cache.options());
  torch::Tensor scratch_slots =
      torch::arange(selected_count, global_topk.block_tables().options());
  xllm::kernel::ReshapePagedCacheParams write_params;
  write_params.key =
      gathered_cache.flatten(/*start_dim=*/0, /*end_dim=*/1).unsqueeze(1);
  write_params.k_cache = scratch_k_cache;
  write_params.slot_mapping = scratch_slots;
  xllm::kernel::reshape_paged_cache(write_params);

  DsaTopkState scratch_topk(
      scratch_slots.view_as(global_topk.block_tables()).contiguous(),
      global_topk.context_lens());
  AttentionMetadata scratch_metadata =
      build_mla_attention_metadata(base_metadata, scratch_topk);
  scratch_metadata.is_prefill = false;
  scratch_metadata.is_chunked_prefill = false;
  torch::Tensor output = torch::empty({token_count, heads.attn * kv_lora_rank_},
                                      q_input.options());
  std::optional<torch::Tensor> output_lse = std::nullopt;
  torch::Tensor query = q_input;
  attn_->decoder_forward(query,
                         output,
                         output_lse,
                         scratch_k_cache,
                         std::nullopt,
                         scratch_metadata,
                         kv_cache.get_k_cache_scale(),
                         std::nullopt,
                         /*return_lse=*/true);
  CHECK(output_lse.has_value())
      << "GLM-5.2 DCP paged attention requires global sparse LSE";
  return neutralize_dcp_padding_rows(
      {output.view({token_count, 1, heads.attn, kv_lora_rank_}),
       output_lse.value()},
      dcp_padding_rows(base_metadata));
}

DcpAttentionResult DeepseekV2AttentionImpl::run_dcp_chunked_prefill_attention(
    const torch::Tensor& q_input,
    const DsaTopkState& global_topk,
    KVCache& kv_cache,
    const AttentionMetadata& base_metadata) {
  CHECK(dcp_decode_context_ != nullptr);
  const HeadInfo& heads = active_heads();
  const DsaTopkState local_topk =
      dcp_decode_context_->localize_topk(global_topk);
  AttentionMetadata local_metadata =
      build_mla_attention_metadata(base_metadata, local_topk);
  torch::Tensor query =
      dcp_spans_tp_ ? parallel_state::gather(q_input, tp_group_, /*dim=*/1)
                    : q_input;
  torch::Tensor unused_key;
  torch::Tensor unused_value;
  Attention& local_attention = dcp_spans_tp_ ? dcp_full_head_attn_ : attn_;
  auto [local_output, local_lse] = local_attention(local_metadata,
                                                   query,
                                                   unused_key,
                                                   unused_value,
                                                   kv_cache,
                                                   /*return_lse=*/true);
  CHECK(local_lse.has_value())
      << "GLM-5.2 DCP chunked prefill requires local LSE";
  CHECK_EQ(local_lse->dim(), 3)
      << "DCP chunked prefill LSE must be [query, heads, 1]";
  CHECK_EQ(local_lse->size(0), q_input.size(0));
  const int64_t attention_heads =
      dcp_spans_tp_ ? full_heads().attn : heads.attn;
  CHECK_EQ(local_lse->size(1), attention_heads);
  CHECK_EQ(local_lse->size(2), 1);
  DcpAttentionResult merged = dcp_decode_context_->merge(
      local_output.view({q_input.size(0), 1, attention_heads, kv_lora_rank_}),
      local_lse.value());
  if (!dcp_spans_tp_) {
    return merged;
  }

  const int64_t first_head = tp_rank_ * heads.attn;
  const int64_t last_head = first_head + heads.attn;
  return {
      merged.output.slice(/*dim=*/2, first_head, last_head).contiguous(),
      merged.lse.slice(/*dim=*/1, first_head, last_head).contiguous(),
  };
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

  const HeadInfo& heads = active_heads();
  const bool enable_dcp_decode = !is_prefill_or_chunked_prefill;
  const bool enable_dcp_chunked_prefill = attn_metadata.is_chunked_prefill;
  const bool enable_dcp_paged_attention =
      enable_dcp_decode || enable_dcp_chunked_prefill;
  const bool enable_dcp_prefill_cache_write = attn_metadata.is_prefill;
  AttentionMetadata cache_metadata = attn_metadata;
  const torch::Tensor padding_rows = dcp_padding_rows(attn_metadata);
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
  prepare_mla_inputs(q,
                     q_norm,
                     q_input,
                     latent_cache,
                     hidden_states,
                     k_cache,
                     k_cache_scale,
                     positions,
                     cache_metadata,
                     enable_fused_qkv,
                     use_prompt_rope);

  torch::Tensor v_input = latent_cache.slice(-1, 0, kv_lora_rank_);
  torch::Tensor k_input = latent_cache;
  q_input = q_input.view({q_input.size(0), -1});
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

  const DsaTopkState* external_topk =
      topk_transfer != nullptr ? topk_transfer->input() : nullptr;
  AttentionMetadata indexer_metadata = attn_metadata;
  if (enable_dcp_paged_attention || attn_metadata.is_prefill) {
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
  std::optional<DsaTopkState> topk_state =
      resolve_dsa_topk_state(positions,
                             hidden_states,
                             q_norm,
                             indexer_metadata,
                             kv_cache,
                             is_prefill_or_chunked_prefill,
                             external_topk);
  if (topk_state.has_value()) {
    topk_state = mask_dcp_padding_topk(topk_state.value(), padding_rows);
  }
  if (topk_transfer != nullptr) {
    topk_transfer->complete(topk_state);
  }
  AttentionMetadata kernel_metadata =
      build_mla_attention_metadata(attn_metadata, topk_state);
  if (enable_dcp_paged_attention) {
    CHECK(topk_state.has_value())
        << "GLM-5.2 DCP paged attention requires global DSA top-k metadata";
  }

  torch::Tensor attn_output;
  if (enable_dcp_decode) {
    const DcpAttentionResult merged = run_dcp_paged_attention(
        q_input, topk_state.value(), kv_cache, attn_metadata);
    attn_output =
        merged.output.view({q_input.size(0), heads.attn * kv_lora_rank_});
  } else if (enable_dcp_chunked_prefill) {
    const DcpAttentionResult merged = run_dcp_chunked_prefill_attention(
        q_input, topk_state.value(), kv_cache, cache_metadata);
    attn_output =
        merged.output.view({q_input.size(0), heads.attn * kv_lora_rank_});
  } else {
    std::vector<int64_t> cache_sizes = k_cache.sizes().vec();
    cache_sizes[0] *= kv_split_size_;
    torch::Tensor replicated_k_cache =
        torch::empty(cache_sizes, k_cache.options());
    xllm::kernel::ReshapePagedCacheParams write_params;
    write_params.key = k_input.unsqueeze(1);
    write_params.k_cache = replicated_k_cache;
    write_params.slot_mapping = attn_metadata.slot_mapping;
    xllm::kernel::reshape_paged_cache(write_params);
    KVCache attention_cache(
        KVCacheTensors{replicated_k_cache, torch::Tensor()});
    std::tie(attn_output, std::ignore) =
        attn_(kernel_metadata, q_input, k_input, v_input, attention_cache);
  }

  if (enable_dcp_prefill_cache_write) {
    update_mla_k_cache(k_input,
                       attn_metadata,
                       kv_cache,
                       k_cache_scale,
                       /*is_prefill_phase=*/true,
                       cache_metadata.slot_mapping);
  }
  return project_output(attn_output, heads);
}

}  // namespace layer
}  // namespace xllm
