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

#include "attention.h"

#include <algorithm>
#include <limits>
#include <vector>

#include "framework/parallel_state/parallel_state.h"
#include "kernels/npu/npu_ops_api.h"
#include "kernels/ops_api.h"
#include "layers/npu_torch/dcp_attention_utils.h"
#include "platform/npu/acl_graph_task_update_context.h"

namespace {

void validate_dcp_decode_lengths(const std::vector<int64_t>& q_cu_seq_lens,
                                 const std::vector<int64_t>& global_kv_seq_lens,
                                 int64_t token_count) {
  CHECK(!q_cu_seq_lens.empty())
      << "DCP decode requires host cumulative query lengths.";
  CHECK_EQ(q_cu_seq_lens.size(), global_kv_seq_lens.size())
      << "DCP decode requires one query and KV length per request.";
  CHECK_EQ(token_count, static_cast<int64_t>(global_kv_seq_lens.size()))
      << "DCP supports only one-token decode requests.";

  int64_t previous_q_end = 0;
  for (const int64_t q_end : q_cu_seq_lens) {
    CHECK_EQ(q_end - previous_q_end, 1)
        << "DCP supports only one-token decode requests.";
    previous_q_end = q_end;
  }
  CHECK_EQ(previous_q_end, token_count)
      << "DCP cumulative query lengths do not match query tokens.";
}

void normalize_zero_dcp_partials(
    torch::Tensor& partial_out,
    torch::Tensor& partial_lse,
    const std::vector<int64_t>& local_kv_seq_lens) {
  CHECK_EQ(partial_out.scalar_type(), torch::kFloat32);
  CHECK_EQ(partial_lse.scalar_type(), torch::kFloat32);
  CHECK_EQ(partial_out.dim(), 3);
  CHECK_EQ(partial_lse.dim(), 3);
  CHECK_EQ(partial_out.size(0), partial_lse.size(0));
  CHECK_EQ(partial_out.size(1), partial_lse.size(1));
  CHECK_EQ(partial_lse.size(2), 1);
  CHECK_EQ(partial_out.size(0), static_cast<int64_t>(local_kv_seq_lens.size()));

  for (int64_t request_index = 0;
       request_index < static_cast<int64_t>(local_kv_seq_lens.size());
       ++request_index) {
    if (local_kv_seq_lens[request_index] == 0) {
      partial_out.select(0, request_index).zero_();
      partial_lse.select(0, request_index)
          .fill_(-std::numeric_limits<float>::infinity());
    }
  }
}

void validate_dcp_partial_shape(const torch::Tensor& partial_out,
                                const torch::Tensor& partial_lse,
                                int64_t token_count,
                                int64_t num_heads,
                                int64_t head_size,
                                const char* partial_name) {
  CHECK_EQ(partial_out.dim(), 3) << partial_name;
  CHECK_EQ(partial_out.size(0), token_count) << partial_name;
  CHECK_EQ(partial_out.size(1), num_heads) << partial_name;
  CHECK_EQ(partial_out.size(2), head_size) << partial_name;
  CHECK_EQ(partial_lse.dim(), 3) << partial_name;
  CHECK_EQ(partial_lse.size(0), token_count) << partial_name;
  CHECK_EQ(partial_lse.size(1), num_heads) << partial_name;
  CHECK_EQ(partial_lse.size(2), 1) << partial_name;
}

}  // namespace

namespace xllm {
namespace layer {

AttentionImpl::AttentionImpl(int64_t num_heads,
                             int64_t head_size,
                             float scale,
                             int64_t num_kv_heads,
                             int64_t sliding_window,
                             int32_t dcp_size,
                             int32_t dcp_rank,
                             ProcessGroup* dcp_group)
    : num_heads_(num_heads),
      head_size_(head_size),
      num_kv_heads_(num_kv_heads),
      sliding_window_(sliding_window),
      scale_(scale),
      dcp_size_(dcp_size),
      dcp_rank_(dcp_rank),
      dcp_group_(dcp_group) {
  CHECK_GT(dcp_size_, 0) << "dcp_size must be positive.";
  CHECK_GE(dcp_rank_, 0) << "dcp_rank must be non-negative.";
  CHECK_LT(dcp_rank_, dcp_size_) << "dcp_rank must be smaller than dcp_size.";
  if (sliding_window_ > -1) {
    sliding_window_ = sliding_window_ - 1;
  }
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>> AttentionImpl::forward(
    const AttentionMetadata& attn_metadata,
    torch::Tensor& query,
    torch::Tensor& key,
    torch::Tensor& value,
    KVCache& kv_cache) {
  std::optional<torch::Tensor> output_lse = std::nullopt;
  torch::Tensor output = torch::empty_like(query);

  if (attn_metadata.is_dummy) {
    return std::make_tuple(output, output_lse);
  }

  bool only_prefill =
      attn_metadata.is_prefill || attn_metadata.is_chunked_prefill;

  torch::Tensor k_cache = kv_cache.get_k_cache();
  torch::Tensor v = value.view({-1, num_kv_heads_, head_size_});
  std::optional<torch::Tensor> v_cache = kv_cache.get_v_cache();

  // Reshape and cache key/value
  xllm::kernel::ReshapePagedCacheParams reshape_paged_cache_params;
  reshape_paged_cache_params.key = key.view({-1, num_kv_heads_, head_size_});
  reshape_paged_cache_params.value = v;
  reshape_paged_cache_params.k_cache = k_cache;
  reshape_paged_cache_params.v_cache = v_cache;
  reshape_paged_cache_params.slot_mapping = attn_metadata.slot_mapping;
  xllm::kernel::reshape_paged_cache(reshape_paged_cache_params);

  if (attn_metadata.use_expanded_decode_for_spec_verify_attention) {
    decoder_forward(query, output, k_cache, v_cache, attn_metadata);
  } else if (dcp_size_ > 1 && attn_metadata.is_chunked_prefill) {
    // Mixed batches also set is_chunked_prefill, but the DCP cache-slot gate in
    // WorkerImpl rejects them upstream, so only pure chunked prefill reaches
    // here under DCP.
    dcp_chunked_prefill_forward(
        query, key, value, output, k_cache, v_cache, attn_metadata);
  } else if (only_prefill) {
    prefill_forward(query, key, value, output, k_cache, v_cache, attn_metadata);
  } else {
    decoder_forward(query, output, k_cache, v_cache, attn_metadata);
  }

  output = output.view({-1, num_heads_ * head_size_});
  return {output, output_lse};
}

void AttentionImpl::prefill_forward(torch::Tensor& query,
                                    torch::Tensor& key,
                                    torch::Tensor& value,
                                    torch::Tensor& output,
                                    const torch::Tensor& k_cache,
                                    const std::optional<torch::Tensor>& v_cache,
                                    const AttentionMetadata& attn_metadata) {
  query = query.view({-1, num_heads_, head_size_});
  output = output.view({-1, num_heads_, head_size_});

  if (attn_metadata.is_prefill) {
    key = key.view({-1, num_kv_heads_, head_size_});
    value = value.view({-1, num_kv_heads_, head_size_});

    auto fia_result = xllm::kernel::npu::npu_fused_infer_attention(
        query,
        key,
        value,
        attn_metadata.fia_attn_mask.defined()
            ? std::make_optional(attn_metadata.fia_attn_mask)
            : std::nullopt,
        std::nullopt,
        attn_metadata.q_cu_seq_lens_host_vec,
        attn_metadata.kv_cu_seq_lens_host_vec,
        num_heads_,
        num_kv_heads_,
        scale_,
        /*block_size=*/0,
        /*sparse_mode=*/3,
        "TND");
    output.copy_(std::get<0>(fia_result).view_as(output));
  } else if (attn_metadata.is_chunked_prefill) {
    torch::Tensor k = k_cache.view({k_cache.size(0), k_cache.size(1), -1});
    torch::Tensor v = v_cache.value().view(
        {v_cache.value().size(0), v_cache.value().size(1), -1});
    auto fia_result = xllm::kernel::npu::npu_fused_infer_attention(
        query,
        k,
        v,
        attn_metadata.fia_attn_mask.defined()
            ? std::make_optional(attn_metadata.fia_attn_mask)
            : std::nullopt,
        attn_metadata.block_table.defined()
            ? std::make_optional(attn_metadata.block_table)
            : std::nullopt,
        attn_metadata.q_cu_seq_lens_host_vec,
        attn_metadata.kv_seq_lens_host_vec,
        num_heads_,
        num_kv_heads_,
        scale_,
        /*block_size=*/k_cache.size(1),
        /*sparse_mode=*/3,
        "TND");
    output.copy_(std::get<0>(fia_result).view_as(output));
  }
}

void AttentionImpl::dcp_decoder_forward(
    torch::Tensor& query,
    torch::Tensor& output,
    const torch::Tensor& k_cache,
    const std::optional<torch::Tensor>& v_cache,
    const AttentionMetadata& attn_metadata) {
  CHECK(dcp_group_ != nullptr) << "DCP decode requires a DCP process group.";
  CHECK_EQ(dcp_group_->world_size(), dcp_size_)
      << "DCP process group size does not match attention DCP size.";
  CHECK_EQ(dcp_group_->rank(), dcp_rank_)
      << "DCP process group rank does not match attention DCP rank.";
  CHECK(!attn_metadata.is_prefill);
  CHECK(!attn_metadata.is_chunked_prefill);
  CHECK(!attn_metadata.is_spec_verify)
      << "DCP-2 does not support speculative decode attention.";
  CHECK(!attn_metadata.use_expanded_decode_for_spec_verify_attention)
      << "DCP-2 does not support speculative decode attention.";
  CHECK(v_cache.has_value() && v_cache.value().defined())
      << "DCP decode requires a defined V cache.";
  CHECK(attn_metadata.block_table.defined())
      << "DCP decode requires a paged KV block table.";

  const int64_t token_count = query.size(0);
  const std::vector<int64_t>& q_cu_seq_lens =
      attn_metadata.q_cu_seq_lens_host_vec;
  const std::vector<int64_t>& global_kv_seq_lens =
      attn_metadata.kv_seq_lens_host_vec;
  validate_dcp_decode_lengths(q_cu_seq_lens, global_kv_seq_lens, token_count);

  const int64_t block_size = k_cache.size(1);
  const std::vector<int64_t> local_kv_seq_lens =
      detail::compute_dcp_local_kv_seq_lens(
          global_kv_seq_lens, dcp_size_, dcp_rank_, block_size);
  const std::shared_ptr<xllm::npu::AclGraphTaskUpdateContext>& graph_context =
      attn_metadata.acl_graph_task_update_context;
  const bool capturing = graph_context != nullptr && graph_context->capturing;
  const torch::Tensor local_block_table = [&]() {
    if (capturing) {
      CHECK(attn_metadata.dcp_local_block_table.defined())
          << "DCP ACL graph capture requires a persistent local block table";
      return attn_metadata.dcp_local_block_table;
    }
    return parallel_state::select_dcp_local_block_table(
        attn_metadata.block_table, dcp_size_, dcp_rank_);
  }();
  CHECK_EQ(local_block_table.size(0), token_count)
      << "DCP local block table batch size does not match decode tokens.";

  const torch::Tensor query_group =
      parallel_state::gather(query, dcp_group_, 1);
  const int64_t group_num_heads = num_heads_ * static_cast<int64_t>(dcp_size_);
  CHECK_EQ(query_group.dim(), 3);
  CHECK_EQ(query_group.size(0), token_count);
  CHECK_EQ(query_group.size(1), group_num_heads);
  CHECK_EQ(query_group.size(2), head_size_);
  CHECK_EQ(group_num_heads % num_kv_heads_, 0)
      << "DCP gathered Q heads must preserve the GQA ratio.";

  const torch::Tensor k = k_cache.view({k_cache.size(0), k_cache.size(1), -1});
  const torch::Tensor v = v_cache.value().view(
      {v_cache.value().size(0), v_cache.value().size(1), -1});
  const std::optional<torch::Tensor> no_mask = std::nullopt;
  const std::optional<torch::Tensor> local_block_table_opt = local_block_table;

  torch::Tensor fia_out;
  torch::Tensor fia_lse;
  if (capturing) {
    // Graph capture: preallocate stable-address output/LSE, wrap the FIA call
    // in a task group, and record a FiaGraphTask so the executor can re-inject
    // the per-step DCP-local KV lengths (and local block table) on replay.
    fia_out = torch::empty({token_count, group_num_heads, head_size_},
                           query_group.options());
    fia_lse = torch::empty({token_count, group_num_heads, 1},
                           query_group.options().dtype(torch::kFloat32));
    // aclnn's default workspace is function-local and released after the call,
    // leaving a dangling address in the captured graph (hangs on replay once
    // the KV length crosses the tiling threshold that needs a non-zero
    // workspace). Query the workspace for the largest local-KV envelope the
    // graph can replay (full local block-table width) and allocate a stable
    // caller-owned buffer reused on every replay.
    const int64_t max_local_kv = local_block_table.size(1) * block_size;
    const std::vector<int64_t> max_local_kv_seq_lens(local_kv_seq_lens.size(),
                                                     max_local_kv);
    const uint64_t fia_workspace_bytes =
        xllm::kernel::npu::npu_fused_infer_attention_workspace_size(
            query_group,
            k,
            v,
            no_mask,
            local_block_table_opt,
            q_cu_seq_lens,
            max_local_kv_seq_lens,
            group_num_heads,
            num_kv_heads_,
            scale_,
            block_size,
            /*sparse_mode=*/0,
            "TND",
            /*softmax_lse_flag=*/true,
            fia_out,
            fia_lse);
    torch::Tensor fia_workspace;
    if (fia_workspace_bytes > 0) {
      fia_workspace = torch::empty({static_cast<int64_t>(fia_workspace_bytes)},
                                   query_group.options().dtype(torch::kByte));
    }
    const std::optional<torch::Tensor> fia_workspace_opt =
        fia_workspace.defined() ? std::optional<torch::Tensor>(fia_workspace)
                                : std::nullopt;
    c10_npu::NPUStream stream = c10_npu::getCurrentNPUStream();
    auto event = std::make_shared<c10_npu::NPUEvent>(ACL_EVENT_EXTERNAL);
    event->block(stream);
    event->reset(stream);
    c10_npu::graph_task_group_begin(stream);
    xllm::kernel::npu::npu_fused_infer_attention_out(query_group,
                                                     k,
                                                     v,
                                                     no_mask,
                                                     local_block_table_opt,
                                                     q_cu_seq_lens,
                                                     local_kv_seq_lens,
                                                     group_num_heads,
                                                     num_kv_heads_,
                                                     scale_,
                                                     block_size,
                                                     /*sparse_mode=*/0,
                                                     "TND",
                                                     /*softmax_lse_flag=*/true,
                                                     fia_out,
                                                     fia_lse,
                                                     fia_workspace_opt);
    c10_npu::NPUTaskGroupHandle handle = c10_npu::graph_task_group_end(stream);
    xllm::npu::FiaGraphTask task;
    task.output = fia_out;
    task.softmax_lse = fia_lse;
    task.query = query_group;
    task.key = k;
    task.value = v;
    task.block_table = local_block_table_opt;
    task.workspace = fia_workspace;
    task.num_heads = group_num_heads;
    task.num_key_value_heads = num_kv_heads_;
    task.scale = scale_;
    task.block_size = block_size;
    task.sparse_mode = 0;
    task.dcp_size = static_cast<int32_t>(dcp_size_);
    task.dcp_rank = static_cast<int32_t>(dcp_rank_);
    task.input_layout = "TND";
    task.softmax_lse_flag = true;
    task.handle = handle;
    task.event = std::move(event);
    graph_context->fia_tasks.emplace_back(std::move(task));
  } else {
    const auto fia_result =
        xllm::kernel::npu::npu_fused_infer_attention(query_group,
                                                     k,
                                                     v,
                                                     no_mask,
                                                     local_block_table_opt,
                                                     q_cu_seq_lens,
                                                     local_kv_seq_lens,
                                                     group_num_heads,
                                                     num_kv_heads_,
                                                     scale_,
                                                     block_size,
                                                     /*sparse_mode=*/0,
                                                     "TND",
                                                     /*softmax_lse_flag=*/true);
    fia_out = std::get<0>(fia_result);
    fia_lse = std::get<1>(fia_result);
  }
  torch::Tensor partial_out = fia_out.to(torch::kFloat32);
  torch::Tensor partial_lse = fia_lse.to(torch::kFloat32);
  if (capturing) {
    detail::normalize_zero_dcp_partials_for_graph(partial_out,
                                                  partial_lse,
                                                  attn_metadata.kv_seq_lens,
                                                  dcp_rank_,
                                                  block_size);
  } else {
    normalize_zero_dcp_partials(partial_out, partial_lse, local_kv_seq_lens);
  }

  const torch::Tensor all_partial_out =
      dcp_group_->allgather_base_sync(partial_out);
  const torch::Tensor all_partial_lse =
      dcp_group_->allgather_base_sync(partial_lse);
  const torch::Tensor merged_out =
      detail::merge_dcp_partials(all_partial_out, all_partial_lse);
  const int64_t head_begin = static_cast<int64_t>(dcp_rank_) * num_heads_;
  const torch::Tensor local_out =
      merged_out.slice(1, head_begin, head_begin + num_heads_);
  output.copy_(local_out.to(output.scalar_type()));
}

void AttentionImpl::dcp_chunked_prefill_forward(
    torch::Tensor& query,
    torch::Tensor& key,
    torch::Tensor& value,
    torch::Tensor& output,
    const torch::Tensor& k_cache,
    const std::optional<torch::Tensor>& v_cache,
    const AttentionMetadata& attn_metadata) {
  CHECK(dcp_group_ != nullptr) << "DCP chunked prefill requires a DCP group.";
  CHECK_EQ(dcp_group_->world_size(), dcp_size_)
      << "DCP process group size does not match attention DCP size.";
  CHECK_EQ(dcp_group_->rank(), dcp_rank_)
      << "DCP process group rank does not match attention DCP rank.";
  CHECK(!attn_metadata.is_spec_verify)
      << "DCP chunked prefill does not support speculative decode attention.";
  CHECK(!attn_metadata.paged_attention_tiling_data.defined())
      << "DCP chunked prefill does not support graph-captured attention.";
  CHECK(v_cache.has_value() && v_cache.value().defined())
      << "DCP chunked prefill requires a defined V cache.";
  CHECK(attn_metadata.block_table.defined())
      << "DCP chunked prefill requires a paged KV block table.";
  CHECK(attn_metadata.fia_attn_mask.defined())
      << "DCP chunked prefill requires a causal attention mask.";

  query = query.view({-1, num_heads_, head_size_});
  output = output.view({-1, num_heads_, head_size_});
  key = key.view({-1, num_kv_heads_, head_size_});
  value = value.view({-1, num_kv_heads_, head_size_});

  const int64_t token_count = query.size(0);
  const std::vector<int64_t>& global_kv_seq_lens =
      attn_metadata.kv_seq_lens_host_vec;
  const std::vector<int64_t> q_cu_seq_lens =
      detail::validate_dcp_chunked_lengths(attn_metadata.q_cu_seq_lens_host_vec,
                                           global_kv_seq_lens,
                                           token_count);

  const int64_t block_size = k_cache.size(1);
  const int64_t group_num_heads = num_heads_ * static_cast<int64_t>(dcp_size_);
  CHECK_EQ(group_num_heads % num_kv_heads_, 0)
      << "DCP gathered Q heads must preserve the GQA ratio.";
  CHECK_EQ(key.size(0), token_count);
  CHECK_EQ(value.size(0), token_count);

  // Gather the query heads across the DCP group so every partial covers the
  // full head-group; each rank slices back its own head range after merge.
  const torch::Tensor query_group =
      parallel_state::gather(query, dcp_group_, 1);
  CHECK_EQ(query_group.dim(), 3);
  CHECK_EQ(query_group.size(0), token_count);
  CHECK_EQ(query_group.size(1), group_num_heads);
  CHECK_EQ(query_group.size(2), head_size_);

  // Current/diagonal part: the current chunk attends to its own KV with a
  // causal mask. The raw key/value projections are identical on every DCP rank
  // (KV heads are replicated within the group), so this part is not sharded.
  const auto current_result = xllm::kernel::npu::npu_fused_infer_attention(
      query_group,
      key,
      value,
      std::make_optional(attn_metadata.fia_attn_mask),
      /*block_table=*/std::nullopt,
      q_cu_seq_lens,
      /*actual_seq_lengths_kv=*/q_cu_seq_lens,
      group_num_heads,
      num_kv_heads_,
      scale_,
      /*block_size=*/0,
      /*sparse_mode=*/3,
      "TND",
      /*softmax_lse_flag=*/true);
  torch::Tensor current_out = std::get<0>(current_result).to(torch::kFloat32);
  torch::Tensor current_lse = std::get<1>(current_result).to(torch::kFloat32);
  validate_dcp_partial_shape(current_out,
                             current_lse,
                             token_count,
                             group_num_heads,
                             head_size_,
                             "DCP chunked current partial shape mismatch.");

  // Context part: the current chunk attends to the previously-cached context
  // KV, which is DCP-sharded round-robin over blocks. Each rank computes a
  // partial over only its local context shard (no mask, full history visible).
  const std::vector<int64_t> context_lens =
      detail::compute_dcp_context_lens(q_cu_seq_lens, global_kv_seq_lens);
  const int64_t head_begin = static_cast<int64_t>(dcp_rank_) * num_heads_;
  if (std::all_of(context_lens.begin(),
                  context_lens.end(),
                  [](int64_t context_len) { return context_len == 0; })) {
    const torch::Tensor local_out =
        current_out.slice(1, head_begin, head_begin + num_heads_);
    CHECK_EQ(local_out.size(0), token_count);
    CHECK_EQ(local_out.size(1), num_heads_);
    CHECK_EQ(local_out.size(2), head_size_);
    output.copy_(local_out.to(output.scalar_type()));
    return;
  }

  const std::vector<int64_t> local_context_lens =
      detail::compute_dcp_local_kv_seq_lens(
          context_lens, dcp_size_, dcp_rank_, block_size);
  const torch::Tensor local_block_table =
      parallel_state::select_dcp_local_block_table(
          attn_metadata.block_table, dcp_size_, dcp_rank_);
  detail::validate_dcp_chunked_block_table(
      local_block_table, static_cast<int64_t>(q_cu_seq_lens.size()));

  const torch::Tensor k = k_cache.view({k_cache.size(0), k_cache.size(1), -1});
  const torch::Tensor v = v_cache.value().view(
      {v_cache.value().size(0), v_cache.value().size(1), -1});
  const auto context_result = xllm::kernel::npu::npu_fused_infer_attention(
      query_group,
      k,
      v,
      /*atten_mask=*/std::nullopt,
      std::make_optional(local_block_table),
      q_cu_seq_lens,
      local_context_lens,
      group_num_heads,
      num_kv_heads_,
      scale_,
      block_size,
      /*sparse_mode=*/0,
      "TND",
      /*softmax_lse_flag=*/true);
  torch::Tensor context_out = std::get<0>(context_result).to(torch::kFloat32);
  torch::Tensor context_lse = std::get<1>(context_result).to(torch::kFloat32);
  validate_dcp_partial_shape(context_out,
                             context_lse,
                             token_count,
                             group_num_heads,
                             head_size_,
                             "DCP chunked context partial shape mismatch.");
  detail::normalize_zero_dcp_chunked_partials(
      context_out, context_lse, local_context_lens, q_cu_seq_lens);

  // Merge context shards from all ranks with the (replicated) current part in a
  // single online-softmax reduction. The formula is mathematically equivalent
  // to softmax over the complete key set. FIA emits each partial output in the
  // query dtype (BF16/FP16) before this fp32 merge, so results match a single
  // monolithic FIA call only up to that partial-output quantization (on the
  // order of a few 1e-3), not bitwise. The current part is identical on every
  // rank, so contributing it once (from this rank) is correct.
  const torch::Tensor all_context_out =
      dcp_group_->allgather_base_sync(context_out);
  const torch::Tensor all_context_lse =
      dcp_group_->allgather_base_sync(context_lse);
  const torch::Tensor stacked_out =
      torch::cat({all_context_out, current_out.unsqueeze(0)}, 0);
  const torch::Tensor stacked_lse =
      torch::cat({all_context_lse, current_lse.unsqueeze(0)}, 0);
  const torch::Tensor merged_out =
      detail::merge_dcp_partials(stacked_out, stacked_lse);

  CHECK_EQ(merged_out.dim(), 3);
  CHECK_EQ(merged_out.size(0), token_count);
  CHECK_EQ(merged_out.size(1), group_num_heads);
  CHECK_EQ(merged_out.size(2), head_size_);
  const torch::Tensor local_out =
      merged_out.slice(1, head_begin, head_begin + num_heads_);
  CHECK_EQ(local_out.size(0), token_count);
  CHECK_EQ(local_out.size(1), num_heads_);
  CHECK_EQ(local_out.size(2), head_size_);
  output.copy_(local_out.to(output.scalar_type()));
}

void AttentionImpl::decoder_forward(torch::Tensor& query,
                                    torch::Tensor& output,
                                    const torch::Tensor& k_cache,
                                    const std::optional<torch::Tensor>& v_cache,
                                    const AttentionMetadata& attn_metadata) {
  query = query.view({-1, num_heads_, head_size_});
  output = output.view({-1, num_heads_, head_size_});

  if (dcp_size_ > 1) {
    dcp_decoder_forward(query, output, k_cache, v_cache, attn_metadata);
    return;
  }

  query = query.view({-1, 1, num_heads_, head_size_});
  output = output.view({-1, 1, num_heads_, head_size_});

  torch::Tensor kv_seq_lens;
  torch::Tensor block_table = attn_metadata.block_table;
  torch::Tensor tiling_data = attn_metadata.paged_attention_tiling_data;
  if (attn_metadata.use_expanded_decode_for_spec_verify_attention) {
    block_table = attn_metadata.expanded_block_table;
    tiling_data = attn_metadata.expanded_paged_attention_tiling_data;
    if (attn_metadata.expanded_kv_seq_lens_host.defined()) {
      kv_seq_lens = attn_metadata.expanded_kv_seq_lens_host;
    } else {
      kv_seq_lens = attn_metadata.expanded_kv_seq_lens;
    }
  } else if (attn_metadata.kv_seq_lens_host.defined()) {
    kv_seq_lens = attn_metadata.kv_seq_lens_host;
  } else {
    // Fallback if host tensor isn't prepared.
    kv_seq_lens = attn_metadata.kv_seq_lens;
  }

  if (tiling_data.defined()) {
    // Use CustomPagedAttention for ACL graph mode to avoid .to(kCPU) operations
    xllm::kernel::npu::batch_decode_acl_graph(query,
                                              k_cache,
                                              v_cache.value_or(torch::Tensor()),
                                              scale_,
                                              block_table,
                                              kv_seq_lens,
                                              tiling_data,
                                              output);
  } else {
    // Standard PagedAttention path
    xllm::kernel::npu::batch_decode(query,
                                    k_cache,
                                    v_cache.value_or(torch::Tensor()),
                                    scale_,
                                    block_table,
                                    kv_seq_lens,
                                    output);
  }
}

}  // namespace layer
}  // namespace xllm
