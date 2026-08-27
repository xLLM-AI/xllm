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

#include "attention.h"

#include <memory>
#include <utility>
#include <vector>

#include "core/platform/npu/acl_graph_task_update_context.h"
#include "kernels/npu/npu_ops_api.h"
#include "kernels/ops_api.h"

namespace {
std::vector<int64_t> make_decode_actual_seq_lengths(int64_t num_tokens) {
  std::vector<int64_t> actual_seq_lengths;
  actual_seq_lengths.reserve(static_cast<size_t>(num_tokens));
  for (int64_t token_idx = 0; token_idx < num_tokens; ++token_idx) {
    actual_seq_lengths.emplace_back(token_idx + 1);
  }
  return actual_seq_lengths;
}

xllm::npu::FusedInferAttentionWorkspaceSignature
make_fused_infer_attention_workspace_signature(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& block_table,
    const std::vector<int64_t>& actual_seq_lengths,
    const std::vector<int64_t>& actual_seq_lengths_kv,
    int64_t num_heads,
    int64_t num_key_value_heads,
    double scale,
    int64_t block_size) {
  return xllm::npu::FusedInferAttentionWorkspaceSignature{
      .query_dtype = query.scalar_type(),
      .key_dtype = key.scalar_type(),
      .value_dtype = value.scalar_type(),
      .block_table_dtype = block_table.scalar_type(),
      .device_index = query.device().index(),
      .query_shape = query.sizes().vec(),
      .key_shape = key.sizes().vec(),
      .value_shape = value.sizes().vec(),
      .block_table_shape = block_table.sizes().vec(),
      .actual_seq_lengths = actual_seq_lengths,
      .actual_seq_lengths_kv = actual_seq_lengths_kv,
      .num_heads = num_heads,
      .num_key_value_heads = num_key_value_heads,
      .block_size = block_size,
      .scale = scale,
  };
}

void run_fused_infer_attention_graph(
    const std::shared_ptr<xllm::npu::AclGraphTaskUpdateContext>& graph_context,
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& block_table,
    const std::vector<int64_t>& actual_seq_lengths,
    const std::vector<int64_t>& actual_seq_lengths_kv,
    int64_t num_heads,
    int64_t num_key_value_heads,
    double scale,
    int64_t block_size,
    xllm::npu::FusedInferAttentionGraphBranch branch,
    torch::Tensor& output) {
  CHECK(graph_context != nullptr && graph_context->capturing)
      << "FIA graph update can only be registered during capture";

  const xllm::npu::FusedInferAttentionWorkspaceSignature workspace_signature =
      make_fused_infer_attention_workspace_signature(query,
                                                     key,
                                                     value,
                                                     block_table,
                                                     actual_seq_lengths,
                                                     actual_seq_lengths_kv,
                                                     num_heads,
                                                     num_key_value_heads,
                                                     scale,
                                                     block_size);
  torch::Tensor workspace = graph_context->fused_infer_attention_workspace;
  if (workspace.defined()) {
    CHECK(graph_context->fused_infer_attention_workspace_signature.has_value());
    CHECK(graph_context->fused_infer_attention_workspace_signature.value() ==
          workspace_signature)
        << "FIA graph layers in one bucket require different workspaces";
  } else {
    workspace =
        xllm::kernel::npu::npu_fused_infer_attention_decode_get_max_workspace(
            query,
            key,
            value,
            block_table,
            actual_seq_lengths,
            actual_seq_lengths_kv,
            num_heads,
            num_key_value_heads,
            scale,
            block_size);
    CHECK(workspace.defined()) << "FIA graph workspace must be defined";
    graph_context->fused_infer_attention_workspace_signature =
        workspace_signature;
    graph_context->fused_infer_attention_workspace = workspace;
  }
  torch::Tensor softmax_lse = torch::empty({0}, query.options());
  c10_npu::NPUStream stream = c10_npu::getCurrentNPUStream();
  auto event = std::make_shared<c10_npu::NPUEvent>(ACL_EVENT_EXTERNAL);
  event->block(stream);
  event->reset(stream);

  c10_npu::graph_task_group_begin(stream);
  xllm::kernel::npu::npu_fused_infer_attention_decode_out(query,
                                                          key,
                                                          value,
                                                          block_table,
                                                          actual_seq_lengths,
                                                          actual_seq_lengths_kv,
                                                          num_heads,
                                                          num_key_value_heads,
                                                          scale,
                                                          block_size,
                                                          workspace,
                                                          output,
                                                          softmax_lse);
  c10_npu::NPUTaskGroupHandle handle = c10_npu::graph_task_group_end(stream);

  xllm::npu::FusedInferAttentionGraphTask task;
  task.output = output;
  task.softmax_lse = std::move(softmax_lse);
  task.query = query;
  task.key = key;
  task.value = value;
  task.block_table = block_table;
  task.workspace = std::move(workspace);
  task.actual_seq_lengths = actual_seq_lengths;
  task.num_heads = num_heads;
  task.num_key_value_heads = num_key_value_heads;
  task.scale = scale;
  task.block_size = block_size;
  task.branch = branch;
  task.capture_order = graph_context->next_capture_order++;
  task.handle = handle;
  task.event = std::move(event);
  graph_context->fused_infer_attention_tasks.emplace_back(std::move(task));
}
}  // namespace

namespace xllm {
namespace layer {

AttentionImpl::AttentionImpl(int64_t num_heads,
                             int64_t head_size,
                             float scale,
                             int64_t num_kv_heads,
                             int64_t sliding_window,
                             bool enable_fia_decode)
    : num_heads_(num_heads),
      head_size_(head_size),
      num_kv_heads_(num_kv_heads),
      sliding_window_(sliding_window),
      scale_(scale),
      enable_fia_decode_(enable_fia_decode) {
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

  const bool only_prefill =
      attn_metadata.is_prefill || attn_metadata.is_chunked_prefill;

  torch::Tensor k_cache = kv_cache.get_k_cache();
  std::optional<torch::Tensor> v_cache = kv_cache.get_v_cache();

  if (!attn_metadata.prefill_without_cache) {
    xllm::kernel::ReshapePagedCacheParams reshape_paged_cache_params;
    reshape_paged_cache_params.key = key.view({-1, num_kv_heads_, head_size_});
    reshape_paged_cache_params.value =
        value.view({-1, num_kv_heads_, head_size_});
    reshape_paged_cache_params.k_cache = k_cache;
    reshape_paged_cache_params.v_cache = v_cache;
    reshape_paged_cache_params.slot_mapping = attn_metadata.slot_mapping;
    xllm::kernel::reshape_paged_cache(reshape_paged_cache_params);
  }

  if (attn_metadata.expanded_decode.enabled) {
    decoder_forward(query, output, k_cache, v_cache, attn_metadata);
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

void AttentionImpl::decoder_forward(torch::Tensor& query,
                                    torch::Tensor& output,
                                    const torch::Tensor& k_cache,
                                    const std::optional<torch::Tensor>& v_cache,
                                    const AttentionMetadata& attn_metadata) {
  query = query.view({-1, 1, num_heads_, head_size_});
  output = output.view({-1, 1, num_heads_, head_size_});

  torch::Tensor kv_seq_lens;
  torch::Tensor block_table = attn_metadata.block_table;
  torch::Tensor tiling_data = attn_metadata.paged_attention_tiling_data;
  if (attn_metadata.expanded_decode.enabled) {
    block_table = attn_metadata.expanded_decode.block_table;
    tiling_data = attn_metadata.expanded_decode.paged_attention_tiling_data;
    if (attn_metadata.expanded_decode.kv_seq_lens_host.defined()) {
      kv_seq_lens = attn_metadata.expanded_decode.kv_seq_lens_host;
    } else {
      kv_seq_lens = attn_metadata.expanded_decode.kv_seq_lens;
    }
  } else if (attn_metadata.kv_seq_lens_host.defined()) {
    kv_seq_lens = attn_metadata.kv_seq_lens_host;
  } else {
    // Fallback if host tensor isn't prepared.
    kv_seq_lens = attn_metadata.kv_seq_lens;
  }

  const bool use_fia_graph_decode =
      enable_fia_decode_ &&
      (!attn_metadata.is_spec_verify || attn_metadata.expanded_decode.enabled);
  if (tiling_data.defined() && !use_fia_graph_decode) {
    xllm::kernel::npu::batch_decode_acl_graph(query,
                                              k_cache,
                                              v_cache.value_or(torch::Tensor()),
                                              scale_,
                                              block_table,
                                              kv_seq_lens,
                                              tiling_data,
                                              output);
    return;
  }

  if (!tiling_data.defined() && !enable_fia_decode_) {
    xllm::kernel::npu::batch_decode(query,
                                    k_cache,
                                    v_cache.value_or(torch::Tensor()),
                                    scale_,
                                    block_table,
                                    kv_seq_lens,
                                    output);
    return;
  }

  std::vector<int64_t> expanded_kv_seq_lens;
  const std::vector<int64_t>* kv_seq_lens_vec =
      &attn_metadata.kv_seq_lens_host_vec;
  if (attn_metadata.expanded_decode.enabled) {
    expanded_kv_seq_lens.reserve(
        attn_metadata.expanded_decode.kv_seq_lens_host_vec.size());
    for (int32_t kv_seq_len :
         attn_metadata.expanded_decode.kv_seq_lens_host_vec) {
      expanded_kv_seq_lens.emplace_back(kv_seq_len);
    }
    kv_seq_lens_vec = &expanded_kv_seq_lens;
  }

  CHECK(v_cache.has_value() && v_cache->defined())
      << "FIA decode requires a value cache";
  CHECK(block_table.defined()) << "FIA decode requires a block table";
  torch::Tensor query_tnd = query.view({-1, num_heads_, head_size_});
  torch::Tensor output_tnd = output.view({-1, num_heads_, head_size_});
  CHECK_EQ(static_cast<int64_t>(kv_seq_lens_vec->size()), query_tnd.size(0))
      << "FIA decode KV lengths must match query tokens";

  torch::Tensor key_view = k_cache.view({k_cache.size(0), k_cache.size(1), -1});
  torch::Tensor value_view =
      v_cache->view({v_cache->size(0), v_cache->size(1), -1});
  std::vector<int64_t> actual_q_lens =
      make_decode_actual_seq_lengths(query_tnd.size(0));

  if (tiling_data.defined()) {
    const xllm::npu::FusedInferAttentionGraphBranch graph_branch =
        attn_metadata.expanded_decode.enabled
            ? xllm::npu::FusedInferAttentionGraphBranch::kSpecVerify
            : xllm::npu::FusedInferAttentionGraphBranch::kDecode;
    run_fused_infer_attention_graph(attn_metadata.acl_graph_task_update_context,
                                    query_tnd,
                                    key_view,
                                    value_view,
                                    block_table,
                                    actual_q_lens,
                                    *kv_seq_lens_vec,
                                    num_heads_,
                                    num_kv_heads_,
                                    scale_,
                                    k_cache.size(1),
                                    graph_branch,
                                    output_tnd);
  } else {
    xllm::kernel::npu::npu_fused_infer_attention_decode_out_cached(
        query_tnd,
        key_view,
        value_view,
        block_table,
        actual_q_lens,
        *kv_seq_lens_vec,
        num_heads_,
        num_kv_heads_,
        scale_,
        /*block_size=*/k_cache.size(1),
        output_tnd);
  }
}

}  // namespace layer
}  // namespace xllm
