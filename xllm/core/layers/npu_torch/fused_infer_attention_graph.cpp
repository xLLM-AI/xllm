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

#include "layers/npu_torch/fused_infer_attention_graph.h"

#include <glog/logging.h>

#include <memory>
#include <utility>

#include "kernels/npu/npu_ops_api.h"
#include "platform/npu/acl_graph_task_update_context.h"

torch::Tensor xllm::layer::detail::run_fused_infer_attention_graph(
    const std::shared_ptr<xllm::npu::AclGraphTaskUpdateContext>& graph_context,
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& block_table,
    const std::vector<int64_t>& actual_seq_lengths,
    const std::vector<int64_t>& actual_seq_lengths_kv,
    int64_t num_key_value_heads,
    double scale,
    int32_t dcp_size,
    int32_t dcp_rank,
    xllm::npu::FusedInferAttentionGraphBranch branch,
    torch::Tensor& output) {
  CHECK(graph_context != nullptr && graph_context->capturing)
      << "FIA graph update can only be registered during capture";
  const int64_t block_size = key.size(1);
  const int64_t num_heads = query.size(1);
  const bool softmax_lse_flag = dcp_size > 1;

  const std::vector<int64_t> workspace_kv_seq_lens =
      dcp_size > 1 ? std::vector<int64_t>(actual_seq_lengths_kv.size(),
                                          block_table.size(1) * block_size)
                   : actual_seq_lengths_kv;

  const xllm::npu::FusedInferAttentionWorkspaceSignature workspace_signature{
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
      .actual_seq_lengths_kv = workspace_kv_seq_lens,
      .num_heads = num_heads,
      .num_key_value_heads = num_key_value_heads,
      .block_size = block_size,
      .scale = scale,
      .softmax_lse_flag = softmax_lse_flag,
  };
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
            workspace_kv_seq_lens,
            num_heads,
            num_key_value_heads,
            scale,
            block_size,
            softmax_lse_flag);
    CHECK(workspace.defined()) << "FIA graph workspace must be defined";
    graph_context->fused_infer_attention_workspace_signature =
        workspace_signature;
    graph_context->fused_infer_attention_workspace = workspace;
  }
  torch::Tensor softmax_lse =
      softmax_lse_flag
          ? torch::empty({query.size(0), num_heads, 1},
                         query.options().dtype(torch::kFloat32))
          : torch::empty({0}, query.options().dtype(torch::kFloat32));
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
                                                          softmax_lse,
                                                          softmax_lse_flag);
  c10_npu::NPUTaskGroupHandle handle = c10_npu::graph_task_group_end(stream);

  xllm::npu::FusedInferAttentionGraphTask task;
  task.output = output;
  task.softmax_lse = softmax_lse;
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
  task.dcp_size = dcp_size;
  task.dcp_rank = dcp_rank;
  task.branch = branch;
  task.capture_order = graph_context->next_capture_order++;
  task.handle = handle;
  task.event = std::move(event);
  graph_context->fused_infer_attention_tasks.emplace_back(std::move(task));
  return softmax_lse;
}
