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

#include "core/kernels/npu/fia_graph_task_batch.h"

#include <glog/logging.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include <exception>
#include <string_view>
#include <utility>

namespace xllm::npu {
namespace {

size_t argument_index(const c10::FunctionSchema& schema,
                      std::string_view name) {
  const auto index = schema.argumentIndexWithName(name);
  CHECK(index.has_value()) << "FIA schema has no argument " << name;
  return static_cast<size_t>(*index);
}

}  // namespace

FiaGraphTaskBatch::FiaGraphTaskBatch(int64_t num_heads,
                                     int64_t num_kv_heads,
                                     double scale,
                                     int64_t block_size,
                                     bool use_v2,
                                     std::vector<int64_t> query_ends,
                                     std::vector<int64_t> kv_lengths)
    : operation_(c10::Dispatcher::singleton().findSchemaOrThrow(
          use_v2 ? "npu::npu_fused_infer_attention_score_v2"
                 : "npu::npu_fused_infer_attention_score",
          "out")),
      kv_lengths_(c10::IValue(std::move(kv_lengths)).toIntList()) {
  const auto& schema = operation_.schema();
  defaults_.reserve(schema.arguments().size());
  for (const auto& argument : schema.arguments()) {
    defaults_.emplace_back(argument.default_value().value_or(c10::IValue()));
  }
  auto set = [&](std::string_view name, c10::IValue value) {
    defaults_[argument_index(schema, name)] = std::move(value);
  };
  set(use_v2 ? "actual_seq_qlen" : "actual_seq_lengths", std::move(query_ends));
  set(use_v2 ? "actual_seq_kvlen" : "actual_seq_lengths_kv", kv_lengths_);
  set(use_v2 ? "num_query_heads" : "num_heads", num_heads);
  set("num_key_value_heads", num_kv_heads);
  set(use_v2 ? "softmax_scale" : "scale", scale);
  set("block_size", block_size);
  set("input_layout", "TND");
  set("sparse_mode", int64_t{0});
  query_index_ = argument_index(schema, "query");
  key_index_ = argument_index(schema, "key");
  value_index_ = argument_index(schema, "value");
  block_table_index_ = argument_index(schema, "block_table");
  workspace_index_ = argument_index(schema, "workspace");
  output_index_ = argument_index(schema, "out");
}

void FiaGraphTaskBatch::invoke(const torch::jit::Stack& arguments) const {
  // Boxed dispatch consumes its stack. Retain the capture-time argument
  // objects and copy their native references, not Python tensor descriptors.
  torch::jit::Stack stack(arguments);
  operation_.callBoxed(&stack);
}

int64_t FiaGraphTaskBatch::capture(torch::Tensor query,
                                   torch::Tensor key,
                                   torch::Tensor value,
                                   torch::Tensor block_table,
                                   torch::Tensor workspace,
                                   torch::Tensor output,
                                   torch::Tensor lse) {
  torch::jit::Stack arguments(defaults_);
  arguments[query_index_] = std::move(query);
  arguments[key_index_] = std::move(key);
  arguments[value_index_] = std::move(value);
  arguments[block_table_index_] = std::move(block_table);
  arguments[workspace_index_] = std::move(workspace);
  arguments[output_index_] =
      std::vector<torch::Tensor>{std::move(output), std::move(lse)};
  c10_npu::NPUStream stream = c10_npu::getCurrentNPUStream();
  auto event = std::make_shared<c10_npu::NPUEvent>(ACL_EVENT_EXTERNAL);
  event->block(stream);
  event->reset(stream);
  c10_npu::graph_task_group_begin(stream);
  try {
    invoke(arguments);
  } catch (const std::exception&) {
    // Close the third-party capture boundary and preserve the operator error.
    c10_npu::graph_task_group_end(stream);
    throw;
  }
  c10_npu::NPUTaskGroupHandle handle = c10_npu::graph_task_group_end(stream);
  const int64_t index = size();
  tasks_.emplace_back(Task{std::move(arguments), handle, std::move(event)});
  return index;
}

void FiaGraphTaskBatch::update(int64_t begin,
                               int64_t end,
                               const std::vector<int64_t>& kv_lengths) {
  CHECK_GE(begin, 0);
  CHECK_LE(begin, end);
  CHECK_LE(end, size());
  CHECK_EQ(kv_lengths.size(), kv_lengths_.size());
  for (size_t index = 0; index < kv_lengths.size(); ++index) {
    kv_lengths_.set(index, kv_lengths[index]);
  }
  c10_npu::NPUStream stream = c10_npu::getCurrentNPUStream();
  for (int64_t index = begin; index < end; ++index) {
    Task& task = tasks_[static_cast<size_t>(index)];
    c10_npu::graph_task_update_begin(stream, task.handle);
    invoke(task.arguments);
    c10_npu::graph_task_update_end(stream);
    task.event->record(stream);
  }
}

int64_t FiaGraphTaskBatch::size() const {
  return static_cast<int64_t>(tasks_.size());
}

}  // namespace xllm::npu
