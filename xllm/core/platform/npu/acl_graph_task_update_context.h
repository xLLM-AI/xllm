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

#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wattributes"
#endif

#include "torch_npu/csrc/core/npu/NPUEvent.h"
#include "torch_npu/csrc/core/npu/NPUGraph.h"

#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

namespace xllm::npu {

constexpr int64_t kCausalConv1dGraphPadSlotId = -1;
constexpr int64_t kCausalConv1dActivationSilu = 1;
constexpr int64_t kCausalConv1dRunModeForward = 0;
constexpr int64_t kCausalConv1dRunModeUpdate = 1;

enum class CausalConv1dGraphBranch {
  kDecode,
  kSpecVerify,
};

enum class FusedInferAttentionGraphBranch {
  kDecode,
  kSpecVerify,
};

struct CausalConv1dGraphTask {
  torch::Tensor output;
  torch::Tensor x;
  torch::Tensor weight;
  torch::Tensor conv_state;
  std::optional<torch::Tensor> bias;
  int64_t activation_mode = kCausalConv1dActivationSilu;
  int64_t pad_slot_id = kCausalConv1dGraphPadSlotId;
  int64_t run_mode = kCausalConv1dRunModeUpdate;
  CausalConv1dGraphBranch branch = CausalConv1dGraphBranch::kDecode;
  uint64_t capture_order = 0;
  c10_npu::NPUTaskGroupHandle handle{};
  std::shared_ptr<c10_npu::NPUEvent> event;
};

class FusedInferAttentionWorkspaceSignature {
 public:
  torch::ScalarType query_dtype;
  torch::ScalarType key_dtype;
  torch::ScalarType value_dtype;
  torch::ScalarType block_table_dtype;
  c10::DeviceIndex device_index;
  std::vector<int64_t> query_shape;
  std::vector<int64_t> key_shape;
  std::vector<int64_t> value_shape;
  std::vector<int64_t> block_table_shape;
  std::vector<int64_t> actual_seq_lengths;
  std::vector<int64_t> actual_seq_lengths_kv;
  int64_t num_heads;
  int64_t num_key_value_heads;
  int64_t block_size;
  double scale;

  bool operator==(const FusedInferAttentionWorkspaceSignature&) const = default;
};

struct FusedInferAttentionGraphTask {
  torch::Tensor output;
  torch::Tensor softmax_lse;
  torch::Tensor query;
  torch::Tensor key;
  torch::Tensor value;
  torch::Tensor block_table;
  torch::Tensor workspace;
  std::vector<int64_t> actual_seq_lengths;
  int64_t num_heads = 0;
  int64_t num_key_value_heads = 0;
  double scale = 0.0;
  int64_t block_size = 0;
  FusedInferAttentionGraphBranch branch =
      FusedInferAttentionGraphBranch::kDecode;
  uint64_t capture_order = 0;
  c10_npu::NPUTaskGroupHandle handle{};
  std::shared_ptr<c10_npu::NPUEvent> event;
};

class AclGraphTaskUpdateContext final {
 public:
  void begin_capture() {
    capturing = true;
    next_capture_order = 0;
    causal_conv1d_tasks.clear();
    fused_infer_attention_tasks.clear();
    fused_infer_attention_workspace_signature.reset();
    fused_infer_attention_workspace = torch::Tensor();
  }

  void end_capture() { capturing = false; }

  bool capturing = false;
  uint64_t next_capture_order = 0;
  std::vector<CausalConv1dGraphTask> causal_conv1d_tasks;
  std::vector<FusedInferAttentionGraphTask> fused_infer_attention_tasks;
  std::optional<FusedInferAttentionWorkspaceSignature>
      fused_infer_attention_workspace_signature;
  torch::Tensor fused_infer_attention_workspace;
};

}  // namespace xllm::npu
