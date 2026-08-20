/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
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
  c10_npu::NPUTaskGroupHandle handle{};
  std::shared_ptr<c10_npu::NPUEvent> event;
};

// Graph-capture record for a DCP fused-infer-attention (FIA) call. The tensors
// (query/key/value/output/softmax_lse) keep stable device addresses across
// replays; the per-step varying host scalar actual_seq_lengths_kv (the
// DCP-local KV lengths) is recomputed at replay from the step's global
// kv_seq_lens via compute_dcp_local_kv_seq_lens(dcp_size, dcp_rank,
// block_size), so only these static DCP params are stored here. `workspace` is
// a caller-owned buffer sized for the largest local-KV envelope at capture and
// reused (same address) on every replay, because aclnn's default workspace is
// function-local and would leave a dangling address in the captured graph. See
// update_graph_tasks in acl_graph_executor_impl.cpp.
struct FiaGraphTask {
  torch::Tensor output;
  torch::Tensor softmax_lse;
  torch::Tensor query;
  torch::Tensor key;
  torch::Tensor value;
  std::optional<torch::Tensor> atten_mask;
  std::optional<torch::Tensor> block_table;
  torch::Tensor workspace;
  int64_t num_heads = 0;
  int64_t num_key_value_heads = 0;
  double scale = 1.0;
  int64_t block_size = 0;
  int64_t sparse_mode = 0;
  int32_t dcp_size = 1;
  int32_t dcp_rank = 0;
  std::string input_layout;
  bool softmax_lse_flag = false;
  c10_npu::NPUTaskGroupHandle handle{};
  std::shared_ptr<c10_npu::NPUEvent> event;
};

class AclGraphTaskUpdateContext final {
 public:
  void begin_capture() {
    capturing = true;
    causal_conv1d_tasks.clear();
    fia_tasks.clear();
  }

  void end_capture() { capturing = false; }

  bool capturing = false;
  std::vector<CausalConv1dGraphTask> causal_conv1d_tasks;
  std::vector<FiaGraphTask> fia_tasks;
};

}  // namespace xllm::npu
