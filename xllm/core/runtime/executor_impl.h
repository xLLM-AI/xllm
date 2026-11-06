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

#pragma once

#include <torch/types.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "common/macros.h"
#include "core/kv_cache/storage/kv_cache.h"
#include "framework/model/causal_lm.h"
#include "framework/model/model_output.h"
#include "framework/model/model_params_view.h"
#include "options.h"
#include "platform/stream.h"
#include "platform/stream_event.h"

namespace xllm {

class SlotBuffer;
class ProcessGroup;
struct TokenResultTensors;

// Borrowed Slot bindings for one complete adaptive DSpark graph. The Target
// inputs/hidden states have physical G rows; results retain [B, draft_width+1].
// Retirement, not graph-cache lifetime, protects these buffers from reuse.
struct AdaptiveDSparkGraphTask {
  SlotBuffer& target;
  SlotBuffer& draft;
  const TokenResultTensors& result;
  torch::Tensor context_slots;
};

struct SpecVerifyGraphTaskSignal {
  int64_t linear_state_id = 0;
  int64_t num_accepted_tokens = 0;
  int64_t spec_width = 0;
  int64_t block_table_width = 0;
  int64_t base_kv_seq_len = 0;
  int64_t max_kv_seq_len = 0;
};

class ExecutorImpl {
 public:
  virtual ~ExecutorImpl() = default;

  virtual bool supports_adaptive_unified_graph() const { return false; }

  virtual void warmup_adaptive_unified_graph(
      ExecutorImpl& /*draft_executor*/,
      const AdaptiveDSparkGraphTask& /*task*/,
      ProcessGroup* /*sampling_group*/) {
    LOG(FATAL) << "Executor does not support adaptive unified graph capture.";
  }

  virtual torch::Tensor run_adaptive_unified_graph(
      ExecutorImpl& /*draft_executor*/,
      const AdaptiveDSparkGraphTask& /*task*/,
      ProcessGroup* /*sampling_group*/) {
    LOG(FATAL) << "Executor does not support adaptive unified graph replay.";
    return {};
  }

  virtual bool supports_prepared_attention_metadata() const { return false; }
  virtual void set_dp_process_group(ProcessGroup* /*group*/) {}
  virtual void prepare_attention_metadata(std::vector<KVCache>& /*kv_caches*/,
                                          ModelParamsView& /*params*/) {
    LOG(FATAL) << "Executor does not support prepared attention metadata.";
  }

  // Regular warmup Tasks capture final Slot inputs before serving uses them.
  virtual void warmup_prepared_graph(const torch::Tensor& /*tokens*/,
                                     const torch::Tensor& /*positions*/,
                                     std::vector<KVCache>& /*kv_caches*/,
                                     const ModelParamsView& /*params*/) {
    LOG(FATAL) << "Executor does not support prepared graph capture.";
  }

  virtual ModelOutput run(const torch::Tensor& tokens,
                          const torch::Tensor& positions,
                          std::vector<KVCache>& kv_caches,
                          const ModelParamsView& params) = 0;

  virtual void prepare_graph_input(const torch::Tensor& /*tokens*/,
                                   const torch::Tensor& /*positions*/,
                                   std::vector<KVCache>& /*kv_caches*/,
                                   const ModelParamsView& /*params*/) {}

  virtual bool prepare_static_mtp_graph_tasks(
      const SpecVerifyGraphTaskSignal& /*signal*/,
      const Stream& /*signal_stream*/) {
    return false;
  }
};

}  // namespace xllm
