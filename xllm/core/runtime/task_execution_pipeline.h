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

#pragma once

#include <folly/futures/Future.h>

#include <deque>
#include <optional>

#include "core/framework/model/causal_lm.h"
#include "core/framework/sampling/sampler.h"
#include "core/platform/device.h"
#include "core/runtime/executor.h"
#include "core/runtime/forward_params.h"
#include "core/runtime/slot_buffer.h"
#include "core/util/concurrent_queue.h"
#include "core/util/threadpool.h"

namespace xllm {

// Model contracts established before task admission. Persistent
// storage is allocated before KV budgeting; KV may be populated afterwards.
struct LlmTaskCapacity {
  uint32_t slot_count = 1;
  ModelInputCapacity model;
  uint32_t max_kv_seq_len = 0;
  uint32_t max_positions = 0;
  uint32_t block_size = 0;
  uint32_t vocab_size = 0;
  uint32_t max_unique_tokens = 0;
  uint32_t max_top_logprobs = 0;
  torch::ScalarType parameter_dtype = torch::kFloat32;
  bool enable_mla = false;
};

struct TaskSubmission {
  Status status;
  uint64_t task_id = 0;
};

struct TaskResult {
  Status status;
  ForwardOutput output;
  uint64_t task_id = 0;
};

// One or two Slots with prepared eager execution. State calls serialize
// Prepare/Consume; the Launch thread submits model work in task order.
// Prepare/Consume may overlap Launch for another Slot. The state executor,
// model, executor and KV vector outlive the pipeline; the owner prevents
// external calls racing with destruction.
class TaskExecutionPipeline final {
 public:
  static Status create(ThreadPool& state_executor,
                       CausalLM& model,
                       Executor& executor,
                       std::vector<KVCache>& kv_caches,
                       const LlmTaskCapacity& capacity,
                       std::unique_ptr<TaskExecutionPipeline>& output);
  ~TaskExecutionPipeline();
  TaskExecutionPipeline(const TaskExecutionPipeline&) = delete;
  TaskExecutionPipeline& operator=(const TaskExecutionPipeline&) = delete;

  // Waits for Prepare Ack only. On success no caller input storage is still
  // borrowed, including unpacked transport storage. Negative tokens encode
  // -(previous sampling output row + 1) with scheduler overlap.
  // Backpressure and validation return before accepting a Task.
  TaskSubmission submit(const ForwardInput& input);
  // Only the oldest accepted TaskId may be consumed. Other IDs are rejected
  // without removing a completion. Slot retirement precedes
  // Future completion; successful outputs own independent, ready CPU values.
  // An unrequested result continues to occupy the Slot.
  folly::Future<TaskResult> take_result_async(uint64_t task_id);
  // The Step/GetLast adapter consumes the same FIFO without another ID queue.
  // Empty FIFO returns INVALID_ARGUMENT. Successful results carry their ID.
  folly::Future<TaskResult> take_result_async();
  uint64_t pinned_bytes() const;
  uint64_t device_bytes() const;

 private:
  friend class TaskExecutionPipelineInputTest;
  static Status validate_input(const ForwardInput& input);

  struct Step {
    uint64_t id = 0;
    uint32_t sample_rows = 0;
  };

  struct Slot {
    Step step;
    Step expected_producer;
    bool is_warmup = false;
    StreamEventPtr input_ready;
    StreamEventPtr output_ready;
    std::unique_ptr<SlotBuffer> buffer;
    // Retain sampler inputs and outputs until the result fence completes.
    SamplingParameters sampling;
    SampleOutput sample_output;
    ModelOutput model_output;
    torch::Tensor logits;
  };

  struct SlotTicket {
    uint32_t slot_id = 0;
    uint64_t task_id = 0;
  };

  TaskExecutionPipeline(ThreadPool& state_executor,
                        CausalLM& model,
                        Executor& executor,
                        std::vector<KVCache>& kv_caches,
                        LlmTaskCapacity capacity);
  Status validate(const Slot& slot,
                  const ModelInputHostView& model,
                  const ModelInputBatch& batch,
                  const SamplingParameters& sampling) const;
  Status prepare(uint32_t slot_id, const ForwardInput& input);
  void launch(uint32_t slot_id);
  ForwardOutput consume(uint32_t slot_id);
  void discard(uint32_t slot_id);
  void release_outputs(Slot& slot);
  folly::Future<TaskResult> take_result_impl(
      std::optional<uint64_t> expected_task_id);
  void launch_loop();
  SlotTicket wait_completed_front();
  void check_external_thread() const;

  CausalLM& model_;
  Executor& executor_;
  std::vector<KVCache>& kv_caches_;
  LlmTaskCapacity capacity_;
  Device device_;
  Stream prepare_stream_;
  Stream task_stream_;
  Stream result_stream_;
  Sampler sampler_;
  // Device reads/writes use only task_stream_. Consume never changes this.
  torch::Tensor previous_tokens_;
  // Prepare and Launch have separate thread-owned descriptors. IDs advance
  // only after successful Prepare; they include empty-output tasks.
  Step accepted_tail_;
  Step published_tail_;
  std::vector<std::unique_ptr<Slot>> slots_;

  ThreadPool& state_executor_;
  ConcurrentQueue<SlotTicket> execution_;
  ConcurrentQueue<SlotTicket> completed_;
  std::thread launch_thread_;
  std::thread::id state_thread_id_;
  // Accessed exclusively on the state executor until the destructor barrier.
  uint64_t next_task_id_ = 1;
  std::deque<SlotTicket> accepted_;
};

}  // namespace xllm
