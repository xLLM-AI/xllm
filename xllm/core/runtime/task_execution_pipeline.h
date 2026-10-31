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

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

#include "core/framework/model/causal_lm.h"
#include "core/framework/sampling/sampling_params.h"
#include "core/framework/speculative/draft_proposal.h"
#include "core/kv_cache/transfer/host_transfer/transfer.h"
#include "core/platform/device.h"
#include "core/runtime/executor.h"
#include "core/runtime/forward_params.h"
#include "core/runtime/slot_buffer.h"
#include "core/runtime/vlm_forward_params.h"
#include "core/util/concurrent_queue.h"
#include "core/util/threadpool.h"

namespace xllm {

class ProcessGroup;
class SpecCostProfileStore;
struct SpecCalibrationCommand;
struct SpecCalibrationGraphCoverage;

// The owner establishes these model contracts before admission. Persistent
// storage is allocated before KV budgeting; KV may be populated afterwards.
struct LlmTaskCapacity {
  uint32_t slot_count = 1;
  ModelInputCapacity model;
  uint32_t max_kv_seq_len = 0;
  uint32_t max_positions = 0;
  // Scheduler page width. DCP keeps logical slots until Python localizes them
  // against each rank's physical KV cache.
  uint32_t logical_block_size = 0;
  uint32_t vocab_size = 0;
  uint32_t max_unique_tokens = 0;
  uint32_t max_top_logprobs = 0;
  uint32_t hidden_size = 0;
  torch::ScalarType parameter_dtype = torch::kFloat32;
  bool chunked_prefill = false;
  bool enable_mla = false;
  // Only a PD Prefill owner may attach per-step KV PUSH mappings.
  bool enable_kv_push = false;
  uint32_t dp_size = 1;
  uint32_t dp_rank = 0;
  // Borrowed groups outlive the pipeline. Orthogonal CP requires consensus
  // along both axes, including greedy sampling, before publishing tokens.
  ProcessGroup* sampling_group = nullptr;
  ProcessGroup* cp_sampling_group = nullptr;
  // DP-local expanded target token limit; zero disables Graph execution.
  uint32_t max_graph_batch_size = 0;
};

using TaskKVPush =
    std::function<folly::SemiFuture<bool>(const std::vector<TransferKVInfo>&,
                                          ModelParamsView&)>;
using TaskKVLoad = std::function<std::optional<HostKVLoadHandle>(uint64_t)>;
using TaskKVFence = std::function<void(const Stream&)>;

enum class TaskInputDomain : uint8_t { LLM, VLM };

// Borrowed resources outlive the pipeline. PUSH binds a fresh layer
// synchronizer to the invocation and resolves KV resources at Launch time.
struct TaskModel {
  CausalLM& model;
  Executor& executor;
  std::vector<KVCache>& kv_caches;
  TaskKVPush push_kv = {};
  TaskKVLoad load_kv = {};
  TaskKVFence fence_kv = {};
  std::function<void(const ModelParamsView&)> restore_request_state = {};
  TaskInputDomain input_domain = TaskInputDomain::LLM;
};

class ProcessGroup;
class WorkerImpl;
enum class SpeculativeTaskKind : uint8_t { MTP, DFLASH, DFLASH2, DSPARK };

struct SpeculativeTaskCapacity {
  LlmTaskCapacity common;
  SpeculativeTaskKind kind = SpeculativeTaskKind::MTP;
  uint32_t num_speculative_tokens = 0;
  uint32_t context_hidden_size = 0;
  int32_t mask_token_id = -1;
  uint32_t selector_top_k = 0;
  uint32_t index_topk = 0;
  bool reuse_topk = false;
  // Restore committed draft KV from verifier hidden states before reuse.
  bool refresh_target_prefix = false;
  bool is_probabilistic_draft = false;
  // Reuse the refreshed final row for the next admitted first proposal.
  bool reuse_prefix_refresh_output = false;
  bool unified_graph = false;
  bool adaptive_target_verify = false;
  bool adaptive_draft_step = false;
  bool sparse_dsa_target = false;
  bool hybrid_recurrent_target = false;
  bool draft_swa_cache = false;
  double adaptive_min_gain = 0.0;
  // Startup calibration installs identical immutable costs on sampling ranks.
  std::shared_ptr<const SpecCostProfileStore> adaptive_costs;
  bool calibration_leader = false;
  bool startup_calibration = false;
};

struct MtpContextStorage;
struct MtpContextView;
struct MtpContextPublication;

struct TaskSubmission {
  Status status;
  uint64_t task_id = 0;
};

struct TaskResult {
  Status status;
  ForwardOutput output;
  uint64_t task_id = 0;
};

// One or two Slots with ordinary or speculative eager/ACL graph execution.
// Both modes share one queue, launch thread and prepare/task/result stream set.
// Cross-round speculative context belongs to this pipeline, not a retiring
// Slot. State calls
// serialize Prepare/Consume; the Launch thread submits model work in task
// order. Prepare/Consume may overlap Launch for another Slot. The state
// executor, model, executor and KV vector outlive the pipeline; the owner
// prevents external calls racing with destruction.
class TaskExecutionPipeline final {
 public:
  static Status create(ThreadPool& state_executor,
                       CausalLM& model,
                       Executor& executor,
                       std::vector<KVCache>& kv_caches,
                       const LlmTaskCapacity& capacity,
                       std::unique_ptr<TaskExecutionPipeline>& output);
  static Status create(ThreadPool& state_executor,
                       TaskModel model,
                       const LlmTaskCapacity& capacity,
                       std::unique_ptr<TaskExecutionPipeline>& output);
  static Status create(ThreadPool& state_executor,
                       TaskModel target,
                       TaskModel draft,
                       const SpeculativeTaskCapacity& capacity,
                       std::unique_ptr<TaskExecutionPipeline>& output);
  // Reuse this same queue for workers whose execution state is owned by the
  // WorkerImpl. The worker supplies only prepare/execute hooks; admission,
  // FIFO ordering, launch and result retirement remain in this class.
  static Status create(ThreadPool& state_executor,
                       TaskModel target,
                       WorkerImpl& worker,
                       const LlmTaskCapacity& capacity,
                       std::unique_ptr<TaskExecutionPipeline>& output);

  ~TaskExecutionPipeline();
  TaskExecutionPipeline(const TaskExecutionPipeline&) = delete;
  TaskExecutionPipeline& operator=(const TaskExecutionPipeline&) = delete;

  // Waits for Prepare Ack only. On success no caller input storage is still
  // borrowed, including unpacked transport storage. Negative tokens encode
  // -(previous sampling output row + 1) with scheduler overlap.
  // Backpressure and validation return before accepting a Task.
  TaskSubmission submit(const LlmForwardInput& input);
  TaskSubmission submit(const VlmForwardInput& input);
  // Publish all previously accepted model work to the hierarchy producer
  // stream. Waits only for Launch submission, without consuming any results.
  void fence_kv_cache();
  std::optional<std::string> spec_calibration(SpecCalibrationCommand command);
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
  friend struct TaskPipelineTestPeer;
  friend class SlotBuffer;
  static Status context_create(MtpContextStorage& pool,
                               uint32_t capacity,
                               std::unique_ptr<MtpContextView>& output);
  static Status context_prepare(MtpContextView& view,
                                std::span<const int32_t> embedding_ids,
                                std::span<const std::string> request_ids,
                                bool read_published_state,
                                const Stream& stream);
  static Status context_prepare_prefill(
      MtpContextView& view,
      std::span<const int32_t> embedding_ids,
      std::span<const std::string> request_ids,
      std::span<const int32_t> extra_token_ids,
      std::span<const uint32_t> published_sequence_rows,
      const Stream& stream);
  static Status context_prepare_decode(MtpContextView& view,
                                       std::span<const int32_t> embedding_ids,
                                       std::span<const std::string> request_ids,
                                       std::span<const int32_t> bootstrap_rows,
                                       const Stream& stream);
  static Status context_prepare_impl(
      MtpContextView& view,
      std::span<const int32_t> embedding_ids,
      std::span<const std::string> request_ids,
      std::span<const uint32_t> published_sequence_rows,
      bool read_published_state,
      const Stream& stream,
      std::span<const int32_t> bootstrap_rows);
  static void context_gather(MtpContextView& view);
  static void context_gather_state(MtpContextView& view);
  static void context_gather_prefix_hidden(MtpContextView& view);
  static void context_publish(MtpContextView& view);
  static void context_advance(MtpContextView& view,
                              const torch::Tensor& accepted_tokens,
                              const torch::Tensor& accepted_lengths,
                              const torch::Tensor& target_hidden);
  static void context_release(MtpContextView& view);
  static uint64_t context_device_bytes(const MtpContextView& view);

  static Status create_mtp_context(uint32_t capacity,
                                   uint32_t hidden_size,
                                   torch::ScalarType dtype,
                                   const torch::Device& device,
                                   std::unique_ptr<MtpContextStorage>& output);
  static uint64_t mtp_context_bytes(const MtpContextStorage& cache);
  void reserve_draft_prefix_context(MtpContextStorage& context,
                                    uint32_t capacity);
  friend class TaskExecutionPipelineInputTest;
  static Status validate_input(const LlmForwardInput& input,
                               const LlmTaskCapacity& capacity,
                               bool speculative = false,
                               bool recurrent = false);
  static std::vector<int64_t> graph_batch_sizes(
      const LlmTaskCapacity& capacity);
  static uint32_t graph_batch_size(const LlmForwardInput& input,
                                   std::span<const int64_t> batch_sizes);
  Status warmup_slot_graphs(const LlmForwardInput& input,
                            uint32_t padded_batch_size);

  struct Step {
    uint64_t id = 0;
    uint32_t sample_rows = 0;
  };

  struct SpeculativeSlot;
  struct Sampling {
    SamplingParameters params;
    SampleOutput output;
    SampleOutput sampled;
  };
  struct Slot {
    Step step;
    Step expected_producer;
    bool is_warmup = false;
    std::vector<TransferKVInfo> transfer_kv_infos;
    std::optional<HostKVLoadHandle> kv_load;
    StreamEventPtr input_ready;
    StreamEventPtr output_ready;
    std::unique_ptr<SlotBuffer> buffer;
    std::unique_ptr<SpeculativeSlot> speculative;
    // Retain sampler inputs and outputs until the result fence completes.
    SamplingParameters sampling;
    SampleOutput sample_output;
    ModelOutput model_output;
    torch::Tensor logits;
    // Used only by the WorkerImpl-owned execution mode. These values stay in
    // the normal task slot until the completed FIFO retires it.
    LlmForwardInput worker_input;
    VlmForwardInput worker_vlm_input;
    // Packed host views borrow this storage even if worker preparation builds
    // a different contiguous H2D buffer.
    torch::Tensor worker_input_host_buffer;
    std::optional<ForwardOutput> worker_output;
  };

  struct SlotTicket {
    uint32_t slot_id = 0;
    uint64_t task_id = 0;
  };

  TaskExecutionPipeline(ThreadPool& state_executor,
                        CausalLM& model,
                        Executor& executor,
                        std::vector<KVCache>& kv_caches,
                        LlmTaskCapacity capacity,
                        TaskKVPush push_kv,
                        TaskKVLoad load_kv,
                        TaskKVFence fence_kv,
                        WorkerImpl* worker_pipeline = nullptr,
                        TaskInputDomain input_domain = TaskInputDomain::LLM);
  Status validate(const Slot& slot, const LlmForwardInput& input) const;
  Status prepare(uint32_t slot_id, const LlmForwardInput& input);
  Status prepare_worker(uint32_t slot_id, const LlmForwardInput& input);
  Status prepare_worker(uint32_t slot_id, const VlmForwardInput& input);
  void launch(uint32_t slot_id);
  void observe_execution_memory(uint32_t slot_id);
  ForwardOutput consume(uint32_t slot_id);
  void discard(uint32_t slot_id);
  void release_outputs(Slot& slot);
  folly::Future<TaskResult> take_result_impl(
      std::optional<uint64_t> expected_task_id);
  void launch_loop();
  void start();
  SlotTicket wait_completed_front();
  void check_external_thread() const;

  Status initialize_speculative();
  Status warmup_speculative_graphs(const LlmForwardInput& input);
  Status prepare_speculative(uint32_t slot_id, const LlmForwardInput& input);
  void launch_speculative(uint32_t slot_id);
  ForwardOutput consume_speculative(uint32_t slot_id);
  void discard_speculative(uint32_t slot_id);
  uint64_t speculative_pinned_bytes() const;
  uint64_t speculative_device_bytes() const;
  const SampleOutput& sample(Sampling& sampling, torch::Tensor& logits);
  Status validate_input(SpeculativeSlot& slot, const LlmForwardInput& input);
  Status plan_sampling(SpeculativeSlot& slot, const LlmForwardInput& input);
  uint32_t draft_calls(const SpeculativeSlot& slot) const;
  Status validate_bootstrap(const LlmForwardInput& input) const;
  void prepare_bootstrap(SpeculativeSlot& slot, const LlmForwardInput& input);
  void apply_bootstrap(SpeculativeSlot& slot);
  void warmup_draft_graphs(const LlmForwardInput& input);
  void prepare_draft_state(SpeculativeSlot& slot, uint32_t step, bool warmup);
  void prepare_prefix_refresh_state(SpeculativeSlot& slot, bool warmup);
  bool prepares_mtp_context() const;
  void prepare_mtp_context(SpeculativeSlot& slot);
  void publish_mtp_context(SpeculativeSlot& slot);
  void bind_mtp_publication(SpeculativeSlot& slot);
  void launch_prefix_refresh(SpeculativeSlot& slot);
  void prepare_empty_shard(SpeculativeSlot& slot, uint64_t batch_id);
  void launch_empty_shard(SpeculativeSlot& slot);
  void launch_prefill(SpeculativeSlot& slot);
  void launch_decode(SpeculativeSlot& slot);
  bool launch_unified_block(SpeculativeSlot& slot);
  void launch_block_draft(SpeculativeSlot& slot);
  void finish_fixed_verify(SpeculativeSlot& slot);
  void generate_adaptive_dspark_proposals(SpeculativeSlot& slot,
                                          const torch::Tensor& anchors);
  void launch_packed_verify(SpeculativeSlot& slot);
  Status plan_adaptive_verify(SpeculativeSlot& slot,
                              const LlmForwardInput& input);
  Status plan_dp_adaptive_dspark(SpeculativeSlot& slot,
                                 const LlmForwardInput& input);
  Status warmup_packed_verify_graphs(const LlmForwardInput& input);
  Status warmup_adaptive_unified_graphs(const LlmForwardInput& input);
  std::vector<SpecCalibrationGraphCoverage> adaptive_verify_graph_coverage()
      const;
  void bind_packed_verify_views(SpeculativeSlot& slot);
  void bind_packed_verify_graph_input(SpeculativeSlot& slot,
                                      bool suppress_kv_writes);
  std::optional<std::string> spec_calibration_impl(
      const SpecCalibrationCommand& command);
  std::optional<std::string> draft_step_calibration_capabilities() const;
  void consume_adaptive_verify(SpeculativeSlot& slot);
  void write_block_context(SpeculativeSlot& slot, bool prefill);
  // skip_sampling_group: rank-local argmax or a prior in-group broadcast.
  void synchronize_tokens(torch::Tensor& tokens, bool skip_sampling_group);
  // Broadcasts the sample result tensors across the TP/CP sampling groups.
  void synchronize_samples(const TokenResultTensors& result,
                           SamplingMode sampling_mode);
  static Status plan_parallel(const ParallelInput& input,
                              uint32_t dp_size,
                              uint32_t dp_rank,
                              uint32_t local_tokens,
                              bool local_decode,
                              ParallelInput& output,
                              bool& run_models,
                              bool& decode);
  bool block_draft() const;
  bool block_draft_graphs() const;
  void restore_recurrent_state(const SpeculativeSlot& slot,
                               const ModelParamsView& model_params);
  Status prepare_recurrent_state(SpeculativeSlot& slot,
                                 const LlmForwardInput& input);
  uint32_t context_hidden_size() const;
  void release_outputs(SpeculativeSlot& slot);

  std::unique_ptr<SpeculativeTaskCapacity> speculative_capacity_;
  struct AdaptiveVerifyState;
  std::unique_ptr<AdaptiveVerifyState> adaptive_verify_;
  // Embedding-row keyed; shared by both fixed and adaptive recurrent tasks.
  torch::Tensor recurrent_accepted_lengths_;
  WorkerImpl* worker_pipeline_ = nullptr;
  TaskInputDomain input_domain_ = TaskInputDomain::LLM;
  std::unique_ptr<TaskModel> draft_;
  torch::ScalarType hidden_dtype_ = torch::kFloat32;
  std::unique_ptr<MtpContextStorage> context_;
  // Only Prepare owns the pool and last-admitted pointer. A lease remains
  // immutable until its producer and consumer Slots have both retired.
  std::vector<std::shared_ptr<MtpContextPublication>> context_publications_;
  std::shared_ptr<MtpContextPublication> last_context_publication_;
  CausalLM& model_;
  Executor& executor_;
  std::vector<KVCache>& kv_caches_;
  TaskKVPush push_kv_;
  TaskKVLoad load_kv_;
  TaskKVFence fence_kv_;
  std::function<void(const ModelParamsView&)> restore_request_state_;
  LlmTaskCapacity capacity_;
  Device device_;
  Stream prepare_stream_;
  Stream task_stream_;
  Stream result_stream_;
  // Device reads/writes use only task_stream_. Consume never changes this.
  torch::Tensor previous_tokens_;
  // Prepare and Launch have separate thread-owned descriptors. IDs advance
  // only after successful Prepare; they include empty-output tasks.
  Step accepted_tail_;
  Step published_tail_;
  std::vector<std::unique_ptr<Slot>> slots_;
  std::vector<int64_t> graph_batch_sizes_;
  // Sorted buckets captured for every Slot through ordinary warmup Tasks.
  // Only these buckets are eligible for serving on every DP rank.
  std::vector<int64_t> captured_graph_batch_sizes_;

  ThreadPool& state_executor_;
  ConcurrentQueue<SlotTicket> execution_;
  ConcurrentQueue<SlotTicket> completed_;
  std::thread launch_thread_;
  std::mutex launch_mutex_;
  std::condition_variable launch_cv_;
  uint64_t launched_task_id_ = 0;
  std::thread::id state_thread_id_;
  // Accessed exclusively on the state executor until the destructor barrier.
  uint64_t next_task_id_ = 1;
  std::deque<SlotTicket> accepted_;
};

}  // namespace xllm
