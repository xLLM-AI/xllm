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

#include "core/runtime/task_execution_pipeline_speculative.h"

#include <glog/logging.h>
#include <glog/vlog_is_on.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <future>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <string_view>
#include <utility>
#include <vector>

#include "core/framework/allocator/virtual_memory/physical_page_pool.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/model_config.h"
#include "core/framework/parallel_state/process_group.h"
#include "core/framework/speculative/dspark/dspark_adaptive_cost_cache.h"
#include "core/framework/speculative/dspark/dspark_sps_cost_model.h"
#include "core/framework/speculative/metrics.h"
#include "core/kernels/npu/tilelang/tilelang_ops_api.h"
#include "core/runtime/executor_impl_factory.h"
#include "core/runtime/task_execution_pipeline.h"
#include "core/runtime/unified_mtp_worker_impl.h"
#include "core/runtime/worker_impl.h"
#include "tests/core/runtime/task_pipeline_test_peer.h"
#include "worker.pb.h"

namespace xllm {

class PhysicalPagePoolTestPeer final {
 public:
  static void set_memory_query(PhysicalPagePool& pool,
                               std::function<int64_t()> query) {
    std::lock_guard<std::mutex> lock(pool.mtx_);
    pool.free_memory_query_ = std::move(query);
  }
};

namespace {

class AdaptiveTraceSink final : public google::LogSink {
 public:
  explicit AdaptiveTraceSink(int32_t level)
      : previous_level_(
            google::SetVLOGLevel("task_execution_pipeline_speculative",
                                 level)) {
    google::AddLogSink(this);
  }

  ~AdaptiveTraceSink() override {
    google::RemoveLogSink(this);
    google::SetVLOGLevel("task_execution_pipeline_speculative",
                         previous_level_);
  }

  void send(google::LogSeverity /*severity*/,
            const char* /*full_filename*/,
            const char* /*base_filename*/,
            int /*line*/,
            const google::LogMessageTime& /*time*/,
            const char* message,
            size_t message_length) override {
    constexpr std::string_view prefix = "DSpark confidence diagnostic: ";
    const std::string_view text(message, message_length);
    if (!text.starts_with(prefix)) {
      return;
    }
    std::lock_guard lock(mutex_);
    records_.emplace_back(text.substr(prefix.size()));
  }

  std::vector<nlohmann::json> snapshots() {
    std::lock_guard lock(mutex_);
    std::vector<nlohmann::json> result;
    result.reserve(records_.size());
    for (const auto& record : records_) {
      result.emplace_back(nlohmann::json::parse(record));
    }
    return result;
  }

 private:
  int32_t previous_level_;
  std::mutex mutex_;
  std::vector<std::string> records_;
};

TEST(UnifiedBlockAdmissionTest, HonorsGraphSwitchAndExpandedBatchLimit) {
  SpeculativeTaskCapacity capacity;
  capacity.kind = SpeculativeTaskKind::DSPARK;
  capacity.unified_graph = true;
  capacity.num_speculative_tokens = 8;
  LlmForwardInput input;
  input.input_params.meta.batch_forward_type = BatchForwardType::DECODE;
  // Worker task_capacity leaves max_graph_batch_size zero if
  // enable_graph=false.
  EXPECT_FALSE(supports_unified_block_graph(capacity, 1, input));
  capacity.common.max_graph_batch_size = 8 * 9;
  EXPECT_TRUE(supports_unified_block_graph(capacity, 8, input));
  capacity.adaptive_dspark = true;
  EXPECT_FALSE(supports_unified_block_graph(capacity, 8, input));
  capacity.adaptive_dspark = false;
  EXPECT_FALSE(supports_unified_block_graph(capacity, 9, input));
  EXPECT_FALSE(supports_unified_block_graph(capacity, 0, input));
  EXPECT_FALSE(supports_unified_block_graph(
      capacity, std::numeric_limits<uint32_t>::max(), input));
  capacity.unified_graph = false;
  EXPECT_FALSE(supports_unified_block_graph(capacity, 1, input));
}

TEST(UnifiedBlockAdmissionTest, RejectsNonGreedyAndTransformedRequests) {
  SpeculativeTaskCapacity capacity;
  capacity.kind = SpeculativeTaskKind::DFLASH2;
  capacity.unified_graph = true;
  capacity.num_speculative_tokens = 7;
  capacity.common.max_graph_batch_size = 8;
  LlmForwardInput input;
  input.input_params.meta.batch_forward_type = BatchForwardType::DECODE;
  EXPECT_TRUE(supports_unified_block_graph(capacity, 1, input));
  input.sampling_params.logprobs = true;
  EXPECT_FALSE(supports_unified_block_graph(capacity, 1, input));
  input.sampling_params.logprobs = false;
  input.sampling_params.sampling_mode = SamplingMode::MIXED;
  EXPECT_FALSE(supports_unified_block_graph(capacity, 1, input));
  input.sampling_params.sampling_mode = SamplingMode::GREEDY;
  for (auto* transform : {&input.sampling_params.top_k,
                          &input.sampling_params.min_p,
                          &input.sampling_params.logits_bias}) {
    *transform = torch::tensor({1});
    EXPECT_FALSE(supports_unified_block_graph(capacity, 1, input));
    *transform = torch::Tensor();
  }
  input.input_params.meta.is_graph_warmup = true;
  EXPECT_FALSE(supports_unified_block_graph(capacity, 1, input));
  input.input_params.meta.is_graph_warmup = false;
  input.input_params.meta.batch_forward_type = BatchForwardType::PREFILL;
  EXPECT_FALSE(supports_unified_block_graph(capacity, 1, input));
}

std::array<uintptr_t, 10> adaptive_graph_binding(
    const DSparkAdaptiveGraphInput& input,
    int64_t graph_rows) {
  return {reinterpret_cast<uintptr_t>(input.arena.data_ptr()),
          reinterpret_cast<uintptr_t>(input.control.data_ptr()),
          reinterpret_cast<uintptr_t>(input.anchors.data_ptr()),
          reinterpret_cast<uintptr_t>(input.base_positions.data_ptr()),
          reinterpret_cast<uintptr_t>(input.base_kv_lengths.data_ptr()),
          reinterpret_cast<uintptr_t>(input.base_block_table.data_ptr()),
          reinterpret_cast<uintptr_t>(input.offsets.data_ptr()),
          reinterpret_cast<uintptr_t>(input.owners.data_ptr()),
          reinterpret_cast<uintptr_t>(input.columns.data_ptr()),
          static_cast<uintptr_t>(graph_rows)};
}

enum class PreparedTestEvent : uint8_t {
  METADATA_PREPARED,
  GRAPH_WARMED,
  MODEL_FORWARDED,
  CONTEXT_WRITTEN,
};

// Encode the consumed token in the hidden state. The real pipeline must patch
// draft inputs and publish accepted context correctly to keep counting upward.
class CountingModel final : public CausalLM {
 public:
  explicit CountingModel(const torch::Device& device)
      : options_(torch::TensorOptions().device(device).dtype(torch::kFloat32)) {
  }

  ModelOutput forward(const torch::Tensor& tokens,
                      const torch::Tensor& positions,
                      std::vector<KVCache>& /*kv_caches*/,
                      const ModelParamsView& params) override {
    {
      std::lock_guard<std::mutex> lock(prepared_mutex);
      prepared_events.emplace_back(PreparedTestEvent::MODEL_FORWARDED);
    }
    forward_inputs.emplace_back(tokens.clone());
    forward_positions.emplace_back(positions.clone());
    forward_slots.emplace_back(params.attention.device.new_cache_slots.clone());
    forward_tables.emplace_back(params.attention.device.block_tables.clone());
    forward_counts.emplace_back(params.parallel.dp_global_token_nums);
    forward_raw_counts.emplace_back(params.parallel.raw_dp_global_token_nums);
    forward_embeddings.emplace_back(
        params.embedding.input_embedding.defined()
            ? params.embedding.input_embedding.clone()
            : torch::Tensor());
    forward_had_sync.emplace_back(params.parallel.layer_synchronizer !=
                                  nullptr);
    auto hidden = (block_draft ? positions : tokens).to(options_).view({-1, 1});
    if (borrowed_hidden_storage.defined()) {
      auto borrowed = borrowed_hidden_storage.narrow(0, 0, hidden.size(0));
      borrowed.copy_(hidden);
      hidden = borrowed;
    }
    // Two columns exercise block drafts whose captured context is wider than
    // either model's hidden state. MTP ignores these auxiliary rows.
    auto auxiliary = torch::cat({hidden, hidden + 100}, /*dim=*/1);
    if (params.parallel.layer_synchronizer != nullptr) {
      CHECK(!params.enable_graph);
      CHECK(params.parallel.layer_synchronizer->record_event(
          /*layer_index=*/0, options_.device().index()));
    }
    if (after_forward) {
      after_forward();
    }
    return ModelOutput(hidden, torch::Tensor(), auxiliary);
  }

  ModelOutput write_context_kv(const torch::Tensor& target_hidden,
                               const torch::Tensor& positions,
                               const torch::Tensor& device_cache_slots,
                               std::vector<KVCache>& kv_caches,
                               const ModelParamsView& params) override {
    {
      std::lock_guard<std::mutex> lock(prepared_mutex);
      bool bound = !prepared_cache_bindings.empty() &&
                   prepared_cache_bindings.size() == kv_caches.size();
      for (size_t layer = 0; bound && layer < kv_caches.size(); ++layer) {
        bound = prepared_cache_bindings[layer] ==
                std::array<uintptr_t, 2>{
                    reinterpret_cast<uintptr_t>(
                        kv_caches[layer].get_k_cache().data_ptr()),
                    reinterpret_cast<uintptr_t>(
                        kv_caches[layer].get_v_cache().data_ptr())};
      }
      context_cache_bound.emplace_back(bound);
      prepared_events.emplace_back(PreparedTestEvent::CONTEXT_WRITTEN);
      if (require_bound_context_cache && params.enable_graph &&
          params.meta.batch_forward_type.is_decode()) {
        EXPECT_TRUE(bound)
            << "Borrowed context capture preceded Draft KV binding";
      }
    }
    context_hidden.emplace_back(target_hidden.clone());
    context_positions.emplace_back(positions.clone());
    context_slots.emplace_back(device_cache_slots.clone());
    context_warmup.emplace_back(params.meta.is_graph_warmup);
    context_bindings.push_back(
        {reinterpret_cast<uintptr_t>(target_hidden.data_ptr()),
         reinterpret_cast<uintptr_t>(positions.data_ptr()),
         reinterpret_cast<uintptr_t>(device_cache_slots.data_ptr()),
         static_cast<uintptr_t>(target_hidden.size(0))});
    return ModelOutput(target_hidden);
  }

  torch::Tensor logits(const torch::Tensor& hidden,
                       const torch::Tensor& selected) override {
    const int32_t call = logits_calls.fetch_add(1);
    const auto next =
        hidden.index_select(0, selected.to(torch::kInt64)).to(torch::kInt64) +
        logit_increments[call % logit_increments.size()];
    auto output = torch::full({selected.numel(), 32}, -20.0, options_);
    output.scatter_(/*dim=*/1, next, /*value=*/20.0);
    return output;
  }

  torch::Tensor dspark_markov_bias(
      const torch::Tensor& previous_tokens) override {
    markov_inputs.emplace_back(previous_tokens.clone());
    auto bias = torch::full({previous_tokens.numel(), 32}, -100.0, options_);
    const int64_t increment = markov_increments[(markov_inputs.size() - 1) %
                                                markov_increments.size()];
    const auto next = row_markov_increments.defined()
                          ? previous_tokens + row_markov_increments.to(
                                                  previous_tokens.options())
                          : previous_tokens + increment;
    bias.scatter_(1, next.view({-1, 1}), 100.0);
    return bias;
  }

  bool has_dspark_confidence_head() const override { return confidence_head; }
  torch::Tensor dspark_confidence_probs(
      const torch::Tensor& hidden,
      const torch::Tensor& previous_tokens) override {
    confidence_hidden.emplace_back(hidden.clone());
    confidence_previous.emplace_back(previous_tokens.clone());
    return torch::full(previous_tokens.sizes(),
                       confidence_value,
                       options_.dtype(torch::kFloat32));
  }

  void load_model(std::unique_ptr<ModelLoader> /*loader*/) override {}
  torch::Device device() const override { return options_.device(); }
  const torch::TensorOptions& options() const override { return options_; }
  void prepare_expert_weight(int32_t /*layer_id*/,
                             const std::vector<int32_t>& /*experts*/) override {
  }
  void update_expert_weight(int32_t /*layer_id*/) override {}

  std::atomic<int32_t> logits_calls{0};
  std::vector<int64_t> logit_increments{1};
  torch::Tensor borrowed_hidden_storage;
  bool block_draft = false;
  std::vector<int64_t> markov_increments{1};
  bool stable_graph_outputs = false;
  bool require_bound_context_cache = false;
  std::mutex prepared_mutex;
  std::vector<std::array<uintptr_t, 2>> prepared_cache_bindings;
  std::vector<PreparedTestEvent> prepared_events;
  std::vector<bool> context_cache_bound;
  torch::Tensor row_markov_increments;
  bool confidence_head = false;
  float confidence_value = 0.0F;
  std::vector<torch::Tensor> confidence_hidden;
  std::vector<torch::Tensor> confidence_previous;
  std::atomic<uint32_t> graph_output_allocations{0};
  std::vector<std::pair<uintptr_t, int64_t>> graph_warmup_bindings;
  std::vector<std::array<uintptr_t, 10>> adaptive_warmup_bindings;
  std::vector<std::array<uintptr_t, 10>> adaptive_forward_bindings;
  std::vector<torch::Tensor> adaptive_controls;
  std::vector<torch::Tensor> markov_inputs;
  std::vector<torch::Tensor> forward_inputs;
  std::vector<torch::Tensor> forward_positions;
  std::vector<torch::Tensor> forward_slots;
  std::vector<torch::Tensor> forward_tables;
  std::vector<std::vector<int32_t>> forward_counts;
  std::vector<std::vector<int32_t>> forward_raw_counts;
  std::vector<torch::Tensor> forward_embeddings;
  std::vector<bool> forward_had_sync;
  std::function<void()> after_forward;
  std::vector<torch::Tensor> context_hidden;
  std::vector<torch::Tensor> context_positions;
  std::vector<torch::Tensor> context_slots;
  std::vector<bool> context_warmup;
  std::vector<std::array<uintptr_t, 4>> context_bindings;

 private:
  torch::TensorOptions options_;
};

class PreparedTestExecutor final : public ExecutorImpl {
 public:
  explicit PreparedTestExecutor(CausalLM* model) : model_(model) {}
  bool supports_prepared_attention_metadata() const override { return true; }
  void prepare_attention_metadata(std::vector<KVCache>& kv_caches,
                                  ModelParamsView& /*params*/) override {
    auto* counting = dynamic_cast<CountingModel*>(model_);
    if (counting == nullptr) {
      return;
    }
    std::lock_guard<std::mutex> lock(counting->prepared_mutex);
    counting->prepared_cache_bindings.clear();
    counting->prepared_cache_bindings.reserve(kv_caches.size());
    for (const auto& cache : kv_caches) {
      counting->prepared_cache_bindings.emplace_back(std::array<uintptr_t, 2>{
          reinterpret_cast<uintptr_t>(cache.get_k_cache().data_ptr()),
          reinterpret_cast<uintptr_t>(cache.get_v_cache().data_ptr())});
    }
    counting->prepared_events.emplace_back(
        PreparedTestEvent::METADATA_PREPARED);
  }
  // The model double stays eager; validate final Slot views during warmup.
  void warmup_prepared_graph(const torch::Tensor& tokens,
                             const torch::Tensor& positions,
                             std::vector<KVCache>& /*kv_caches*/,
                             const ModelParamsView& params) override {
    EXPECT_GT(tokens.numel(), 0);
    EXPECT_EQ(tokens.numel(), positions.numel());
    EXPECT_EQ(tokens.numel(), params.attention.device.new_cache_slots.numel());
    for (const int32_t count : params.parallel.raw_dp_global_token_nums) {
      EXPECT_GE(count, 0);
      EXPECT_LE(count, tokens.numel());
    }
    auto* counting = dynamic_cast<CountingModel*>(model_);
    if (counting != nullptr) {
      {
        std::lock_guard<std::mutex> lock(counting->prepared_mutex);
        counting->prepared_events.emplace_back(PreparedTestEvent::GRAPH_WARMED);
      }
      const auto& adaptive = params.graph.dspark_adaptive;
      if (adaptive.arena.defined()) {
        counting->adaptive_warmup_bindings.emplace_back(
            adaptive_graph_binding(adaptive, tokens.numel()));
        EXPECT_EQ(adaptive.arena.numel(),
                  adaptive.anchors.numel() * (12 * adaptive.draft_width + 4));
        EXPECT_EQ(adaptive.control.numel(), 1);
        EXPECT_EQ(adaptive.offsets.numel(), adaptive.anchors.numel() + 1);
        EXPECT_EQ(adaptive.base_block_table.size(0), adaptive.anchors.numel());
        EXPECT_EQ(adaptive.owners.numel(), tokens.numel());
        EXPECT_EQ(adaptive.columns.numel(), tokens.numel());
      }
      counting->graph_warmup_bindings.emplace_back(
          reinterpret_cast<uintptr_t>(tokens.data_ptr()), tokens.numel());
      if (counting->stable_graph_outputs) {
        const auto& binding = counting->graph_warmup_bindings.back();
        auto [entry, inserted] = graph_outputs_.try_emplace(binding);
        if (inserted) {
          entry->second.hidden_states =
              torch::empty({tokens.numel(), 1}, counting->options());
          entry->second.aux_hidden_states =
              torch::empty({tokens.numel(), 2}, counting->options());
          ++counting->graph_output_allocations;
        }
      }
    }
  }
  ModelOutput run(const torch::Tensor& tokens,
                  const torch::Tensor& positions,
                  std::vector<KVCache>& kv_caches,
                  const ModelParamsView& params) override {
    auto* counting = dynamic_cast<CountingModel*>(model_);
    const auto& adaptive = params.graph.dspark_adaptive;
    if (adaptive.arena.defined()) {
      // Emulate the prepared graph prehook with the actual NPU operators.
      // This eager executor double verifies pipeline data flow, not capture.
      auto arena = adaptive.arena;
      auto offsets = adaptive.offsets;
      auto packed_tokens = tokens;
      auto packed_positions = positions;
      auto lengths = params.attention.device.kv_seq_lens;
      auto slots = params.attention.device.new_cache_slots;
      auto tables = params.attention.device.block_tables;
      auto queries = params.attention.device.q_seq_lens;
      auto owners = adaptive.owners;
      auto columns = adaptive.columns;
      kernel::npu::tilelang::dspark_allocate_verify_prefixes_graph(
          arena, adaptive.control, tokens.numel(), adaptive.draft_width);
      kernel::npu::tilelang::dspark_pack_verify_inputs_graph(
          adaptive.anchors,
          arena,
          adaptive.base_positions,
          adaptive.base_kv_lengths,
          adaptive.base_block_table,
          adaptive.control,
          offsets,
          packed_tokens,
          packed_positions,
          lengths,
          slots,
          tables,
          queries,
          owners,
          columns,
          adaptive.block_size,
          adaptive.draft_width);
      if (counting != nullptr) {
        counting->adaptive_forward_bindings.emplace_back(
            adaptive_graph_binding(adaptive, tokens.numel()));
        counting->adaptive_controls.emplace_back(adaptive.control.clone());
        EXPECT_NE(std::find(counting->adaptive_warmup_bindings.begin(),
                            counting->adaptive_warmup_bindings.end(),
                            counting->adaptive_forward_bindings.back()),
                  counting->adaptive_warmup_bindings.end());
      }
    }
    ModelOutput actual = model_->forward(tokens, positions, kv_caches, params);
    if (counting == nullptr || !counting->stable_graph_outputs ||
        !params.enable_graph) {
      return actual;
    }
    const std::pair<uintptr_t, int64_t> binding{
        reinterpret_cast<uintptr_t>(tokens.data_ptr()), tokens.numel()};
    EXPECT_NE(std::find(counting->graph_warmup_bindings.begin(),
                        counting->graph_warmup_bindings.end(),
                        binding),
              counting->graph_warmup_bindings.end());
    // Emulate the fixed output addresses of an ACL graph, without pretending
    // this executor double captures or replays a real device graph.
    auto [entry, inserted] = graph_outputs_.try_emplace(binding);
    if (inserted) {
      entry->second.hidden_states = torch::empty_like(actual.hidden_states);
      entry->second.aux_hidden_states =
          torch::empty_like(actual.aux_hidden_states);
      ++counting->graph_output_allocations;
    }
    entry->second.hidden_states.copy_(actual.hidden_states);
    entry->second.aux_hidden_states.copy_(actual.aux_hidden_states);
    return entry->second;
  }

 private:
  CausalLM* model_;
  std::map<std::pair<uintptr_t, int64_t>, ModelOutput> graph_outputs_;
};

// The composite worker deliberately owns no model: the shared queue must use
// the explicit leaf TaskModel, as UnifiedMtpWorkerImpl does in production.
class PreparedTestWorker final : public WorkerImpl {
 public:
  explicit PreparedTestWorker(const torch::Device& device)
      : WorkerImpl(ParallelArgs(0, 1, 1, nullptr), device, runtime::Options()),
        release_future_(release_.get_future().share()) {}

  bool init_model(ModelContext& /*context*/) override { return false; }

  void prepare_task_pipeline_input(const LlmForwardInput& input,
                                   LlmForwardInput& prepared) override {
    auto guard = prepare_stream_->set_stream_guard();
    if (contiguous_input_) {
      prepared = input.to(device_.unwrap(), torch::kFloat32);
    } else {
      prepared = input.clone();
      prepared.token_ids = input.token_ids.to(device_.unwrap());
    }
    prepared.runtime.metadata_ready_event = prepare_stream_->record_event();
    CHECK(prepared.runtime.metadata_ready_event != nullptr);
    prepared.runtime.retained_device_tensors.emplace_back(torch::from_blob(
        new int32_t(0),
        {1},
        [this](void* data) {
          if (!finished_.load()) {
            released_early_.store(true);
          }
          delete static_cast<int32_t*>(data);
          ++released_inputs_;
        },
        torch::TensorOptions().dtype(torch::kInt32)));
  }

  std::optional<ForwardOutput> step(const LlmForwardInput& /*input*/) override {
    LOG(FATAL) << "Shared task queue must call the direct execute hook.";
    return std::nullopt;
  }

  std::optional<ForwardOutput> execute_task_pipeline(
      const LlmForwardInput& prepared) override {
    auto guard = compute_stream_->set_stream_guard();
    CHECK(compute_stream_->wait_event(prepared.runtime.metadata_ready_event));
    CHECK_EQ(aclrtLaunchHostFunc(
                 compute_stream_->get_stream()->stream(),
                 [](void* data) {
                   auto* worker = static_cast<PreparedTestWorker*>(data);
                   if (!worker->entered_signaled_.exchange(true)) {
                     worker->entered_.set_value();
                   }
                   worker->release_future_.wait_for(std::chrono::seconds(10));
                   worker->finished_.store(true);
                 },
                 this),
             ACL_SUCCESS);
    ForwardOutput output;
    if (with_tokens_) {
      output.sample_output.next_tokens = prepared.token_ids + 1;
    }
    output.ready_event = compute_stream_->record_event();
    CHECK(output.ready_event != nullptr);
    return output;
  }

  std::future<void> entered() { return entered_.get_future(); }
  void release() {
    if (!release_signaled_.exchange(true)) {
      release_.set_value();
    }
  }
  bool finished() const { return finished_.load(); }
  void without_tokens() { with_tokens_ = false; }
  void with_contiguous_input() { contiguous_input_ = true; }
  bool released_early() const { return released_early_.load(); }
  int32_t released_inputs() const { return released_inputs_.load(); }

 private:
  bool with_tokens_ = true;
  bool contiguous_input_ = false;
  std::promise<void> entered_;
  std::promise<void> release_;
  std::shared_future<void> release_future_;
  std::atomic<bool> entered_signaled_{false};
  std::atomic<bool> release_signaled_{false};
  std::atomic<bool> finished_{false};
  std::atomic<bool> released_early_{false};
  std::atomic<int32_t> released_inputs_{0};
};

// Exercise the production Unified execute hook without loading model weights.
class JsonTaskTestWorker final : public UnifiedMtpWorkerImpl {
 public:
  explicit JsonTaskTestWorker(const torch::Device& device)
      : UnifiedMtpWorkerImpl(ParallelArgs(0, 1, 1, nullptr),
                             device,
                             options(),
                             WorkerType::LLM) {}

  std::optional<ForwardOutput> step(const LlmForwardInput& input) override {
    if (!input.json_object_states.empty()) {
      observed_tokens = input.json_object_states.front().snapshot().token_ids;
    }
    if (!with_output) {
      return std::nullopt;
    }
    ForwardOutput output;
    output.sample_output.next_tokens =
        torch::tensor({next_token}, torch::kInt64);
    return output;
  }

  int64_t next_token = 0;
  bool with_output = true;
  std::vector<int32_t> observed_tokens;

 private:
  static runtime::Options options() {
    runtime::Options result;
    result.enable_schedule_overlap(true);
    return result;
  }
};

// Supply a leader's values that differ from the local draws. Downstream model
// inputs and the next round must use these values, including accepted length.
class LeaderSamples final : public ProcessGroup {
 public:
  explicit LeaderSamples(const torch::Device& device,
                         int64_t token_offset = 0,
                         int32_t world_size = 2)
      : ProcessGroup(/*rank=*/world_size - 1, world_size, device),
        values_{torch::tensor({7 + token_offset}, torch::kInt64),
                torch::tensor({12 + token_offset}, torch::kInt64),
                torch::tensor({14 + token_offset}, torch::kInt64),
                torch::tensor({{8 + token_offset, int64_t{-1}, int64_t{-1}}},
                              torch::kInt64),
                torch::tensor({18 + token_offset}, torch::kInt64),
                torch::tensor({20 + token_offset}, torch::kInt64),
                torch::tensor({{9 + token_offset, int64_t{-1}, int64_t{-1}}},
                              torch::kInt64)} {
    received_.reserve(values_.size());
  }

  void broadcast(torch::Tensor& tensor, int32_t root_rank) override {
    ASSERT_EQ(root_rank, 0);
    ASSERT_LT(next_, values_.size());
    ASSERT_EQ(tensor.numel(), values_[next_].numel());
    received_.emplace_back(tensor.clone());
    tensor.copy_(values_[next_++].to(tensor.options()).view(tensor.sizes()));
  }

  uint32_t consumed() const { return next_; }
  const torch::Tensor& received(uint32_t index) const {
    return received_.at(index);
  }

 private:
  std::vector<torch::Tensor> values_;
  std::vector<torch::Tensor> received_;
  uint32_t next_ = 0;
};

class RecordingSamples final : public ProcessGroup {
 public:
  explicit RecordingSamples(const torch::Device& device,
                            std::vector<torch::Tensor> values = {})
      : ProcessGroup(/*rank=*/1, /*world_size=*/2, device),
        values_(std::move(values)) {
    received_.reserve(32);
  }

  void broadcast(torch::Tensor& tensor, int32_t root_rank) override {
    ASSERT_EQ(root_rank, 0);
    received_.emplace_back(tensor.clone());
    if (values_.empty()) {
      return;
    }
    const size_t index = received_.size() - 1;
    ASSERT_LT(index, values_.size());
    ASSERT_EQ(tensor.scalar_type(), values_[index].scalar_type());
    ASSERT_EQ(tensor.numel(), values_[index].numel());
    tensor.copy_(values_[index].to(tensor.options()).view(tensor.sizes()));
  }

  size_t broadcasts() const { return received_.size(); }
  const torch::Tensor& received(size_t index) const {
    return received_.at(index);
  }

 private:
  std::vector<torch::Tensor> values_;
  std::vector<torch::Tensor> received_;
};

class SpeculativePipelineTest : public ::testing::Test {
 protected:
  void SetUp() override {
    static const bool registered =
        ExecutorImplFactory::get_instance().register_creator(
            "speculative_pipeline_test",
            [](CausalLM* model,
               const ModelArgs& /*args*/,
               const torch::Device& /*device*/,
               const runtime::Options& /*options*/) {
              return std::make_unique<PreparedTestExecutor>(model);
            });
    ASSERT_TRUE(registered);
    previous_model_impl_ = ModelConfig::get_instance().model_impl();
    ModelConfig::get_instance().model_impl("auto");
    runtime::Options options;
    options.backend("speculative_pipeline_test").enable_graph(false);
    target_ = std::make_unique<CountingModel>(device_);
    draft_ = std::make_unique<CountingModel>(device_);
    target_executor_ = std::make_unique<Executor>(
        target_.get(), ModelArgs{}, device_, options);
    draft_executor_ =
        std::make_unique<Executor>(draft_.get(), ModelArgs{}, device_, options);
    const auto cache_options = target_->options();
    for (auto* caches : {&target_cache_, &draft_cache_}) {
      caches->emplace_back(
          KVCacheTensors{torch::zeros({4, 16, 1, 1}, cache_options),
                         torch::zeros({4, 16, 1, 1}, cache_options)});
    }
    capacity_.common.model = {16, 2, 2};
    capacity_.common.slot_count = 1;
    capacity_.common.max_kv_seq_len = 32;
    capacity_.common.max_positions = 32;
    capacity_.common.logical_block_size = 16;
    capacity_.common.vocab_size = 32;
    capacity_.common.max_unique_tokens = 32;
    capacity_.common.max_top_logprobs = 2;
    capacity_.common.hidden_size = 1;
    capacity_.common.enable_mla = true;
    capacity_.num_speculative_tokens = 2;
    ASSERT_EQ(aclrtSynchronizeDevice(), ACL_SUCCESS);
  }

  void TearDown() override {
    pipeline_.reset();
    EXPECT_EQ(aclrtSynchronizeDevice(), ACL_SUCCESS);
    ModelConfig::get_instance().model_impl(previous_model_impl_);
  }

  void create(TaskKVPush target_push = {}, TaskKVPush draft_push = {}) {
    const Status status = TaskExecutionPipeline::create(
        state_thread_,
        {*target_, *target_executor_, target_cache_, std::move(target_push)},
        {*draft_, *draft_executor_, draft_cache_, std::move(draft_push)},
        capacity_,
        pipeline_);
    ASSERT_TRUE(status.ok()) << status.message();
  }

  void configure_dcp() {
    // Four local physical blocks of eight tokens represent four logical
    // sixteen-token pages shared by two DCP ranks.
    for (auto* caches : {&target_cache_, &draft_cache_}) {
      caches->clear();
      caches->emplace_back(
          KVCacheTensors{torch::zeros({4, 8, 1, 1}, target_->options()),
                         torch::zeros({4, 8, 1, 1}, target_->options())});
    }
    capacity_.common.logical_block_size = 16;
    capacity_.common.slot_count = 2;
    capacity_.common.chunked_prefill = true;
  }

  LlmForwardInput input(bool decode, int32_t token = 3, int32_t position = 2) {
    LlmForwardInput input;
    const int32_t query = decode ? 1 : 2;
    input.token_ids = decode ? torch::tensor({token}, torch::kInt32)
                             : torch::tensor({1, 2}, torch::kInt32);
    input.positions = decode ? torch::tensor({position}, torch::kInt32)
                             : torch::tensor({0, 1}, torch::kInt32);
    auto& params = input.input_params;
    params.meta.batch_forward_type =
        decode ? BatchForwardType::DECODE : BatchForwardType::PREFILL;
    params.meta.num_sequences = 1;
    params.meta.actual_num_sequences = 1;
    params.attention.host.q_seq_lens = {query};
    params.attention.host.kv_seq_lens = {decode ? position + 1 : query};
    params.attention.host.q_cu_seq_lens = {query};
    params.attention.host.new_cache_slots =
        decode ? std::vector<int32_t>{position} : std::vector<int32_t>{0, 1};
    params.attention.host.block_tables = torch::tensor({{0, 1}}, torch::kInt32);
    params.embedding.embedding_ids = {1};
    params.embedding.request_ids = {"counting-request"};
    if (!decode) {
      params.embedding.extra_token_ids = {-1};
    }
    input.sampling_params.selected_token_idxes =
        torch::tensor({query - 1}, torch::kInt32);
    input.sampling_params.do_sample = torch::tensor({false}, torch::kBool);
    return input;
  }

  ForwardOutput execute(const LlmForwardInput& input) {
    const auto submitted = pipeline_->submit(input);
    EXPECT_TRUE(submitted.status.ok()) << submitted.status.message();
    if (!submitted.status.ok()) {
      return {};
    }
    auto result = pipeline_->take_result_async(submitted.task_id).get();
    EXPECT_TRUE(result.status.ok()) << result.status.message();
    EXPECT_EQ(result.task_id, submitted.task_id);
    return std::move(result.output);
  }

  LlmForwardInput dcp_input(bool decode,
                            int32_t token = 3,
                            int32_t position = 15) {
    auto value = input(decode, token, position);
    auto& host = value.input_params.attention.host;
    host.block_tables = torch::tensor({{3, 1}}, torch::kInt32);
    if (decode) {
      const int32_t page = position < 16 ? 3 : 1;
      host.new_cache_slots = {page * 16 + position % 16};
    } else {
      value.input_params.meta.batch_forward_type =
          BatchForwardType::CHUNKED_PREFILL;
      value.positions = torch::tensor({13, 14}, torch::kInt32);
      host.kv_seq_lens = {15};
      host.new_cache_slots = {61, 62};
    }
    return value;
  }

  void configure_dspark_batch_graphs() {
    capacity_.kind = SpeculativeTaskKind::DSPARK;
    capacity_.common.model = {64, 4, 2};
    capacity_.common.slot_count = 2;
    capacity_.common.max_graph_batch_size = 32;
    capacity_.common.chunked_prefill = true;
    capacity_.num_speculative_tokens = 7;
    capacity_.context_hidden_size = 2;
    capacity_.mask_token_id = 0;
    draft_->block_draft = true;
    target_->stable_graph_outputs = true;
    draft_->stable_graph_outputs = true;
    for (auto* caches : {&target_cache_, &draft_cache_}) {
      caches->clear();
      caches->emplace_back(
          KVCacheTensors{torch::zeros({8, 16, 1, 1}, target_->options()),
                         torch::zeros({8, 16, 1, 1}, target_->options())});
    }
  }

  void configure_adaptive_dspark(bool profiled = true,
                                 uint32_t draft_width = 7) {
    configure_dspark_batch_graphs();
    capacity_.num_speculative_tokens = draft_width;
    capacity_.adaptive_dspark = true;
    capacity_.sparse_dsa_target = true;
    draft_->confidence_head = true;
    auto costs = std::make_shared<DSparkAdaptiveCostCache>();
    if (profiled) {
      for (const uint32_t requests : {1U, 2U, 4U}) {
        DSparkAdaptiveCostProfile profile;
        profile.draft_width = draft_width;
        profile.draft_graph_rows = requests * draft_width;
        profile.max_kv_length = 32;
        profile.fixed_draft_time_ms = 1.0;
        profile.commit_time_ms = 0.1;
        profile.target_graph_costs = {{1, 1.0},
                                      {2, 2.0},
                                      {4, 4.0},
                                      {6, 6.0},
                                      {8, 8.0},
                                      {12, 12.0},
                                      {16, 16.0},
                                      {24, 24.0},
                                      {32, 32.0}};
        ASSERT_TRUE(costs->store(std::move(profile)));
      }
    }
    capacity_.adaptive_costs = std::move(costs);
  }

  LlmForwardInput dspark_batch_input(uint32_t batch_size,
                                     bool decode,
                                     bool placeholder = false,
                                     int32_t position = 2) {
    const int32_t query = decode ? 1 : 2;
    std::vector<int32_t> tokens;
    std::vector<int32_t> positions;
    std::vector<int32_t> slots;
    std::vector<int32_t> block_tables;
    std::vector<int32_t> selected;
    tokens.reserve(batch_size * query);
    positions.reserve(batch_size * query);
    slots.reserve(batch_size * query);
    block_tables.reserve(batch_size * 2);
    selected.reserve(batch_size);
    LlmForwardInput value;
    auto& params = value.input_params;
    params.meta.batch_forward_type =
        decode ? BatchForwardType::DECODE : BatchForwardType::PREFILL;
    params.meta.num_sequences = batch_size;
    params.meta.actual_num_sequences = batch_size;
    params.embedding.embedding_ids.reserve(batch_size);
    params.embedding.request_ids.reserve(batch_size);
    params.embedding.extra_token_ids.reserve(batch_size);
    auto& host = params.attention.host;
    host.q_seq_lens.reserve(batch_size);
    host.kv_seq_lens.reserve(batch_size);
    host.q_cu_seq_lens.reserve(batch_size);
    for (uint32_t row = 0; row < batch_size; ++row) {
      const int32_t first_page = static_cast<int32_t>(2 * row + 1);
      block_tables.push_back(first_page);
      block_tables.push_back(first_page + 1);
      params.embedding.embedding_ids.push_back(static_cast<int32_t>(row + 1));
      params.embedding.request_ids.emplace_back("batch-request-" +
                                                std::to_string(row));
      host.q_seq_lens.push_back(query);
      host.kv_seq_lens.push_back(decode ? position + 1 : query);
      host.q_cu_seq_lens.push_back(static_cast<int32_t>((row + 1) * query));
      for (int32_t item = 0; item < query; ++item) {
        const int32_t token =
            decode ? (placeholder ? -1 : static_cast<int32_t>(2 * row + 3))
                   : static_cast<int32_t>(2 * row + item + 1);
        const int32_t token_position = decode ? position : item;
        tokens.push_back(token);
        positions.push_back(token_position);
        slots.push_back(first_page * 16 + token_position);
      }
      selected.push_back(static_cast<int32_t>((row + 1) * query - 1));
      if (!decode) {
        params.embedding.extra_token_ids.push_back(-1);
      }
    }
    value.token_ids = torch::tensor(tokens, torch::kInt32);
    value.positions = torch::tensor(positions, torch::kInt32);
    host.new_cache_slots = std::move(slots);
    host.block_tables =
        torch::tensor(block_tables, torch::kInt32).view({batch_size, 2});
    value.sampling_params.selected_token_idxes =
        torch::tensor(selected, torch::kInt32);
    value.sampling_params.sampling_mode = SamplingMode::GREEDY;
    return value;
  }

  void warmup_dspark_batch_graphs() {
    for (uint32_t batch_size = 1;
         batch_size <=
         std::min<uint32_t>(3, capacity_.common.model.max_sequences);
         ++batch_size) {
      execute(dspark_batch_input(batch_size, /*decode=*/false));
      auto warmup = dspark_batch_input(batch_size, /*decode=*/true);
      warmup.input_params.meta.is_graph_warmup = true;
      const auto output = execute(warmup);
      EXPECT_TRUE(output.is_graph_warmup);
      ASSERT_TRUE(output.sample_output.next_tokens.defined());
      EXPECT_EQ(output.sample_output.next_tokens.size(0), batch_size);
      EXPECT_EQ(output.sample_output.next_tokens.size(1),
                capacity_.num_speculative_tokens + 1);
    }
  }

  const torch::Device device_{torch::kPrivateUse1, 0};
  std::string previous_model_impl_;
  ThreadPool state_thread_{1};
  std::unique_ptr<CountingModel> target_;
  std::unique_ptr<CountingModel> draft_;
  std::unique_ptr<Executor> target_executor_;
  std::unique_ptr<Executor> draft_executor_;
  std::vector<KVCache> target_cache_;
  std::vector<KVCache> draft_cache_;
  SpeculativeTaskCapacity capacity_;
  std::unique_ptr<TaskExecutionPipeline> pipeline_;
};

class PipelineExecutionMemoryTest : public SpeculativePipelineTest {
 protected:
  static constexpr int64_t kPageSize = 4096;

  void SetUp() override {
    SpeculativePipelineTest::SetUp();
    auto& config = KVCacheConfig::get_instance();
    previous_virtual_memory_ = config.enable_virtual_memory();
    previous_elastic_ = config.vmm_elastic();
    config.enable_virtual_memory(true).vmm_elastic(true);
    auto& pool = PhysicalPagePool::get_instance();
    ASSERT_TRUE(pool.reset());
    pool.init(torch::Device(torch::kCPU),
              /*num_pages=*/8,
              kPageSize,
              /*lazy=*/true);
    PhysicalPagePoolTestPeer::set_memory_query(pool, [this] {
      if (on_query_) {
        on_query_();
      }
      return free_bytes_.load();
    });
  }

  void TearDown() override {
    if (worker_ != nullptr) {
      worker_->release();
    }
    SpeculativePipelineTest::TearDown();
    EXPECT_TRUE(PhysicalPagePool::get_instance().reset());
    auto& config = KVCacheConfig::get_instance();
    config.enable_virtual_memory(previous_virtual_memory_)
        .vmm_elastic(previous_elastic_);
  }

  void create_worker() {
    worker_ = std::make_unique<PreparedTestWorker>(device_);
    ASSERT_TRUE(TaskExecutionPipeline::create(
                    state_thread_,
                    {*target_, *target_executor_, target_cache_},
                    *worker_,
                    capacity_.common,
                    pipeline_)
                    .ok());
  }

  void verify_worker_sampling(bool warmup) {
    create_worker();
    ASSERT_NE(pipeline_, nullptr);
    auto entered = worker_->entered();
    auto sampled = std::make_shared<std::promise<bool>>();
    auto sampling = sampled->get_future();
    on_query_ = [this, sampled] {
      if (!sampled_.exchange(true)) {
        sampled->set_value(worker_->finished());
      }
    };
    auto request = input(true);
    request.input_params.meta.is_graph_warmup = warmup;
    const auto submitted = pipeline_->submit(request);
    ASSERT_TRUE(submitted.status.ok()) << submitted.status.message();
    EXPECT_EQ(entered.wait_for(std::chrono::seconds(5)),
              std::future_status::ready);
    if (warmup) {
      EXPECT_EQ(sampling.wait_for(std::chrono::milliseconds(50)),
                std::future_status::timeout);
    } else {
      const auto status = sampling.wait_for(std::chrono::seconds(5));
      EXPECT_EQ(status, std::future_status::ready);
      if (status == std::future_status::ready) {
        EXPECT_FALSE(sampling.get());
      }
    }
    worker_->release();
    if (warmup) {
      const auto status = sampling.wait_for(std::chrono::seconds(5));
      EXPECT_EQ(status, std::future_status::ready);
      if (status == std::future_status::ready) {
        EXPECT_TRUE(sampling.get());
      }
    }
    EXPECT_TRUE(
        pipeline_->take_result_async(submitted.task_id).get().status.ok());
    pipeline_.reset();
    on_query_ = {};
  }

  bool previous_virtual_memory_ = false;
  bool previous_elastic_ = false;
  std::atomic<int64_t> free_bytes_{3 * kPageSize};
  std::atomic<bool> sampled_{false};
  std::function<void()> on_query_;
  std::unique_ptr<PreparedTestWorker> worker_;
};

enum class BudgetExecutionPath { ORDINARY, SPECULATIVE, WORKER };

class PipelineExecutionMemoryPathTest
    : public PipelineExecutionMemoryTest,
      public ::testing::WithParamInterface<BudgetExecutionPath> {
 protected:
  void create_path() {
    switch (GetParam()) {
      case BudgetExecutionPath::ORDINARY:
        ASSERT_TRUE(TaskExecutionPipeline::create(state_thread_,
                                                  *target_,
                                                  *target_executor_,
                                                  target_cache_,
                                                  capacity_.common,
                                                  pipeline_)
                        .ok());
        break;
      case BudgetExecutionPath::SPECULATIVE:
        create();
        break;
      case BudgetExecutionPath::WORKER:
        create_worker();
        worker_->release();
        break;
    }
  }

  void execute_warmup() {
    ASSERT_NE(pipeline_, nullptr);
    auto request = input(false);
    if (GetParam() == BudgetExecutionPath::ORDINARY) {
      request.input_params.embedding = {};
    }
    request.input_params.meta.is_graph_warmup = true;
    execute(request);
    free_bytes_.store(6 * kPageSize);
  }
};

TEST_P(PipelineExecutionMemoryPathTest, WarmupPreservesExecutionBudget) {
  create_path();
  execute_warmup();
  EXPECT_EQ(PhysicalPagePool::get_instance().num_allocatable(), 3U);
}

TEST_P(PipelineExecutionMemoryPathTest, FixedPoolSkipsExecutionSampling) {
  create_path();
  KVCacheConfig::get_instance().vmm_elastic(false);
  execute_warmup();
  EXPECT_EQ(PhysicalPagePool::get_instance().num_allocatable(), 6U);
}

TEST_P(PipelineExecutionMemoryPathTest, DisabledVmmSkipsExecutionSampling) {
  create_path();
  KVCacheConfig::get_instance().enable_virtual_memory(false);
  execute_warmup();
  EXPECT_EQ(PhysicalPagePool::get_instance().num_allocatable(), 6U);
}

INSTANTIATE_TEST_SUITE_P(ExecutionPaths,
                         PipelineExecutionMemoryPathTest,
                         ::testing::Values(BudgetExecutionPath::ORDINARY,
                                           BudgetExecutionPath::SPECULATIVE,
                                           BudgetExecutionPath::WORKER));

TEST_F(PipelineExecutionMemoryTest, WorkerWarmupSamplesAfterExecutionFinishes) {
  verify_worker_sampling(/*warmup=*/true);
}

TEST_F(PipelineExecutionMemoryTest, WorkerServingSamplesWithoutStreamWait) {
  verify_worker_sampling(/*warmup=*/false);
}

TransferKVInfo push_info() {
  TransferKVInfo info;
  info.request_id = "pd-request";
  info.rank_local_mapping = true;
  info.mappings.emplace_back(KVTransferMapping{0, {1}, {3}, 0});
  return info;
}

TEST_F(SpeculativePipelineTest,
       OrdinaryKvPushOwnsMappingsAndGatesResultAndSlotReuse) {
  capacity_.common.enable_kv_push = true;
  std::promise<void> push_started;
  auto started = push_started.get_future();
  std::promise<void> read_mappings;
  auto readable = read_mappings.get_future();
  std::promise<void> model_finished;
  auto finished = model_finished.get_future();
  folly::Promise<bool> transfer_finished;
  std::vector<TransferKVInfo> observed;
  std::weak_ptr<NPULayerSynchronizerImpl> synchronizer;
  target_->after_forward = [&model_finished]() { model_finished.set_value(); };
  TaskKVPush push = [&](const std::vector<TransferKVInfo>& infos,
                        ModelParamsView& params) {
    push_started.set_value();
    readable.wait();
    observed = infos;
    params.parallel.layer_synchronizer =
        std::make_shared<NPULayerSynchronizerImpl>(/*num_layers=*/1);
    synchronizer = params.parallel.layer_synchronizer;
    return transfer_finished.getSemiFuture();
  };
  ASSERT_TRUE(TaskExecutionPipeline::create(
                  state_thread_,
                  {*target_, *target_executor_, target_cache_, std::move(push)},
                  capacity_.common,
                  pipeline_)
                  .ok());
  auto prefill = input(false);
  prefill.input_params.embedding = {};
  prefill.transfer_kv_infos.emplace_back(push_info());
  const auto submitted = pipeline_->submit(prefill);
  ASSERT_TRUE(submitted.status.ok()) << submitted.status.message();
  EXPECT_EQ(started.wait_for(std::chrono::seconds(10)),
            std::future_status::ready);
  // Submit acknowledges ownership of nested transfer mappings, even while
  // Launch has not read them and the caller immediately reuses its storage.
  prefill.transfer_kv_infos.front().request_id = "replacement";
  prefill.transfer_kv_infos.front().mappings.front().local_ids.front() = 99;
  prefill.transfer_kv_infos.clear();
  read_mappings.set_value();
  EXPECT_EQ(finished.wait_for(std::chrono::seconds(10)),
            std::future_status::ready);
  EXPECT_FALSE(pipeline_->submit(prefill).status.ok());
  auto pending = pipeline_->take_result_async(submitted.task_id);
  auto result =
      std::async(std::launch::async, [future = std::move(pending)]() mutable {
        return std::move(future).get();
      });
  EXPECT_EQ(result.wait_for(std::chrono::milliseconds(50)),
            std::future_status::timeout);
  transfer_finished.setValue(true);
  const auto completed = result.get();
  ASSERT_TRUE(completed.status.ok()) << completed.status.message();
  ASSERT_EQ(observed.size(), 1U);
  EXPECT_EQ(observed.front().request_id, "pd-request");
  ASSERT_EQ(observed.front().mappings.size(), 1U);
  EXPECT_EQ(observed.front().mappings.front().local_ids,
            (std::vector<uint64_t>{1}));
  EXPECT_EQ(observed.front().mappings.front().remote_ids,
            (std::vector<uint64_t>{3}));
  EXPECT_TRUE(synchronizer.expired());
  target_->after_forward = {};
  const auto reused = execute(prefill);
  EXPECT_TRUE(torch::equal(reused.sample_output.next_tokens,
                           completed.output.sample_output.next_tokens));
  EXPECT_EQ(target_->forward_had_sync, (std::vector<bool>{true, false}));
}

class MtpKvPushPipelineTest : public SpeculativePipelineTest,
                              public ::testing::WithParamInterface<bool> {};

TEST_P(MtpKvPushPipelineTest,
       MtpKvPushWaitsForIndependentTargetAndDraftEvents) {
  capacity_.common.enable_kv_push = true;
  folly::Promise<bool> target_transfer;
  folly::Promise<bool> draft_transfer;
  std::promise<void> draft_finished;
  auto finished = draft_finished.get_future();
  std::weak_ptr<NPULayerSynchronizerImpl> target_sync;
  std::weak_ptr<NPULayerSynchronizerImpl> draft_sync;
  const auto push =
      [](folly::Promise<bool>& completion,
         std::weak_ptr<NPULayerSynchronizerImpl>& observed) -> TaskKVPush {
    return [&completion, &observed](const std::vector<TransferKVInfo>& infos,
                                    ModelParamsView& params) {
      EXPECT_EQ(infos.size(), 1U);
      params.parallel.layer_synchronizer =
          std::make_shared<NPULayerSynchronizerImpl>(/*num_layers=*/1);
      observed = params.parallel.layer_synchronizer;
      return completion.getSemiFuture();
    };
  };
  draft_->after_forward = [&draft_finished]() { draft_finished.set_value(); };
  create(push(target_transfer, target_sync), push(draft_transfer, draft_sync));
  auto prefill = input(false);
  prefill.transfer_kv_infos.emplace_back(push_info());
  const auto submitted = pipeline_->submit(prefill);
  ASSERT_TRUE(submitted.status.ok()) << submitted.status.message();
  const std::future_status forward_status =
      finished.wait_for(std::chrono::seconds(10));
  EXPECT_EQ(forward_status, std::future_status::ready);
  if (forward_status == std::future_status::ready) {
    const auto target = target_sync.lock();
    const auto draft = draft_sync.lock();
    EXPECT_NE(target, nullptr);
    EXPECT_NE(draft, nullptr);
    EXPECT_NE(target, draft);
    if (target != nullptr && draft != nullptr) {
      EXPECT_TRUE(target->get_event_flag(0)->load());
      EXPECT_TRUE(draft->get_event_flag(0)->load());
    }
  }
  auto pending = pipeline_->take_result_async(submitted.task_id);
  auto result =
      std::async(std::launch::async, [future = std::move(pending)]() mutable {
        return std::move(future).get();
      });
  folly::Promise<bool>& first_transfer =
      GetParam() ? target_transfer : draft_transfer;
  folly::Promise<bool>& second_transfer =
      GetParam() ? draft_transfer : target_transfer;
  first_transfer.setValue(true);
  EXPECT_EQ(result.wait_for(std::chrono::milliseconds(50)),
            std::future_status::timeout);
  second_transfer.setValue(true);
  const auto completed = result.get();
  ASSERT_TRUE(completed.status.ok()) << completed.status.message();
  EXPECT_TRUE(torch::equal(completed.output.sample_output.embeddings,
                           torch::tensor({{2.0f}})));
  EXPECT_TRUE(target_sync.expired());
  EXPECT_TRUE(draft_sync.expired());
  draft_->after_forward = {};
  const auto decoded = execute(input(true));
  EXPECT_TRUE(torch::equal(decoded.sample_output.next_tokens,
                           torch::tensor({{4, 5, 6}}, torch::kInt64)));
  EXPECT_EQ(target_->forward_had_sync, (std::vector<bool>{true, false}));
  EXPECT_EQ(draft_->forward_had_sync, (std::vector<bool>{true, false, false}));
}

INSTANTIATE_TEST_SUITE_P(CompletionOrder,
                         MtpKvPushPipelineTest,
                         ::testing::Bool());

TEST_F(SpeculativePipelineTest,
       UnfinishedMtpPrefillPushesBothCachesWithoutSampling) {
  capacity_.common.enable_kv_push = true;
  capacity_.common.chunked_prefill = true;
  uint32_t target_pushes = 0;
  uint32_t draft_pushes = 0;
  const auto push = [](uint32_t& count) -> TaskKVPush {
    return [&count](const std::vector<TransferKVInfo>& infos,
                    ModelParamsView& params) {
      EXPECT_EQ(infos.size(), 1U);
      ++count;
      params.parallel.layer_synchronizer =
          std::make_shared<NPULayerSynchronizerImpl>(/*num_layers=*/1);
      return folly::makeSemiFuture(true);
    };
  };
  create(push(target_pushes), push(draft_pushes));
  auto unfinished = input(false);
  unfinished.input_params.meta.batch_forward_type =
      BatchForwardType::CHUNKED_PREFILL;
  unfinished.input_params.embedding.extra_token_ids = {3};
  unfinished.input_params.embedding.embedding_ids.clear();
  unfinished.input_params.embedding.request_ids.clear();
  unfinished.sampling_params = {};
  unfinished.transfer_kv_infos.emplace_back(push_info());
  unfinished.input_params.parallel.layer_synchronizer =
      std::make_shared<NPULayerSynchronizerImpl>(/*num_layers=*/1);
  EXPECT_FALSE(pipeline_->submit(unfinished).status.ok());
  unfinished.input_params.parallel.layer_synchronizer.reset();
  const auto output = execute(unfinished);
  EXPECT_FALSE(output.sample_output.next_tokens.defined());
  EXPECT_FALSE(output.sample_output.embeddings.defined());
  EXPECT_EQ(target_pushes, 1U);
  EXPECT_EQ(draft_pushes, 1U);
  EXPECT_EQ(target_->logits_calls.load(), 0);
  EXPECT_EQ(draft_->logits_calls.load(), 0);
}

TEST_F(SpeculativePipelineTest,
       FreshDecodeImportsBootstrapAndReplacesReusedRequest) {
  create();
  auto decode = input(true);
  EXPECT_FALSE(pipeline_->submit(decode).status.ok());
  decode.input_params.embedding.mtp_bootstrap_row_idxes = {0};
  decode.input_params.embedding.mtp_bootstrap_embeddings =
      torch::tensor({{2.0f}});
  const auto first = execute(decode);
  EXPECT_TRUE(torch::equal(first.sample_output.next_tokens,
                           torch::tensor({{4, 5, 6}}, torch::kInt64)));
  ASSERT_EQ(draft_->forward_embeddings.size(), 2U);
  EXPECT_TRUE(torch::equal(draft_->forward_embeddings.front().cpu(),
                           torch::tensor({{0.0f}, {2.0f}})));

  auto replacement = input(true, /*token=*/10, /*position=*/5);
  replacement.input_params.embedding.request_ids = {"new-pd-request"};
  EXPECT_FALSE(pipeline_->submit(replacement).status.ok());
  replacement.input_params.embedding.mtp_bootstrap_row_idxes = {0};
  replacement.input_params.embedding.mtp_bootstrap_embeddings =
      torch::tensor({{9.0f}});
  const auto second = execute(replacement);
  EXPECT_TRUE(torch::equal(second.sample_output.next_tokens,
                           torch::tensor({{11, 12, 13}}, torch::kInt64)));
  ASSERT_EQ(draft_->forward_embeddings.size(), 4U);
  EXPECT_TRUE(torch::equal(draft_->forward_embeddings[2].cpu(),
                           torch::tensor({{0.0f}, {9.0f}})));
  EXPECT_TRUE(torch::equal(first.sample_output.next_tokens,
                           torch::tensor({{4, 5, 6}}, torch::kInt64)));
}

TEST_F(SpeculativePipelineTest, PrefillDecodeAndSlotReuseKeepAcceptedContext) {
  create();
  ASSERT_NE(pipeline_, nullptr);
  const auto prefill = execute(input(false));
  ASSERT_TRUE(prefill.sample_output.next_tokens.defined());
  EXPECT_TRUE(torch::equal(prefill.sample_output.next_tokens,
                           torch::tensor({3}, torch::kInt64)));
  EXPECT_TRUE(
      torch::equal(prefill.sample_output.embeddings, torch::tensor({{2.0f}})));
  EXPECT_EQ(draft_->logits_calls.load(), 0);
  const auto decoded = execute(input(true));
  ASSERT_TRUE(decoded.sample_output.next_tokens.defined());
  EXPECT_TRUE(torch::equal(decoded.sample_output.next_tokens,
                           torch::tensor({{4, 5, 6}}, torch::kInt64)));
  EXPECT_EQ(draft_->logits_calls.load(), 2);
  const auto reused = execute(input(true, 6, 5));
  ASSERT_TRUE(reused.sample_output.next_tokens.defined());
  EXPECT_TRUE(torch::equal(reused.sample_output.next_tokens,
                           torch::tensor({{7, 8, 9}}, torch::kInt64)));
  EXPECT_EQ(decoded.sample_output.next_tokens[0][2].item<int64_t>(), 6);
  EXPECT_TRUE(decoded.sample_output.next_tokens.device().is_cpu());
  EXPECT_EQ(decoded.ready_event, nullptr);
}

TEST_F(SpeculativePipelineTest,
       DcpOrdinaryInputUsesLogicalSlotsAndLocalPageBounds) {
  configure_dcp();
  const Status status = TaskExecutionPipeline::create(state_thread_,
                                                      *target_,
                                                      *target_executor_,
                                                      target_cache_,
                                                      capacity_.common,
                                                      pipeline_);
  ASSERT_TRUE(status.ok()) << status.message();
  auto prefill = dcp_input(false);
  prefill.input_params.embedding = {};
  prefill.token_ids = torch::tensor({1, 2, 3, 4}, torch::kInt32);
  prefill.positions = torch::tensor({14, 15, 16, 17}, torch::kInt32);
  auto& host = prefill.input_params.attention.host;
  host.q_seq_lens = {4};
  host.kv_seq_lens = {18};
  host.q_cu_seq_lens = {4};
  host.new_cache_slots = {62, 63, 16, 17};
  prefill.sampling_params.selected_token_idxes =
      torch::tensor({3}, torch::kInt32);

  auto invalid = prefill.clone();
  invalid.input_params.attention.host.new_cache_slots = {30, 31, 8, 9};
  EXPECT_FALSE(pipeline_->submit(invalid).status.ok());
  invalid = prefill.clone();
  invalid.input_params.attention.host.block_tables =
      torch::tensor({{3, 4}}, torch::kInt32);
  invalid.input_params.attention.host.new_cache_slots = {62, 63, 64, 65};
  EXPECT_FALSE(pipeline_->submit(invalid).status.ok());

  const auto output = execute(prefill);
  ASSERT_TRUE(output.sample_output.next_tokens.defined());
  EXPECT_TRUE(torch::equal(output.sample_output.next_tokens,
                           torch::tensor({5}, torch::kInt64)));
  ASSERT_EQ(target_->forward_slots.size(), 1U);
  EXPECT_TRUE(torch::equal(target_->forward_slots.front().cpu(),
                           torch::tensor({62, 63, 16, 17}, torch::kInt32)));
}

TEST_F(SpeculativePipelineTest, DcpMtpKeepsLogicalPagesAcrossPendingSlots) {
  configure_dcp();
  create();
  ASSERT_NE(pipeline_, nullptr);
  const auto prefill = execute(dcp_input(false));
  ASSERT_TRUE(prefill.sample_output.next_tokens.defined());
  EXPECT_TRUE(torch::equal(prefill.sample_output.next_tokens,
                           torch::tensor({3}, torch::kInt64)));

  auto invalid = dcp_input(true);
  invalid.input_params.attention.host.new_cache_slots = {31};
  EXPECT_FALSE(pipeline_->submit(invalid).status.ok());
  invalid = dcp_input(true);
  invalid.input_params.attention.host.block_tables =
      torch::tensor({{3, 4}}, torch::kInt32);
  EXPECT_FALSE(pipeline_->submit(invalid).status.ok());

  const auto first = pipeline_->submit(dcp_input(true));
  ASSERT_TRUE(first.status.ok()) << first.status.message();
  // The next scheduler row advances by one pending token; the accepted
  // device state advances by all three verified tokens instead.
  const auto second =
      pipeline_->submit(dcp_input(true, /*token=*/-1, /*position=*/16));
  ASSERT_TRUE(second.status.ok()) << second.status.message();
  auto first_result = pipeline_->take_result_async(first.task_id).get();
  ASSERT_TRUE(first_result.status.ok()) << first_result.status.message();
  auto second_result = pipeline_->take_result_async(second.task_id).get();
  ASSERT_TRUE(second_result.status.ok()) << second_result.status.message();
  EXPECT_TRUE(torch::equal(first_result.output.sample_output.next_tokens,
                           torch::tensor({{4, 5, 6}}, torch::kInt64)));
  EXPECT_TRUE(torch::equal(second_result.output.sample_output.next_tokens,
                           torch::tensor({{7, 8, 9}}, torch::kInt64)));
  ASSERT_EQ(target_->forward_slots.size(), 3U);
  EXPECT_TRUE(torch::equal(target_->forward_slots[1].cpu(),
                           torch::tensor({63, 16, 17}, torch::kInt32)));
  EXPECT_TRUE(torch::equal(target_->forward_slots[2].cpu(),
                           torch::tensor({18, 19, 20}, torch::kInt32)));
  ASSERT_EQ(draft_->forward_slots.size(), 5U);
  // Draft pairs use the target-hidden position, one before target decode.
  EXPECT_TRUE(torch::equal(draft_->forward_slots[1].cpu(),
                           torch::tensor({63, 62}, torch::kInt32)));
  EXPECT_TRUE(torch::equal(draft_->forward_slots[3].cpu(),
                           torch::tensor({16, 17}, torch::kInt32)));

  const auto reused = execute(dcp_input(true, /*token=*/-1, /*position=*/19));
  ASSERT_TRUE(reused.sample_output.next_tokens.defined());
  EXPECT_TRUE(torch::equal(reused.sample_output.next_tokens,
                           torch::tensor({{10, 11, 12}}, torch::kInt64)));
  EXPECT_TRUE(torch::equal(target_->forward_slots.back().cpu(),
                           torch::tensor({21, 22, 23}, torch::kInt32)));
  EXPECT_TRUE(torch::equal(first_result.output.sample_output.next_tokens,
                           torch::tensor({{4, 5, 6}}, torch::kInt64)));
}

TEST_F(SpeculativePipelineTest, LogprobsFollowAcceptedTokensAndPreserveShape) {
  create();
  ASSERT_NE(pipeline_, nullptr);
  execute(input(false));
  auto decode = input(true);
  decode.sampling_params.logprobs = true;
  decode.sampling_params.max_top_logprobs = 2;
  const auto output = execute(decode);
  ASSERT_TRUE(output.sample_output.next_tokens.defined());
  EXPECT_TRUE(torch::equal(output.sample_output.next_tokens,
                           torch::tensor({{4, 5, 6}}, torch::kInt64)));
  EXPECT_TRUE(output.logprobs);
  EXPECT_EQ(output.max_top_logprobs, 2);
  EXPECT_TRUE(
      torch::allclose(output.sample_output.logprobs, torch::zeros({1, 3})));
  EXPECT_TRUE(torch::equal(output.sample_output.top_tokens.select(2, 0),
                           output.sample_output.next_tokens));
}

TEST_F(SpeculativePipelineTest, RejectedTransportDoesNotConsumeRequestContext) {
  create();
  ASSERT_NE(pipeline_, nullptr);
  execute(input(false));
  auto malformed = input(true);
  malformed.input_params.embedding.request_ids.clear();
  EXPECT_FALSE(pipeline_->submit(malformed).status.ok());
  const auto output = execute(input(true));
  ASSERT_TRUE(output.sample_output.next_tokens.defined());
  EXPECT_TRUE(torch::equal(output.sample_output.next_tokens,
                           torch::tensor({{4, 5, 6}}, torch::kInt64)));
}

TEST_F(SpeculativePipelineTest,
       LeaderSamplesDriveLaterDraftsAndAcceptedContext) {
  LeaderSamples leader(device_);
  capacity_.common.sampling_group = &leader;
  capacity_.draft_sampling_mode = DraftSamplingMode::PROBABILISTIC;
  create();
  ASSERT_NE(pipeline_, nullptr);
  auto prefill = input(false);
  prefill.sampling_params.do_sample.fill_(true);
  prefill.sampling_params.sampling_mode = SamplingMode::RANDOM;
  const auto first = execute(prefill);
  ASSERT_TRUE(first.sample_output.next_tokens.defined());
  EXPECT_EQ(first.sample_output.next_tokens.item<int64_t>(), 7);
  auto decode = input(true, 7, 2);
  decode.sampling_params.do_sample.fill_(true);
  decode.sampling_params.sampling_mode = SamplingMode::RANDOM;
  const auto accepted = execute(decode);
  ASSERT_TRUE(accepted.sample_output.next_tokens.defined());
  EXPECT_TRUE(torch::equal(accepted.sample_output.next_tokens,
                           torch::tensor({{8, -1, -1}}, torch::kInt64)));
  ASSERT_EQ(draft_->forward_inputs.size(), 3U);
  EXPECT_TRUE(torch::equal(draft_->forward_inputs[0].cpu(),
                           torch::tensor({2, 7}, torch::kInt32)));
  EXPECT_EQ(draft_->forward_inputs[2].cpu().item<int32_t>(), 12);
  EXPECT_TRUE(torch::equal(target_->forward_inputs.back().cpu(),
                           torch::tensor({7, 12, 14}, torch::kInt32)));
  EXPECT_EQ(leader.consumed(), 4U);
  const auto next = execute(input(true, 8, 3));
  ASSERT_TRUE(next.sample_output.next_tokens.defined());
  EXPECT_TRUE(torch::equal(next.sample_output.next_tokens,
                           torch::tensor({{9, 10, 11}}, torch::kInt64)));
  // Greedy rounds keep the existing no-broadcast policy.
  EXPECT_EQ(leader.consumed(), 4U);
  pipeline_.reset();
}

class CpSpeculativePipelineTest : public SpeculativePipelineTest,
                                  public ::testing::WithParamInterface<bool> {};

TEST_P(CpSpeculativePipelineTest,
       CpLeaderSamplesDriveDraftsAndGreedyAcceptedContext) {
  const bool tensor_parallel = GetParam();
  LeaderSamples tp_leader(device_,
                          /*token_offset=*/1,
                          /*world_size=*/tensor_parallel ? 2 : 1);
  LeaderSamples cp_leader(device_);
  capacity_.common.sampling_group = &tp_leader;
  capacity_.common.cp_sampling_group = &cp_leader;
  capacity_.draft_sampling_mode = DraftSamplingMode::PROBABILISTIC;
  create();
  ASSERT_NE(pipeline_, nullptr);
  auto prefill = input(false);
  prefill.sampling_params.do_sample.fill_(true);
  prefill.sampling_params.sampling_mode = SamplingMode::RANDOM;
  const auto first = execute(prefill);
  ASSERT_TRUE(first.sample_output.next_tokens.defined());
  EXPECT_EQ(first.sample_output.next_tokens.item<int64_t>(), 7);
  auto decode = input(true, /*token=*/7, /*position=*/2);
  decode.sampling_params.do_sample.fill_(true);
  decode.sampling_params.sampling_mode = SamplingMode::RANDOM;
  const auto accepted = execute(decode);
  ASSERT_TRUE(accepted.sample_output.next_tokens.defined());
  EXPECT_TRUE(torch::equal(accepted.sample_output.next_tokens,
                           torch::tensor({{8, -1, -1}}, torch::kInt64)));
  ASSERT_EQ(draft_->forward_inputs.size(), 3U);
  EXPECT_TRUE(torch::equal(draft_->forward_inputs[0].cpu(),
                           torch::tensor({2, 7}, torch::kInt32)));
  EXPECT_EQ(draft_->forward_inputs[2].cpu().item<int32_t>(), 12);
  EXPECT_TRUE(torch::equal(target_->forward_inputs.back().cpu(),
                           torch::tensor({7, 12, 14}, torch::kInt32)));
  EXPECT_EQ(cp_leader.consumed(), 4U);
  EXPECT_EQ(tp_leader.consumed(), tensor_parallel ? 4U : 0U);
  EXPECT_EQ(cp_leader.received(0).cpu().item<int64_t>(),
            tensor_parallel ? 8 : 3);
  if (tensor_parallel) {
    EXPECT_TRUE(torch::equal(cp_leader.received(3).cpu(),
                             torch::tensor({{9, -1, -1}}, torch::kInt64)));
  }

  // CP also synchronizes greedy rounds; the preceding accepted token and
  // length must come from the CP leader before the next state is gathered.
  const auto next = execute(input(true, /*token=*/8, /*position=*/3));
  ASSERT_TRUE(next.sample_output.next_tokens.defined());
  EXPECT_TRUE(torch::equal(next.sample_output.next_tokens,
                           torch::tensor({{9, -1, -1}}, torch::kInt64)));
  EXPECT_TRUE(torch::equal(target_->forward_inputs.back().cpu(),
                           torch::tensor({8, 18, 20}, torch::kInt32)));
  EXPECT_EQ(cp_leader.consumed(), 7U);
  EXPECT_EQ(tp_leader.consumed(), tensor_parallel ? 7U : 0U);
  pipeline_.reset();
}

TEST_P(CpSpeculativePipelineTest, OrdinaryCpLeaderTokensFeedOverlappedDecode) {
  const bool tensor_parallel = GetParam();
  LeaderSamples tp_leader(device_,
                          /*token_offset=*/1,
                          /*world_size=*/tensor_parallel ? 2 : 1);
  LeaderSamples cp_leader(device_);
  capacity_.common.sampling_group = &tp_leader;
  capacity_.common.cp_sampling_group = &cp_leader;
  capacity_.common.slot_count = 2;
  const Status status = TaskExecutionPipeline::create(state_thread_,
                                                      *target_,
                                                      *target_executor_,
                                                      target_cache_,
                                                      capacity_.common,
                                                      pipeline_);
  ASSERT_TRUE(status.ok()) << status.message();
  auto prefill = input(false);
  prefill.input_params.embedding = {};
  auto decode = input(true, /*token=*/-1, /*position=*/2);
  decode.input_params.embedding = {};
  const auto first = pipeline_->submit(prefill);
  ASSERT_TRUE(first.status.ok()) << first.status.message();
  const auto second = pipeline_->submit(decode);
  ASSERT_TRUE(second.status.ok()) << second.status.message();
  const auto first_result = pipeline_->take_result_async(first.task_id).get();
  ASSERT_TRUE(first_result.status.ok()) << first_result.status.message();
  const auto second_result = pipeline_->take_result_async(second.task_id).get();
  ASSERT_TRUE(second_result.status.ok()) << second_result.status.message();
  EXPECT_EQ(first_result.output.sample_output.next_tokens.item<int64_t>(), 7);
  EXPECT_EQ(second_result.output.sample_output.next_tokens.item<int64_t>(), 12);
  EXPECT_TRUE(torch::equal(target_->forward_inputs.back().cpu(),
                           torch::tensor({7}, torch::kInt32)));
  EXPECT_EQ(cp_leader.consumed(), 2U);
  EXPECT_EQ(tp_leader.consumed(), tensor_parallel ? 2U : 0U);
  EXPECT_EQ(cp_leader.received(0).cpu().item<int64_t>(),
            tensor_parallel ? 8 : 3);
  pipeline_.reset();
}

INSTANTIATE_TEST_SUITE_P(TensorParallel,
                         CpSpeculativePipelineTest,
                         ::testing::Values(false, true));

TEST_F(SpeculativePipelineTest, ExpandedModelCountsPreserveIdlePeerMetadata) {
  capacity_.common.dp_size = 2;
  create();
  ASSERT_NE(pipeline_, nullptr);
  auto prefill = input(false);
  prefill.input_params.parallel.dp_global_token_nums = {2, 0};
  prefill.input_params.parallel.raw_dp_global_token_nums = {2, 0};
  prefill.input_params.parallel.dp_is_decode = {0, 0};
  execute(prefill);
  auto decode = input(true);
  decode.input_params.parallel.dp_global_token_nums = {1, 0};
  decode.input_params.parallel.raw_dp_global_token_nums = {1, 0};
  decode.input_params.parallel.dp_is_decode = {1, 0};
  const auto output = execute(decode);
  ASSERT_TRUE(output.sample_output.next_tokens.defined());
  EXPECT_TRUE(torch::equal(output.sample_output.next_tokens,
                           torch::tensor({{4, 5, 6}}, torch::kInt64)));
  EXPECT_EQ(target_->forward_counts.back(), (std::vector<int32_t>{3, 0}));
  ASSERT_EQ(draft_->forward_counts.size(), 3U);
  EXPECT_EQ(draft_->forward_counts[1], (std::vector<int32_t>{2, 0}));
  EXPECT_EQ(draft_->forward_counts[2], (std::vector<int32_t>{1, 0}));
  decode.input_params.parallel.dp_global_token_nums[1] =
      std::numeric_limits<int32_t>::max() / 3 + 1;
  decode.input_params.parallel.raw_dp_global_token_nums =
      decode.input_params.parallel.dp_global_token_nums;
  decode.input_params.parallel.dp_is_decode[1] = 1;
  EXPECT_FALSE(pipeline_->submit(decode).status.ok());
}

TEST_F(SpeculativePipelineTest,
       DFlashPrefillScattersContextAndDecodeReusesAcceptedPosition) {
  RecordingSamples leader(device_);
  capacity_.common.sampling_group = &leader;
  capacity_.kind = SpeculativeTaskKind::DFLASH;
  capacity_.context_hidden_size = 2;
  capacity_.mask_token_id = 0;
  capacity_.common.enable_mla = false;
  draft_->block_draft = true;
  create();
  ASSERT_NE(pipeline_, nullptr);

  const auto prefill = execute(input(false));
  ASSERT_TRUE(prefill.sample_output.next_tokens.defined());
  EXPECT_TRUE(torch::equal(prefill.sample_output.next_tokens,
                           torch::tensor({3}, torch::kInt64)));
  EXPECT_TRUE(torch::equal(prefill.sample_output.embeddings,
                           torch::tensor({{2.0f, 102.0f}})));
  // Context-only prefill performs a scatter, with no draft model invocation.
  EXPECT_TRUE(draft_->forward_inputs.empty());
  EXPECT_EQ(draft_->logits_calls.load(), 0);
  ASSERT_EQ(draft_->context_hidden.size(), 1U);
  EXPECT_TRUE(torch::equal(draft_->context_hidden.front().cpu(),
                           torch::tensor({{1.0f, 101.0f}, {2.0f, 102.0f}})));
  EXPECT_TRUE(torch::equal(draft_->context_slots.front().cpu(),
                           torch::tensor({0, 1}, torch::kInt32)));

  const auto decoded = execute(input(true));
  ASSERT_TRUE(decoded.sample_output.next_tokens.defined());
  EXPECT_TRUE(torch::equal(decoded.sample_output.next_tokens,
                           torch::tensor({{4, 5, 6}}, torch::kInt64)));
  ASSERT_EQ(draft_->forward_inputs.size(), 1U);
  EXPECT_TRUE(torch::equal(draft_->forward_inputs.front().cpu(),
                           torch::tensor({3, 0, 0}, torch::kInt32)));
  ASSERT_EQ(draft_->context_slots.size(), 2U);
  EXPECT_TRUE(torch::equal(draft_->context_slots.back().cpu(),
                           torch::tensor({2, 3, 4}, torch::kInt32)));

  // The scheduler's placeholder advances by one; accepted context advances
  // by three. The next round must consume the device-published position.
  const auto reused = execute(input(true, /*token=*/-1, /*position=*/3));
  ASSERT_TRUE(reused.sample_output.next_tokens.defined());
  EXPECT_TRUE(torch::equal(reused.sample_output.next_tokens,
                           torch::tensor({{7, 8, 9}}, torch::kInt64)));
  ASSERT_EQ(draft_->forward_inputs.size(), 2U);
  EXPECT_TRUE(torch::equal(draft_->forward_inputs.back().cpu(),
                           torch::tensor({6, 0, 0}, torch::kInt32)));
  EXPECT_TRUE(torch::equal(draft_->forward_positions.back().cpu(),
                           torch::tensor({5, 6, 7}, torch::kInt32)));
  EXPECT_TRUE(torch::equal(draft_->context_positions.back().cpu(),
                           torch::tensor({5, 6, 7}, torch::kInt32)));
  EXPECT_TRUE(torch::equal(decoded.sample_output.next_tokens,
                           torch::tensor({{4, 5, 6}}, torch::kInt64)));
  // DFlash retains its existing result consensus even for greedy CP=1.
  EXPECT_EQ(leader.broadcasts(), 3U);
  pipeline_.reset();
}

TEST_F(SpeculativePipelineTest, DSparkPartialAcceptanceAdvancesAcrossTwoSlots) {
  capacity_.kind = SpeculativeTaskKind::DSPARK;
  capacity_.common.model.max_sequences = 1;
  capacity_.common.slot_count = 2;
  capacity_.context_hidden_size = 2;
  capacity_.mask_token_id = 0;
  draft_->block_draft = true;
  draft_->markov_increments = {1, 2};
  create();
  ASSERT_NE(pipeline_, nullptr);
  execute(input(false));
  // Keep both results unconsumed while submitting, so the second task must
  // use Slot 1 and consume Slot 0's device-published accepted context.
  const auto first_submission = pipeline_->submit(input(true));
  ASSERT_TRUE(first_submission.status.ok());
  const auto second_submission =
      pipeline_->submit(input(true, /*token=*/-1, /*position=*/3));
  ASSERT_TRUE(second_submission.status.ok());
  auto first_result =
      pipeline_->take_result_async(first_submission.task_id).get();
  auto second_result =
      pipeline_->take_result_async(second_submission.task_id).get();
  ASSERT_TRUE(first_result.status.ok());
  ASSERT_TRUE(second_result.status.ok());
  const auto& first = first_result.output;
  const auto& second = second_result.output;
  ASSERT_EQ(draft_->forward_inputs.size(), 2U);
  ASSERT_EQ(draft_->context_slots.size(), 3U);
  EXPECT_TRUE(torch::equal(first.sample_output.next_tokens,
                           torch::tensor({{4, 5, -1}}, torch::kInt64)));
  EXPECT_TRUE(torch::equal(draft_->forward_inputs.front().cpu(),
                           torch::tensor({3, 0}, torch::kInt32)));
  EXPECT_TRUE(torch::equal(draft_->context_slots[1].cpu(),
                           torch::tensor({2, 3, -1}, torch::kInt32)));
  EXPECT_TRUE(torch::equal(second.sample_output.next_tokens,
                           torch::tensor({{6, 7, -1}}, torch::kInt64)));
  EXPECT_TRUE(torch::equal(draft_->forward_inputs.back().cpu(),
                           torch::tensor({5, 0}, torch::kInt32)));
  EXPECT_TRUE(torch::equal(draft_->forward_positions.back().cpu(),
                           torch::tensor({4, 5}, torch::kInt32)));
  EXPECT_TRUE(torch::equal(draft_->context_slots.back().cpu(),
                           torch::tensor({4, 5, -1}, torch::kInt32)));
  const auto reused = execute(input(true, /*token=*/-1, /*position=*/4));
  EXPECT_TRUE(torch::equal(reused.sample_output.next_tokens,
                           torch::tensor({{8, 9, -1}}, torch::kInt64)));
  // Reusing Slot 0 must not overwrite its already consumed result.
  EXPECT_TRUE(torch::equal(first.sample_output.next_tokens,
                           torch::tensor({{4, 5, -1}}, torch::kInt64)));
}

TEST_F(SpeculativePipelineTest,
       DSparkGraphWarmupPlansDraftBindingsOnlyForDecode) {
  capacity_.kind = SpeculativeTaskKind::DSPARK;
  capacity_.common.model.max_sequences = 1;
  capacity_.common.slot_count = 2;
  capacity_.common.max_graph_batch_size = 8;
  capacity_.num_speculative_tokens = 7;
  capacity_.context_hidden_size = 2;
  capacity_.mask_token_id = 0;
  draft_->block_draft = true;
  create();
  ASSERT_NE(pipeline_, nullptr);

  for (const BatchForwardType::Value phase :
       {BatchForwardType::PREFILL, BatchForwardType::CHUNKED_PREFILL}) {
    auto warmup = input(false);
    warmup.input_params.meta.batch_forward_type = phase;
    warmup.input_params.meta.is_graph_warmup = true;
    const auto output = execute(warmup);
    EXPECT_TRUE(output.is_graph_warmup);
    EXPECT_TRUE(torch::equal(output.sample_output.next_tokens,
                             torch::tensor({3}, torch::kInt64)));
    EXPECT_TRUE(draft_->forward_inputs.empty());
  }
  ASSERT_EQ(draft_->context_slots.size(), 2U);

  auto decode_warmup = input(true);
  decode_warmup.input_params.meta.is_graph_warmup = true;
  const auto output = execute(decode_warmup);
  EXPECT_TRUE(output.is_graph_warmup);
  EXPECT_TRUE(
      torch::equal(output.sample_output.next_tokens,
                   torch::tensor({{4, 5, 6, 7, 8, 9, 10, 11}}, torch::kInt64)));
  // Both Slots warm their final eight-row context binding without KV writes.
  ASSERT_EQ(draft_->context_slots.size(), 5U);
  for (uint32_t slot = 2; slot < 4; ++slot) {
    EXPECT_TRUE(draft_->context_warmup[slot]);
    EXPECT_TRUE(torch::equal(draft_->context_slots[slot].cpu(),
                             torch::full({8}, -1, torch::kInt32)));
  }
  ASSERT_EQ(draft_->forward_inputs.size(), 1U);
  EXPECT_EQ(draft_->forward_inputs.front().numel(), 7);

  const auto next = execute(input(true, /*token=*/-1, /*position=*/3));
  EXPECT_FALSE(next.is_graph_warmup);
  EXPECT_TRUE(torch::equal(
      next.sample_output.next_tokens,
      torch::tensor({{12, 13, 14, 15, 16, 17, 18, 19}}, torch::kInt64)));
  ASSERT_EQ(draft_->context_slots.size(), 6U);
  EXPECT_FALSE(draft_->context_warmup.back());
}

TEST_F(SpeculativePipelineTest,
       DSparkMarkovChainPublishesAcceptedContextAcrossBothSlots) {
  capacity_.kind = SpeculativeTaskKind::DSPARK;
  capacity_.common.model.max_sequences = 1;
  capacity_.common.slot_count = 2;
  capacity_.context_hidden_size = 2;
  capacity_.mask_token_id = 0;
  draft_->block_draft = true;
  create();
  ASSERT_NE(pipeline_, nullptr);
  execute(input(false));
  auto first = pipeline_->submit(input(true));
  ASSERT_TRUE(first.status.ok()) << first.status.message();
  // The second Task reads the first Task's device state, not this stale
  // scheduler placeholder. Both Slots remain live until their results retire.
  auto second = pipeline_->submit(input(true, /*token=*/-1, /*position=*/3));
  ASSERT_TRUE(second.status.ok()) << second.status.message();
  auto first_result = pipeline_->take_result_async(first.task_id).get();
  auto second_result = pipeline_->take_result_async(second.task_id).get();
  ASSERT_TRUE(first_result.status.ok()) << first_result.status.message();
  ASSERT_TRUE(second_result.status.ok()) << second_result.status.message();
  EXPECT_TRUE(torch::equal(first_result.output.sample_output.next_tokens,
                           torch::tensor({{4, 5, 6}}, torch::kInt64)));
  EXPECT_TRUE(torch::equal(second_result.output.sample_output.next_tokens,
                           torch::tensor({{7, 8, 9}}, torch::kInt64)));
  ASSERT_EQ(draft_->forward_inputs.size(), 2U);
  EXPECT_TRUE(torch::equal(draft_->forward_inputs.front().cpu(),
                           torch::tensor({3, 0}, torch::kInt32)));
  EXPECT_TRUE(torch::equal(draft_->forward_positions.back().cpu(),
                           torch::tensor({5, 6}, torch::kInt32)));
  ASSERT_EQ(draft_->markov_inputs.size(), 4U);
  for (uint32_t step = 0; step < 4; ++step) {
    EXPECT_EQ(draft_->markov_inputs[step].cpu().item<int64_t>(),
              step < 2 ? 3 + step : 4 + step);
  }
  EXPECT_TRUE(torch::equal(draft_->context_slots.back().cpu(),
                           torch::tensor({5, 6, 7}, torch::kInt32)));
}

TEST_F(SpeculativePipelineTest,
       DSparkRejectsWrongMarkovProposalsWithoutPublishingRejectedKv) {
  capacity_.kind = SpeculativeTaskKind::DSPARK;
  capacity_.common.model.max_sequences = 1;
  capacity_.context_hidden_size = 2;
  capacity_.mask_token_id = 0;
  draft_->block_draft = true;
  draft_->markov_increments = {2};
  create();
  execute(input(false));
  const auto rejected = execute(input(true));
  EXPECT_TRUE(torch::equal(rejected.sample_output.next_tokens,
                           torch::tensor({{4, -1, -1}}, torch::kInt64)));
  EXPECT_TRUE(torch::equal(draft_->context_slots.back().cpu(),
                           torch::tensor({2, -1, -1}, torch::kInt32)));
  const auto next = execute(input(true, /*token=*/-1, /*position=*/3));
  EXPECT_TRUE(torch::equal(next.sample_output.next_tokens,
                           torch::tensor({{5, -1, -1}}, torch::kInt64)));
  EXPECT_TRUE(torch::equal(draft_->forward_inputs.back().cpu(),
                           torch::tensor({4, 0}, torch::kInt32)));
}

TEST_F(SpeculativePipelineTest,
       DSparkBatchGraphsReuseWarmBindingsAcrossShapeChanges) {
  configure_dspark_batch_graphs();
  create();
  ASSERT_NE(pipeline_, nullptr);
  warmup_dspark_batch_graphs();
  const size_t target_warmups = target_->graph_warmup_bindings.size();
  const size_t draft_warmups = draft_->graph_warmup_bindings.size();
  const uint32_t target_outputs = target_->graph_output_allocations.load();
  const uint32_t draft_outputs = draft_->graph_output_allocations.load();
  std::vector<std::array<uintptr_t, 4>> warmed_context;
  warmed_context.reserve(draft_->context_bindings.size());
  for (size_t index = 0; index < draft_->context_bindings.size(); ++index) {
    if (draft_->context_warmup[index]) {
      warmed_context.emplace_back(draft_->context_bindings[index]);
    }
  }
  ASSERT_EQ(target_outputs, 6U);
  ASSERT_EQ(draft_outputs, 6U);
  ASSERT_EQ(warmed_context.size(), 6U);

  for (uint32_t batch_size : {1U, 2U, 3U, 2U, 3U}) {
    execute(dspark_batch_input(batch_size, /*decode=*/false));
    const auto output =
        execute(dspark_batch_input(batch_size, /*decode=*/true));
    ASSERT_TRUE(output.sample_output.next_tokens.defined());
    EXPECT_EQ(output.sample_output.next_tokens.size(0), batch_size);
    EXPECT_EQ(output.sample_output.next_tokens.size(1), 8);
    for (uint32_t row = 0; row < batch_size; ++row) {
      const int64_t first_token = 2 * row + 4;
      EXPECT_TRUE(torch::equal(
          output.sample_output.next_tokens.select(0, row),
          torch::arange(first_token, first_token + 8, torch::kInt64)));
    }
    const int64_t target_rows = batch_size == 3 ? 32 : batch_size * 8;
    const int64_t draft_rows = batch_size == 3 ? 28 : batch_size * 7;
    EXPECT_EQ(target_->forward_inputs.back().numel(), target_rows);
    EXPECT_EQ(draft_->forward_inputs.back().numel(), draft_rows);
    EXPECT_EQ(draft_->context_hidden.back().size(0), target_rows);
    const auto& binding = draft_->context_bindings.back();
    EXPECT_NE(std::find(warmed_context.begin(), warmed_context.end(), binding),
              warmed_context.end());
    EXPECT_FALSE(draft_->context_warmup.back());
    if (batch_size == 3) {
      EXPECT_TRUE(
          torch::equal(draft_->context_slots.back().narrow(0, 24, 8).cpu(),
                       torch::full({8}, -1, torch::kInt32)));
    }
    const auto metrics = make_spec_metrics(output.sample_output.next_tokens,
                                           output.spec_verify_layouts);
    EXPECT_EQ(metrics.num_draft_tokens, batch_size * 7);
    EXPECT_EQ(metrics.num_accepted_tokens, batch_size * 7);
    EXPECT_EQ(metrics.num_committed_tokens, batch_size * 8);
    EXPECT_EQ(target_->graph_warmup_bindings.size(), target_warmups);
    EXPECT_EQ(draft_->graph_warmup_bindings.size(), draft_warmups);
    EXPECT_EQ(target_->graph_output_allocations.load(), target_outputs);
    EXPECT_EQ(draft_->graph_output_allocations.load(), draft_outputs);
  }
}

TEST_F(SpeculativePipelineTest,
       DSparkBatchGraphSlotsPreserveIndependentAcceptedContexts) {
  configure_dspark_batch_graphs();
  RecordingSamples leader(device_);
  capacity_.common.sampling_group = &leader;
  create();
  ASSERT_NE(pipeline_, nullptr);
  warmup_dspark_batch_graphs();
  const auto with_logprobs = [](LlmForwardInput input) {
    input.sampling_params.logprobs = true;
    input.sampling_params.max_top_logprobs = 2;
    // The normalized greedy decision comes from SamplingMode, not a
    // temperature shortcut or the presence of output probability fields.
    input.sampling_params.temperatures = torch::ones({3});
    return input;
  };
  const auto prefill =
      execute(with_logprobs(dspark_batch_input(3, /*decode=*/false)));
  EXPECT_TRUE(prefill.logprobs);
  EXPECT_EQ(leader.broadcasts(), 0U);
  draft_->row_markov_increments = torch::tensor({1, 2, 1}, torch::kInt64);
  const size_t context_start = draft_->context_bindings.size();
  const size_t target_warmups = target_->graph_warmup_bindings.size();
  const size_t draft_warmups = draft_->graph_warmup_bindings.size();
  const uint32_t target_outputs = target_->graph_output_allocations.load();
  const uint32_t draft_outputs = draft_->graph_output_allocations.load();
  std::vector<std::array<uintptr_t, 4>> warmed_context;
  warmed_context.reserve(context_start);
  for (size_t index = 0; index < context_start; ++index) {
    if (draft_->context_warmup[index]) {
      warmed_context.emplace_back(draft_->context_bindings[index]);
    }
  }

  const auto first =
      pipeline_->submit(with_logprobs(dspark_batch_input(3, /*decode=*/true)));
  ASSERT_TRUE(first.status.ok()) << first.status.message();
  const auto second = pipeline_->submit(with_logprobs(dspark_batch_input(
      3, /*decode=*/true, /*placeholder=*/true, /*position=*/3)));
  ASSERT_TRUE(second.status.ok()) << second.status.message();
  const auto first_result = pipeline_->take_result_async(first.task_id).get();
  const auto second_result = pipeline_->take_result_async(second.task_id).get();
  ASSERT_TRUE(first_result.status.ok()) << first_result.status.message();
  ASSERT_TRUE(second_result.status.ok()) << second_result.status.message();
  const auto expected_first = torch::tensor({{4, 5, 6, 7, 8, 9, 10, 11},
                                             {6, -1, -1, -1, -1, -1, -1, -1},
                                             {8, 9, 10, 11, 12, 13, 14, 15}},
                                            torch::kInt64);
  const auto expected_second = torch::tensor({{12, 13, 14, 15, 16, 17, 18, 19},
                                              {7, -1, -1, -1, -1, -1, -1, -1},
                                              {16, 17, 18, 19, 20, 21, 22, 23}},
                                             torch::kInt64);
  EXPECT_TRUE(torch::equal(first_result.output.sample_output.next_tokens,
                           expected_first));
  EXPECT_TRUE(torch::equal(second_result.output.sample_output.next_tokens,
                           expected_second));
  for (const auto* result : {&first_result.output, &second_result.output}) {
    EXPECT_TRUE(result->logprobs);
    EXPECT_EQ(result->max_top_logprobs, 2);
    EXPECT_TRUE(torch::allclose(
        result->sample_output.logprobs.select(/*dim=*/0, /*index=*/0),
        torch::zeros({8})));
    EXPECT_TRUE(torch::equal(
        result->sample_output.top_tokens.select(/*dim=*/0, /*index=*/0)
            .select(/*dim=*/1, /*index=*/0),
        result->sample_output.next_tokens.select(/*dim=*/0, /*index=*/0)));
  }
  ASSERT_EQ(draft_->context_bindings.size(), context_start + 2);
  for (size_t index = context_start; index < context_start + 2; ++index) {
    EXPECT_NE(std::find(warmed_context.begin(),
                        warmed_context.end(),
                        draft_->context_bindings[index]),
              warmed_context.end());
  }
  EXPECT_EQ(draft_->context_hidden.back().size(0), 32);
  EXPECT_TRUE(torch::equal(
      draft_->context_slots[context_start].cpu(),
      torch::tensor(
          {18, 19, 20, 21, 22, 23, 24, 25, 50, -1, -1, -1, -1, -1, -1, -1,
           82, 83, 84, 85, 86, 87, 88, 89, -1, -1, -1, -1, -1, -1, -1, -1},
          torch::kInt32)));
  EXPECT_TRUE(torch::equal(
      draft_->context_slots.back().cpu(),
      torch::tensor(
          {26, 27, 28, 29, 30, 31, 32, 33, 51, -1, -1, -1, -1, -1, -1, -1,
           90, 91, 92, 93, 94, 95, 96, 97, -1, -1, -1, -1, -1, -1, -1, -1},
          torch::kInt32)));
  EXPECT_TRUE(torch::equal(
      draft_->context_positions.back().narrow(0, 0, 24).cpu(),
      torch::tensor({10, 11, 12, 13, 14, 15, 16, 17, 3,  4,  5,  6,
                     7,  8,  9,  10, 10, 11, 12, 13, 14, 15, 16, 17},
                    torch::kInt32)));
  // Padding in the B=4 physical draft bucket must not create a fourth request.
  EXPECT_EQ(draft_->forward_inputs.back().numel(), 28);
  EXPECT_EQ(draft_->markov_inputs.back().numel(), 3);
  const auto metrics =
      make_spec_metrics(second_result.output.sample_output.next_tokens,
                        second_result.output.spec_verify_layouts);
  ASSERT_EQ(metrics.sequence_stats.size(), 3U);
  EXPECT_EQ(metrics.num_draft_tokens, 21);
  EXPECT_EQ(metrics.num_accepted_tokens, 14);
  EXPECT_EQ(metrics.num_committed_tokens, 17);
  EXPECT_EQ(metrics.sequence_stats[0].accepted_tokens, 7);
  EXPECT_EQ(metrics.sequence_stats[1].accepted_tokens, 0);
  EXPECT_EQ(metrics.sequence_stats[2].accepted_tokens, 7);
  EXPECT_EQ(target_->graph_output_allocations.load(), target_outputs);
  EXPECT_EQ(draft_->graph_output_allocations.load(), draft_outputs);
  EXPECT_EQ(target_->graph_warmup_bindings.size(), target_warmups);
  EXPECT_EQ(draft_->graph_warmup_bindings.size(), draft_warmups);
  // Retiring the second Slot must not mutate the first Slot's CPU result.
  EXPECT_TRUE(torch::equal(first_result.output.sample_output.next_tokens,
                           expected_first));
  EXPECT_EQ(leader.broadcasts(), 0U);
  pipeline_.reset();
}

TEST_F(SpeculativePipelineTest,
       DSparkMixedTargetsKeepLeaderConsensusWithGreedyDraft) {
  configure_dspark_batch_graphs();
  capacity_.draft_sampling_mode = DraftSamplingMode::GREEDY;
  const auto first_tokens = torch::tensor(
      {{8, -1, -1, -1, -1, -1, -1, -1}, {10, -1, -1, -1, -1, -1, -1, -1}},
      torch::kInt64);
  const auto second_tokens = torch::tensor(
      {{9, -1, -1, -1, -1, -1, -1, -1}, {11, -1, -1, -1, -1, -1, -1, -1}},
      torch::kInt64);
  const auto first_top =
      first_tokens.clamp_min(0).unsqueeze(/*dim=*/2).expand({2, 8, 2}).clone();
  const auto second_top =
      second_tokens.clamp_min(0).unsqueeze(/*dim=*/2).expand({2, 8, 2}).clone();
  RecordingSamples leader(device_,
                          {torch::tensor({7, 9}, torch::kInt64),
                           torch::tensor({-3.0F, -5.0F}),
                           torch::tensor({{7, 8}, {9, 10}}, torch::kInt64),
                           torch::tensor({{-3.0F, -4.0F}, {-5.0F, -6.0F}}),
                           first_tokens,
                           torch::full({2, 8}, -6.0F),
                           first_top,
                           torch::full({2, 8, 2}, -8.0F),
                           second_tokens,
                           torch::full({2, 8}, -7.0F),
                           second_top,
                           torch::full({2, 8, 2}, -9.0F)});
  capacity_.common.sampling_group = &leader;
  create();
  ASSERT_NE(pipeline_, nullptr);
  warmup_dspark_batch_graphs();
  EXPECT_EQ(leader.broadcasts(), 0U);
  const auto mixed = [](LlmForwardInput input) {
    input.sampling_params.do_sample =
        torch::tensor({false, true}, torch::kBool);
    // Zero temperatures must not bypass consensus for an explicit mixed batch.
    input.sampling_params.sampling_mode = SamplingMode::MIXED;
    input.sampling_params.temperatures = torch::zeros({2});
    input.sampling_params.logprobs = true;
    input.sampling_params.max_top_logprobs = 2;
    return input;
  };
  const auto check_probabilities = [](const ForwardOutput& output,
                                      const torch::Tensor& top_tokens,
                                      float logprob,
                                      float top_logprob) {
    const auto& sampled = output.sample_output;
    EXPECT_TRUE(torch::equal(sampled.logprobs.select(/*dim=*/1, /*index=*/0),
                             torch::full({2}, logprob)));
    EXPECT_TRUE(torch::equal(sampled.top_tokens.select(/*dim=*/1, /*index=*/0),
                             top_tokens.select(/*dim=*/1, /*index=*/0)));
    EXPECT_TRUE(
        torch::equal(sampled.top_logprobs.select(/*dim=*/1, /*index=*/0),
                     torch::full({2, 2}, top_logprob)));
    // The leader commits one token per row. Host snapshots copy only that
    // valid prefix and reset the remaining probability fields to padding.
    EXPECT_TRUE(torch::equal(
        sampled.logprobs.narrow(/*dim=*/1, /*start=*/1, /*length=*/7),
        torch::full({2, 7}, -std::numeric_limits<float>::infinity())));
    EXPECT_TRUE(torch::equal(
        sampled.top_tokens.narrow(/*dim=*/1, /*start=*/1, /*length=*/7),
        torch::full({2, 7, 2}, -1, torch::kInt64)));
    EXPECT_TRUE(torch::equal(
        sampled.top_logprobs.narrow(/*dim=*/1, /*start=*/1, /*length=*/7),
        torch::full({2, 7, 2}, -std::numeric_limits<float>::infinity())));
  };
  const auto prefill = execute(mixed(dspark_batch_input(2, /*decode=*/false)));
  EXPECT_TRUE(torch::equal(prefill.sample_output.next_tokens,
                           torch::tensor({7, 9}, torch::kInt64)));
  EXPECT_TRUE(torch::equal(prefill.sample_output.logprobs,
                           torch::tensor({-3.0F, -5.0F})));
  EXPECT_EQ(leader.broadcasts(), 4U);
  const size_t markov_start = draft_->markov_inputs.size();
  const auto first = execute(mixed(dspark_batch_input(2, /*decode=*/true)));
  EXPECT_TRUE(torch::equal(first.sample_output.next_tokens, first_tokens));
  check_probabilities(
      first, first_top, /*logprob=*/-6.0F, /*top_logprob=*/-8.0F);
  // Only the four accepted-result fields broadcast; seven greedy draft
  // proposals do not. The target leader's anchor drives each local chain.
  EXPECT_EQ(leader.broadcasts(), 8U);
  ASSERT_GT(draft_->markov_inputs.size(), markov_start);
  EXPECT_TRUE(torch::equal(draft_->markov_inputs[markov_start].cpu(),
                           torch::tensor({7, 9}, torch::kInt64)));
  const auto second = execute(mixed(dspark_batch_input(
      2, /*decode=*/true, /*placeholder=*/true, /*position=*/3)));
  EXPECT_TRUE(torch::equal(second.sample_output.next_tokens, second_tokens));
  check_probabilities(
      second, second_top, /*logprob=*/-7.0F, /*top_logprob=*/-9.0F);
  EXPECT_EQ(leader.broadcasts(), 12U);
  EXPECT_TRUE(torch::equal(draft_->markov_inputs[markov_start + 7].cpu(),
                           torch::tensor({8, 10}, torch::kInt64)));
  EXPECT_TRUE(torch::equal(
      draft_->context_slots.back().cpu(),
      torch::tensor(
          {19, -1, -1, -1, -1, -1, -1, -1, 51, -1, -1, -1, -1, -1, -1, -1},
          torch::kInt32)));
  EXPECT_TRUE(torch::equal(
      draft_->context_positions.back().cpu().view({2, 8}).select(1, 0),
      torch::tensor({3, 3}, torch::kInt32)));
  EXPECT_TRUE(torch::equal(first.sample_output.next_tokens, first_tokens));
  pipeline_.reset();
}

TEST_F(SpeculativePipelineTest, DSparkRejectsUnsupportedTopologyBeforeLaunch) {
  capacity_.kind = SpeculativeTaskKind::DSPARK;
  capacity_.context_hidden_size = 0;
  capacity_.mask_token_id = 0;
  EXPECT_FALSE(TaskExecutionPipeline::create(
                   state_thread_,
                   {*target_, *target_executor_, target_cache_},
                   {*draft_, *draft_executor_, draft_cache_},
                   capacity_,
                   pipeline_)
                   .ok());
  capacity_.common.model.max_sequences = 1;
  capacity_.context_hidden_size = 2;
  capacity_.common.dp_size = 2;
  EXPECT_FALSE(TaskExecutionPipeline::create(
                   state_thread_,
                   {*target_, *target_executor_, target_cache_},
                   {*draft_, *draft_executor_, draft_cache_},
                   capacity_,
                   pipeline_)
                   .ok());
  EXPECT_EQ(pipeline_, nullptr);
}

TEST_F(SpeculativePipelineTest,
       AdaptiveTargetBindingsReuseFullCapacitySourcesAcrossRequestCounts) {
  // Without measured planner costs every round remains full width, so B=3
  // and B=4 both choose G=32 and exercise the same per-Slot target binding.
  configure_adaptive_dspark(/*profiled=*/false);
  // Four one-based page pairs include page 8 even before the second page is
  // used. Admission validates the whole table, including reserved page 0.
  for (auto* caches : {&target_cache_, &draft_cache_}) {
    caches->clear();
    caches->emplace_back(
        KVCacheTensors{torch::zeros({9, 16, 1, 1}, target_->options()),
                       torch::zeros({9, 16, 1, 1}, target_->options())});
  }
  create();
  ASSERT_NE(pipeline_, nullptr);
  warmup_dspark_batch_graphs();
  const auto warmed = target_->adaptive_warmup_bindings;
  const auto target_warmups = target_->graph_warmup_bindings;
  const auto draft_warmups = draft_->graph_warmup_bindings;
  const uint32_t target_outputs = target_->graph_output_allocations.load();
  ASSERT_FALSE(target_->adaptive_controls.empty());
  for (const auto& control : target_->adaptive_controls) {
    EXPECT_EQ(static_cast<uint64_t>(control.item<int64_t>()) >> 32, 1U);
  }
  execute(dspark_batch_input(4, /*decode=*/false));
  const size_t first_decode = target_->adaptive_forward_bindings.size();
  // Retain the first result until both Tasks are submitted, so the second
  // Decode must borrow the other Slot rather than the just-retired first one.
  const auto first_task =
      pipeline_->submit(dspark_batch_input(4, /*decode=*/true));
  ASSERT_TRUE(first_task.status.ok()) << first_task.status.message();
  const auto second_task = pipeline_->submit(dspark_batch_input(
      3, /*decode=*/true, /*placeholder=*/true, /*position=*/10));
  ASSERT_TRUE(second_task.status.ok()) << second_task.status.message();
  const auto first_result =
      pipeline_->take_result_async(first_task.task_id).get();
  const auto second_result =
      pipeline_->take_result_async(second_task.task_id).get();
  ASSERT_TRUE(first_result.status.ok()) << first_result.status.message();
  ASSERT_TRUE(second_result.status.ok()) << second_result.status.message();
  const auto& first = first_result.output;
  const auto& second = second_result.output;
  // Both preceding results have retired. These are known current anchors,
  // not a pending placeholder that needs another seven-token upper bound.
  auto third_input = dspark_batch_input(
      3, /*decode=*/true, /*placeholder=*/false, /*position=*/18);
  third_input.token_ids = torch::tensor({19, 21, 23}, torch::kInt32);
  const auto third = execute(third_input);
  ASSERT_EQ(target_->adaptive_forward_bindings.size(), first_decode + 3);
  ASSERT_EQ(target_->adaptive_controls.size(), first_decode + 3);
  const auto& first_binding = target_->adaptive_forward_bindings[first_decode];
  const auto& second_binding =
      target_->adaptive_forward_bindings[first_decode + 1];
  const auto& third_binding =
      target_->adaptive_forward_bindings[first_decode + 2];
  EXPECT_EQ(first_binding.back(), 32U);
  EXPECT_EQ(second_binding.back(), 32U);
  EXPECT_EQ(first_binding, third_binding);
  EXPECT_NE(first_binding[0], second_binding[0]);
  EXPECT_NE(first_binding[1], second_binding[1]);
  for (size_t index = 0; index < 3; ++index) {
    EXPECT_EQ(target_->adaptive_controls[first_decode + index].item<int64_t>(),
              index == 0 ? 4 : 3);
  }
  EXPECT_EQ(target_->adaptive_warmup_bindings, warmed);
  EXPECT_EQ(target_->graph_warmup_bindings, target_warmups);
  EXPECT_EQ(draft_->graph_warmup_bindings, draft_warmups);
  EXPECT_EQ(target_->graph_output_allocations.load(), target_outputs);
  EXPECT_TRUE(
      torch::equal(first.sample_output.next_tokens[0],
                   torch::tensor({4, 5, 6, 7, 8, 9, 10, 11}, torch::kInt64)));
  EXPECT_TRUE(torch::equal(
      second.sample_output.next_tokens[0],
      torch::tensor({12, 13, 14, 15, 16, 17, 18, 19}, torch::kInt64)));
  EXPECT_TRUE(torch::equal(
      third.sample_output.next_tokens[0],
      torch::tensor({20, 21, 22, 23, 24, 25, 26, 27}, torch::kInt64)));
  EXPECT_TRUE(torch::equal(
      draft_->context_slots.back().narrow(0, 0, 8).cpu(),
      torch::tensor({34, 35, 36, 37, 38, 39, 40, 41}, torch::kInt32)));
}

TEST_F(SpeculativePipelineTest,
       AdaptiveDSparkTraceUsesRetiredHostFeedbackAndKeepsZeroBudgetRows) {
  AdaptiveTraceSink trace(/*level=*/2);
  configure_adaptive_dspark();
  create();
  ASSERT_NE(pipeline_, nullptr);
  warmup_dspark_batch_graphs();
  execute(dspark_batch_input(3, /*decode=*/false));
  EXPECT_TRUE(trace.snapshots().empty());
  const auto first = pipeline_->submit(dspark_batch_input(3, /*decode=*/true));
  ASSERT_TRUE(first.status.ok());
  const auto second = pipeline_->submit(dspark_batch_input(
      3, /*decode=*/true, /*placeholder=*/true, /*position=*/3));
  ASSERT_TRUE(second.status.ok());
  ASSERT_TRUE(pipeline_->take_result_async(first.task_id).get().status.ok());
  ASSERT_TRUE(pipeline_->take_result_async(second.task_id).get().status.ok());
  execute(dspark_batch_input(
      3, /*decode=*/true, /*placeholder=*/false, /*position=*/18));
  pipeline_.reset();
  const auto records = trace.snapshots();
  ASSERT_EQ(records.size(), 3U);
  EXPECT_EQ(records[0]["draft_counts"], nlohmann::json({7, 7, 7}));
  EXPECT_TRUE(records[0]["lagged_conditional_confidence"].empty());
  const auto& last = records.back();
  EXPECT_EQ(last["logical_requests"], 3);
  EXPECT_EQ(last["physical_requests"], 4);
  EXPECT_EQ(last["target_graph_rows"], 4);
  EXPECT_EQ(last["selected_draft_tokens"], 1);
  EXPECT_EQ(last["draft_counts"], nlohmann::json({1, 0, 0}));
  EXPECT_EQ(last["accepted_counts"], nlohmann::json({1, 0, 0}));
  EXPECT_EQ(last["request_ids"].size(), 3U);
  EXPECT_EQ(last["conditional_confidence"].size(), 21U);
  EXPECT_EQ(last["lagged_conditional_confidence"].size(), 21U);
  EXPECT_EQ(last["eligible_feedback_generation"], records[0]["generation"]);
}

TEST_F(SpeculativePipelineTest,
       AdaptiveDSparkTraceIsDisabledAtNormalVerbosity) {
  AdaptiveTraceSink trace(/*level=*/0);
  configure_adaptive_dspark();
  create();
  ASSERT_NE(pipeline_, nullptr);
  warmup_dspark_batch_graphs();
  execute(dspark_batch_input(3, /*decode=*/false));
  execute(dspark_batch_input(3, /*decode=*/true));
  pipeline_.reset();
  EXPECT_TRUE(trace.snapshots().empty());
}

TEST_F(SpeculativePipelineTest,
       AdaptiveDSparkTraceDoesNotDuplicatePeerRecords) {
  AdaptiveTraceSink trace(/*level=*/2);
  configure_adaptive_dspark();
  RecordingSamples peer(device_);
  capacity_.common.sampling_group = &peer;
  create();
  ASSERT_NE(pipeline_, nullptr);
  warmup_dspark_batch_graphs();
  execute(dspark_batch_input(3, /*decode=*/false));
  execute(dspark_batch_input(3, /*decode=*/true));
  pipeline_.reset();
  EXPECT_TRUE(trace.snapshots().empty());
}

TEST_F(SpeculativePipelineTest,
       AdaptiveDSparkRetiresLaggedFeedbackWithoutChangingQueuedPlans) {
  configure_adaptive_dspark();
  RecordingSamples leader(device_);
  capacity_.common.sampling_group = &leader;
  create();
  ASSERT_NE(pipeline_, nullptr);
  warmup_dspark_batch_graphs();
  const size_t startup_broadcasts = leader.broadcasts();
  execute(dspark_batch_input(3, /*decode=*/false));
  EXPECT_EQ(leader.broadcasts(), startup_broadcasts);
  const size_t target_warmups = target_->graph_warmup_bindings.size();
  const size_t draft_warmups = draft_->graph_warmup_bindings.size();
  const uint32_t target_outputs = target_->graph_output_allocations.load();
  const uint32_t draft_outputs = draft_->graph_output_allocations.load();

  const auto first = pipeline_->submit(dspark_batch_input(3, /*decode=*/true));
  ASSERT_TRUE(first.status.ok()) << first.status.message();
  const auto second = pipeline_->submit(dspark_batch_input(
      3, /*decode=*/true, /*placeholder=*/true, /*position=*/3));
  ASSERT_TRUE(second.status.ok()) << second.status.message();
  const auto first_result = pipeline_->take_result_async(first.task_id).get();
  const auto second_result = pipeline_->take_result_async(second.task_id).get();
  ASSERT_TRUE(first_result.status.ok());
  ASSERT_TRUE(second_result.status.ok());
  EXPECT_EQ(leader.broadcasts(), startup_broadcasts + 2);
  // Both plans were immutable full width before any usable feedback retired.
  EXPECT_EQ(first_result.output.spec_verify_layouts[0].verify_len, 7);
  EXPECT_EQ(second_result.output.spec_verify_layouts[0].verify_len, 7);
  EXPECT_EQ(
      second_result.output.sample_output.next_tokens[0][7].item<int64_t>(), 19);
  const auto output = execute(dspark_batch_input(
      3, /*decode=*/true, /*placeholder=*/false, /*position=*/18));
  EXPECT_TRUE(torch::equal(output.sample_output.next_tokens,
                           torch::tensor({{20, 21, -1, -1, -1, -1, -1, -1},
                                          {22, -1, -1, -1, -1, -1, -1, -1},
                                          {24, -1, -1, -1, -1, -1, -1, -1}},
                                         torch::kInt64)));
  ASSERT_EQ(output.spec_verify_layouts.size(), 3U);
  EXPECT_EQ(output.spec_verify_layouts[0].verify_len, 1);
  EXPECT_EQ(output.spec_verify_layouts[1].verify_len, 0);
  EXPECT_EQ(output.spec_verify_layouts[2].verify_len, 0);
  EXPECT_EQ(target_->forward_inputs.back().numel(), 4);
  EXPECT_EQ(draft_->forward_inputs.back().numel(), 28);
  EXPECT_EQ(draft_->context_hidden.back().size(0), 4);
  EXPECT_TRUE(torch::equal(draft_->context_slots.back().cpu(),
                           torch::tensor({34, 35, 66, 98}, torch::kInt32)));
  EXPECT_TRUE(torch::equal(draft_->context_positions.back().cpu(),
                           torch::tensor({18, 19, 18, 18}, torch::kInt32)));
  EXPECT_TRUE(torch::equal(draft_->confidence_previous.back().cpu(),
                           torch::tensor({{19, 20, 21, 22, 23, 24, 25},
                                          {21, 22, 23, 24, 25, 26, 27},
                                          {23, 24, 25, 26, 27, 28, 29}},
                                         torch::kInt64)));
  const auto metrics = make_spec_metrics(output.sample_output.next_tokens,
                                         output.spec_verify_layouts);
  EXPECT_EQ(metrics.num_draft_tokens, 1);
  EXPECT_EQ(metrics.num_accepted_tokens, 1);
  EXPECT_EQ(metrics.num_committed_tokens, 4);
  EXPECT_EQ(target_->graph_warmup_bindings.size(), target_warmups);
  EXPECT_EQ(draft_->graph_warmup_bindings.size(), draft_warmups);
  EXPECT_EQ(target_->graph_output_allocations.load(), target_outputs);
  EXPECT_EQ(draft_->graph_output_allocations.load(), draft_outputs);
  // Returned snapshots outlive both Slot reuse and smaller physical graphs.
  EXPECT_EQ(first_result.output.sample_output.next_tokens[0][7].item<int64_t>(),
            11);
  ASSERT_EQ(leader.broadcasts(), startup_broadcasts + 3);
  // Each of the three serving Decodes publishes exactly one authoritative
  // proposal/confidence arena. Greedy target IDs still require no broadcast.
  for (size_t index = startup_broadcasts; index < startup_broadcasts + 3;
       ++index) {
    EXPECT_EQ(leader.received(index).scalar_type(), torch::kUInt8);
    EXPECT_EQ(leader.received(index).numel(), 3 * 84);
  }
  pipeline_.reset();
}

TEST_F(SpeculativePipelineTest,
       AdaptiveDraftFiveSupportsOneRequestSlotAndLaggedFeedback) {
  configure_adaptive_dspark(/*profiled=*/true, /*draft_width=*/5);
  capacity_.common.model.max_sequences = 1;
  create();
  ASSERT_NE(pipeline_, nullptr);
  warmup_dspark_batch_graphs();
  execute(dspark_batch_input(1, /*decode=*/false));
  const size_t warmups = target_->graph_warmup_bindings.size();
  for (uint32_t round = 0; round < 4; ++round) {
    const auto output = execute(dspark_batch_input(1, /*decode=*/true));
    ASSERT_EQ(output.sample_output.next_tokens.size(0), 1);
    ASSERT_EQ(output.sample_output.next_tokens.size(1), 6);
    ASSERT_EQ(output.spec_verify_layouts.size(), 1U);
    EXPECT_LE(output.spec_verify_layouts.front().verify_len, 5);
    EXPECT_EQ(draft_->confidence_previous.back().size(1), 5);
  }
  EXPECT_EQ(target_->graph_warmup_bindings.size(), warmups);
  pipeline_.reset();
}

TEST_F(SpeculativePipelineTest,
       AdaptiveDSparkRequiresMeasuredCostsAndFreshRequestFeedback) {
  configure_adaptive_dspark(/*profiled=*/false);
  create();
  ASSERT_NE(pipeline_, nullptr);
  warmup_dspark_batch_graphs();
  execute(dspark_batch_input(2, /*decode=*/false));
  execute(dspark_batch_input(2, /*decode=*/true));
  execute(dspark_batch_input(2,
                             /*decode=*/true,
                             /*placeholder=*/false,
                             /*position=*/10));
  const auto full = execute(dspark_batch_input(2,
                                               /*decode=*/true,
                                               /*placeholder=*/false,
                                               /*position=*/18));
  ASSERT_EQ(full.spec_verify_layouts.size(), 2U);
  EXPECT_EQ(full.spec_verify_layouts[0].verify_len, 7);
  EXPECT_EQ(full.spec_verify_layouts[1].verify_len, 7);
  EXPECT_EQ(target_->forward_inputs.back().numel(), 16);

  // A new Prefill generation using the same embedding ids must discard old
  // confidence even when a test deliberately repeats its request-id strings.
  pipeline_.reset();
  configure_adaptive_dspark();
  create();
  ASSERT_NE(pipeline_, nullptr);
  warmup_dspark_batch_graphs();
  execute(dspark_batch_input(2, /*decode=*/false));
  execute(dspark_batch_input(2, /*decode=*/true));
  execute(dspark_batch_input(2,
                             /*decode=*/true,
                             /*placeholder=*/false,
                             /*position=*/10));
  const auto pruned = execute(dspark_batch_input(2,
                                                 /*decode=*/true,
                                                 /*placeholder=*/false,
                                                 /*position=*/18));
  EXPECT_EQ(pruned.spec_verify_layouts[0].verify_len, 0);
  EXPECT_EQ(pruned.spec_verify_layouts[1].verify_len, 0);
  EXPECT_EQ(target_->forward_inputs.back().numel(), 2);
  execute(dspark_batch_input(2, /*decode=*/false));
  const auto reset = execute(dspark_batch_input(2, /*decode=*/true));
  EXPECT_EQ(reset.spec_verify_layouts[0].verify_len, 7);
  EXPECT_EQ(reset.spec_verify_layouts[1].verify_len, 7);
  EXPECT_EQ(target_->forward_inputs.back().numel(), 16);
}

TEST_F(SpeculativePipelineTest,
       AdaptiveDSparkGreedyTargetIdsSkipBroadcastAfterArenaConsensus) {
  configure_adaptive_dspark();
  RecordingSamples leader(device_);
  capacity_.common.sampling_group = &leader;
  create();
  ASSERT_NE(pipeline_, nullptr);
  warmup_dspark_batch_graphs();
  const size_t startup_broadcasts = leader.broadcasts();
  execute(dspark_batch_input(3, /*decode=*/false));
  EXPECT_EQ(leader.broadcasts(), startup_broadcasts);
  const auto output = execute(dspark_batch_input(3, /*decode=*/true));
  ASSERT_EQ(leader.broadcasts(), startup_broadcasts + 1);
  const auto& consensus = leader.received(startup_broadcasts);
  ASSERT_EQ(consensus.scalar_type(), torch::kUInt8);
  ASSERT_EQ(consensus.numel(), 3 * 84);
  const auto arena = consensus.cpu();
  EXPECT_TRUE(
      torch::equal(arena.narrow(/*dim=*/0, /*start=*/0, /*length=*/3 * 56)
                       .view(torch::kInt64)
                       .view({3, 7}),
                   torch::tensor({{4, 5, 6, 7, 8, 9, 10},
                                  {6, 7, 8, 9, 10, 11, 12},
                                  {8, 9, 10, 11, 12, 13, 14}},
                                 torch::kInt64)));
  EXPECT_TRUE(
      torch::equal(arena.narrow(/*dim=*/0, /*start=*/3 * 56, /*length=*/3 * 28)
                       .view(torch::kFloat32)
                       .view({3, 7}),
                   torch::zeros({3, 7})));
  EXPECT_TRUE(torch::equal(
      output.sample_output.next_tokens.select(/*dim=*/0, /*index=*/0),
      torch::tensor({4, 5, 6, 7, 8, 9, 10, 11}, torch::kInt64)));
  EXPECT_TRUE(torch::equal(
      draft_->context_slots.back()
          .narrow(/*dim=*/0, /*start=*/0, /*length=*/8)
          .cpu(),
      torch::tensor({18, 19, 20, 21, 22, 23, 24, 25}, torch::kInt32)));
  const auto metrics = make_spec_metrics(output.sample_output.next_tokens,
                                         output.spec_verify_layouts);
  EXPECT_EQ(metrics.num_draft_tokens, 21);
  EXPECT_EQ(metrics.num_accepted_tokens, 21);
  EXPECT_EQ(metrics.num_committed_tokens, 24);
  const auto next = execute(dspark_batch_input(
      3, /*decode=*/true, /*placeholder=*/true, /*position=*/3));
  ASSERT_EQ(leader.broadcasts(), startup_broadcasts + 2);
  EXPECT_EQ(leader.received(startup_broadcasts + 1).scalar_type(),
            torch::kUInt8);
  EXPECT_EQ(leader.received(startup_broadcasts + 1).numel(), 3 * 84);
  EXPECT_TRUE(torch::equal(
      next.sample_output.next_tokens.select(/*dim=*/0, /*index=*/0),
      torch::tensor({12, 13, 14, 15, 16, 17, 18, 19}, torch::kInt64)));
  EXPECT_TRUE(torch::equal(
      draft_->context_slots.back().narrow(0, 0, 8).cpu(),
      torch::tensor({26, 27, 28, 29, 30, 31, 32, 33}, torch::kInt32)));
  pipeline_.reset();
}

TEST_F(SpeculativePipelineTest, AdaptiveDSparkRejectsUnsupportedAdmission) {
  configure_adaptive_dspark();
  capacity_.sparse_dsa_target = false;
  const auto try_create = [this]() {
    return TaskExecutionPipeline::create(
        state_thread_,
        {*target_, *target_executor_, target_cache_},
        {*draft_, *draft_executor_, draft_cache_},
        capacity_,
        pipeline_);
  };
  EXPECT_FALSE(try_create().ok());
  capacity_.sparse_dsa_target = true;
  capacity_.num_speculative_tokens = 6;
  EXPECT_FALSE(try_create().ok());
  capacity_.num_speculative_tokens = 7;
  draft_->confidence_head = false;
  EXPECT_FALSE(try_create().ok());
  draft_->confidence_head = true;
  create();
  ASSERT_NE(pipeline_, nullptr);
  auto stochastic = dspark_batch_input(2, /*decode=*/false);
  stochastic.sampling_params.sampling_mode = SamplingMode::RANDOM;
  EXPECT_FALSE(pipeline_->submit(stochastic).status.ok());
  auto penalty = dspark_batch_input(2, /*decode=*/false);
  penalty.sampling_params.repetition_penalties = torch::tensor({1.0F, 1.1F});
  EXPECT_FALSE(pipeline_->submit(penalty).status.ok());
  auto logprobs = dspark_batch_input(2, /*decode=*/false);
  logprobs.sampling_params.logprobs = true;
  EXPECT_FALSE(pipeline_->submit(logprobs).status.ok());
  auto mask = dspark_batch_input(2, /*decode=*/false);
  mask.sampling_params.filter_mask = torch::zeros({2, 32});
  EXPECT_FALSE(pipeline_->submit(mask).status.ok());
  auto bitmask = dspark_batch_input(2, /*decode=*/false);
  bitmask.sampling_params.filter_bitmask = torch::ones({2, 1}, torch::kInt64);
  EXPECT_FALSE(pipeline_->submit(bitmask).status.ok());
  auto beam = dspark_batch_input(2, /*decode=*/false);
  beam.sampling_params.use_beam_search = true;
  EXPECT_FALSE(pipeline_->submit(beam).status.ok());
  auto embeddings = dspark_batch_input(2, /*decode=*/false);
  embeddings.sampling_params.is_embeddings = true;
  EXPECT_FALSE(pipeline_->submit(embeddings).status.ok());
  auto multiple_returns = dspark_batch_input(2, /*decode=*/false);
  multiple_returns.sampling_params.num_return_sequences = 2;
  EXPECT_FALSE(pipeline_->submit(multiple_returns).status.ok());
  EXPECT_TRUE(target_->forward_inputs.empty());
  EXPECT_TRUE(draft_->forward_inputs.empty());
  execute(dspark_batch_input(2, /*decode=*/false));
  EXPECT_FALSE(target_->forward_inputs.empty());
}

TEST_F(SpeculativePipelineTest,
       AdaptiveDSparkRejectsTokenBiasBeforePrefillAndDecodeLaunch) {
  configure_adaptive_dspark();
  create();
  ASSERT_NE(pipeline_, nullptr);
  warmup_dspark_batch_graphs();
  const auto reject_bias = [this](bool decode) {
    const size_t target_forwards = target_->forward_inputs.size();
    const size_t draft_forwards = draft_->forward_inputs.size();
    for (const float bias : {0.0F,
                             4.0F,
                             -4.0F,
                             -std::numeric_limits<float>::infinity(),
                             std::numeric_limits<float>::quiet_NaN()}) {
      auto request = dspark_batch_input(2, decode);
      request.sampling_params.logits_bias = torch::zeros({2, 32});
      request.sampling_params.logits_bias[0][3] = bias;
      EXPECT_FALSE(pipeline_->submit(request).status.ok());
    }
    auto device_bias = dspark_batch_input(2, decode);
    device_bias.sampling_params.logits_bias =
        torch::zeros({2, 32}, target_->options());
    EXPECT_FALSE(pipeline_->submit(device_bias).status.ok());
    EXPECT_EQ(target_->forward_inputs.size(), target_forwards);
    EXPECT_EQ(draft_->forward_inputs.size(), draft_forwards);
  };
  reject_bias(/*decode=*/false);
  execute(dspark_batch_input(2, /*decode=*/false));
  reject_bias(/*decode=*/true);
  // Unsupported requests must not poison a later ordinary greedy Task or
  // bypass the shared Slot sampling validation when adaptive sampling strips
  // the original fields for its target argmax.
  const auto output = execute(dspark_batch_input(2, /*decode=*/true));
  EXPECT_TRUE(
      torch::equal(output.sample_output.next_tokens[0],
                   torch::tensor({4, 5, 6, 7, 8, 9, 10, 11}, torch::kInt64)));
}

TEST_F(SpeculativePipelineTest,
       AdaptiveDSpark8192PageCapacityRunsShortStartupRequests) {
  configure_adaptive_dspark();
  capacity_.common.model.max_blocks_per_sequence = 8192;
  capacity_.common.logical_block_size = 128;
  capacity_.common.max_positions = 1U << 20;
  capacity_.common.max_kv_seq_len = 1U << 20;
  for (auto* caches : {&target_cache_, &draft_cache_}) {
    caches->clear();
    caches->emplace_back(
        KVCacheTensors{torch::zeros({8, 128, 1, 1}, target_->options()),
                       torch::zeros({8, 128, 1, 1}, target_->options())});
  }
  create();
  ASSERT_NE(pipeline_, nullptr);
  const auto request = [this](bool decode, bool warmup) {
    auto value = dspark_batch_input(3, decode);
    value.input_params.meta.is_graph_warmup = warmup;
    auto& host = value.input_params.attention.host;
    const int32_t* positions = value.positions.const_data_ptr<int32_t>();
    const int32_t* tables = host.block_tables.const_data_ptr<int32_t>();
    const uint32_t query_width = decode ? 1 : 2;
    for (uint32_t row = 0; row < 3; ++row) {
      for (uint32_t column = 0; column < query_width; ++column) {
        const uint32_t token = row * query_width + column;
        const int32_t position = positions[token];
        host.new_cache_slots[token] =
            tables[row * 2 + position / 128] * 128 + position % 128;
      }
    }
    return value;
  };
  const auto prefill = execute(request(/*decode=*/false, /*warmup=*/true));
  EXPECT_TRUE(prefill.is_graph_warmup);
  const auto warmup = execute(request(/*decode=*/true, /*warmup=*/true));
  ASSERT_TRUE(warmup.sample_output.next_tokens.defined());
  EXPECT_TRUE(warmup.is_graph_warmup);
  const auto target_bindings = target_->graph_warmup_bindings;
  const auto draft_bindings = draft_->graph_warmup_bindings;
  execute(request(/*decode=*/false, /*warmup=*/false));
  const auto decoded = execute(request(/*decode=*/true, /*warmup=*/false));
  EXPECT_FALSE(decoded.is_graph_warmup);
  EXPECT_TRUE(torch::equal(decoded.sample_output.next_tokens,
                           torch::tensor({{4, 5, 6, 7, 8, 9, 10, 11},
                                          {6, 7, 8, 9, 10, 11, 12, 13},
                                          {8, 9, 10, 11, 12, 13, 14, 15}},
                                         torch::kInt64)));
  EXPECT_EQ(target_->graph_warmup_bindings, target_bindings);
  EXPECT_EQ(draft_->graph_warmup_bindings, draft_bindings);
  ASSERT_FALSE(target_->forward_tables.empty());
  const auto table = target_->forward_tables.back().cpu();
  ASSERT_EQ(table.size(/*dim=*/0), 32);
  ASSERT_EQ(table.size(/*dim=*/1), 8192);
  EXPECT_TRUE(table.is_contiguous());
  EXPECT_EQ(table.stride(/*dim=*/0), 8192);
  for (int64_t row = 0; row < 24; ++row) {
    const int64_t owner = row / 8;
    EXPECT_EQ(table[row][0].item<int32_t>(), owner * 2 + 1);
    EXPECT_EQ(table[row][1].item<int32_t>(), owner * 2 + 2);
  }
  // Short Host tables still produce full-stride Slot rows. Every unused
  // column and padded request must be initialized, not truncated to 4096.
  EXPECT_EQ(torch::count_nonzero(table.narrow(1, 2, 8190)).item<int64_t>(), 0);
  EXPECT_EQ(torch::count_nonzero(table.narrow(0, 24, 8)).item<int64_t>(), 0);
  EXPECT_TRUE(torch::equal(
      draft_->context_slots.back().cpu(),
      torch::tensor({130, 131, 132, 133, 134, 135, 136, 137, 386, 387, 388,
                     389, 390, 391, 392, 393, 642, 643, 644, 645, 646, 647,
                     648, 649, -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
                    torch::kInt32)));
}

TEST_F(SpeculativePipelineTest,
       AdaptiveDSparkRejectsPageOffsetOverflowBeforeAllocation) {
  configure_adaptive_dspark();
  const uint64_t maximum_offset = std::numeric_limits<int32_t>::max();
  const uint64_t expanded_rows =
      static_cast<uint64_t>(capacity_.common.model.max_sequences) * 8;
  const uint32_t too_many_columns =
      static_cast<uint32_t>(maximum_offset / expanded_rows + 1);
  const uint32_t too_large_block =
      static_cast<uint32_t>(maximum_offset / 2 + 1);
  const std::array<std::pair<uint32_t, uint32_t>, 2> geometries{
      std::pair{too_many_columns, uint32_t{1}},
      std::pair{uint32_t{2}, too_large_block}};
  for (const auto& [columns, block_size] : geometries) {
    SCOPED_TRACE(columns);
    SCOPED_TRACE(block_size);
    auto overflow = capacity_;
    overflow.common.model.max_blocks_per_sequence = columns;
    overflow.common.logical_block_size = block_size;
    EXPECT_FALSE(TaskExecutionPipeline::create(
                     state_thread_,
                     {*target_, *target_executor_, target_cache_},
                     {*draft_, *draft_executor_, draft_cache_},
                     overflow,
                     pipeline_)
                     .ok());
    EXPECT_EQ(pipeline_, nullptr);
  }
  EXPECT_TRUE(target_->forward_inputs.empty());
  EXPECT_TRUE(draft_->forward_inputs.empty());
  EXPECT_TRUE(target_->graph_warmup_bindings.empty());
  EXPECT_TRUE(draft_->graph_warmup_bindings.empty());
}

TEST_F(SpeculativePipelineTest,
       DSparkGraphsRejectContextBeyondFirstWarmedCompactTable) {
  for (const bool adaptive : {false, true}) {
    SCOPED_TRACE(adaptive);
    pipeline_.reset();
    if (adaptive) {
      configure_adaptive_dspark();
    } else {
      configure_dspark_batch_graphs();
      capacity_.adaptive_dspark = false;
    }
    capacity_.common.max_positions = 1024;
    capacity_.common.max_kv_seq_len = 1024;
    capacity_.common.model.max_blocks_per_sequence = 64;
    create();
    ASSERT_NE(pipeline_, nullptr);
    warmup_dspark_batch_graphs();
    execute(dspark_batch_input(1, /*decode=*/false));
    const size_t target_forwards = target_->forward_inputs.size();
    const size_t draft_forwards = draft_->forward_inputs.size();
    const size_t target_warmups = target_->graph_warmup_bindings.size();
    const size_t draft_warmups = draft_->graph_warmup_bindings.size();

    // First warmup fits in 16 context pages plus Draft7: power-of-two
    // rounding captures 32 sixteen-token pages (512 tokens), not all 64.
    auto outside = dspark_batch_input(1,
                                      /*decode=*/true,
                                      /*placeholder=*/false,
                                      /*position=*/506);
    outside.input_params.attention.host.block_tables =
        torch::ones({1, 64}, torch::kInt32);
    outside.input_params.attention.host.new_cache_slots = {26};
    // KV=507 plus six later query rows crosses that 512-token bound.
    const auto rejected = pipeline_->submit(outside);
    EXPECT_FALSE(rejected.status.ok());
    EXPECT_EQ(rejected.task_id, 0U);
    EXPECT_EQ(target_->forward_inputs.size(), target_forwards);
    EXPECT_EQ(draft_->forward_inputs.size(), draft_forwards);

    // Repeating startup warmup with the same tensor binding must not pretend
    // to enlarge a graph that the Python runner has already captured.
    outside.input_params.meta.is_graph_warmup = true;
    EXPECT_FALSE(pipeline_->submit(outside).status.ok());
    EXPECT_EQ(target_->graph_warmup_bindings.size(), target_warmups);
    EXPECT_EQ(draft_->graph_warmup_bindings.size(), draft_warmups);
    const auto inside = execute(dspark_batch_input(1, /*decode=*/true));
    EXPECT_TRUE(torch::equal(
        inside.sample_output.next_tokens,
        torch::tensor({{4, 5, 6, 7, 8, 9, 10, 11}}, torch::kInt64)));
    EXPECT_EQ(target_->graph_warmup_bindings.size(), target_warmups);
    EXPECT_EQ(draft_->graph_warmup_bindings.size(), draft_warmups);
  }
}

TEST_F(SpeculativePipelineTest,
       AdaptiveDSparkColdStartupBindsDraftCachesBeforeBorrowedContextCapture) {
  configure_adaptive_dspark(/*profiled=*/false);
  draft_->require_bound_context_cache = true;
  create();
  ASSERT_NE(pipeline_, nullptr);
  {
    std::lock_guard<std::mutex> lock(draft_->prepared_mutex);
    ASSERT_TRUE(draft_->prepared_cache_bindings.empty());
  }

  auto prefill = dspark_batch_input(3, /*decode=*/false);
  prefill.input_params.meta.is_graph_warmup = true;
  const auto prefilled = execute(prefill);
  ASSERT_TRUE(prefilled.is_graph_warmup);
  ASSERT_TRUE(draft_->forward_inputs.empty());
  ASSERT_TRUE(draft_->graph_warmup_bindings.empty());
  const size_t prefill_writes = draft_->context_hidden.size();
  size_t prefill_events = 0;
  {
    std::lock_guard<std::mutex> lock(draft_->prepared_mutex);
    ASSERT_TRUE(draft_->prepared_cache_bindings.empty());
    prefill_events = draft_->prepared_events.size();
  }

  auto decode = dspark_batch_input(3, /*decode=*/true, /*placeholder=*/true);
  decode.input_params.meta.is_graph_warmup = true;
  const auto warmed = execute(decode);
  ASSERT_TRUE(warmed.is_graph_warmup);
  ASSERT_GT(draft_->context_hidden.size(), prefill_writes);
  {
    std::lock_guard<std::mutex> lock(draft_->prepared_mutex);
    const auto begin = draft_->prepared_events.begin() + prefill_events;
    const auto metadata = std::find(begin,
                                    draft_->prepared_events.end(),
                                    PreparedTestEvent::METADATA_PREPARED);
    const auto capture = std::find(
        begin, draft_->prepared_events.end(), PreparedTestEvent::GRAPH_WARMED);
    const auto context = std::find(begin,
                                   draft_->prepared_events.end(),
                                   PreparedTestEvent::CONTEXT_WRITTEN);
    ASSERT_NE(metadata, draft_->prepared_events.end());
    ASSERT_NE(capture, draft_->prepared_events.end());
    ASSERT_NE(context, draft_->prepared_events.end());
    EXPECT_LT(metadata, capture);
    EXPECT_LT(capture, context);
    EXPECT_GE(std::count(begin, context, PreparedTestEvent::METADATA_PREPARED),
              2);
    EXPECT_GE(std::count(begin, context, PreparedTestEvent::GRAPH_WARMED), 2);
    for (size_t index = prefill_writes;
         index < draft_->context_cache_bound.size();
         ++index) {
      EXPECT_TRUE(draft_->context_cache_bound[index]);
    }
  }
  std::vector<std::array<uintptr_t, 4>> full_width_bindings;
  full_width_bindings.reserve(draft_->context_bindings.size());
  for (const auto& binding : draft_->context_bindings) {
    if (binding[3] == 32) {
      full_width_bindings.emplace_back(binding);
    }
  }
  std::sort(full_width_bindings.begin(), full_width_bindings.end());
  full_width_bindings.erase(
      std::unique(full_width_bindings.begin(), full_width_bindings.end()),
      full_width_bindings.end());
  EXPECT_EQ(full_width_bindings.size(), 2U);

  const auto target_bindings = target_->graph_warmup_bindings;
  const uint32_t target_allocations = target_->graph_output_allocations.load();
  const uint32_t draft_allocations = draft_->graph_output_allocations.load();
  auto repeated = dspark_batch_input(
      3, /*decode=*/true, /*placeholder=*/true, /*position=*/10);
  repeated.input_params.meta.is_graph_warmup = true;
  EXPECT_TRUE(execute(repeated).is_graph_warmup);
  EXPECT_EQ(target_->graph_warmup_bindings, target_bindings);
  EXPECT_EQ(target_->graph_output_allocations.load(), target_allocations);
  EXPECT_EQ(draft_->graph_output_allocations.load(), draft_allocations);

  const auto draft_bindings = draft_->graph_warmup_bindings;
  execute(dspark_batch_input(3, /*decode=*/false));
  const auto first = pipeline_->submit(dspark_batch_input(3, /*decode=*/true));
  ASSERT_TRUE(first.status.ok()) << first.status.message();
  const auto second = pipeline_->submit(dspark_batch_input(
      3, /*decode=*/true, /*placeholder=*/true, /*position=*/3));
  ASSERT_TRUE(second.status.ok()) << second.status.message();
  const auto first_result = pipeline_->take_result_async(first.task_id).get();
  const auto second_result = pipeline_->take_result_async(second.task_id).get();
  ASSERT_TRUE(first_result.status.ok());
  ASSERT_TRUE(second_result.status.ok());
  EXPECT_EQ(first_result.output.sample_output.next_tokens[0][7].item<int64_t>(),
            11);
  EXPECT_EQ(
      second_result.output.sample_output.next_tokens[0][7].item<int64_t>(), 19);
  EXPECT_EQ(target_->graph_warmup_bindings, target_bindings);
  EXPECT_EQ(draft_->graph_warmup_bindings, draft_bindings);
  EXPECT_EQ(target_->graph_output_allocations.load(), target_allocations);
  EXPECT_EQ(draft_->graph_output_allocations.load(), draft_allocations);
}

TEST_F(SpeculativePipelineTest,
       AdaptiveDSparkStartupCaptureDoesNotMeasureCosts) {
  configure_adaptive_dspark();
  capacity_.adaptive_costs.reset();
  create();
  ASSERT_NE(pipeline_, nullptr);

  // ProfileManager's unified startup submits a real Prefill warmup first.
  auto prefill = dspark_batch_input(3, /*decode=*/false);
  prefill.input_params.meta.is_graph_warmup = true;
  const auto prefilled = execute(prefill);
  ASSERT_TRUE(prefilled.sample_output.next_tokens.defined());
  EXPECT_TRUE(prefilled.is_graph_warmup);
  EXPECT_TRUE(torch::equal(prefilled.sample_output.next_tokens,
                           torch::tensor({3, 5, 7}, torch::kInt64)));
  EXPECT_EQ(target_->forward_inputs.size(), 1U);
  EXPECT_TRUE(draft_->forward_inputs.empty());
  EXPECT_TRUE(target_->graph_warmup_bindings.empty());
  EXPECT_TRUE(draft_->graph_warmup_bindings.empty());
  EXPECT_TRUE(draft_->confidence_hidden.empty());
  EXPECT_TRUE(draft_->markov_inputs.empty());
  ASSERT_EQ(draft_->context_hidden.size(), 1U);
  const auto hidden = torch::arange(1, 7, torch::kFloat32).view({6, 1});
  EXPECT_TRUE(torch::equal(draft_->context_hidden.front().cpu(),
                           torch::cat({hidden, hidden + 100}, /*dim=*/1)));
  EXPECT_TRUE(torch::equal(draft_->context_positions.front().cpu(),
                           torch::tensor({0, 1, 0, 1, 0, 1}, torch::kInt32)));
  EXPECT_TRUE(
      torch::equal(draft_->context_slots.front().cpu(),
                   torch::tensor({16, 17, 48, 49, 80, 81}, torch::kInt32)));
  for (const uint32_t draft_rows : {7U, 14U, 28U}) {
    EXPECT_FALSE(TaskPipelineTestPeer::adaptive_costs(*pipeline_)
                     .lookup(draft_rows, /*max_kv_length=*/8));
  }

  auto decode = dspark_batch_input(3, /*decode=*/true, /*placeholder=*/true);
  decode.input_params.meta.is_graph_warmup = true;
  const auto warmed = execute(decode);
  ASSERT_TRUE(warmed.sample_output.next_tokens.defined());
  EXPECT_TRUE(warmed.is_graph_warmup);
  EXPECT_TRUE(torch::equal(warmed.sample_output.next_tokens,
                           torch::tensor({{4, 5, 6, 7, 8, 9, 10, 11},
                                          {6, 7, 8, 9, 10, 11, 12, 13},
                                          {8, 9, 10, 11, 12, 13, 14, 15}},
                                         torch::kInt64)));
  const auto profile = TaskPipelineTestPeer::adaptive_costs(*pipeline_)
                           .lookup(/*draft_graph_rows=*/28,
                                   /*max_kv_length=*/18);
  EXPECT_FALSE(profile.has_value());
  const size_t target_forwards = target_->forward_inputs.size();
  const size_t draft_forwards = draft_->forward_inputs.size();
  const size_t context_writes = draft_->context_hidden.size();
  const size_t confidence_calls = draft_->confidence_hidden.size();
  const size_t target_warmups = target_->graph_warmup_bindings.size();
  const auto draft_bindings = draft_->graph_warmup_bindings;
  const uint32_t target_allocations = target_->graph_output_allocations.load();
  const uint32_t draft_allocations = draft_->graph_output_allocations.load();

  auto repeat = dspark_batch_input(3,
                                   /*decode=*/true,
                                   /*placeholder=*/true,
                                   /*position=*/10);
  repeat.input_params.meta.is_graph_warmup = true;
  const auto repeated = execute(repeat);
  ASSERT_TRUE(repeated.sample_output.next_tokens.defined());
  EXPECT_TRUE(repeated.is_graph_warmup);
  EXPECT_TRUE(torch::equal(repeated.sample_output.next_tokens,
                           torch::tensor({{12, 13, 14, 15, 16, 17, 18, 19},
                                          {14, 15, 16, 17, 18, 19, 20, 21},
                                          {16, 17, 18, 19, 20, 21, 22, 23}},
                                         torch::kInt64)));
  // Only the real Decode Task executes again: no additional profiling sweep
  // or new prepared output allocation is needed for the existing bindings.
  EXPECT_EQ(target_->forward_inputs.size(), target_forwards + 1);
  EXPECT_EQ(draft_->forward_inputs.size(), draft_forwards + 1);
  EXPECT_EQ(draft_->context_hidden.size(), context_writes + 1);
  EXPECT_EQ(draft_->confidence_hidden.size(), confidence_calls + 1);
  EXPECT_EQ(target_->graph_warmup_bindings.size(), target_warmups);
  EXPECT_EQ(target_->graph_output_allocations.load(), target_allocations);
  EXPECT_EQ(draft_->graph_output_allocations.load(), draft_allocations);
  for (size_t index = draft_bindings.size();
       index < draft_->graph_warmup_bindings.size();
       ++index) {
    EXPECT_NE(std::find(draft_bindings.begin(),
                        draft_bindings.end(),
                        draft_->graph_warmup_bindings[index]),
              draft_bindings.end());
  }
  const auto reused_profile = TaskPipelineTestPeer::adaptive_costs(*pipeline_)
                                  .lookup(/*draft_graph_rows=*/28,
                                          /*max_kv_length=*/18);
  EXPECT_FALSE(reused_profile.has_value());
}

TEST_F(SpeculativePipelineTest,
       StartupCalibrationRequiresDrainedSlotsAndInstallsCosts) {
  configure_adaptive_dspark();
  capacity_.adaptive_costs.reset();
  capacity_.startup_calibration = true;
  capacity_.calibration_leader = true;
  capacity_.common.model.max_blocks_per_sequence = 8;
  capacity_.common.max_positions = 128;
  capacity_.common.max_kv_seq_len = 128;
  // Four one-based page pairs include reserved page 8, even when Decode has
  // not reached the second page. Admission validates the complete page table.
  for (auto* caches : {&target_cache_, &draft_cache_}) {
    caches->clear();
    caches->emplace_back(
        KVCacheTensors{torch::zeros({9, 16, 1, 1}, target_->options()),
                       torch::zeros({9, 16, 1, 1}, target_->options())});
  }
  create();
  ASSERT_NE(pipeline_, nullptr);
  execute(dspark_batch_input(4, /*decode=*/false));
  auto warmup = dspark_batch_input(4, /*decode=*/true);
  warmup.input_params.meta.is_graph_warmup = true;
  execute(warmup);
  EXPECT_FALSE(TaskPipelineTestPeer::adaptive_costs(*pipeline_).lookup(28, 32));
  const auto pending =
      pipeline_->submit(dspark_batch_input(4, /*decode=*/true));
  ASSERT_TRUE(pending.status.ok());
  EXPECT_FALSE(pipeline_->dspark_calibration(
      {DSparkCalibrationAction::CAPABILITIES, {}}));
  ASSERT_TRUE(pipeline_->take_result_async(pending.task_id).get().status.ok());
  const auto capabilities = pipeline_->dspark_calibration(
      {DSparkCalibrationAction::CAPABILITIES, {}});
  ASSERT_TRUE(capabilities);
  const auto geometry = nlohmann::json::parse(*capabilities);
  EXPECT_EQ(geometry["request_buckets"], nlohmann::json({4}));
  const nlohmann::json configuration{
      {"model", "sps"},
      {"request_buckets", {4}},
      {"warmup_steps", 0},
      {"window_steps", 1},
      {"windows_per_point", 1},
      {"contexts",
       {{{"prefill_tokens", 1}, {"min_kv_length", 1}, {"max_kv_length", 64}}}}};
  const std::string begin = nlohmann::json{
      {"configuration", configuration},
      {"point", uint32_t{0}}}.dump();
  EXPECT_FALSE(pipeline_->dspark_calibration(
      {DSparkCalibrationAction::BEGIN_SAMPLE, "{}"}));
  ASSERT_TRUE(pipeline_->dspark_calibration(
      {DSparkCalibrationAction::BEGIN_SAMPLE, begin}));
  EXPECT_FALSE(pipeline_->dspark_calibration(
      {DSparkCalibrationAction::BEGIN_SAMPLE, begin}));
  EXPECT_FALSE(pipeline_->dspark_calibration(
      {DSparkCalibrationAction::INSTALL_COSTS, "{}"}));
  EXPECT_FALSE(pipeline_->dspark_calibration(
      {DSparkCalibrationAction::FINISH_SAMPLE, {}}));
  execute(dspark_batch_input(
      4, /*decode=*/true, /*placeholder=*/false, /*position=*/10));
  execute(dspark_batch_input(
      4, /*decode=*/true, /*placeholder=*/false, /*position=*/11));
  const auto completed = pipeline_->dspark_calibration(
      {DSparkCalibrationAction::FINISH_SAMPLE, {}});
  ASSERT_TRUE(completed);
  const auto windows = nlohmann::json::parse(*completed);
  ASSERT_EQ(windows.size(), 1U);
  EXPECT_EQ(windows[0]["target_graph_rows"], 4);
  EXPECT_EQ(windows[0]["draft_tokens"], 0);
  EXPECT_EQ(windows[0]["accepted_tokens"], 0);
  EXPECT_EQ(windows[0]["steps"], 1);
  const nlohmann::json table{{"sample_batch_tokens", {4, 8, 16, 32}},
                             {"sample_steps_per_sec", {100, 90, 80, 70}},
                             {"max_batch_tokens", 32},
                             {"max_kv_length", 64}};
  const std::string bundle = nlohmann::json{
      {"schema_version", 1},
      {"measurement", "steady_full_step"},
      {"compatibility", {{"test", "runtime-lifecycle"}}},
      {"profiles",
       {{{"min_kv_length", 1},
         {"max_kv_length", 64},
         {"request_buckets", {4}},
         {"table", table}}}}}.dump();
  EXPECT_FALSE(pipeline_->dspark_calibration(
      {DSparkCalibrationAction::INSTALL_COSTS, "{}"}));
  EXPECT_FALSE(TaskPipelineTestPeer::adaptive_costs(*pipeline_).lookup(28, 32));
  const auto installed = pipeline_->dspark_calibration(
      {DSparkCalibrationAction::INSTALL_COSTS, bundle});
  ASSERT_TRUE(installed);
  EXPECT_EQ(*installed, bundle);
  const auto profile =
      TaskPipelineTestPeer::adaptive_costs(*pipeline_).lookup(28, 32);
  ASSERT_TRUE(profile);
  EXPECT_EQ(profile->cost_scope, DSparkCostScope::FULL_STEP);
  EXPECT_EQ(profile->fixed_draft_time_ms, 0.0);
  EXPECT_FALSE(pipeline_->dspark_calibration(
      {DSparkCalibrationAction::BEGIN_SAMPLE, begin}));
  execute(dspark_batch_input(4, /*decode=*/false));
  EXPECT_TRUE(execute(dspark_batch_input(4, /*decode=*/true))
                  .sample_output.next_tokens.defined());
}

TEST_F(SpeculativePipelineTest, AdaptiveDSparkUsesExternalFullStepCosts) {
  configure_adaptive_dspark();
  capacity_.adaptive_costs.reset();
  capacity_.startup_calibration = true;
  const std::string table = R"json({
    "bias_seconds":0.010,
    "bs_probes":[1,4], "alpha_seconds":[0.001,0.004],
    "m_probes":[0,32], "theta_seconds":[0.0,0.032],
    "max_kv_length":64
  })json";
  create();
  ASSERT_NE(pipeline_, nullptr);
  // External costs do not replace the request-owned Prefill context.
  const auto prefilled = execute(dspark_batch_input(3, /*decode=*/false));
  ASSERT_TRUE(prefilled.sample_output.next_tokens.defined());
  auto warmup = dspark_batch_input(3, /*decode=*/true);
  warmup.input_params.meta.is_graph_warmup = true;
  const auto warmed = execute(warmup);
  ASSERT_TRUE(warmed.sample_output.next_tokens.defined());
  EXPECT_TRUE(warmed.is_graph_warmup);
  EXPECT_FALSE(TaskPipelineTestPeer::adaptive_costs(*pipeline_).lookup(28, 32));
  ASSERT_TRUE(pipeline_->dspark_calibration(
      {DSparkCalibrationAction::INSTALL_COSTS, table}));
  const auto profile =
      TaskPipelineTestPeer::adaptive_costs(*pipeline_)
          .lookup(/*draft_graph_rows=*/28, /*max_kv_length=*/32);
  ASSERT_TRUE(profile);
  EXPECT_EQ(profile->cost_scope, DSparkCostScope::FULL_STEP);
  EXPECT_EQ(profile->max_kv_length, 32U);
  EXPECT_DOUBLE_EQ(profile->fixed_draft_time_ms, 0.0);
  EXPECT_DOUBLE_EQ(profile->commit_time_ms, 0.0);
  ASSERT_FALSE(profile->target_graph_costs.empty());
  // Logical B=3 warms Draft's physical Bcap=4. Price that physical bucket.
  EXPECT_EQ(profile->target_graph_costs.front().graph_rows, 4U);
  EXPECT_NEAR(profile->target_graph_costs.front().target_time_ms, 18.0, 1e-5);
  EXPECT_NEAR(profile->target_graph_costs.back().target_time_ms, 46.0, 1e-5);
  EXPECT_FALSE(TaskPipelineTestPeer::adaptive_costs(*pipeline_)
                   .lookup(/*draft_graph_rows=*/28, /*max_kv_length=*/65));
}

TEST_F(SpeculativePipelineTest,
       LocalIdleShardRunsPeerForwardsWithoutSamplingAndCanBeReused) {
  capacity_.common.dp_size = 2;
  create();
  ASSERT_NE(pipeline_, nullptr);
  LlmForwardInput idle;
  idle.input_params.meta.batch_forward_type = BatchForwardType::EMPTY;
  auto& parallel = idle.input_params.parallel;
  parallel.dp_global_token_nums = {0, 2};
  parallel.raw_dp_global_token_nums = {0, 2};
  parallel.dp_is_decode = {0, 0};
  const auto prefill = execute(idle);
  EXPECT_FALSE(prefill.sample_output.next_tokens.defined());
  ASSERT_EQ(target_->forward_inputs.size(), 1U);
  ASSERT_EQ(draft_->forward_inputs.size(), 1U);
  EXPECT_EQ(target_->forward_inputs.front().numel(), 1);
  EXPECT_EQ(draft_->forward_inputs.front().numel(), 1);

  parallel.dp_global_token_nums = {0, 1};
  parallel.raw_dp_global_token_nums = {0, 1};
  parallel.dp_is_decode = {0, 1};
  const auto decode = execute(idle);
  EXPECT_FALSE(decode.sample_output.next_tokens.defined());
  ASSERT_EQ(target_->forward_inputs.size(), 2U);
  ASSERT_EQ(draft_->forward_inputs.size(), 3U);
  EXPECT_EQ(target_->forward_counts.back(), (std::vector<int32_t>{0, 3}));
  EXPECT_EQ(draft_->forward_counts[1], (std::vector<int32_t>{0, 2}));
  EXPECT_EQ(draft_->forward_counts[2], (std::vector<int32_t>{0, 1}));
  EXPECT_EQ(target_->logits_calls.load(), 0);
  EXPECT_EQ(draft_->logits_calls.load(), 0);

  auto active = input(false);
  active.input_params.parallel.dp_global_token_nums = {2, 0};
  active.input_params.parallel.raw_dp_global_token_nums = {2, 0};
  active.input_params.parallel.dp_is_decode = {0, 0};
  const auto result = execute(active);
  ASSERT_TRUE(result.sample_output.next_tokens.defined());
  EXPECT_TRUE(torch::equal(result.sample_output.next_tokens,
                           torch::tensor({3}, torch::kInt64)));
}

TEST_F(SpeculativePipelineTest,
       UnfinishedPrefillRetiresWithoutSamplingBeforeCompletedChunk) {
  capacity_.common.chunked_prefill = true;
  create();
  ASSERT_NE(pipeline_, nullptr);
  auto unfinished = input(false);
  unfinished.input_params.meta.batch_forward_type =
      BatchForwardType::CHUNKED_PREFILL;
  unfinished.input_params.embedding.extra_token_ids = {3};
  unfinished.input_params.embedding.embedding_ids.clear();
  unfinished.input_params.embedding.request_ids.clear();
  unfinished.sampling_params = SamplingParameters();
  const auto pending = execute(unfinished);
  EXPECT_FALSE(pending.sample_output.next_tokens.defined());
  EXPECT_FALSE(pending.sample_output.embeddings.defined());
  EXPECT_EQ(target_->logits_calls.load(), 0);
  EXPECT_EQ(draft_->logits_calls.load(), 0);
  ASSERT_EQ(draft_->forward_inputs.size(), 1U);
  EXPECT_TRUE(torch::equal(draft_->forward_inputs.front().cpu(),
                           torch::tensor({2, 3}, torch::kInt32)));

  auto completed = input(false);
  completed.token_ids = torch::tensor({3, 4}, torch::kInt32);
  completed.positions = torch::tensor({2, 3}, torch::kInt32);
  completed.input_params.meta.batch_forward_type =
      BatchForwardType::CHUNKED_PREFILL;
  completed.input_params.attention.host.kv_seq_lens = {4};
  completed.input_params.attention.host.new_cache_slots = {2, 3};
  const auto sampled = execute(completed);
  ASSERT_TRUE(sampled.sample_output.next_tokens.defined());
  EXPECT_TRUE(torch::equal(sampled.sample_output.next_tokens,
                           torch::tensor({5}, torch::kInt64)));
  EXPECT_TRUE(
      torch::equal(sampled.sample_output.embeddings, torch::tensor({{4.0f}})));
  const auto decoded = execute(input(true, /*token=*/5, /*position=*/4));
  ASSERT_TRUE(decoded.sample_output.next_tokens.defined());
  EXPECT_TRUE(torch::equal(decoded.sample_output.next_tokens,
                           torch::tensor({{6, 7, 8}}, torch::kInt64)));
}

TEST_F(SpeculativePipelineTest,
       WorkerResultWaitsForDeviceBeforeReleasingPreparedInput) {
  PreparedTestWorker worker(device_);
  auto entered = worker.entered();
  ASSERT_TRUE(TaskExecutionPipeline::create(
                  state_thread_,
                  {*target_, *target_executor_, target_cache_},
                  worker,
                  capacity_.common,
                  pipeline_)
                  .ok());
  const auto submitted = pipeline_->submit(input(true));
  ASSERT_TRUE(submitted.status.ok());
  EXPECT_EQ(entered.wait_for(std::chrono::seconds(5)),
            std::future_status::ready);
  auto result = pipeline_->take_result_async(submitted.task_id);
  result.wait(std::chrono::milliseconds(50));
  EXPECT_FALSE(result.isReady());
  EXPECT_EQ(worker.released_inputs(), 0);
  worker.release();
  auto completed = std::move(result).get();
  EXPECT_TRUE(completed.status.ok());
  EXPECT_EQ(completed.task_id, submitted.task_id);
  EXPECT_FALSE(worker.released_early());
  EXPECT_EQ(worker.released_inputs(), 1);
  EXPECT_TRUE(completed.output.retained_inputs.empty());
  EXPECT_TRUE(torch::equal(completed.output.sample_output.next_tokens.cpu(),
                           torch::tensor({4}, torch::kInt32)));
  pipeline_.reset();
}

TEST_F(SpeculativePipelineTest,
       WorkerDiscardWaitsForTokenlessDeviceOutputBeforeReleasingInput) {
  PreparedTestWorker worker(device_);
  worker.without_tokens();
  auto entered = worker.entered();
  ASSERT_TRUE(TaskExecutionPipeline::create(
                  state_thread_,
                  {*target_, *target_executor_, target_cache_},
                  worker,
                  capacity_.common,
                  pipeline_)
                  .ok());
  ASSERT_TRUE(pipeline_->submit(input(true)).status.ok());
  EXPECT_EQ(entered.wait_for(std::chrono::seconds(5)),
            std::future_status::ready);
  auto destroyed =
      std::async(std::launch::async, [this] { pipeline_.reset(); });
  EXPECT_EQ(destroyed.wait_for(std::chrono::milliseconds(50)),
            std::future_status::timeout);
  EXPECT_EQ(worker.released_inputs(), 0);
  worker.release();
  destroyed.get();
  EXPECT_FALSE(worker.released_early());
  EXPECT_EQ(worker.released_inputs(), 1);
}

TEST_F(SpeculativePipelineTest,
       WorkerKeepsPackedHostViewsUntilResultCompletes) {
  PreparedTestWorker worker(device_);
  worker.with_contiguous_input();
  auto entered = worker.entered();
  ASSERT_TRUE(TaskExecutionPipeline::create(
                  state_thread_,
                  {*target_, *target_executor_, target_cache_},
                  worker,
                  capacity_.common,
                  pipeline_)
                  .ok());
  proto::PackedForwardInput payload;
  ASSERT_TRUE(input(false).to_packed_proto(&payload));
  LlmForwardInput packed;
  ASSERT_TRUE(packed.from_packed_proto(payload));
  auto released = std::make_shared<std::atomic<bool>>(false);
  torch::Tensor owner = packed.runtime.input_host_buffer;
  packed.runtime.input_host_buffer = torch::from_blob(
      owner.data_ptr(),
      owner.sizes(),
      [owner, released](void* /*data*/) mutable {
        owner = torch::Tensor();
        released->store(true);
      },
      owner.options());
  owner = torch::Tensor();
  const auto submitted = pipeline_->submit(packed);
  ASSERT_TRUE(submitted.status.ok());
  EXPECT_EQ(entered.wait_for(std::chrono::seconds(5)),
            std::future_status::ready);
  packed = LlmForwardInput();
  EXPECT_FALSE(released->load());
  worker.release();
  auto result = pipeline_->take_result_async(submitted.task_id).get();
  EXPECT_TRUE(result.status.ok());
  EXPECT_TRUE(released->load());
  EXPECT_TRUE(torch::equal(result.output.sample_output.next_tokens.cpu(),
                           torch::tensor({2, 3}, torch::kInt32)));
  pipeline_.reset();
}

TEST_F(SpeculativePipelineTest, WorkerOwnedPipelineUsesBothSlotsAndFifo) {
  capacity_.common.slot_count = 2;
  PreparedTestWorker worker(device_);
  auto entered = worker.entered();
  ASSERT_TRUE(TaskExecutionPipeline::create(
                  state_thread_,
                  {*target_, *target_executor_, target_cache_},
                  worker,
                  capacity_.common,
                  pipeline_)
                  .ok());

  const auto first = pipeline_->submit(input(true));
  ASSERT_TRUE(first.status.ok()) << first.status.message();
  ASSERT_EQ(entered.wait_for(std::chrono::seconds(5)),
            std::future_status::ready);
  const auto second =
      pipeline_->submit(input(true, /*token=*/6, /*position=*/5));
  ASSERT_TRUE(second.status.ok()) << second.status.message();
  const auto full = pipeline_->submit(input(true, /*token=*/7, /*position=*/6));
  EXPECT_EQ(full.status.code(), StatusCode::RESOURCE_EXHAUSTED);

  worker.release();
  const auto first_result = pipeline_->take_result_async(first.task_id).get();
  ASSERT_TRUE(first_result.status.ok()) << first_result.status.message();
  const auto second_result = pipeline_->take_result_async(second.task_id).get();
  ASSERT_TRUE(second_result.status.ok()) << second_result.status.message();

  const auto reused =
      pipeline_->submit(input(true, /*token=*/8, /*position=*/7));
  EXPECT_TRUE(reused.status.ok()) << reused.status.message();
  pipeline_->take_result_async(reused.task_id).get();
}

TEST_F(SpeculativePipelineTest, UnifiedTaskAdvancesJsonFromOwnedPriorTokens) {
  JsonTaskTestWorker worker(device_);
  JsonObjectGrammar grammar({"{", "}", "stop"}, {2});
  auto prefill = input(false);
  prefill.json_object_states = {grammar.initial_state()};
  prefill.sample_sequence_ids = {"json#0"};
  prefill.sample_prior_output_rows = {-1};
  auto first = worker.execute_task_pipeline(prefill);
  ASSERT_TRUE(first.has_value());
  EXPECT_TRUE(worker.observed_tokens.empty());
  // The consumer may release/reuse a pinned result before the next launch.
  first->sample_output.next_tokens.fill_(2);

  auto decode = input(true);
  decode.json_object_states = {grammar.initial_state()};
  decode.sample_sequence_ids = {"json#0"};
  decode.sample_prior_output_rows = {0};
  worker.next_token = 1;
  auto second = worker.execute_task_pipeline(decode);
  ASSERT_TRUE(second.has_value());
  EXPECT_TRUE(second->json_object_errors.empty());
  EXPECT_EQ(worker.observed_tokens, (std::vector<int32_t>{0}));
  EXPECT_TRUE(decode.json_object_states.front().snapshot().token_ids.empty());

  ASSERT_TRUE(decode.json_object_states.front().accept_token(0));
  worker.next_token = 2;
  auto third = worker.execute_task_pipeline(decode);
  ASSERT_TRUE(third.has_value());
  EXPECT_TRUE(third->json_object_errors.empty());
  EXPECT_EQ(worker.observed_tokens, (std::vector<int32_t>{0, 1}));

  // A new prefill has no prior row and must not inherit the old grammar.
  prefill.sample_sequence_ids = {"new-json#0"};
  worker.next_token = 0;
  worker.execute_task_pipeline(prefill);
  EXPECT_TRUE(worker.observed_tokens.empty());
  worker.with_output = false;
  worker.execute_task_pipeline(input(true));
  worker.with_output = true;
  decode.json_object_states = {grammar.initial_state()};
  decode.sample_sequence_ids = {"new-json#0"};
  decode.token_ids = torch::tensor({-1}, torch::kInt32);
  auto missing = worker.execute_task_pipeline(decode);
  ASSERT_TRUE(missing.has_value());
  ASSERT_EQ(missing->json_object_errors.size(), 1U);
  EXPECT_EQ(missing->json_object_errors.front().sample_sequence_id,
            "new-json#0");
}

TEST_F(SpeculativePipelineTest, MinimumLengthConstrainsTargetButNotDraft) {
  create();
  execute(input(false));
  auto decode = input(true);
  decode.sampling_params.min_token_positions =
      torch::tensor({100}, torch::kLong);
  decode.sampling_params.min_tokens_remaining =
      torch::tensor({100}, torch::kLong);
  decode.sampling_params.min_token_ids = torch::tensor({{4, -1}}, torch::kLong);
  const auto result = execute(decode);
  // The first draft still proposes 4; the target must reject that blocked id.
  ASSERT_GE(draft_->forward_inputs.size(), 3U);
  EXPECT_EQ(draft_->forward_inputs[2].item<int32_t>(), 4);
  EXPECT_EQ(result.sample_output.next_tokens[0][0].item<int64_t>(), 0);
}

class TargetPrefixLengthSpeculativePipelineTest
    : public SpeculativePipelineTest,
      public ::testing::WithParamInterface<int32_t> {};

TEST_P(TargetPrefixLengthSpeculativePipelineTest,
       RefreshPairsEveryAcceptedTokenWithItsVerifierHidden) {
  const int32_t accepted = GetParam();
  capacity_.num_speculative_tokens = 3;
  capacity_.refresh_target_prefix = true;
  draft_->logit_increments.assign(3, 1);
  if (accepted < 4) {
    draft_->logit_increments[accepted - 1] = 2;
  }
  create();
  execute(input(false));
  const auto result = execute(input(true));
  auto expected = torch::arange(4, 8, torch::kInt64);
  expected.narrow(0, accepted, 4 - accepted).fill_(-1);
  EXPECT_TRUE(
      torch::equal(result.sample_output.next_tokens, expected.view({1, 4})));
  ASSERT_EQ(draft_->forward_inputs.size(), 5U);
  expected.clamp_min_(0);
  EXPECT_TRUE(torch::equal(draft_->forward_inputs.back().cpu(), expected));
  EXPECT_TRUE(torch::equal(draft_->forward_positions.back().cpu(),
                           torch::arange(2, 6, torch::kInt32)));
  EXPECT_TRUE(torch::equal(
      draft_->forward_embeddings.back().narrow(0, 0, accepted).cpu(),
      torch::arange(3, 3 + accepted, torch::kFloat32).view({accepted, 1})));
  EXPECT_EQ(draft_->logits_calls.load(), 3);
}

INSTANTIATE_TEST_SUITE_P(AcceptedLengths,
                         TargetPrefixLengthSpeculativePipelineTest,
                         ::testing::Values(1, 2, 3, 4));

TEST_F(SpeculativePipelineTest,
       PrefixRefreshPreservesContextBeforeAnotherBodyReusesThePool) {
  capacity_.num_speculative_tokens = 3;
  capacity_.refresh_target_prefix = true;
  capacity_.common.slot_count = 2;
  auto storage = torch::empty({16, 1}, target_->options());
  for (auto* model : {target_.get(), draft_.get()}) {
    model->borrowed_hidden_storage = storage;
  }
  create();
  execute(input(false));
  auto first = input(true);
  auto second = input(true, -1, 3);
  const auto a = pipeline_->submit(first);
  const auto b = pipeline_->submit(second);
  ASSERT_TRUE(a.status.ok()) << a.status.message();
  ASSERT_TRUE(b.status.ok()) << b.status.message();
  const auto ar = pipeline_->take_result_async(a.task_id).get();
  const auto br = pipeline_->take_result_async(b.task_id).get();
  ASSERT_TRUE(ar.status.ok()) << ar.status.message();
  ASSERT_TRUE(br.status.ok()) << br.status.message();
  EXPECT_TRUE(torch::equal(ar.output.sample_output.next_tokens,
                           torch::tensor({{4, 5, 6, 7}}, torch::kInt64)));
  EXPECT_TRUE(torch::equal(br.output.sample_output.next_tokens,
                           torch::tensor({{8, 9, 10, 11}}, torch::kInt64)));
  ASSERT_EQ(draft_->forward_embeddings.size(), 9U);
  EXPECT_TRUE(torch::equal(draft_->forward_embeddings[4].cpu(),
                           torch::tensor({{3.f}, {4.f}, {5.f}, {6.f}})));
  EXPECT_TRUE(torch::equal(draft_->forward_embeddings[5].cpu(),
                           torch::tensor({{5.f}, {6.f}})));
  EXPECT_TRUE(torch::equal(draft_->forward_inputs[4].cpu(),
                           torch::tensor({4, 5, 6, 7}, torch::kInt64)));
}

TEST_F(SpeculativePipelineTest, PrefixRefreshKeepsEmptyDpPeersInBodyOrder) {
  capacity_.common.dp_size = 2;
  capacity_.common.dp_rank = 1;
  capacity_.num_speculative_tokens = 3;
  capacity_.refresh_target_prefix = true;
  create();
  LlmForwardInput empty;
  empty.token_ids = torch::empty({0}, torch::kInt32);
  empty.positions = torch::empty({0}, torch::kInt32);
  auto& p = empty.input_params;
  p.meta.batch_forward_type = BatchForwardType::EMPTY;
  p.attention.host.block_tables = torch::empty({0, 2}, torch::kInt32);
  p.parallel.dp_global_token_nums = {1, 0};
  p.parallel.raw_dp_global_token_nums = {1, 0};
  p.parallel.dp_is_decode = {1, 0};
  const auto result = execute(empty);
  EXPECT_FALSE(result.sample_output.next_tokens.defined());
  ASSERT_EQ(draft_->forward_inputs.size(), 4U);
  EXPECT_EQ(draft_->forward_counts.back(), (std::vector<int32_t>{4, 0}));
  EXPECT_EQ(draft_->logits_calls.load(), 0);
  EXPECT_TRUE(torch::equal(draft_->forward_embeddings.back().cpu(),
                           torch::zeros({1, 1})));
}

}  // namespace
}  // namespace xllm
