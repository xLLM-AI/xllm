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

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <functional>
#include <future>
#include <limits>
#include <memory>
#include <vector>

#include "core/framework/config/model_config.h"
#include "core/framework/parallel_state/process_group.h"
#include "core/runtime/executor_impl_factory.h"
#include "core/runtime/params_utils.h"
#include "core/runtime/task_execution_pipeline.h"
#include "core/runtime/unified_mtp_worker_impl.h"
#include "core/runtime/worker_impl.h"

namespace xllm {
namespace {

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
                      const ModelInputParams& params) override {
    forward_inputs.emplace_back(tokens.clone());
    forward_positions.emplace_back(positions.clone());
    forward_slots.emplace_back(params.attention.device.new_cache_slots.clone());
    forward_counts.emplace_back(params.parallel.dp_global_token_nums);
    forward_embeddings.emplace_back(
        params.embedding.input_embedding.defined()
            ? params.embedding.input_embedding.clone()
            : torch::Tensor());
    forward_had_sync.emplace_back(params.parallel.layer_synchronizer !=
                                  nullptr);
    auto hidden = (block_draft ? positions : tokens).to(options_).view({-1, 1});
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
                               std::vector<KVCache>& /*kv_caches*/,
                               const ModelInputParams& /*params*/) override {
    context_hidden.emplace_back(target_hidden.clone());
    context_positions.emplace_back(positions.clone());
    context_slots.emplace_back(device_cache_slots.clone());
    return ModelOutput(target_hidden);
  }

  torch::Tensor logits(const torch::Tensor& hidden,
                       const torch::Tensor& selected) override {
    ++logits_calls;
    const auto next =
        hidden.index_select(0, selected.to(torch::kInt64)).to(torch::kInt64) +
        1;
    auto output = torch::full({selected.numel(), 32}, -20.0, options_);
    output.scatter_(/*dim=*/1, next, /*value=*/20.0);
    return output;
  }

  void load_model(std::unique_ptr<ModelLoader> /*loader*/) override {}
  torch::Device device() const override { return options_.device(); }
  const torch::TensorOptions& options() const override { return options_; }
  void prepare_expert_weight(int32_t /*layer_id*/,
                             const std::vector<int32_t>& /*experts*/) override {
  }
  void update_expert_weight(int32_t /*layer_id*/) override {}

  std::atomic<int32_t> logits_calls{0};
  bool block_draft = false;
  std::vector<torch::Tensor> forward_inputs;
  std::vector<torch::Tensor> forward_positions;
  std::vector<torch::Tensor> forward_slots;
  std::vector<std::vector<int32_t>> forward_counts;
  std::vector<torch::Tensor> forward_embeddings;
  std::vector<bool> forward_had_sync;
  std::function<void()> after_forward;
  std::vector<torch::Tensor> context_hidden;
  std::vector<torch::Tensor> context_positions;
  std::vector<torch::Tensor> context_slots;

 private:
  torch::TensorOptions options_;
};

class PreparedTestExecutor final : public ExecutorImpl {
 public:
  explicit PreparedTestExecutor(CausalLM* model) : model_(model) {}
  bool supports_prepared_attention_metadata() const override { return true; }
  void prepare_attention_metadata(std::vector<KVCache>& /*kv_caches*/,
                                  ModelInputParams& /*params*/) override {}
  ModelOutput run(const torch::Tensor& tokens,
                  const torch::Tensor& positions,
                  std::vector<KVCache>& kv_caches,
                  const ModelInputParams& params) override {
    return model_->forward(tokens, positions, kv_caches, params);
  }

 private:
  CausalLM* model_;
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
  void release() { release_.set_value(); }
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
    input.sampling_params.sample_idxes = torch::tensor({0}, torch::kInt32);
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
                        ModelInputParams& params) {
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
                                    ModelInputParams& params) {
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
                    ModelInputParams& params) {
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
  EXPECT_TRUE(torch::equal(draft_->forward_slots[1].cpu(),
                           torch::tensor({16, 63}, torch::kInt32)));
  EXPECT_TRUE(torch::equal(draft_->forward_slots[3].cpu(),
                           torch::tensor({17, 18}, torch::kInt32)));

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
  prefill.sampling_params.all_greedy_sample = false;
  prefill.sampling_params.all_random_sample = true;
  const auto first = execute(prefill);
  ASSERT_TRUE(first.sample_output.next_tokens.defined());
  EXPECT_EQ(first.sample_output.next_tokens.item<int64_t>(), 7);
  auto decode = input(true, 7, 2);
  decode.sampling_params.do_sample.fill_(true);
  decode.sampling_params.all_greedy_sample = false;
  decode.sampling_params.all_random_sample = true;
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
  prefill.sampling_params.all_greedy_sample = false;
  prefill.sampling_params.all_random_sample = true;
  const auto first = execute(prefill);
  ASSERT_TRUE(first.sample_output.next_tokens.defined());
  EXPECT_EQ(first.sample_output.next_tokens.item<int64_t>(), 7);
  auto decode = input(true, /*token=*/7, /*position=*/2);
  decode.sampling_params.do_sample.fill_(true);
  decode.sampling_params.all_greedy_sample = false;
  decode.sampling_params.all_random_sample = true;
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
  EXPECT_FALSE(decode.do_sample.defined());
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
  ASSERT_TRUE(forward_input_to_packed_proto(input(false), &payload));
  LlmForwardInput packed;
  packed_proto_to_forward_input(payload, packed, device_, nullptr);
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

}  // namespace
}  // namespace xllm
