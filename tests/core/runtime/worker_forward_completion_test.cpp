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

#include <acl/acl.h>
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "core/framework/config/eplb_config.h"
#include "core/framework/config/execution_config.h"
#include "core/framework/config/kernel_config.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/load_config.h"
#include "core/framework/config/model_config.h"
#include "core/framework/sampling/sampling_params.h"
#include "core/kv_cache/layout/kv_cache_capacity.h"
#include "core/kv_cache/layout/kv_cache_shape.h"
#include "core/kv_cache/transfer/kv_transfer.h"
#include "core/platform/device.h"
#include "core/runtime/llm_worker_impl.h"
#include "core/runtime/vlm_worker_impl.h"
#include "models/model_registry.h"

namespace xllm {
namespace {

using ForwardCompletion = WorkerImpl::ForwardCompletion;

template <typename T>
class ScopedConfigValue final {
 public:
  ScopedConfigValue(T& value, T new_value)
      : value_(value), old_value_(std::exchange(value, std::move(new_value))) {}

  ~ScopedConfigValue() { value_ = std::move(old_value_); }

 private:
  T& value_;
  T old_value_;
};

struct ForwardObservation {
  torch::Tensor hidden_states;
  torch::Tensor logits;
  const SamplingParameters* sampling_params = nullptr;
  bool forward_saw_sampling_history = false;
  bool logits_saw_sampling_history = false;
  aclrtStream execution_stream = nullptr;
  bool weights_loaded = false;
  bool emit_hidden = true;
  std::function<void()> on_forward;
  std::shared_ptr<KVPushSynchronizerImpl> push_synchronizer;
};

class CompletionLanguageModel final : public CausalLM {
 public:
  CompletionLanguageModel(const ModelContext& context,
                          std::shared_ptr<ForwardObservation> observation)
      : options_(context.get_tensor_options()),
        observation_(std::move(observation)) {}

  ModelOutput forward(const torch::Tensor& tokens,
                      const torch::Tensor& /*positions*/,
                      std::vector<KVCache>& /*kv_caches*/,
                      const ModelParamsView& parameters) override {
    observation_->forward_saw_sampling_history =
        observation_->sampling_params != nullptr &&
        observation_->sampling_params->prefix_tokens().defined();
    observation_->execution_stream =
        c10_npu::getCurrentNPUStream(tokens.device().index()).stream();
    observation_->hidden_states = tokens.add(/*other=*/5).unsqueeze(/*dim=*/1);
    observation_->push_synchronizer = parameters.parallel.layer_synchronizer;
    if (observation_->push_synchronizer != nullptr) {
      EXPECT_TRUE(observation_->push_synchronizer->record_event(
          /*layer_index=*/0, options_.device().index()));
    }
    if (observation_->on_forward) {
      observation_->on_forward();
    }
    ModelOutput output;
    if (observation_->emit_hidden) {
      output.hidden_states = observation_->hidden_states;
    }
    return output;
  }

  void load_model(std::unique_ptr<ModelLoader> /*loader*/) override {
    observation_->weights_loaded = true;
  }

  torch::Device device() const override { return options_.device(); }

  const torch::TensorOptions& options() const override { return options_; }

  void prepare_expert_weight(
      int32_t /*layer_id*/,
      const std::vector<int32_t>& /*expert_ids*/) override {}

  void update_expert_weight(int32_t /*layer_id*/) override {}

  torch::Tensor logits_processor(const torch::Tensor&) override {
    observation_->logits_saw_sampling_history =
        observation_->sampling_params != nullptr &&
        observation_->sampling_params->prefix_tokens().defined();
    return observation_->logits;
  }

 private:
  torch::TensorOptions options_;
  std::shared_ptr<ForwardObservation> observation_;
};

class CompletionTestModel final : public CausalVLM {
 public:
  CompletionTestModel(const ModelContext& context,
                      std::shared_ptr<ForwardObservation> observation)
      : language_model_(context, std::move(observation)) {}

  CausalLM& language_model() override { return language_model_; }

  void load_model(std::unique_ptr<ModelLoader> loader) override {
    language_model_.load_model(std::move(loader));
  }

  torch::Device device() const override { return language_model_.device(); }

  const torch::TensorOptions& options() const override {
    return language_model_.options();
  }

  MMDict encode(const ModelParamsView&) override { return {}; }

  torch::Tensor get_input_embeddings(const torch::Tensor&,
                                     const ModelParamsView&) override {
    return {};
  }

 private:
  CompletionLanguageModel language_model_;
};

class DeferredPushTransfer final : public KVTransfer {
 public:
  void register_kv_cache(std::vector<KVCache>& caches,
                         const KVCacheShape& /*shape*/,
                         torch::ScalarType /*dtype*/) override {
    registered_layers_ = static_cast<int32_t>(caches.size());
  }
  int32_t registered_layers() const { return registered_layers_; }

  void get_cache_info(uint64_t& /*cluster_id*/,
                      std::string& /*addr*/) override {}
  bool link_clusters(const std::vector<uint64_t>& /*cluster_ids*/,
                     const std::vector<std::string>& /*remote_addrs*/,
                     const std::vector<uint16_t>& /*ports*/) override {
    return true;
  }
  bool unlink_cluster(const uint64_t& /*cluster_id*/,
                      const std::string& /*remote_addr*/,
                      const uint16_t /*port*/,
                      bool /*force*/) override {
    return true;
  }
  bool pull_kv_blocks(
      const uint64_t /*cluster_id*/,
      const std::string& /*addr*/,
      const std::vector<KVTransferMapping>& /*mappings*/) override {
    return true;
  }
  void merge_kv_blocks(
      std::unordered_map<std::string, KVCacheInfo>& /*infos*/,
      const std::vector<TransferKVInfo>& /*transfers*/) override {}
  std::vector<std::string> push_kv_blocks(
      std::unordered_map<std::string, KVCacheInfo>& /*infos*/,
      std::shared_ptr<KVPushSynchronizerImpl>& /*synchronizer*/,
      bool /*is_spec_draft*/,
      int32_t /*kv_split_rank*/,
      int32_t /*kv_split_size*/) override {
    return {};
  }
  folly::SemiFuture<std::vector<std::string>> push_kv_blocks_async(
      const std::vector<TransferKVInfo>& /*transfers*/,
      const ParallelArgs& /*parallel_args*/,
      std::shared_ptr<KVPushSynchronizerImpl> synchronizer,
      bool is_spec_draft) override {
    EXPECT_FALSE(is_spec_draft);
    ++calls_;
    synchronizer_ = std::move(synchronizer);
    return completion_.getSemiFuture();
  }
  void finish(std::vector<std::string> failed_request_ids) {
    completion_.setValue(std::move(failed_request_ids));
  }
  int32_t calls() const { return calls_; }
  const std::shared_ptr<KVPushSynchronizerImpl>& synchronizer() const {
    return synchronizer_;
  }

 private:
  int32_t registered_layers_ = 0;
  int32_t calls_ = 0;
  folly::Promise<std::vector<std::string>> completion_;
  std::shared_ptr<KVPushSynchronizerImpl> synchronizer_;
};

template <typename Worker>
class PushTestWorker final : public Worker {
 public:
  using Worker::Worker;

  bool allocate_registered_cache(const KVCacheShape& shape,
                                 std::shared_ptr<KVTransfer> transfer) {
    this->dtype_ = torch::kFloat32;
    return this->allocate_kv_cache_with_transfer(std::move(transfer), shape);
  }
  void disable_result_driver() {
    this->driver_ = false;
    this->dp_driver_ = false;
  }
};

enum class WorkerKind : int8_t { LLM, VLM };

using CompletionCase = std::tuple<WorkerKind, ForwardCompletion, bool>;

class WorkerForwardCompletionTest
    : public ::testing::TestWithParam<CompletionCase> {
 protected:
  template <typename Worker, typename Input>
  void verify_completion(const std::string& backend) {
    const ForwardCompletion completion = std::get<1>(GetParam());
    const bool overlap_enabled = std::get<2>(GetParam());
    const torch::Device device(torch::kPrivateUse1, /*index=*/0);
    const ParallelArgs parallel_args(/*rank=*/0,
                                     /*world_size=*/1,
                                     /*process_group=*/nullptr);
    runtime::Options options;
    options.backend(backend);
    options.enable_schedule_overlap(overlap_enabled);
    options.enable_graph(false);
    options.enable_speculative_decode(false);
    Worker worker(parallel_args, device, options);
    Device execution_device(device);
    std::unique_ptr<Stream> stream = execution_device.get_stream_from_pool();

    auto observation = std::make_shared<ForwardObservation>();
    ASSERT_TRUE(
        initialize_model(worker, parallel_args, device, backend, observation));

    auto owner_released = std::make_shared<std::atomic<bool>>(false);
    Input input;
    {
      auto guard = stream->set_stream_guard();
      input.token_ids = torch::tensor(
          {7}, torch::TensorOptions().dtype(torch::kInt32).device(device));
      input.positions = torch::zeros_like(input.token_ids);
    }
    input.input_params.meta.num_sequences = 1;
    input.input_params.meta.batch_forward_type = BatchForwardType::DECODE;
    auto owner = std::make_unique<int32_t>(42);
    input.runtime.retained_device_tensors.emplace_back(torch::from_blob(
        owner.release(),
        {1},
        [owner_released](void* data) {
          delete static_cast<int32_t*>(data);
          owner_released->store(true);
        },
        torch::TensorOptions().dtype(torch::kInt32)));

    WorkerImpl& leaf = worker;
    std::optional<ForwardOutput> output =
        leaf.execute_no_sync_on_stream(input, *stream, completion);
    input.runtime.retained_device_tensors.clear();
    const bool owner_released_before_completion = owner_released->load();
    bool event_synchronized = true;
    if (output.has_value() && output->ready_event != nullptr) {
      event_synchronized = output->ready_event->synchronize();
    }
    const int32_t stream_result = stream->synchronize();

    EXPECT_EQ(observation->execution_stream, stream->get_stream()->stream());
    EXPECT_FALSE(observation->weights_loaded);
    EXPECT_FALSE(owner_released_before_completion);
    EXPECT_EQ(stream_result, 0);
    ASSERT_TRUE(output.has_value());
    if (completion == ForwardCompletion::RECORD_EVENT) {
      ASSERT_NE(output->ready_event, nullptr);
      EXPECT_TRUE(event_synchronized);
    } else {
      EXPECT_EQ(output->ready_event, nullptr);
    }
    ASSERT_EQ(output->retained_inputs.size(), 1);
    auto retained =
        std::static_pointer_cast<const Input>(output->retained_inputs.front());
    ASSERT_EQ(retained->runtime.retained_device_tensors.size(), 1);
    EXPECT_EQ(retained->runtime.retained_device_tensors.front()
                  .template item<int32_t>(),
              42);
    EXPECT_EQ(
        observation->hidden_states.to(torch::kCPU).template item<int32_t>(),
        12);
    EXPECT_FALSE(owner_released->load());
    retained.reset();
    output.reset();
    EXPECT_TRUE(owner_released->load());
  }

  template <typename Worker, typename Input>
  void verify_sampling_history(const std::string& backend) {
    const ForwardCompletion completion = std::get<1>(GetParam());
    const torch::Device device(torch::kPrivateUse1, /*index=*/0);
    const ParallelArgs parallel_args(/*rank=*/0,
                                     /*world_size=*/1,
                                     /*process_group=*/nullptr);
    runtime::Options options;
    options.backend(backend);
    options.enable_schedule_overlap(std::get<2>(GetParam()));
    options.enable_graph(false);
    options.enable_speculative_decode(false);
    Worker worker(parallel_args, device, options);
    Device execution_device(device);
    std::unique_ptr<Stream> stream = execution_device.get_stream_from_pool();
    auto observation = std::make_shared<ForwardObservation>();
    ASSERT_TRUE(
        initialize_model(worker, parallel_args, device, backend, observation));

    Input input;
    torch::Tensor prior_tokens;
    {
      auto guard = stream->set_stream_guard();
      const auto int_options =
          torch::TensorOptions().dtype(torch::kInt32).device(device);
      const auto long_options = int_options.dtype(torch::kInt64);
      const auto float_options = int_options.dtype(torch::kFloat32);
      input.token_ids = torch::tensor({2, 1, 7}, int_options);
      input.positions = torch::zeros_like(input.token_ids);
      input.input_params.meta.num_sequences = 3;
      input.input_params.meta.batch_forward_type = BatchForwardType::DECODE;
      // Reorder continuing rows and leave the inserted row without history.
      input.sample_prior_output_rows = {1, 0, -1};
      auto& sampling = input.sampling_params;
      sampling.logits_indices() = torch::arange(3, long_options);
      sampling.bad_words() =
          torch::tensor({{{2, 3}}, {{2, 3}}, {{2, 3}}}, long_options);
      sampling.bad_word_suffix() = torch::full({3, 1}, -1, long_options);
      sampling.repetition_penalties() =
          torch::tensor({1.0F, 2.0F, 1.0F}, float_options);
      sampling.frequency_penalties() =
          torch::tensor({0.0F, 1.0F, 0.0F}, float_options);
      sampling.presence_penalties() =
          torch::tensor({0.0F, 1.0F, 0.0F}, float_options);
      sampling.unique_token_ids() = torch::zeros({3, 1}, long_options);
      sampling.unique_token_counts() = torch::zeros({3, 1}, int_options);
      sampling.unique_token_ids_lens() = torch::zeros({3}, int_options);
      sampling.seeds() = torch::tensor({100, 200, 300}, long_options);
      sampling.seed_offsets() = torch::tensor({7, 11, 13}, long_options);
      prior_tokens = torch::tensor({{1, 1}, {2, -1}}, int_options);
      // History must block token 3 in row zero and penalize token 1 enough
      // to select token 2 in row one. The inserted row still selects token 3.
      observation->logits = torch::tensor({{4.0F, 0.0F, 0.0F, 10.0F},
                                           {0.0F, 8.0F, 2.0F, 0.0F},
                                           {3.0F, 0.0F, 0.0F, 8.0F}},
                                          float_options);
    }
    observation->sampling_params = &input.sampling_params;
    WorkerImpl& leaf = worker;
    std::optional<ForwardOutput> output = leaf.execute_no_sync_on_stream(
        input, *stream, completion, [&input, &prior_tokens]() {
          advance_sampling_history(input.sampling_params,
                                   prior_tokens,
                                   input.sample_prior_output_rows);
        });
    EXPECT_EQ(stream->synchronize(), 0);

    ASSERT_TRUE(output.has_value());
    ASSERT_TRUE(output->sample_output.next_tokens.defined());
    EXPECT_FALSE(observation->forward_saw_sampling_history);
    EXPECT_FALSE(observation->logits_saw_sampling_history);
    EXPECT_TRUE(torch::equal(output->sample_output.next_tokens.to(torch::kCPU),
                             torch::tensor({0, 2, 3}, torch::kInt64)));
    EXPECT_TRUE(torch::equal(
        input.sampling_params.prefix_tokens().to(torch::kCPU),
        torch::tensor({{2, -1}, {1, 1}, {-1, -1}}, torch::kInt64)));
    EXPECT_TRUE(
        torch::equal(input.sampling_params.bad_word_suffix().to(torch::kCPU),
                     torch::tensor({{2}, {1}, {-1}}, torch::kInt64)));
    EXPECT_TRUE(
        torch::equal(input.sampling_params.seed_offsets().to(torch::kCPU),
                     torch::tensor({8, 13, 13}, torch::kInt64)));
  }

  template <typename Worker, typename Input>
  void verify_kv_push(const std::string& backend,
                      bool emit_hidden,
                      bool non_driver,
                      bool transfer_fails) {
    const ForwardCompletion completion = std::get<1>(GetParam());
    const torch::Device device(torch::kPrivateUse1, /*index=*/0);
    const ParallelArgs parallel_args(/*rank=*/0,
                                     /*world_size=*/1,
                                     /*process_group=*/nullptr);
    runtime::Options options;
    options.backend(backend)
        .enable_schedule_overlap(std::get<2>(GetParam()))
        .enable_graph(false)
        .enable_speculative_decode(false)
        .enable_disagg_pd(true)
        .instance_role(InstanceRole::PREFILL)
        .block_size(4);
    PushTestWorker<Worker> worker(parallel_args, device, options);
    Device execution_device(device);
    std::unique_ptr<Stream> stream = execution_device.get_stream_from_pool();
    auto observation = std::make_shared<ForwardObservation>();
    observation->emit_hidden = emit_hidden;
    ASSERT_TRUE(
        initialize_model(worker, parallel_args, device, backend, observation));
    if (non_driver) {
      worker.disable_result_driver();
    }
    auto transfer = std::make_shared<DeferredPushTransfer>();
    KVCacheCapacity capacity;
    capacity.n_blocks(2).block_size(4);
    const KVCacheShape shape(
        capacity, worker.context_.get_model_args(), /*world_size=*/1);
    ASSERT_TRUE(worker.allocate_registered_cache(shape, transfer));
    EXPECT_EQ(transfer->registered_layers(), 1);
    Input input;
    {
      auto guard = stream->set_stream_guard();
      input.token_ids = torch::tensor(
          {7}, torch::TensorOptions().dtype(torch::kInt32).device(device));
      input.positions = torch::zeros_like(input.token_ids);
    }
    input.input_params.meta.num_sequences = 1;
    input.input_params.meta.batch_forward_type = BatchForwardType::PREFILL;
    TransferKVInfo info;
    info.request_id = "request";
    input.transfer_kv_infos.emplace_back(std::move(info));
    std::promise<void> forward_entered;
    std::future<void> entered = forward_entered.get_future();
    observation->on_forward = [&]() { forward_entered.set_value(); };
    WorkerImpl& leaf = worker;
    auto result = std::async(std::launch::async, [&]() {
      return leaf.execute_no_sync_on_stream(input, *stream, completion);
    });
    EXPECT_EQ(entered.wait_for(std::chrono::seconds(5)),
              std::future_status::ready);
    EXPECT_EQ(result.wait_for(std::chrono::milliseconds(50)),
              std::future_status::timeout);
    std::vector<std::string> failures;
    if (transfer_fails) {
      failures.emplace_back("request");
    }
    transfer->finish(failures);
    std::optional<ForwardOutput> output = result.get();
    EXPECT_EQ(stream->synchronize(), 0);
    ASSERT_TRUE(output.has_value());
    EXPECT_EQ(output->failed_request_ids, failures);
    EXPECT_EQ(transfer->calls(), 1);
    ASSERT_NE(transfer->synchronizer(), nullptr);
    EXPECT_EQ(transfer->synchronizer()->get_event_size(), 1u);
    EXPECT_EQ(observation->push_synchronizer, transfer->synchronizer());
  }

 private:
  bool initialize_model(
      WorkerImpl& worker,
      const ParallelArgs& parallel_args,
      const torch::Device& device,
      const std::string& backend,
      const std::shared_ptr<ForwardObservation>& observation) {
    static std::atomic<int32_t> next_model_id{0};
    const std::string model_name = "worker_forward_completion_" + backend +
                                   "_" +
                                   std::to_string(next_model_id.fetch_add(1));
    const std::weak_ptr<ForwardObservation> weak_observation = observation;
    if (!ModelRegistry::register_causallm_factory(
            model_name,
            /*implementation=*/"native",
            [weak_observation](const ModelContext& context) {
              return std::make_unique<CompletionLanguageModel>(
                  context, weak_observation.lock());
            })) {
      return false;
    }
    if (!ModelRegistry::register_causalvlm_factory(
            model_name,
            /*implementation=*/"native",
            [weak_observation](const ModelContext& context) {
              return std::make_unique<CompletionTestModel>(
                  context, weak_observation.lock());
            })) {
      return false;
    }
    ModelArgs args;
    args.model_type(model_name)
        .n_layers(1)
        .hidden_size(16)
        .n_heads(2)
        .n_kv_heads(1)
        .head_dim(8);
    ModelContext context(parallel_args,
                         args,
                         QuantArgs{},
                         torch::TensorOptions().device(device));
    context.set_model_impl("native");
    worker.context_ = std::move(context);
    return worker.init_model(worker.context_);
  }

  ScopedConfigValue<bool> rolling_load_{
      LoadConfig::get_instance().enable_rolling_load(),
      false};
  ScopedConfigValue<bool> virtual_memory_{
      KVCacheConfig::get_instance().enable_vmm_kvcache(),
      false};
  ScopedConfigValue<bool> eplb_{EPLBConfig::get_instance().enable_eplb(),
                                false};
  ScopedConfigValue<bool> graph_{ExecutionConfig::get_instance().enable_graph(),
                                 false};
  ScopedConfigValue<std::string> model_impl_{
      ModelConfig::get_instance().model_impl(),
      "native"};
  ScopedConfigValue<std::string> kernel_backend_{
      KernelConfig::get_instance().npu_kernel_backend(),
      "TORCH"};
};

TEST_P(WorkerForwardCompletionTest, HonorsCompletionAndRetainsInputOwners) {
  if (std::get<0>(GetParam()) == WorkerKind::LLM) {
    verify_completion<LLMWorkerImpl, LlmForwardInput>("llm");
  } else {
    verify_completion<VLMWorkerImpl, VlmForwardInput>("vlm");
  }
}

TEST_P(WorkerForwardCompletionTest, PreparesSamplingAfterForwardAndLogits) {
  if (std::get<0>(GetParam()) == WorkerKind::LLM) {
    verify_sampling_history<LLMWorkerImpl, LlmForwardInput>("llm");
  } else {
    verify_sampling_history<VLMWorkerImpl, VlmForwardInput>("vlm");
  }
}

TEST_P(WorkerForwardCompletionTest, WaitsForKVPushBeforeReturning) {
  if (std::get<0>(GetParam()) == WorkerKind::LLM) {
    verify_kv_push<LLMWorkerImpl, LlmForwardInput>("llm",
                                                   /*emit_hidden=*/true,
                                                   /*non_driver=*/false,
                                                   /*transfer_fails=*/false);
  } else {
    verify_kv_push<VLMWorkerImpl, VlmForwardInput>("vlm",
                                                   /*emit_hidden=*/true,
                                                   /*non_driver=*/false,
                                                   /*transfer_fails=*/false);
  }
}

TEST_P(WorkerForwardCompletionTest, ReportsKVPushFailureOnNonDriver) {
  if (std::get<0>(GetParam()) == WorkerKind::LLM) {
    verify_kv_push<LLMWorkerImpl, LlmForwardInput>("llm",
                                                   /*emit_hidden=*/true,
                                                   /*non_driver=*/true,
                                                   /*transfer_fails=*/true);
  } else {
    verify_kv_push<VLMWorkerImpl, VlmForwardInput>("vlm",
                                                   /*emit_hidden=*/true,
                                                   /*non_driver=*/true,
                                                   /*transfer_fails=*/true);
  }
}

TEST_P(WorkerForwardCompletionTest, ReportsKVPushFailureWithoutHiddenOutput) {
  if (std::get<0>(GetParam()) == WorkerKind::LLM) {
    verify_kv_push<LLMWorkerImpl, LlmForwardInput>("llm",
                                                   /*emit_hidden=*/false,
                                                   /*non_driver=*/false,
                                                   /*transfer_fails=*/true);
  } else {
    verify_kv_push<VLMWorkerImpl, VlmForwardInput>("vlm",
                                                   /*emit_hidden=*/false,
                                                   /*non_driver=*/false,
                                                   /*transfer_fails=*/true);
  }
}

INSTANTIATE_TEST_SUITE_P(
    LeafWorkers,
    WorkerForwardCompletionTest,
    ::testing::Combine(::testing::Values(WorkerKind::LLM, WorkerKind::VLM),
                       ::testing::Values(ForwardCompletion::CALLER_MANAGED,
                                         ForwardCompletion::RECORD_EVENT),
                       ::testing::Bool()),
    [](const ::testing::TestParamInfo<CompletionCase>& info) {
      const std::string worker =
          std::get<0>(info.param) == WorkerKind::LLM ? "Llm" : "Vlm";
      const std::string completion =
          std::get<1>(info.param) == ForwardCompletion::RECORD_EVENT
              ? "RecordEvent"
              : "CallerManaged";
      const std::string overlap =
          std::get<2>(info.param) ? "OverlapEnabled" : "OverlapDisabled";
      return worker + completion + overlap;
    });

}  // namespace
}  // namespace xllm
