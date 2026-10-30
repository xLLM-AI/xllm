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

#include <gtest/gtest.h>
#include <pybind11/embed.h>
#include <torch/python.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include <chrono>
#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/framework/config/model_config.h"
#include "core/framework/parallel_state/process_group.h"
#include "core/layers/common/attention_metadata.h"
#include "core/platform/layer_synchronizer.h"
#include "core/runtime/executor_impl_factory.h"
#include "core/runtime/task_execution_pipeline.h"
#include "models/llm/py_causal_lm.h"
#include "models/py_model_helper.h"

namespace py = pybind11;

namespace xllm {
namespace {

struct RecurrentObservation {
  torch::Tensor read_indices;
  torch::Tensor write_indices;
  torch::Tensor query_starts;
  torch::Tensor warm;
  torch::Tensor conv;
  torch::Tensor ssm;
  bool enable_graph = false;
};

// Advance real device state and retain the values each forward consumes. The
// first layer intentionally has no full-attention cache.
class RecurrentTestModel final : public CausalLM {
 public:
  explicit RecurrentTestModel(const torch::Device& device)
      : options_(torch::TensorOptions().device(device).dtype(torch::kFloat32)) {
    observations.reserve(16);
  }

  ModelOutput forward(const torch::Tensor& tokens,
                      const torch::Tensor& /*positions*/,
                      std::vector<KVCache>& caches,
                      const ModelParamsView& params) override {
    if (before_forward) {
      before_forward(static_cast<uint32_t>(observations.size()));
    }
    const auto& metadata = *params.attn_metadata;
    const torch::Tensor write_indices =
        params.embedding.linear_state_indices.to(torch::kInt64);
    const torch::Tensor read_indices =
        metadata.linear_state_read_indices.defined()
            ? metadata.linear_state_read_indices.to(torch::kInt64)
            : write_indices;
    const torch::Tensor conv = caches.front().get_conv_cache();
    const torch::Tensor ssm = caches.front().get_ssm_cache();
    const torch::Tensor read_conv = conv.index_select(/*dim=*/0, read_indices);
    const torch::Tensor read_ssm = ssm.index_select(/*dim=*/0, read_indices);
    const torch::Tensor previous_conv =
        torch::where(metadata.has_initial_states.view({-1, 1, 1}),
                     read_conv,
                     torch::zeros_like(read_conv));
    const torch::Tensor previous_ssm =
        torch::where(metadata.has_initial_states.view({-1, 1, 1}),
                     read_ssm,
                     torch::zeros_like(read_ssm));
    observations.emplace_back(
        RecurrentObservation{read_indices.clone(),
                             write_indices.clone(),
                             metadata.q_cu_seq_lens.clone(),
                             metadata.has_initial_states.clone(),
                             previous_conv.clone(),
                             previous_ssm.clone(),
                             params.enable_graph});
    const torch::Tensor active = write_indices.gt(0).view({-1, 1, 1});
    const torch::Tensor delta =
        params.attention.device.q_seq_lens.to(options_).view({-1, 1, 1});
    conv.index_copy_(
        /*dim=*/0,
        write_indices,
        torch::where(active, previous_conv + delta, read_conv));
    ssm.index_copy_(
        /*dim=*/0,
        write_indices,
        torch::where(active, previous_ssm + delta * 10, read_ssm));
    return ModelOutput(tokens.to(options_).view({-1, 1}));
  }

  torch::Tensor logits(const torch::Tensor& hidden,
                       const torch::Tensor& selected) override {
    torch::Tensor result =
        torch::zeros({selected.numel(), 32}, hidden.options());
    result.select(/*dim=*/1, /*index=*/1).fill_(20);
    return result;
  }
  void load_model(std::unique_ptr<ModelLoader> /*loader*/) override {}
  torch::Device device() const override { return options_.device(); }
  const torch::TensorOptions& options() const override { return options_; }
  bool is_hybrid_linear_attention() override { return true; }
  void prepare_expert_weight(int32_t /*layer_id*/,
                             const std::vector<int32_t>& /*experts*/) override {
  }
  void update_expert_weight(int32_t /*layer_id*/) override {}

  std::function<void(uint32_t)> before_forward;
  std::vector<RecurrentObservation> observations;

 private:
  torch::TensorOptions options_;
};

class RecurrentTestExecutor final : public ExecutorImpl {
 public:
  explicit RecurrentTestExecutor(CausalLM* model) : model_(model) {}
  bool supports_prepared_attention_metadata() const override { return true; }
  void prepare_attention_metadata(std::vector<KVCache>& /*caches*/,
                                  ModelParamsView& /*params*/) override {}
  void warmup_prepared_graph(const torch::Tensor& tokens,
                             const torch::Tensor& positions,
                             std::vector<KVCache>& caches,
                             const ModelParamsView& params) override {
    // Capture executes the model too. Exercise the state effects here; the
    // Python runner tests cover actual graph storage and replay.
    model_->forward(tokens, positions, caches, params);
  }
  ModelOutput run(const torch::Tensor& tokens,
                  const torch::Tensor& positions,
                  std::vector<KVCache>& caches,
                  const ModelParamsView& params) override {
    return model_->forward(tokens, positions, caches, params);
  }

 private:
  CausalLM* model_;
};

// Publish a marker immediately before the first consumer wait. The producer
// can finish that marker before submitting H2D, making an early restore read
// stale state deterministically without a device wait on an unrecorded event.
class DelayedHostLoadSynchronizer final : public LayerSynchronizer {
 public:
  DelayedHostLoadSynchronizer()
      : synchronizer_(create_layer_synchronizer(/*num_layers=*/1)) {}

  std::future<StreamEventPtr> entered() { return entered_.get_future(); }

  bool synchronize_layer(int64_t /*layer_index*/) override {
    ADD_FAILURE() << "Host reload must wait on the consumer stream.";
    return false;
  }

  bool wait_layer_on_current_stream(int64_t layer_index) override {
    if (!waiting_) {
      waiting_ = true;
      Stream consumer(c10_npu::getCurrentNPUStream());
      entered_.set_value(consumer.record_event());
    }
    return synchronizer_->wait_layer_on_current_stream(layer_index);
  }

  bool record_stream(int64_t layer_index, Stream* stream) override {
    return synchronizer_->record_stream(layer_index, stream);
  }

  void abort() override { synchronizer_->abort(); }
  uint32_t size() const override { return synchronizer_->size(); }

 private:
  std::shared_ptr<LayerSynchronizer> synchronizer_;
  std::promise<StreamEventPtr> entered_;
  bool waiting_ = false;
};

class RecurrentPipelineTest : public ::testing::Test {
 protected:
  void SetUp() override {
    static const bool registered =
        ExecutorImplFactory::get_instance().register_creator(
            "recurrent_pipeline_test",
            [](CausalLM* model,
               const ModelArgs& /*args*/,
               const torch::Device& /*device*/,
               const runtime::Options& /*options*/) {
              return std::make_unique<RecurrentTestExecutor>(model);
            });
    ASSERT_TRUE(registered);
    previous_model_impl_ = ModelConfig::get_instance().model_impl();
    ModelConfig::get_instance().model_impl("auto");
    model_ = std::make_unique<RecurrentTestModel>(device_);
    runtime::Options options;
    options.backend("recurrent_pipeline_test").enable_graph(false);
    executor_ =
        std::make_unique<Executor>(model_.get(), ModelArgs{}, device_, options);
    caches_.emplace_back(LinearAttentionKVCacheTensors{
        torch::full({8, 1, 1}, 7.0, model_->options()),
        torch::full({8, 1, 1}, 11.0, model_->options())});
    caches_.emplace_back(
        KVCacheTensors{torch::zeros({4, 16, 1, 1}, model_->options()),
                       torch::zeros({4, 16, 1, 1}, model_->options())});
    capacity_.slot_count = 2;
    capacity_.model = {16, 4, 2};
    capacity_.model.enable_linear_attention = true;
    capacity_.max_kv_seq_len = 32;
    capacity_.max_positions = 32;
    capacity_.logical_block_size = 16;
    capacity_.vocab_size = 32;
    capacity_.max_unique_tokens = 32;
    capacity_.hidden_size = 1;
    capacity_.chunked_prefill = true;
    ASSERT_EQ(aclrtSynchronizeDevice(), ACL_SUCCESS);
    ASSERT_TRUE(
        TaskExecutionPipeline::create(
            state_thread_, *model_, *executor_, caches_, capacity_, pipeline_)
            .ok());
  }

  void TearDown() override {
    pipeline_.reset();
    EXPECT_EQ(aclrtSynchronizeDevice(), ACL_SUCCESS);
    ModelConfig::get_instance().model_impl(previous_model_impl_);
  }

  LlmForwardInput input(int32_t slot,
                        int32_t query,
                        int32_t cached = 0,
                        bool decode = false) const {
    LlmForwardInput input;
    input.token_ids = torch::ones({query}, torch::kInt32);
    input.positions = torch::arange(cached, cached + query, torch::kInt32);
    auto& params = input.input_params;
    params.meta.batch_forward_type = decode ? BatchForwardType::DECODE
                                     : cached == 0
                                         ? BatchForwardType::PREFILL
                                         : BatchForwardType::CHUNKED_PREFILL;
    params.meta.num_sequences = 1;
    params.meta.actual_num_sequences = 1;
    params.attention.host.q_seq_lens = {query};
    params.attention.host.kv_seq_lens = {cached + query};
    params.attention.host.q_cu_seq_lens = {query};
    params.attention.host.block_tables = torch::tensor({{0, 1}}, torch::kInt32);
    params.attention.device.new_cache_slots = input.positions.clone();
    params.embedding.linear_state_ids = {slot};
    params.embedding.linear_state_indices =
        torch::tensor({slot}, torch::kInt32);
    params.linear_state_validity_mask = {cached == 0 ? 0 : 1};
    return input;
  }

  void finish(const TaskSubmission& submission) {
    ASSERT_TRUE(submission.status.ok()) << submission.status.message();
    const auto result = pipeline_->take_result_async(submission.task_id).get();
    ASSERT_TRUE(result.status.ok()) << result.status.message();
  }

  const torch::Device device_{torch::kPrivateUse1, 0};
  std::string previous_model_impl_;
  ThreadPool state_thread_{1};
  LlmTaskCapacity capacity_;
  std::unique_ptr<RecurrentTestModel> model_;
  std::unique_ptr<Executor> executor_;
  std::vector<KVCache> caches_;
  std::unique_ptr<TaskExecutionPipeline> pipeline_;
};

TEST_F(RecurrentPipelineTest, ColdChunkedDecodeAndReusedSlotPreserveState) {
  finish(pipeline_->submit(input(/*slot=*/1, /*query=*/2)));
  finish(pipeline_->submit(input(/*slot=*/1, /*query=*/2, /*cached=*/2)));
  finish(pipeline_->submit(
      input(/*slot=*/1, /*query=*/1, /*cached=*/4, /*decode=*/true)));
  finish(pipeline_->submit(input(/*slot=*/1, /*query=*/1)));
  ASSERT_EQ(model_->observations.size(), 4);
  const std::vector<float> conv{0, 2, 4, 0};
  const std::vector<float> ssm{0, 20, 40, 0};
  for (uint32_t index = 0; index < conv.size(); ++index) {
    const auto& observed = model_->observations[index];
    EXPECT_EQ(observed.conv.cpu().item<float>(), conv[index]);
    EXPECT_EQ(observed.ssm.cpu().item<float>(), ssm[index]);
    EXPECT_EQ(observed.warm.cpu().item<bool>(), index == 1 || index == 2);
    EXPECT_TRUE(torch::equal(
        observed.query_starts.cpu(),
        torch::tensor({0, index == 0 || index == 1 ? 2 : 1}, torch::kInt32)));
  }
  EXPECT_EQ(caches_.front().get_conv_cache()[1].cpu().item<float>(), 1);
  EXPECT_EQ(caches_.front().get_conv_cache()[2].cpu().item<float>(), 7);
}

TEST_F(RecurrentPipelineTest,
       RestoreWaitsForEarlierForwardAndOwnsInputOperations) {
  std::promise<void> entered;
  auto entered_future = entered.get_future();
  std::promise<void> release;
  auto release_future = release.get_future();
  model_->before_forward = [&](uint32_t call) {
    if (call == 0) {
      entered.set_value();
      release_future.wait();
    }
  };
  const auto first = pipeline_->submit(input(/*slot=*/1, /*query=*/2));
  const auto entered_status = entered_future.wait_for(std::chrono::seconds(10));
  EXPECT_EQ(entered_status, std::future_status::ready);
  auto restore = input(/*slot=*/2, /*query=*/1, /*cached=*/2);
  restore.input_params.embedding.linear_state_read_ids = {1};
  const auto second = pipeline_->submit(restore);
  restore.input_params.embedding.linear_state_read_ids.clear();
  restore.input_params.embedding.linear_state_ids.front() = 3;
  restore.input_params.embedding.linear_state_indices.fill_(3);
  release.set_value();
  finish(first);
  finish(second);
  ASSERT_EQ(model_->observations.size(), 2);
  const auto& observed = model_->observations[1];
  EXPECT_EQ(observed.write_indices.cpu().item<int64_t>(), 2);
  EXPECT_EQ(observed.read_indices.cpu().item<int64_t>(), 1);
  EXPECT_EQ(observed.conv.cpu().item<float>(), 2);
  EXPECT_EQ(observed.ssm.cpu().item<float>(), 20);
  EXPECT_TRUE(observed.warm.cpu().item<bool>());
  EXPECT_EQ(caches_.front().get_ssm_cache()[2].cpu().item<float>(), 30);
  EXPECT_EQ(caches_.front().get_ssm_cache()[3].cpu().item<float>(), 11);
}

TEST_F(RecurrentPipelineTest,
       PythonRestoreWaitsForDelayedHostCheckpointReload) {
  pipeline_.reset();
  auto load = std::make_shared<DelayedHostLoadSynchronizer>();
  auto entered = load->entered();
  torch::Tensor observed_conv;
  torch::Tensor observed_ssm;
  ensure_python_interpreter();
  py::gil_scoped_acquire gil;
  py::dict scope;
  scope["native_forward"] = py::cpp_function(
      [&](const torch::Tensor& tokens, const torch::Tensor& indices) {
        const torch::Tensor rows = indices.to(torch::kInt64);
        const torch::Tensor conv = caches_.front().get_conv_cache();
        const torch::Tensor ssm = caches_.front().get_ssm_cache();
        observed_conv = conv.index_select(/*dim=*/0, rows).clone();
        observed_ssm = ssm.index_select(/*dim=*/0, rows).clone();
        conv.index_copy_(/*dim=*/0, rows, observed_conv + 1);
        ssm.index_copy_(/*dim=*/0, rows, observed_ssm + 10);
        return tokens.to(model_->options()).view({-1, 1});
      });
  py::exec(R"PY(
import sys
import types
import unittest.mock

class TestModel:
    def __init__(self, config: dict) -> None:
        pass
    def eval(self) -> "TestModel":
        return self

class TestExecutor:
    supports_prepared_metadata = True
    supports_layerwise_kv_load = False
    context_kv_graph_runner = None
    def __init__(self, *args: object) -> None:
        pass
    def bind_kv_caches(self, caches: object) -> None:
        pass
    def prepare_metadata(self, metadata: object) -> None:
        pass
    def execute(self, tokens: object, positions: object, metadata: object,
                *args: object, **kwargs: object) -> object:
        return native_forward(tokens, metadata.linear_state_write_indices)

registry = types.ModuleType('xllm.python.registry')
registry.get_model_class = lambda name: TestModel
executor_module = types.ModuleType('xllm.python.model_executor.executor')
executor_module.ModelExecutor = TestExecutor
distributed = types.ModuleType('xllm.python.distributed')
distributed.init_process_group = lambda *args: None
patch = unittest.mock.patch.dict(sys.modules, {
    'xllm.python.registry': registry,
    'xllm.python.model_executor.executor': executor_module,
    'xllm.python.distributed': distributed,
})
patch.__enter__()
)PY",
           scope);
  {
    ProcessGroup group(/*rank=*/0, /*world_size=*/1, device_);
    ParallelArgs parallel(/*rank=*/0, /*world_size=*/1, &group);
    parallel.moe_tp_group_ = &group;
    parallel.python_rendezvous_host_ = "127.0.0.1";
    parallel.python_rendezvous_port_ = 1;
    ModelArgs args;
    args.model_type("qwen3_5_text")
        .dtype("float32")
        .n_heads(1)
        .n_kv_heads(1)
        .head_dim(8)
        .n_layers(2)
        .vocab_size(32);
    ModelContext context(parallel,
                         args,
                         QuantArgs(),
                         model_->options(),
                         /*context=*/nullptr);
    PyCausalLM model(context, "test_recurrent_reload");
    ModelConfig::get_instance().model_impl("python");
    runtime::Options options;
    options.enable_task_pipeline(true).enable_graph(false);
    Executor executor(&model, args, device_, options);
    // Native executors already wait before restore; use the real Python
    // bridge so this test covers its later model-consumer fence as well.
    EXPECT_NE(executor.python_impl(), nullptr);
    TaskModel resources{model, executor, caches_};
    resources.load_kv = [load](uint64_t batch_id) {
      EXPECT_EQ(batch_id, 101U);
      return HostKVLoadHandle{load, /*layers_per_event=*/1};
    };
    {
      py::gil_scoped_release release;
      const Status created = TaskExecutionPipeline::create(
          state_thread_, std::move(resources), capacity_, pipeline_);
      EXPECT_TRUE(created.ok()) << created.message();
      if (created.ok()) {
        auto restore = input(/*slot=*/2, /*query=*/1, /*cached=*/2);
        restore.input_params.meta.batch_id = 101;
        restore.input_params.embedding.linear_state_read_ids = {1};
        restore.input_params.linear_state_validity_mask = {1};
        const auto submitted = pipeline_->submit(restore);
        EXPECT_TRUE(submitted.status.ok()) << submitted.status.message();
        if (submitted.status.ok()) {
          const auto waiting = entered.wait_for(std::chrono::seconds(10));
          EXPECT_EQ(waiting, std::future_status::ready);
          if (waiting == std::future_status::ready) {
            const auto marker = entered.get();
            EXPECT_NE(marker, nullptr);
            if (marker != nullptr) {
              EXPECT_TRUE(marker->synchronize());
            }
          }
          Stream copy_stream(device_);
          const auto host_options =
              torch::TensorOptions().dtype(torch::kFloat32).pinned_memory(true);
          const torch::Tensor conv_host =
              torch::full({1, 1}, 23.0, host_options);
          const torch::Tensor ssm_host =
              torch::full({1, 1}, 47.0, host_options);
          {
            auto guard = copy_stream.set_stream_guard();
            caches_.front().get_conv_cache()[1].copy_(conv_host,
                                                      /*non_blocking=*/true);
            caches_.front().get_ssm_cache()[1].copy_(ssm_host,
                                                     /*non_blocking=*/true);
          }
          const bool recorded =
              load->record_stream(/*layer_index=*/0, &copy_stream);
          EXPECT_TRUE(recorded);
          if (!recorded) {
            load->abort();
          }
          finish(submitted);
        }
      }
      pipeline_.reset();
    }
  }
  scope["patch"].attr("__exit__")(py::none(), py::none(), py::none());
  ASSERT_TRUE(observed_conv.defined());
  ASSERT_TRUE(observed_ssm.defined());
  EXPECT_EQ(observed_conv.cpu().item<float>(), 23);
  EXPECT_EQ(observed_ssm.cpu().item<float>(), 47);
  EXPECT_EQ(caches_.front().get_conv_cache()[1].cpu().item<float>(), 23);
  EXPECT_EQ(caches_.front().get_ssm_cache()[1].cpu().item<float>(), 47);
  EXPECT_EQ(caches_.front().get_conv_cache()[2].cpu().item<float>(), 24);
  EXPECT_EQ(caches_.front().get_ssm_cache()[2].cpu().item<float>(), 57);
}

TEST_F(RecurrentPipelineTest,
       DirectReadAdvancesOnlyLiveSlotBeforeContinuingDecode) {
  finish(pipeline_->submit(input(/*slot=*/1, /*query=*/2)));
  auto direct = input(/*slot=*/2, /*query=*/2, /*cached=*/2);
  direct.input_params.embedding.linear_state_read_ids = {1};
  direct.input_params.linear_state_validity_mask = {1};
  finish(pipeline_->submit(direct));
  ASSERT_EQ(model_->observations.size(), 2);
  const auto& observed = model_->observations[1];
  EXPECT_EQ(observed.read_indices.cpu().item<int64_t>(), 1);
  EXPECT_EQ(observed.write_indices.cpu().item<int64_t>(), 2);
  EXPECT_EQ(observed.conv.cpu().item<float>(), 2);
  EXPECT_EQ(observed.ssm.cpu().item<float>(), 20);
  EXPECT_TRUE(observed.warm.cpu().item<bool>());
  EXPECT_EQ(caches_.front().get_conv_cache()[1].cpu().item<float>(), 2);
  EXPECT_EQ(caches_.front().get_ssm_cache()[1].cpu().item<float>(), 20);
  EXPECT_EQ(caches_.front().get_conv_cache()[2].cpu().item<float>(), 4);
  EXPECT_EQ(caches_.front().get_ssm_cache()[2].cpu().item<float>(), 40);
  finish(pipeline_->submit(
      input(/*slot=*/2, /*query=*/1, /*cached=*/4, /*decode=*/true)));
  ASSERT_EQ(model_->observations.size(), 3);
  EXPECT_EQ(model_->observations.back().read_indices.cpu().item<int64_t>(), 2);
  EXPECT_EQ(model_->observations.back().ssm.cpu().item<float>(), 40);
  EXPECT_EQ(caches_.front().get_ssm_cache()[2].cpu().item<float>(), 50);
}

TEST_F(RecurrentPipelineTest,
       InvalidStateOperationsLeaveCacheAndPipelineUsable) {
  const torch::Tensor initial_conv =
      caches_.front().get_conv_cache().cpu().clone();
  const torch::Tensor initial_ssm =
      caches_.front().get_ssm_cache().cpu().clone();
  auto invalid = input(/*slot=*/2, /*query=*/1, /*cached=*/2);
  invalid.input_params.embedding.linear_state_read_ids = {8};
  EXPECT_FALSE(pipeline_->submit(invalid).status.ok());
  invalid.input_params.embedding.linear_state_read_ids = {0};
  EXPECT_FALSE(pipeline_->submit(invalid).status.ok());
  invalid.input_params.embedding.linear_state_read_ids = {-1};
  EXPECT_FALSE(pipeline_->submit(invalid).status.ok());
  invalid.input_params.embedding.linear_state_read_ids = {1, 2};
  EXPECT_FALSE(pipeline_->submit(invalid).status.ok());
  invalid.input_params.embedding.linear_state_read_ids.clear();
  invalid.input_params.embedding.linear_state_ids = {3, 2};
  EXPECT_FALSE(pipeline_->submit(invalid).status.ok());
  invalid.input_params.embedding.linear_state_ids = {2};
  auto decode = input(/*slot=*/2, /*query=*/1, /*cached=*/2, /*decode=*/true);
  // Cold checkpoint cannot back a direct Decode read.
  decode.input_params.embedding.linear_state_read_ids = {1};
  decode.input_params.linear_state_validity_mask = {0};
  EXPECT_FALSE(pipeline_->submit(decode).status.ok());
  auto padding = input(/*slot=*/0, /*query=*/1);
  EXPECT_FALSE(pipeline_->submit(padding).status.ok());
  EXPECT_TRUE(model_->observations.empty());
  EXPECT_TRUE(
      torch::equal(caches_.front().get_conv_cache().cpu(), initial_conv));
  EXPECT_TRUE(torch::equal(caches_.front().get_ssm_cache().cpu(), initial_ssm));
  finish(pipeline_->submit(input(/*slot=*/2, /*query=*/1)));
  EXPECT_EQ(caches_.front().get_ssm_cache()[2].cpu().item<float>(), 10);
}

TEST_F(RecurrentPipelineTest,
       GraphWarmupPreservesLiveStateAndPaddedDecodeRestoresOnlyActualRows) {
  pipeline_.reset();
  capacity_.max_graph_batch_size = 4;
  ASSERT_TRUE(
      TaskExecutionPipeline::create(
          state_thread_, *model_, *executor_, caches_, capacity_, pipeline_)
          .ok());
  const torch::Tensor initial_conv =
      caches_.front().get_conv_cache().cpu().clone();
  const torch::Tensor initial_ssm =
      caches_.front().get_ssm_cache().cpu().clone();
  auto warmup = input(/*slot=*/1, /*query=*/1, /*cached=*/1, /*decode=*/true);
  warmup.token_ids = torch::ones({4}, torch::kInt32);
  warmup.positions = torch::ones({4}, torch::kInt32);
  auto& params = warmup.input_params;
  params.meta.is_graph_warmup = true;
  params.meta.num_sequences = 4;
  params.meta.actual_num_sequences = 4;
  params.attention.host.q_seq_lens.assign(4, 1);
  params.attention.host.kv_seq_lens.assign(4, 2);
  params.attention.host.q_cu_seq_lens = {1, 2, 3, 4};
  params.attention.host.block_tables =
      torch::tensor({{0, 1}}, torch::kInt32).repeat({4, 1});
  params.attention.device.new_cache_slots = warmup.positions.clone();
  params.embedding.linear_state_ids = {1, 2, 3, 4};
  params.embedding.linear_state_indices = torch::arange(1, 5, torch::kInt32);
  params.linear_state_validity_mask.assign(4, 0);
  finish(pipeline_->submit(warmup));
  // Both Slots are captured and the admitted warmup Task then executes.
  ASSERT_EQ(model_->observations.size(), 3);
  for (const auto& observed : model_->observations) {
    EXPECT_TRUE(observed.enable_graph);
    EXPECT_EQ(observed.write_indices.cpu().count_nonzero().item<int64_t>(), 0);
    EXPECT_EQ(observed.read_indices.cpu().count_nonzero().item<int64_t>(), 0);
    EXPECT_EQ(observed.warm.cpu().count_nonzero().item<int64_t>(), 0);
    EXPECT_TRUE(torch::equal(observed.query_starts.cpu(),
                             torch::tensor({0, 1, 2, 3, 4}, torch::kInt32)));
  }
  EXPECT_TRUE(
      torch::equal(caches_.front().get_conv_cache().cpu(), initial_conv));
  EXPECT_TRUE(torch::equal(caches_.front().get_ssm_cache().cpu(), initial_ssm));
  finish(pipeline_->submit(input(/*slot=*/1, /*query=*/2)));
  auto decode = input(/*slot=*/2, /*query=*/1, /*cached=*/2, /*decode=*/true);
  decode.input_params.embedding.linear_state_read_ids = {1};
  decode.input_params.linear_state_validity_mask = {1};
  finish(pipeline_->submit(decode));
  ASSERT_EQ(model_->observations.size(), 5);
  const auto& observed = model_->observations.back();
  EXPECT_TRUE(observed.enable_graph);
  EXPECT_TRUE(torch::equal(observed.write_indices.cpu(),
                           torch::tensor({2, 0, 0, 0}, torch::kInt64)));
  EXPECT_TRUE(
      torch::equal(observed.warm.cpu(),
                   torch::tensor({true, false, false, false}, torch::kBool)));
  EXPECT_EQ(observed.conv.cpu()[0].item<float>(), 2);
  EXPECT_EQ(observed.ssm.cpu()[0].item<float>(), 20);
  EXPECT_EQ(caches_.front().get_conv_cache()[2].cpu().item<float>(), 3);
  EXPECT_EQ(caches_.front().get_ssm_cache()[2].cpu().item<float>(), 30);
  EXPECT_EQ(caches_.front().get_conv_cache()[0].cpu().item<float>(), 7);
  EXPECT_EQ(caches_.front().get_ssm_cache()[0].cpu().item<float>(), 11);
}

}  // namespace
}  // namespace xllm
