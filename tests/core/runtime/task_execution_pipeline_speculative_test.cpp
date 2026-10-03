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
#include <limits>
#include <memory>
#include <vector>

#include "core/framework/config/model_config.h"
#include "core/framework/parallel_state/process_group.h"
#include "core/runtime/executor_impl_factory.h"
#include "core/runtime/task_execution_pipeline.h"

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
    auto hidden = (block_draft ? positions : tokens).to(options_).view({-1, 1});
    // Two columns exercise block drafts whose captured context is wider than
    // either model's hidden state. MTP ignores these auxiliary rows.
    auto auxiliary = torch::cat({hidden, hidden + 100}, /*dim=*/1);
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
  std::vector<torch::Tensor> context_hidden;
  std::vector<torch::Tensor> context_positions;
  std::vector<torch::Tensor> context_slots;

 private:
  torch::TensorOptions options_;
};

class PreparedTestExecutor final : public ExecutorImpl {
 public:
  explicit PreparedTestExecutor(CausalLM* model) : model_(model) {}
  ForwardInput prepare_inputs(Batch& /*batch*/) override { return {}; }
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

  void create() {
    const Status status = TaskExecutionPipeline::create(
        state_thread_,
        {*target_, *target_executor_, target_cache_},
        {*draft_, *draft_executor_, draft_cache_},
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

  ForwardInput input(bool decode, int32_t token = 3, int32_t position = 2) {
    ForwardInput input;
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

  ForwardOutput execute(const ForwardInput& input) {
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

  ForwardInput dcp_input(bool decode,
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

  auto invalid = prefill;
  invalid.input_params.attention.host.new_cache_slots = {30, 31, 8, 9};
  EXPECT_FALSE(pipeline_->submit(invalid).status.ok());
  invalid = prefill;
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
  ForwardInput idle;
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

}  // namespace
}  // namespace xllm
