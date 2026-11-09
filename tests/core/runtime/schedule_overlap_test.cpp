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
#include <utility>
#include <vector>

#include "core/framework/config/eplb_config.h"
#include "core/framework/constrained/json_object_grammar.h"
#include "core/framework/sampling/sampler.h"
#include "core/framework/sampling/sampling_params.h"
#include "core/runtime/worker_impl.h"

namespace xllm {
namespace {

using namespace std::chrono_literals;

template <typename T>
class ScopedConfigValue final {
 public:
  ScopedConfigValue(T& value, T new_value) : value_(value), old_value_(value) {
    value_ = new_value;
  }

  ~ScopedConfigValue() { value_ = old_value_; }

 private:
  T& value_;
  T old_value_;
};

ForwardOutput token_output(int32_t token_id) {
  ForwardOutput output;
  output.sample_output.next_tokens = torch::tensor({token_id}, torch::kInt64);
  return output;
}

class OverlapTestWorker final : public WorkerImpl {
 public:
  enum class ContinuationMode : int8_t {
    CUSTOM_REBUILD,
    DEFAULT_REPLACEMENT,
    MODEL_MANAGED
  };

  explicit OverlapTestWorker(
      std::vector<std::optional<ForwardOutput>> outputs,
      bool driver = true,
      ContinuationMode continuation_mode = ContinuationMode::CUSTOM_REBUILD)
      : OverlapTestWorker(std::move(outputs),
                          ParallelArgs(/*rank=*/driver ? 0 : 1,
                                       /*world_size=*/2,
                                       /*dp_size=*/1,
                                       /*process_group=*/nullptr),
                          continuation_mode) {}

  OverlapTestWorker(
      std::vector<std::optional<ForwardOutput>> outputs,
      const ParallelArgs& parallel_args,
      ContinuationMode continuation_mode = ContinuationMode::CUSTOM_REBUILD,
      bool speculative_decode = false)
      : WorkerImpl(parallel_args,
                   torch::Device(torch::kPrivateUse1, /*index=*/0),
                   options(speculative_decode)),
        outputs_(std::move(outputs)),
        continuation_mode_(continuation_mode) {}

  ~OverlapTestWorker() override { drain_pending_execution(); }

  bool init_model(ModelContext& /*context*/) override { return false; }

  void prepare_work_before_execute(const LlmForwardInput& input,
                                   LlmForwardInput& prepared) override {
    prepare_input(input, prepared);
  }

  void prepare_work_before_execute(const VlmForwardInput& input,
                                   VlmForwardInput& prepared) override {
    prepare_input(input, prepared);
  }

  int32_t continuation_count() const { return continuation_count_; }
  const torch::Tensor& executed_tokens() const { return executed_tokens_; }
  const std::vector<int32_t>& executed_grammar_tokens() const {
    return executed_grammar_tokens_;
  }
  bool executed_grammar_initialized() const {
    return executed_grammar_initialized_;
  }
  void set_output_hook(std::function<void(ForwardOutput&, Stream&)> hook) {
    output_hook_ = std::move(hook);
  }
  void set_empty_output_hook(std::function<void()> hook) {
    empty_output_hook_ = std::move(hook);
  }
  const torch::Tensor& executed_filter_bitmask() const {
    return executed_filter_bitmask_;
  }
  const torch::Tensor& forward_prefix_tokens() const {
    return forward_prefix_tokens_;
  }
  const SamplingParameters& executed_sampling_params() const {
    return executed_sampling_params_;
  }
  void set_sampling_logits(torch::Tensor logits) {
    sampling_logits_ = std::move(logits);
  }

 protected:
  bool prepare_model_decode_continuation(
      LlmForwardInput& input,
      const ForwardOutput& prior_output) override {
    return prepare_continuation(input, prior_output);
  }

  bool prepare_model_decode_continuation(
      VlmForwardInput& input,
      const ForwardOutput& prior_output) override {
    return prepare_continuation(input, prior_output);
  }

  std::optional<ForwardOutput> step_for_schedule_overlap(
      const LlmForwardInput& input,
      const std::function<void()>& prepare_sampling = {}) override {
    return execute(input, prepare_sampling);
  }

  std::optional<ForwardOutput> step_for_schedule_overlap(
      const VlmForwardInput& input,
      const std::function<void()>& prepare_sampling = {}) override {
    return execute(input, prepare_sampling);
  }

 private:
  static runtime::Options options(bool speculative_decode) {
    runtime::Options options;
    options.enable_schedule_overlap(true);
    options.enable_speculative_decode(speculative_decode);
    return options;
  }

  template <typename Input>
  void prepare_input(const Input& input, Input& prepared) const {
    prepared = input.clone();
    if (continuation_mode_ == ContinuationMode::DEFAULT_REPLACEMENT) {
      prepared.token_ids = input.token_ids.to(device());
    }
  }

  template <typename Input>
  bool prepare_continuation(Input& input, const ForwardOutput& prior_output) {
    ++continuation_count_;
    if (continuation_mode_ == ContinuationMode::DEFAULT_REPLACEMENT) {
      return WorkerImpl::prepare_model_decode_continuation(input, prior_output);
    }
    if (continuation_mode_ == ContinuationMode::CUSTOM_REBUILD) {
      input.token_ids =
          prior_output.sample_output.next_tokens.to(torch::kInt32);
    }
    return true;
  }

  template <typename Input>
  std::optional<ForwardOutput> execute(
      const Input& input,
      const std::function<void()>& prepare_sampling) {
    executed_tokens_ = input.token_ids;
    executed_filter_bitmask_ = input.sampling_params.filter_bitmask();
    executed_grammar_initialized_ =
        !input.json_object_states.empty() &&
        input.json_object_states.front().initialized();
    executed_grammar_tokens_.clear();
    if (executed_grammar_initialized_) {
      executed_grammar_tokens_ =
          input.json_object_states.front().snapshot().token_ids;
    }
    forward_prefix_tokens_ = input.sampling_params.prefix_tokens();
    if (prepare_sampling) {
      prepare_sampling();
    }
    executed_sampling_params_ = input.sampling_params;
    CHECK_LT(step_index_, outputs_.size());
    auto output = std::move(outputs_[step_index_++]);
    if (output.has_value() && sampling_logits_.defined() &&
        input.sampling_params.num_sample_seqs() > 0) {
      auto stream_guard = compute_stream_->set_stream_guard();
      torch::Tensor logits = sampling_logits_.clone();
      output->sample_output = sample(logits, input.sampling_params);
    }
    if (output.has_value() && output_hook_) {
      output_hook_(*output, *compute_stream_);
    } else if (!output.has_value() && empty_output_hook_) {
      empty_output_hook_();
    }
    return output;
  }

  std::vector<std::optional<ForwardOutput>> outputs_;
  ContinuationMode continuation_mode_;
  size_t step_index_ = 0;
  int32_t continuation_count_ = 0;
  torch::Tensor executed_tokens_;
  torch::Tensor executed_filter_bitmask_;
  torch::Tensor forward_prefix_tokens_;
  SamplingParameters executed_sampling_params_;
  torch::Tensor sampling_logits_;
  std::vector<int32_t> executed_grammar_tokens_;
  bool executed_grammar_initialized_ = false;
  std::function<void(ForwardOutput&, Stream&)> output_hook_;
  std::function<void()> empty_output_hook_;
};

template <typename Input>
Input step_input(bool decode,
                 int32_t token_id = 7,
                 std::string request_id = "request") {
  Input input;
  input.token_ids = torch::tensor({token_id}, torch::kInt32);
  input.input_params.meta.batch_forward_type =
      decode ? BatchForwardType::DECODE : BatchForwardType::PREFILL;
  input.input_params.embedding.request_ids = {request_id};
  input.sample_sequence_ids = {request_id + "#0"};
  return input;
}

template <typename Input>
void expect_rejected_json_continuation(torch::Tensor prior_tokens) {
  JsonObjectGrammar grammar({"{", "}", "stop"}, /*stop_token_ids=*/{2});
  ForwardOutput prior_output;
  prior_output.sample_output.next_tokens = std::move(prior_tokens);
  OverlapTestWorker worker({std::move(prior_output), token_output(2)},
                           /*driver=*/true,
                           OverlapTestWorker::ContinuationMode::MODEL_MANAGED);
  auto first = worker.step_async(step_input<Input>(/*decode=*/false));
  ASSERT_TRUE(std::move(first).get().has_value());
  worker.get_last_step_result();

  auto input = step_input<Input>(/*decode=*/true, /*token_id=*/-1);
  input.json_object_states = {grammar.initial_state()};
  input.sample_prior_output_rows = {0};
  auto step = worker.step_async(input);
  const auto output = std::move(step).get();
  ASSERT_TRUE(output.has_value());
  EXPECT_FALSE(worker.executed_grammar_initialized());
  EXPECT_EQ(worker.executed_tokens().template item<int32_t>(), 0);
  ASSERT_EQ(output->json_object_errors.size(), 1U);
  EXPECT_EQ(output->json_object_errors.front().sample_sequence_id, "request#0");
  EXPECT_EQ(output->json_object_errors.front().message,
            "prior token violates json_object grammar");
  const ForwardOutput published = worker.get_last_step_result();
  ASSERT_EQ(published.json_object_errors.size(), 1U);
  EXPECT_EQ(published.json_object_errors.front().sample_sequence_id,
            "request#0");
  EXPECT_EQ(published.json_object_errors.front().message,
            output->json_object_errors.front().message);
  EXPECT_EQ(input.token_ids.template item<int32_t>(), -1);
  EXPECT_TRUE(input.json_object_states.front().initialized());
  EXPECT_TRUE(input.json_object_states.front().snapshot().token_ids.empty());
  EXPECT_TRUE(input.json_object_errors.empty());
}

template <typename Input>
class ScheduleOverlapTest : public ::testing::Test {};

using ForwardInputTypes = ::testing::Types<LlmForwardInput, VlmForwardInput>;
TYPED_TEST_SUITE(ScheduleOverlapTest, ForwardInputTypes);

TYPED_TEST(ScheduleOverlapTest, DecodeWithoutPriorTokensSkipsContinuation) {
  OverlapTestWorker worker({token_output(17)});
  auto step = worker.step_async(step_input<TypeParam>(/*decode=*/true));
  ASSERT_TRUE(std::move(step).get().has_value());
  EXPECT_EQ(worker.continuation_count(), 0);
  EXPECT_EQ(worker.executed_tokens().template item<int32_t>(), 7);
  worker.get_last_step_result();
}

TYPED_TEST(ScheduleOverlapTest, PrefillIgnoresPriorContinuation) {
  OverlapTestWorker worker({token_output(17), token_output(23)});
  auto first = worker.step_async(step_input<TypeParam>(/*decode=*/false));
  ASSERT_TRUE(std::move(first).get().has_value());
  worker.get_last_step_result();

  auto second = worker.step_async(step_input<TypeParam>(/*decode=*/false));
  ASSERT_TRUE(std::move(second).get().has_value());
  EXPECT_EQ(worker.continuation_count(), 0);
  EXPECT_EQ(worker.executed_tokens().template item<int32_t>(), 7);
  worker.get_last_step_result();
}

TYPED_TEST(ScheduleOverlapTest, ConsumedResultRemainsAvailableForContinuation) {
  OverlapTestWorker worker({token_output(17), token_output(23)});
  auto first = worker.step_async(step_input<TypeParam>(/*decode=*/false));
  ASSERT_TRUE(std::move(first).get().has_value());
  ASSERT_EQ(worker.get_last_step_result()
                .sample_output.next_tokens.template item<int64_t>(),
            17);

  auto second = worker.step_async(
      step_input<TypeParam>(/*decode=*/true, /*token_id=*/-1));
  ASSERT_TRUE(std::move(second).get().has_value());
  EXPECT_EQ(worker.continuation_count(), 1);
  EXPECT_EQ(worker.executed_tokens().template item<int32_t>(), 17);
  EXPECT_EQ(worker.get_last_step_result()
                .sample_output.next_tokens.template item<int64_t>(),
            23);
}

TYPED_TEST(ScheduleOverlapTest,
           DefaultContinuationReplacesIndexedPlaceholders) {
  const torch::Device device(torch::kPrivateUse1, /*index=*/0);
  ForwardOutput prior_output;
  prior_output.sample_output.next_tokens =
      torch::tensor({17, 23}, torch::kInt64).to(device);
  OverlapTestWorker worker(
      {std::move(prior_output), token_output(31)},
      /*driver=*/true,
      OverlapTestWorker::ContinuationMode::DEFAULT_REPLACEMENT);
  auto first = worker.step_async(step_input<TypeParam>(/*decode=*/false));
  ASSERT_TRUE(std::move(first).get().has_value());
  worker.get_last_step_result();

  auto input = step_input<TypeParam>(/*decode=*/true);
  input.token_ids = torch::tensor({-2, 9, -1}, torch::kInt32);
  auto second = worker.step_async(input);
  ASSERT_TRUE(std::move(second).get().has_value());
  EXPECT_EQ(worker.continuation_count(), 1);
  EXPECT_EQ(worker.executed_tokens().device(), device);
  EXPECT_TRUE(torch::equal(worker.executed_tokens().to(torch::kCPU),
                           torch::tensor({23, 9, 17}, torch::kInt32)));
  EXPECT_TRUE(
      torch::equal(input.token_ids, torch::tensor({-2, 9, -1}, torch::kInt32)));
  worker.get_last_step_result();
}

TYPED_TEST(ScheduleOverlapTest, ModelManagedContinuationPreservesModelInput) {
  JsonObjectGrammar grammar({"{", "}", "stop"}, /*stop_token_ids=*/{2});
  OverlapTestWorker worker({token_output(0), token_output(1)},
                           /*driver=*/true,
                           OverlapTestWorker::ContinuationMode::MODEL_MANAGED);
  auto first = worker.step_async(step_input<TypeParam>(/*decode=*/false));
  ASSERT_TRUE(std::move(first).get().has_value());
  worker.get_last_step_result();

  auto input = step_input<TypeParam>(/*decode=*/true, /*token_id=*/-1);
  input.json_object_states = {grammar.initial_state()};
  input.sample_prior_output_rows = {0};
  auto second = worker.step_async(input);
  const auto output = std::move(second).get();
  ASSERT_TRUE(output.has_value());
  EXPECT_TRUE(output->json_object_errors.empty());
  EXPECT_EQ(worker.continuation_count(), 1);
  EXPECT_EQ(worker.executed_tokens().template item<int32_t>(), -1);
  EXPECT_EQ(worker.executed_grammar_tokens(), std::vector<int32_t>({0}));
  EXPECT_EQ(input.token_ids.template item<int32_t>(), -1);
  EXPECT_TRUE(input.json_object_states.front().snapshot().token_ids.empty());
  worker.get_last_step_result();
}

TYPED_TEST(ScheduleOverlapTest,
           SpeculativeContinuationStopsAtSentinelAndRefreshesMasks) {
  JsonObjectGrammar grammar({"{", "}", "stop"}, /*stop_token_ids=*/{2});
  ForwardOutput prior_output;
  prior_output.sample_output.next_tokens =
      torch::tensor({{0, 1, -1, 0}}, torch::kInt64);
  OverlapTestWorker worker({std::move(prior_output), token_output(2)},
                           /*driver=*/true,
                           OverlapTestWorker::ContinuationMode::MODEL_MANAGED);
  auto first = worker.step_async(step_input<TypeParam>(/*decode=*/false));
  ASSERT_TRUE(std::move(first).get().has_value());
  worker.get_last_step_result();

  auto input = step_input<TypeParam>(/*decode=*/true, /*token_id=*/-1);
  input.json_object_states = {grammar.initial_state()};
  input.sample_prior_output_rows = {0};
  input.sampling_params.filter_bitmask() = torch::zeros({1, 1}, torch::kInt32);
  auto step = worker.step_async(input);
  const auto output = std::move(step).get();
  ASSERT_TRUE(output.has_value());
  EXPECT_TRUE(output->json_object_errors.empty());
  EXPECT_EQ(worker.executed_grammar_tokens(), std::vector<int32_t>({0, 1}));
  ASSERT_TRUE(worker.executed_filter_bitmask().defined());
  EXPECT_EQ(worker.executed_filter_bitmask().sizes(),
            torch::IntArrayRef({1, 1}));
  EXPECT_EQ(worker.executed_filter_bitmask().scalar_type(), torch::kInt32);
  EXPECT_EQ(
      worker.executed_filter_bitmask().to(torch::kCPU).template item<int32_t>(),
      1 << 2);
  EXPECT_TRUE(input.json_object_states.front().snapshot().token_ids.empty());
  EXPECT_EQ(input.sampling_params.filter_bitmask().template item<int32_t>(), 0);
  worker.get_last_step_result();
}

TYPED_TEST(ScheduleOverlapTest,
           InvalidPriorTokenRejectsConstrainedContinuation) {
  expect_rejected_json_continuation<TypeParam>(
      torch::tensor({1}, torch::kInt64));
}

TYPED_TEST(ScheduleOverlapTest,
           InvalidSpeculativeTokenRejectsConstrainedContinuation) {
  expect_rejected_json_continuation<TypeParam>(
      torch::tensor({{0, 0, -1}}, torch::kInt64));
}

TYPED_TEST(ScheduleOverlapTest, ExistingErrorsSanitizeOnlyTheirPlaceholders) {
  ForwardOutput prior_output;
  prior_output.sample_output.next_tokens =
      torch::tensor({17, 23}, torch::kInt64);
  OverlapTestWorker worker({std::move(prior_output), token_output(31)},
                           /*driver=*/true,
                           OverlapTestWorker::ContinuationMode::MODEL_MANAGED);
  auto initial_input = step_input<TypeParam>(/*decode=*/false);
  initial_input.input_params.embedding.request_ids = {"request", "other"};
  initial_input.sample_sequence_ids = {"request#0", "other#0"};
  auto first = worker.step_async(initial_input);
  ASSERT_TRUE(std::move(first).get().has_value());
  worker.get_last_step_result();

  auto input = step_input<TypeParam>(/*decode=*/true);
  input.token_ids = torch::tensor({-1, -2, 9}, torch::kInt32);
  input.input_params.embedding.request_ids = {"request", "other"};
  input.sample_sequence_ids = {"request#0", "other#0"};
  input.sample_prior_output_rows = {0, 1};
  input.json_object_errors.emplace_back(
      JsonObjectOutputError{"request#0", "existing error"});
  auto step = worker.step_async(input);
  const auto output = std::move(step).get();
  ASSERT_TRUE(output.has_value());
  EXPECT_TRUE(torch::equal(worker.executed_tokens(),
                           torch::tensor({0, -2, 9}, torch::kInt32)));
  ASSERT_EQ(output->json_object_errors.size(), 1U);
  EXPECT_EQ(output->json_object_errors.front().sample_sequence_id, "request#0");
  EXPECT_EQ(output->json_object_errors.front().message, "existing error");
  EXPECT_TRUE(
      torch::equal(input.token_ids, torch::tensor({-1, -2, 9}, torch::kInt32)));
  ASSERT_EQ(input.json_object_errors.size(), 1U);
  EXPECT_EQ(input.json_object_errors.front().message, "existing error");
  worker.get_last_step_result();
}

TYPED_TEST(ScheduleOverlapTest, EmptyStepClearsPriorContinuation) {
  OverlapTestWorker worker({token_output(17), std::nullopt, token_output(23)});
  auto first = worker.step_async(step_input<TypeParam>(/*decode=*/false));
  ASSERT_TRUE(std::move(first).get().has_value());
  worker.get_last_step_result();

  auto empty = step_input<TypeParam>(/*decode=*/true);
  empty.token_ids = torch::empty({0}, torch::kInt32);
  auto idle = worker.step_async(empty);
  EXPECT_FALSE(std::move(idle).get().has_value());
  EXPECT_FALSE(
      worker.get_last_step_result().sample_output.next_tokens.defined());

  auto next = worker.step_async(
      step_input<TypeParam>(/*decode=*/true, /*token_id=*/-1));
  ASSERT_TRUE(std::move(next).get().has_value());
  EXPECT_EQ(worker.continuation_count(), 0);
  EXPECT_EQ(worker.executed_tokens().template item<int32_t>(), -1);
  worker.get_last_step_result();
}

TYPED_TEST(ScheduleOverlapTest, AddsWarmupAndJsonErrorsToBothResultPaths) {
  ForwardOutput produced = token_output(17);
  produced.json_object_errors.emplace_back(
      JsonObjectOutputError{"model#0", "model error"});
  OverlapTestWorker worker({std::move(produced)});
  auto input = step_input<TypeParam>(/*decode=*/false);
  input.input_params.meta.is_graph_warmup = true;
  input.json_object_errors.emplace_back(
      JsonObjectOutputError{"request#0", "input error"});

  auto step = worker.step_async(input);
  const auto returned = std::move(step).get();
  ASSERT_TRUE(returned.has_value());
  EXPECT_TRUE(returned->is_graph_warmup);
  ASSERT_EQ(returned->json_object_errors.size(), 2U);
  EXPECT_EQ(returned->json_object_errors[0].message, "model error");
  EXPECT_EQ(returned->json_object_errors[1].message, "input error");

  const ForwardOutput published = worker.get_last_step_result();
  EXPECT_TRUE(published.is_graph_warmup);
  ASSERT_EQ(published.json_object_errors.size(), 2U);
  EXPECT_EQ(published.json_object_errors[0].message, "model error");
  EXPECT_EQ(published.json_object_errors[1].message, "input error");
}

TYPED_TEST(ScheduleOverlapTest, ConstrainedDecodeRejectsUnrelatedPriorResult) {
  JsonObjectGrammar grammar({"{", "}", "stop"}, /*stop_token_ids=*/{2});
  OverlapTestWorker worker({token_output(0), token_output(1)});
  auto first = worker.step_async(step_input<TypeParam>(
      /*decode=*/false, /*token_id=*/7, /*request_id=*/"previous"));
  ASSERT_TRUE(std::move(first).get().has_value());
  worker.get_last_step_result();

  auto input = step_input<TypeParam>(
      /*decode=*/true, /*token_id=*/-1, /*request_id=*/"current");
  input.json_object_states = {grammar.initial_state()};
  input.sample_prior_output_rows = {0};
  auto step = worker.step_async(input);
  const auto output = std::move(step).get();
  ASSERT_TRUE(output.has_value());
  EXPECT_EQ(worker.continuation_count(), 0);
  EXPECT_EQ(worker.executed_tokens().template item<int32_t>(), 0);
  EXPECT_FALSE(worker.executed_grammar_initialized());
  ASSERT_EQ(output->json_object_errors.size(), 1U);
  EXPECT_EQ(output->json_object_errors.front().sample_sequence_id, "current#0");
  EXPECT_NE(output->json_object_errors.front().message.find("missing"),
            std::string::npos);
  ASSERT_EQ(worker.get_last_step_result().json_object_errors.size(), 1U);
}

TYPED_TEST(ScheduleOverlapTest, ConstrainedContinuationAdvancesGrammar) {
  JsonObjectGrammar grammar({"{", "}", "stop"}, /*stop_token_ids=*/{2});
  OverlapTestWorker worker({token_output(0), token_output(1)});
  auto first = worker.step_async(step_input<TypeParam>(/*decode=*/false));
  ASSERT_TRUE(std::move(first).get().has_value());
  worker.get_last_step_result();

  auto input = step_input<TypeParam>(/*decode=*/true, /*token_id=*/-1);
  input.json_object_states = {grammar.initial_state()};
  input.sample_prior_output_rows = {0};
  auto step = worker.step_async(input);
  const auto output = std::move(step).get();
  ASSERT_TRUE(output.has_value());
  EXPECT_TRUE(output->json_object_errors.empty());
  EXPECT_EQ(worker.continuation_count(), 1);
  EXPECT_EQ(worker.executed_grammar_tokens(), std::vector<int32_t>({0}));
  EXPECT_TRUE(input.json_object_states.front().snapshot().token_ids.empty());
  worker.get_last_step_result();
}

TYPED_TEST(ScheduleOverlapTest, NewConstrainedSequenceStartsWithoutPriorRow) {
  JsonObjectGrammar grammar({"{", "}", "stop"}, /*stop_token_ids=*/{2});
  OverlapTestWorker worker({token_output(0), token_output(1)},
                           /*driver=*/true,
                           OverlapTestWorker::ContinuationMode::MODEL_MANAGED);
  auto first = worker.step_async(step_input<TypeParam>(/*decode=*/false));
  ASSERT_TRUE(std::move(first).get().has_value());
  worker.get_last_step_result();

  auto input = step_input<TypeParam>(/*decode=*/true);
  input.sample_sequence_ids = {"request#new"};
  input.sample_prior_output_rows = {-1};
  input.json_object_states = {grammar.initial_state()};
  auto step = worker.step_async(input);
  const auto output = std::move(step).get();
  ASSERT_TRUE(output.has_value());
  EXPECT_TRUE(output->json_object_errors.empty());
  EXPECT_TRUE(worker.executed_grammar_initialized());
  EXPECT_TRUE(worker.executed_grammar_tokens().empty());
  ASSERT_TRUE(worker.executed_filter_bitmask().defined());
  EXPECT_TRUE(torch::equal(worker.executed_filter_bitmask().to(torch::kCPU),
                           torch::tensor({{1 << 0}}, torch::kInt32)));
  EXPECT_EQ(worker.continuation_count(), 1);
  worker.get_last_step_result();
}

TYPED_TEST(ScheduleOverlapTest, MixedConstrainedSequencesAdvancePriorRow) {
  JsonObjectGrammar grammar({"{", "}", "stop"}, /*stop_token_ids=*/{2});
  ForwardOutput next_output;
  next_output.sample_output.next_tokens = torch::tensor({1, 0}, torch::kInt64);
  OverlapTestWorker worker({token_output(0), std::move(next_output)},
                           /*driver=*/true,
                           OverlapTestWorker::ContinuationMode::MODEL_MANAGED);
  auto first = worker.step_async(step_input<TypeParam>(/*decode=*/false));
  ASSERT_TRUE(std::move(first).get().has_value());
  worker.get_last_step_result();

  auto input = step_input<TypeParam>(/*decode=*/true);
  input.token_ids = torch::tensor({-1, 7}, torch::kInt32);
  input.input_params.embedding.request_ids = {"request", "new"};
  input.sample_sequence_ids = {"request#0", "new#0"};
  input.sample_prior_output_rows = {0, -1};
  input.json_object_states = {grammar.initial_state(), grammar.initial_state()};
  auto step = worker.step_async(input);
  const auto output = std::move(step).get();
  ASSERT_TRUE(output.has_value());
  EXPECT_TRUE(output->json_object_errors.empty());
  EXPECT_EQ(worker.executed_grammar_tokens(), std::vector<int32_t>({0}));
  ASSERT_TRUE(worker.executed_filter_bitmask().defined());
  EXPECT_TRUE(torch::equal(worker.executed_filter_bitmask().to(torch::kCPU),
                           torch::tensor({{1 << 1}, {1 << 0}}, torch::kInt32)));
  EXPECT_EQ(worker.continuation_count(), 1);
  worker.get_last_step_result();
}

TYPED_TEST(ScheduleOverlapTest,
           SamplingHistoryFollowsReorderedSequenceRowsAfterForward) {
  for (const torch::Device& device :
       {torch::Device(torch::kCPU),
        torch::Device(torch::kPrivateUse1, /*index=*/0)}) {
    SCOPED_TRACE(device.str());
    ForwardOutput prior_output;
    prior_output.sample_output.next_tokens =
        torch::tensor({{2, -1}, {1, 1}}, torch::kInt64).to(device);
    OverlapTestWorker worker(
        {std::move(prior_output), token_output(7)},
        /*driver=*/true,
        OverlapTestWorker::ContinuationMode::MODEL_MANAGED);
    if (!device.is_cpu()) {
      worker.set_output_hook([](ForwardOutput& output, Stream& stream) {
        auto guard = stream.set_stream_guard();
        output.ready_event = stream.record_event();
        CHECK(output.ready_event != nullptr);
      });
    }
    auto initial_input = step_input<TypeParam>(/*decode=*/false);
    initial_input.input_params.embedding.request_ids = {"first", "second"};
    initial_input.sample_sequence_ids = {"first#0", "second#0"};
    auto first = worker.step_async(initial_input);
    ASSERT_TRUE(std::move(first).get().has_value());
    worker.get_last_step_result();

    auto input = step_input<TypeParam>(/*decode=*/true);
    input.token_ids = torch::tensor({1, 2, 7}, torch::kInt32).to(device);
    input.input_params.embedding.request_ids = {"second", "first", "new"};
    input.sample_sequence_ids = {"second#0", "first#0", "new#0"};
    input.sample_prior_output_rows = {1, 0, -1};
    auto& sampling = input.sampling_params;
    sampling.selected_token_idxes() =
        torch::arange(3, torch::kInt64).to(device);
    sampling.bad_words() =
        torch::tensor({{{1, 3}}, {{1, 3}}, {{1, 3}}}, torch::kInt64).to(device);
    sampling.bad_word_suffix() =
        torch::full({3, 1}, -1, torch::kInt64).to(device);
    sampling.seeds() = torch::tensor({7, 8, 9}, torch::kInt64).to(device);
    sampling.seed_offsets() =
        torch::tensor({10, 20, 30}, torch::kInt64).to(device);
    worker.set_sampling_logits(torch::tensor({{4.0F, 0.0F, 0.0F, 10.0F},
                                              {4.0F, 0.0F, 0.0F, 10.0F},
                                              {4.0F, 0.0F, 0.0F, 10.0F}})
                                   .to(device));

    auto second = worker.step_async(input);
    const auto output = std::move(second).get();
    ASSERT_TRUE(output.has_value());
    if (!device.is_cpu()) {
      ASSERT_NE(output->ready_event, nullptr);
      ASSERT_FALSE(output->retained_inputs.empty());
      const auto retained_rows = std::static_pointer_cast<const torch::Tensor>(
          output->retained_inputs.front());
      ASSERT_TRUE(retained_rows->device().is_cpu());
      EXPECT_EQ(retained_rows->scalar_type(), torch::kInt64);
      EXPECT_TRUE(retained_rows->is_pinned());
      EXPECT_TRUE(torch::equal(*retained_rows,
                               torch::tensor({1, 0, -1}, torch::kInt64)));
      ASSERT_TRUE(output->ready_event->synchronize());
    }
    EXPECT_FALSE(worker.forward_prefix_tokens().defined());
    const SamplingParameters& executed = worker.executed_sampling_params();
    EXPECT_TRUE(torch::equal(
        executed.prefix_tokens().to(torch::kCPU),
        torch::tensor({{1, 1}, {2, -1}, {-1, -1}}, torch::kInt64)));
    EXPECT_TRUE(torch::equal(executed.bad_word_suffix().to(torch::kCPU),
                             torch::tensor({{1}, {2}, {-1}}, torch::kInt64)));
    EXPECT_TRUE(torch::equal(executed.seed_offsets().to(torch::kCPU),
                             torch::tensor({12, 21, 30}, torch::kInt64)));
    // Only the reordered second sequence ends in the forbidden-word prefix.
    EXPECT_TRUE(torch::equal(output->sample_output.next_tokens.to(torch::kCPU),
                             torch::tensor({0, 3, 3}, torch::kInt64)));
    EXPECT_FALSE(input.sampling_params.prefix_tokens().defined());
    EXPECT_TRUE(
        torch::equal(input.sampling_params.bad_word_suffix().to(torch::kCPU),
                     torch::full({3, 1}, -1, torch::kInt64)));
    EXPECT_TRUE(
        torch::equal(input.sampling_params.seed_offsets().to(torch::kCPU),
                     torch::tensor({10, 20, 30}, torch::kInt64)));
    EXPECT_TRUE(torch::equal(
        worker.get_last_step_result().sample_output.next_tokens.to(torch::kCPU),
        output->sample_output.next_tokens.to(torch::kCPU)));
  }
}

TYPED_TEST(ScheduleOverlapTest, NonDriverContinuesWithoutHostConsumption) {
  OverlapTestWorker worker({token_output(17), token_output(23)},
                           /*driver=*/false);
  auto first = worker.step_async(step_input<TypeParam>(/*decode=*/false))
                   .toUnsafeFuture();
  first.wait(2s);
  const bool first_ready = first.isReady();
  if (!first_ready) {
    worker.begin_shutdown();
  }
  ASSERT_TRUE(first_ready);
  ASSERT_TRUE(std::move(first).get().has_value());

  auto second =
      worker.step_async(step_input<TypeParam>(/*decode=*/true, /*token_id=*/-1))
          .toUnsafeFuture();
  second.wait(2s);
  const bool second_ready = second.isReady();
  if (!second_ready) {
    worker.begin_shutdown();
  }
  EXPECT_TRUE(second_ready);
  ASSERT_TRUE(std::move(second).get().has_value());
  EXPECT_EQ(worker.continuation_count(), 1);
  EXPECT_EQ(worker.executed_tokens().template item<int32_t>(), 17);
}

TYPED_TEST(ScheduleOverlapTest,
           EplbNonDriverPublishesTokenlessMetadataWithFixedPolicy) {
  ScopedConfigValue<bool> eplb(EPLBConfig::get_instance().enable_eplb(), true);
  ForwardOutput output;
  output.expert_load_data = torch::tensor({3, 5}, torch::kInt64);
  output.prepared_token = 11;
  OverlapTestWorker worker({std::move(output)}, /*driver=*/false);
  ScopedConfigValue<bool> changed_eplb(EPLBConfig::get_instance().enable_eplb(),
                                       false);
  auto step = worker.step_async(step_input<TypeParam>(/*decode=*/false))
                  .toUnsafeFuture();
  step.wait(2s);
  const bool step_ready = step.isReady();
  if (!step_ready) {
    worker.begin_shutdown();
  }
  ASSERT_TRUE(step_ready);
  ASSERT_TRUE(std::move(step).get().has_value());

  auto publication = std::async(std::launch::async, [&worker]() {
    return worker.get_last_step_result();
  });
  const std::future_status publication_status = publication.wait_for(2s);
  if (publication_status != std::future_status::ready) {
    worker.begin_shutdown();
  }
  ASSERT_EQ(publication_status, std::future_status::ready);
  const ForwardOutput published = publication.get();
  EXPECT_FALSE(published.sample_output.next_tokens.defined());
  ASSERT_TRUE(published.expert_load_data.defined());
  EXPECT_TRUE(torch::equal(published.expert_load_data,
                           torch::tensor({3, 5}, torch::kInt64)));
  EXPECT_EQ(published.prepared_token, 11);
}

TYPED_TEST(ScheduleOverlapTest, DpDriverPublishesWithoutEplb) {
  ScopedConfigValue<bool> eplb(EPLBConfig::get_instance().enable_eplb(), false);
  const ParallelArgs parallel_args(/*rank=*/2,
                                   /*world_size=*/4,
                                   /*dp_size=*/2,
                                   /*process_group=*/nullptr);
  OverlapTestWorker worker({token_output(17)}, parallel_args);
  auto step = worker.step_async(step_input<TypeParam>(/*decode=*/false))
                  .toUnsafeFuture();
  step.wait(2s);
  const bool step_ready = step.isReady();
  if (!step_ready) {
    worker.begin_shutdown();
  }
  ASSERT_TRUE(step_ready);
  ASSERT_TRUE(std::move(step).get().has_value());

  auto publication = std::async(std::launch::async, [&worker]() {
    return worker.get_last_step_result();
  });
  const std::future_status publication_status = publication.wait_for(2s);
  if (publication_status != std::future_status::ready) {
    worker.begin_shutdown();
  }
  ASSERT_EQ(publication_status, std::future_status::ready);
  const ForwardOutput published = publication.get();
  ASSERT_TRUE(published.sample_output.next_tokens.defined());
  EXPECT_EQ(published.sample_output.next_tokens.template item<int64_t>(), 17);
}

TYPED_TEST(ScheduleOverlapTest,
           DriverDefersTokenlessInputsAfterFilteredPublication) {
  ScopedConfigValue<bool> eplb(EPLBConfig::get_instance().enable_eplb(), false);
  std::promise<void> callback_entered;
  auto entered = callback_entered.get_future();
  std::promise<void> release_callback;
  auto callback_release = release_callback.get_future().share();
  std::atomic<bool> device_finished{false};
  std::atomic<bool> owner_released{false};
  std::atomic<bool> owner_released_early{false};
  std::function<void()> callback = [&]() {
    callback_entered.set_value();
    callback_release.wait_for(10s);
    device_finished.store(true);
  };
  StreamEventPtr completion;
  OverlapTestWorker worker({ForwardOutput{}, std::nullopt, token_output(31)});
  worker.set_output_hook([&](ForwardOutput& output, Stream& stream) {
    auto guard = stream.set_stream_guard();
    CHECK_EQ(
        aclrtLaunchHostFunc(
            stream.get_stream()->stream(),
            [](void* data) { (*static_cast<std::function<void()>*>(data))(); },
            &callback),
        ACL_SUCCESS);
    output.ready_event = stream.record_event();
    CHECK(output.ready_event != nullptr);
    completion = output.ready_event;
    output.retained_inputs.emplace_back(
        std::shared_ptr<int32_t>(new int32_t(42), [&](int32_t* value) {
          owner_released_early.store(!device_finished.load());
          owner_released.store(true);
          delete value;
        }));
  });
  auto first = worker.step_async(step_input<TypeParam>(/*decode=*/false))
                   .toUnsafeFuture()
                   .thenValue([](std::optional<ForwardOutput> output) {
                     return output.has_value();
                   });
  first.wait(2s);
  const bool first_ready_without_completion = first.isReady();
  const std::future_status callback_status = entered.wait_for(2s);
  auto publication = std::async(std::launch::async, [&worker]() {
    return worker.get_last_step_result();
  });
  const std::future_status publication_status = publication.wait_for(2s);
  if (publication_status != std::future_status::ready) {
    worker.begin_shutdown();
  }
  const ForwardOutput published = publication.get();
  auto second = worker.step_async(step_input<TypeParam>(/*decode=*/false))
                    .toUnsafeFuture()
                    .thenValue([](std::optional<ForwardOutput> output) {
                      return output.has_value();
                    });
  second.wait(2s);
  const bool cleared_before_completion = second.isReady();
  const bool released_before_completion = owner_released.load();

  // Unblock device work before assertions so failures can drain the worker.
  release_callback.set_value();
  first.wait(2s);
  second.wait(2s);
  EXPECT_TRUE(first_ready_without_completion);
  EXPECT_EQ(callback_status, std::future_status::ready);
  EXPECT_EQ(publication_status, std::future_status::ready);
  EXPECT_FALSE(published.sample_output.next_tokens.defined());
  EXPECT_EQ(published.ready_event, nullptr);
  EXPECT_TRUE(published.retained_inputs.empty());
  EXPECT_TRUE(cleared_before_completion);
  EXPECT_FALSE(released_before_completion);
  ASSERT_TRUE(first.isReady());
  ASSERT_TRUE(second.isReady());
  EXPECT_TRUE(std::move(first).get());
  EXPECT_FALSE(std::move(second).get());
  ASSERT_NE(completion, nullptr);
  ASSERT_TRUE(completion->synchronize());
  (void)worker.get_last_step_result();
  worker.set_output_hook({});
  const auto next =
      worker.step_async(step_input<TypeParam>(/*decode=*/false)).get();
  EXPECT_TRUE(next.has_value());
  EXPECT_TRUE(owner_released.load());
  EXPECT_FALSE(owner_released_early.load());
}

TYPED_TEST(ScheduleOverlapTest,
           DpSpeculativeNonDriverDefersInputsWithoutBlockingNextForward) {
  ScopedConfigValue<bool> eplb(EPLBConfig::get_instance().enable_eplb(), false);
  std::promise<void> callback_entered;
  auto entered = callback_entered.get_future();
  std::promise<void> release_callback;
  auto callback_release = release_callback.get_future().share();
  std::atomic<bool> device_finished{false};
  std::atomic<bool> owner_released{false};
  std::atomic<bool> owner_released_early{false};
  std::atomic<bool> returned_after_device_completion{false};
  StreamEventPtr prior_completion;
  std::function<void()> callback = [&]() {
    callback_entered.set_value();
    callback_release.wait_for(10s);
    device_finished.store(true);
  };
  std::promise<void> second_output_produced;
  auto produced = second_output_produced.get_future();
  const ParallelArgs parallel_args(/*rank=*/1,
                                   /*world_size=*/4,
                                   /*dp_size=*/2,
                                   /*process_group=*/nullptr);
  int32_t output_index = 0;
  OverlapTestWorker worker(
      {token_output(17), token_output(23), token_output(31)},
      parallel_args,
      OverlapTestWorker::ContinuationMode::CUSTOM_REBUILD,
      /*speculative_decode=*/true);
  worker.set_output_hook([&](ForwardOutput& output, Stream& stream) {
    if (output_index++ > 0) {
      second_output_produced.set_value();
      return;
    }
    auto guard = stream.set_stream_guard();
    CHECK_EQ(
        aclrtLaunchHostFunc(
            stream.get_stream()->stream(),
            [](void* data) { (*static_cast<std::function<void()>*>(data))(); },
            &callback),
        ACL_SUCCESS);
    output.ready_event = stream.record_event();
    CHECK(output.ready_event != nullptr);
    prior_completion = output.ready_event;
    output.retained_inputs.emplace_back(
        std::shared_ptr<int32_t>(new int32_t(42), [&](int32_t* value) {
          owner_released_early.store(!device_finished.load());
          owner_released.store(true);
          delete value;
        }));
  });
  auto first = worker.step_async(step_input<TypeParam>(/*decode=*/false))
                   .toUnsafeFuture()
                   .thenValue([](std::optional<ForwardOutput> output) {
                     return output.has_value();
                   });
  first.wait(2s);
  const bool first_ready_without_device_completion = first.isReady();
  const std::future_status callback_status = entered.wait_for(2s);
  auto second =
      worker.step_async(step_input<TypeParam>(/*decode=*/false))
          .toUnsafeFuture()
          .thenValue([&](std::optional<ForwardOutput> output) {
            returned_after_device_completion.store(device_finished.load());
            return output.has_value();
          });
  const std::future_status produced_status = produced.wait_for(2s);
  second.wait(2s);
  const bool returned_early = second.isReady();
  const bool released_before_device_completion = owner_released.load();

  // Unblock device work before assertions so failures can drain the worker.
  release_callback.set_value();
  first.wait(2s);
  second.wait(2s);
  const bool first_ready = first.isReady();
  const bool second_ready = second.isReady();
  if (!first_ready || !second_ready) {
    worker.begin_shutdown();
  }
  EXPECT_TRUE(first_ready_without_device_completion);
  EXPECT_EQ(callback_status, std::future_status::ready);
  EXPECT_EQ(produced_status, std::future_status::ready);
  EXPECT_TRUE(returned_early);
  EXPECT_FALSE(released_before_device_completion);
  ASSERT_TRUE(first_ready);
  ASSERT_TRUE(second_ready);
  EXPECT_TRUE(std::move(first).get());
  EXPECT_TRUE(std::move(second).get());
  EXPECT_FALSE(returned_after_device_completion.load());
  ASSERT_NE(prior_completion, nullptr);
  ASSERT_TRUE(prior_completion->synchronize());
  worker.set_output_hook({});
  const auto next =
      worker.step_async(step_input<TypeParam>(/*decode=*/false)).get();
  EXPECT_TRUE(next.has_value());
  EXPECT_TRUE(owner_released.load());
  EXPECT_FALSE(owner_released_early.load());
}

TYPED_TEST(ScheduleOverlapTest,
           CacheOnlyNonDriverDefersTokenlessInputsAfterClearingOutput) {
  ScopedConfigValue<bool> eplb(EPLBConfig::get_instance().enable_eplb(), false);
  for (const ParallelArgs& parallel_args :
       {ParallelArgs(/*rank=*/1,
                     /*world_size=*/2,
                     /*dp_size=*/1,
                     /*process_group=*/nullptr),
        ParallelArgs(/*rank=*/1,
                     /*world_size=*/2,
                     /*dp_size=*/1,
                     /*cp_size=*/2,
                     /*process_group=*/nullptr,
                     /*ep_size=*/1),
        ParallelArgs(/*rank=*/1,
                     /*world_size=*/4,
                     /*dp_size=*/2,
                     /*process_group=*/nullptr)}) {
    SCOPED_TRACE("dp_size=" + std::to_string(parallel_args.dp_size()) +
                 ", cp_size=" + std::to_string(parallel_args.cp_size()));
    std::promise<void> callback_entered;
    auto entered = callback_entered.get_future();
    std::promise<void> release_callback;
    auto callback_release = release_callback.get_future().share();
    std::atomic<bool> device_finished{false};
    std::atomic<bool> owner_released{false};
    std::atomic<bool> owner_released_early{false};
    std::function<void()> callback = [&]() {
      callback_entered.set_value();
      callback_release.wait_for(10s);
      device_finished.store(true);
    };
    StreamEventPtr completion;
    OverlapTestWorker worker(
        {ForwardOutput{}, std::nullopt, token_output(31)},
        parallel_args,
        OverlapTestWorker::ContinuationMode::CUSTOM_REBUILD,
        /*speculative_decode=*/parallel_args.dp_size() > 1);
    worker.set_output_hook([&](ForwardOutput& output, Stream& stream) {
      auto guard = stream.set_stream_guard();
      CHECK_EQ(aclrtLaunchHostFunc(
                   stream.get_stream()->stream(),
                   [](void* data) {
                     (*static_cast<std::function<void()>*>(data))();
                   },
                   &callback),
               ACL_SUCCESS);
      output.ready_event = stream.record_event();
      CHECK(output.ready_event != nullptr);
      completion = output.ready_event;
      output.retained_inputs.emplace_back(
          std::shared_ptr<int32_t>(new int32_t(42), [&](int32_t* value) {
            owner_released_early.store(!device_finished.load());
            owner_released.store(true);
            delete value;
          }));
    });
    auto first = worker.step_async(step_input<TypeParam>(/*decode=*/false))
                     .toUnsafeFuture()
                     .thenValue([](std::optional<ForwardOutput> output) {
                       return output.has_value();
                     });
    first.wait(2s);
    const bool first_ready_without_completion = first.isReady();
    const std::future_status callback_status = entered.wait_for(2s);
    auto second = worker.step_async(step_input<TypeParam>(/*decode=*/false))
                      .toUnsafeFuture()
                      .thenValue([](std::optional<ForwardOutput> output) {
                        return output.has_value();
                      });
    second.wait(2s);
    const bool cleared_before_completion = second.isReady();
    const bool released_before_completion = owner_released.load();
    release_callback.set_value();
    first.wait(2s);
    second.wait(2s);
    EXPECT_TRUE(first_ready_without_completion);
    EXPECT_EQ(callback_status, std::future_status::ready);
    EXPECT_TRUE(cleared_before_completion);
    EXPECT_FALSE(released_before_completion);
    ASSERT_TRUE(first.isReady());
    ASSERT_TRUE(second.isReady());
    EXPECT_TRUE(std::move(first).get());
    EXPECT_FALSE(std::move(second).get());
    ASSERT_NE(completion, nullptr);
    ASSERT_TRUE(completion->synchronize());
    worker.set_output_hook({});
    const auto next =
        worker.step_async(step_input<TypeParam>(/*decode=*/false)).get();
    EXPECT_TRUE(next.has_value());
    EXPECT_TRUE(owner_released.load());
    EXPECT_FALSE(owner_released_early.load());
  }
}

TYPED_TEST(ScheduleOverlapTest,
           NonDriverShutdownWaitsBeforeReturningRetainedInputs) {
  std::promise<void> callback_entered;
  auto entered = callback_entered.get_future();
  std::promise<void> release_callback;
  auto callback_release = release_callback.get_future().share();
  std::atomic<bool> device_finished{false};
  std::atomic<bool> returned_after_device_completion{false};
  std::atomic<bool> owner_released{false};
  std::atomic<bool> owner_released_early{false};
  std::function<void()> callback = [&]() {
    callback_entered.set_value();
    callback_release.wait_for(10s);
    device_finished.store(true);
  };
  std::promise<void> forward_entered;
  auto executing = forward_entered.get_future();
  std::promise<void> release_forward;
  auto forward_release = release_forward.get_future().share();
  std::promise<void> forward_exiting;
  auto exiting = forward_exiting.get_future();
  OverlapTestWorker worker({token_output(17)}, /*driver=*/false);
  worker.set_output_hook([&](ForwardOutput& output, Stream& stream) {
    auto guard = stream.set_stream_guard();
    CHECK_EQ(
        aclrtLaunchHostFunc(
            stream.get_stream()->stream(),
            [](void* data) { (*static_cast<std::function<void()>*>(data))(); },
            &callback),
        ACL_SUCCESS);
    output.ready_event = stream.record_event();
    CHECK(output.ready_event != nullptr);
    output.retained_inputs.emplace_back(
        std::shared_ptr<int32_t>(new int32_t(42), [&](int32_t* value) {
          owner_released_early.store(!device_finished.load());
          owner_released.store(true);
          delete value;
        }));
    forward_entered.set_value();
    forward_release.wait_for(10s);
    forward_exiting.set_value();
  });
  auto step =
      worker.step_async(step_input<TypeParam>(/*decode=*/false))
          .toUnsafeFuture()
          .thenValue([&](std::optional<ForwardOutput> output) {
            returned_after_device_completion.store(device_finished.load());
            const bool has_output = output.has_value();
            output.reset();
            return has_output;
          });
  const std::future_status executing_status = executing.wait_for(2s);
  const std::future_status callback_status = entered.wait_for(2s);
  worker.begin_shutdown();
  release_forward.set_value();
  const std::future_status exiting_status = exiting.wait_for(2s);
  step.wait(50ms);
  const bool returned_early = step.isReady();

  // Release both gates before assertions so a failure can still drain safely.
  release_callback.set_value();
  step.wait(2s);
  const bool step_ready = step.isReady();
  EXPECT_EQ(executing_status, std::future_status::ready);
  EXPECT_EQ(callback_status, std::future_status::ready);
  EXPECT_EQ(exiting_status, std::future_status::ready);
  EXPECT_FALSE(returned_early);
  ASSERT_TRUE(step_ready);
  EXPECT_TRUE(std::move(step).get());
  EXPECT_TRUE(returned_after_device_completion.load());
  EXPECT_TRUE(owner_released.load());
  EXPECT_FALSE(owner_released_early.load());
}

}  // namespace
}  // namespace xllm
