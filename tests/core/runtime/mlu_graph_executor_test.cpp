/* Copyright 2025-2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <framework/core/device.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "base_executor_impl.h"
#include "core/common/constants.h"
#include "core/framework/batch/batch.h"
#include "core/framework/config/execution_config.h"
#include "core/framework/kv_cache/kv_cache.h"
#include "core/framework/model/model_args.h"
#include "core/framework/model/model_output.h"
#include "mlu_graph_executor_impl.h"
#include "models/llm/mlu/mtp_topk_state.h"
#include "platform/device.h"
#include "runtime/options.h"

namespace xllm {
namespace {

class ScopedConfigSnapshot final {
 public:
  ScopedConfigSnapshot()
      : max_tokens_for_graph_mode_(
            ExecutionConfig::get_instance().max_tokens_for_graph_mode()) {}

  ~ScopedConfigSnapshot() {
    ExecutionConfig::get_instance().max_tokens_for_graph_mode(
        max_tokens_for_graph_mode_);
  }

 private:
  int32_t max_tokens_for_graph_mode_;
};

}  // namespace

class MockCausalLM : public CausalLM {
 public:
  MockCausalLM(const torch::TensorOptions& options) : options_(options) {
    auto weight = torch::randn({1024, 1024}, options_) * 0.02;
    weight_ = register_parameter("weight", weight, false);
  }

  ModelOutput forward(const torch::Tensor& tokens,
                      const torch::Tensor& positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelInputParams& params) override {
    (void)tokens;
    (void)positions;
    (void)kv_caches;
    ++forward_cnt_;
    last_tokens_size_ = tokens.size(0);
    last_dp_token_nums_ = params.parallel.dp_global_token_nums;
    auto hidden_states = params.embedding.input_embedding.matmul(weight_);
    ModelOutput output(hidden_states);
    if (return_aux_hidden_states_) {
      output.aux_hidden_states = hidden_states + 1;
    }
    output.mtp_topk_state = mtp_topk_state_;
    return output;
  }
  torch::Tensor logits(const torch::Tensor& hidden_states,
                       const torch::Tensor& seleted_idxes) override {
    (void)seleted_idxes;
    return hidden_states;
  }
  int32_t forward_cnt() const { return forward_cnt_; }
  int64_t last_tokens_size() const { return last_tokens_size_; }
  const std::vector<int32_t>& last_dp_token_nums() const {
    return last_dp_token_nums_;
  }
  void return_aux_hidden_states(bool value) {
    return_aux_hidden_states_ = value;
  }
  void set_mtp_topk_state(MtpTopkStatePtr state) {
    mtp_topk_state_ = std::move(state);
  }
  void load_model(std::unique_ptr<ModelLoader> loader) override {}
  torch::Device device() const override { return options_.device(); }
  void prepare_expert_weight(int32_t layer_id,
                             const std::vector<int32_t>& expert_ids) override {}
  void update_expert_weight(int32_t layer_id) override {}
  const torch::TensorOptions& options() const override { return options_; }

 private:
  torch::Tensor input_;
  torch::Tensor weight_;
  MtpTopkStatePtr mtp_topk_state_;
  torch::TensorOptions options_;
  bool return_aux_hidden_states_ = false;
  int32_t forward_cnt_ = 0;
  int64_t last_tokens_size_ = 0;
  std::vector<int32_t> last_dp_token_nums_;
};

class MluGraphExecutorTest : public ::testing::Test {
 protected:
  MluGraphExecutorTest() = default;

  void SetUp() override {
    torch::Device device("mlu:0");
    tensor_options_ = torch::TensorOptions(torch::kBFloat16).device(device);

    model_args_.model_type("test_model");
    model_args_.dtype("bfloat16");
    model_args_.hidden_size(1024);
    model_args_.max_position_embeddings(2048);

    const uint32_t block_size = 16;
    options_.num_decoding_tokens(1);
    options_.block_size(block_size);

    model_ = std::make_unique<MockCausalLM>(tensor_options_);
    rebuild_impl();
  }

  ForwardInput prepare_inputs(int32_t batch_size, uint64_t seed) {
    Device device(tensor_options_.device());
    device.set_seed(seed);
    const int64_t max_seq_len = model_args_.max_position_embeddings();
    const uint32_t block_size = options_.block_size();
    const int64_t num_blocks_per_req =
        (max_seq_len + block_size - 1) / block_size + 1;
    auto int_tensor_options = tensor_options_.dtype(torch::kInt32);
    auto token_ids = torch::full({batch_size}, 1, int_tensor_options);
    auto positions = torch::full({batch_size}, 1, int_tensor_options);
    auto new_cache_slots =
        torch::randint(0, 10, {batch_size}, int_tensor_options);
    auto block_table = torch::randint(
        0, 10, {batch_size, num_blocks_per_req}, int_tensor_options);
    std::vector<int32_t> q_seq_lens_vec(batch_size + 1, 0);
    std::vector<int32_t> kv_seq_lens_vec(batch_size + 1, 0);
    for (int32_t i = 0; i < batch_size; ++i) {
      q_seq_lens_vec[i + 1] = q_seq_lens_vec[i] + 1;
      kv_seq_lens_vec[i + 1] = kv_seq_lens_vec[i] + 1;
    }
    auto q_seq_lens = torch::tensor(q_seq_lens_vec, int_tensor_options);
    auto kv_seq_lens = torch::tensor(kv_seq_lens_vec, int_tensor_options);
    auto input_embedding =
        torch::randn({batch_size, model_args_.hidden_size()}, tensor_options_) *
        0.1;
    ModelInputParams input_params;
    input_params.meta.batch_forward_type = BatchForwardType::DECODE;
    input_params.meta.num_sequences = batch_size;
    input_params.meta.kv_max_seq_len = 1;
    input_params.meta.q_max_seq_len = 1;
    input_params.parallel.dp_global_token_nums = {1};
    input_params.parallel.dp_is_decode = {1};
    input_params.attention.device.new_cache_slots = new_cache_slots;
    input_params.attention.device.block_tables = block_table;
    input_params.attention.device.q_seq_lens = q_seq_lens;
    input_params.attention.device.kv_seq_lens = kv_seq_lens;
    input_params.attention.host.q_seq_lens = q_seq_lens_vec;
    input_params.attention.host.kv_seq_lens = kv_seq_lens_vec;
    input_params.embedding.input_embedding = input_embedding;

    kv_caches_.resize(batch_size);
    ForwardInput input;
    input.token_ids = token_ids;
    input.positions = positions;
    input.input_params = input_params;
    return input;
  }

  void rebuild_impl() {
    const torch::Device device("mlu:0");
    impl_ = std::make_unique<::xllm::mlu::MluGraphExecutorImpl>(
        model_.get(), model_args_, device, options_);
    base_impl_ = std::make_unique<BaseExecutorImpl>(
        model_.get(), model_args_, device, options_);
  }

  ModelArgs model_args_;
  torch::TensorOptions tensor_options_;
  runtime::Options options_;
  std::unique_ptr<MockCausalLM> model_;
  std::vector<KVCache> kv_caches_;
  std::unique_ptr<::xllm::mlu::MluGraphExecutorImpl> impl_;
  std::unique_ptr<BaseExecutorImpl> base_impl_;
};

// Test graph creation and execution with different batch sizes
TEST_F(MluGraphExecutorTest, DifferentBatchSizes) {
  // Test with different batch sizes to ensure graph creation works
  const std::vector<uint32_t> batch_sizes = {1, 3, 13, 21, 65};
  for (auto batch_size : batch_sizes) {
    auto forward_input = prepare_inputs(batch_size, 1);
    auto eager_model_output = base_impl_->run({forward_input.token_ids},
                                              {forward_input.positions},
                                              kv_caches_,
                                              {forward_input.input_params});
    auto eager_output = eager_model_output.hidden_states;

    auto graph_model_output = impl_->run({forward_input.token_ids},
                                         {forward_input.positions},
                                         kv_caches_,
                                         {forward_input.input_params});
    auto graph_output = graph_model_output.hidden_states;

    auto replay_model_output = impl_->run({forward_input.token_ids},
                                          {forward_input.positions},
                                          kv_caches_,
                                          {forward_input.input_params});
    auto replay_output = replay_model_output.hidden_states;

    CHECK_EQ(eager_output.sizes(), graph_output.sizes());
    CHECK_EQ(eager_output.sizes(), replay_output.sizes());
    // Compare outputs - should be identical
    torch_mlu::synchronize();
    EXPECT_TRUE(torch::allclose(eager_output, graph_output, 1e-5, 1e-6));
    EXPECT_TRUE(torch::allclose(eager_output, replay_output, 1e-5, 1e-6));
  }
}

// Test multiple runs to verify consistency across different execution modes
TEST_F(MluGraphExecutorTest, MluGraphExecutorVsBaseExecutorImplMultipleRuns) {
  int32_t batch_size = 5;
  int32_t seed = 42;
  auto forward_input = prepare_inputs(batch_size, seed);
  auto eager_model_output = base_impl_->run({forward_input.token_ids},
                                            {forward_input.positions},
                                            kv_caches_,
                                            {forward_input.input_params});
  auto eager_output = eager_model_output.hidden_states;

  auto graph_model_output = impl_->run({forward_input.token_ids},
                                       {forward_input.positions},
                                       kv_caches_,
                                       {forward_input.input_params});
  auto graph_output = graph_model_output.hidden_states;

  CHECK_EQ(eager_output.sizes(), graph_output.sizes());
  // Compare outputs - should be identical
  torch_mlu::synchronize();
  EXPECT_TRUE(torch::allclose(eager_output, graph_output, 1e-5, 1e-6));

  // Run multiple times and compare results
  const int num_runs = 5;
  auto base_forward_input = prepare_inputs(batch_size + 1, seed);
  auto replay_forward_input = prepare_inputs(batch_size + 1, seed);
  EXPECT_TRUE(torch::allclose(
      base_forward_input.input_params.embedding.input_embedding,
      replay_forward_input.input_params.embedding.input_embedding,
      1e-5,
      1e-6));

  for (int i = 0; i < num_runs; ++i) {
    auto base_model_output = base_impl_->run({base_forward_input.token_ids},
                                             {base_forward_input.positions},
                                             kv_caches_,
                                             {base_forward_input.input_params});
    auto base_output = base_model_output.hidden_states;

    auto replay_model_output = impl_->run({replay_forward_input.token_ids},
                                          {replay_forward_input.positions},
                                          kv_caches_,
                                          {replay_forward_input.input_params});
    auto replay_output = replay_model_output.hidden_states;
    base_forward_input.input_params.embedding.input_embedding = base_output;
    replay_forward_input.input_params.embedding.input_embedding = replay_output;
    CHECK_EQ(base_output.sizes(), replay_output.sizes());
  }

  torch_mlu::synchronize();
  EXPECT_TRUE(torch::allclose(
      base_forward_input.input_params.embedding.input_embedding,
      replay_forward_input.input_params.embedding.input_embedding,
      1e-5,
      1e-6));
}

TEST_F(MluGraphExecutorTest, DraftDecodeFallsBackToEager) {
  options_.is_draft_engine(true);
  rebuild_impl();

  const int32_t batch_size = 5;
  const uint64_t seed = 7;
  auto forward_input = prepare_inputs(batch_size, seed);

  auto eager_model_output = base_impl_->run({forward_input.token_ids},
                                            {forward_input.positions},
                                            kv_caches_,
                                            {forward_input.input_params});
  auto eager_output = eager_model_output.hidden_states;

  auto first_impl_output = impl_
                               ->run({forward_input.token_ids},
                                     {forward_input.positions},
                                     kv_caches_,
                                     {forward_input.input_params})
                               .hidden_states;
  auto second_impl_output = impl_
                                ->run({forward_input.token_ids},
                                      {forward_input.positions},
                                      kv_caches_,
                                      {forward_input.input_params})
                                .hidden_states;

  torch_mlu::synchronize();
  EXPECT_TRUE(torch::allclose(eager_output, first_impl_output, 1e-5, 1e-6));
  EXPECT_TRUE(
      torch::allclose(first_impl_output, second_impl_output, 1e-5, 1e-6));
  EXPECT_EQ(model_->forward_cnt(), 3);
}

TEST_F(MluGraphExecutorTest, DraftEagerDoesNotExposeAuxWhenDisabled) {
  model_->return_aux_hidden_states(true);
  options_.is_draft_engine(true);
  options_.enable_graph_aux_hidden_states(false);
  rebuild_impl();

  const int32_t batch_size = 5;
  const uint64_t seed = 17;
  auto forward_input = prepare_inputs(batch_size, seed);

  ModelOutput output = impl_->run({forward_input.token_ids},
                                  {forward_input.positions},
                                  kv_caches_,
                                  {forward_input.input_params});

  EXPECT_FALSE(output.aux_hidden_states.defined());
  EXPECT_EQ(model_->forward_cnt(), 1);
}

TEST_F(MluGraphExecutorTest, DraftEagerPreservesTypedTopkWhenAuxIsDisabled) {
  mlu::model::MluMtpTopkState::LayerStates expected_layers;
  expected_layers.emplace_back(layer::DsaTopkState(
      torch::tensor({{1, 2}, {3, 4}}, tensor_options_.dtype(torch::kInt32)),
      torch::tensor({5, 6}, tensor_options_.dtype(torch::kInt32))));
  const MtpTopkStatePtr expected_state =
      std::make_shared<mlu::model::MluMtpTopkState>(std::move(expected_layers));
  model_->return_aux_hidden_states(true);
  model_->set_mtp_topk_state(expected_state);
  options_.is_draft_engine(true);
  options_.enable_graph_aux_hidden_states(false);
  rebuild_impl();

  const int32_t batch_size = 2;
  const uint64_t seed = 23;
  auto forward_input = prepare_inputs(batch_size, seed);

  ModelOutput output = impl_->run({forward_input.token_ids},
                                  {forward_input.positions},
                                  kv_caches_,
                                  {forward_input.input_params});

  EXPECT_EQ(output.mtp_topk_state.get(), expected_state.get());
  EXPECT_FALSE(output.aux_hidden_states.defined());
  EXPECT_EQ(model_->forward_cnt(), 1);
}

TEST_F(MluGraphExecutorTest, TargetDecodeCapturesThenReplays) {
  options_.is_draft_engine(false);
  rebuild_impl();

  const int32_t batch_size = 5;
  const uint64_t seed = 11;
  auto forward_input = prepare_inputs(batch_size, seed);

  auto first_impl_output = impl_
                               ->run({forward_input.token_ids},
                                     {forward_input.positions},
                                     kv_caches_,
                                     {forward_input.input_params})
                               .hidden_states;
  auto second_impl_output = impl_
                                ->run({forward_input.token_ids},
                                      {forward_input.positions},
                                      kv_caches_,
                                      {forward_input.input_params})
                                .hidden_states;

  torch_mlu::synchronize();
  EXPECT_TRUE(
      torch::allclose(first_impl_output, second_impl_output, 1e-5, 1e-6));
  EXPECT_EQ(model_->forward_cnt(), 1);
}

TEST_F(MluGraphExecutorTest, LargeDecodeBucketCapturesThenReplays) {
  ScopedConfigSnapshot config_snapshot;
  ExecutionConfig::get_instance().max_tokens_for_graph_mode(128);
  options_.is_draft_engine(false);
  options_.max_seqs_per_batch(128);
  rebuild_impl();

  const int32_t batch_size = 65;
  const uint64_t seed = 19;
  auto forward_input = prepare_inputs(batch_size, seed);
  const int32_t start_cnt = model_->forward_cnt();

  auto first_output = impl_
                          ->run({forward_input.token_ids},
                                {forward_input.positions},
                                kv_caches_,
                                {forward_input.input_params})
                          .hidden_states;
  auto second_output = impl_
                           ->run({forward_input.token_ids},
                                 {forward_input.positions},
                                 kv_caches_,
                                 {forward_input.input_params})
                           .hidden_states;

  torch_mlu::synchronize();
  EXPECT_TRUE(torch::allclose(first_output, second_output, 1e-5, 1e-6));
  EXPECT_EQ(model_->forward_cnt(), start_cnt + 1);
  EXPECT_EQ(model_->last_tokens_size(), 80);
}

TEST_F(MluGraphExecutorTest, OverConfiguredTokenLimitFallsBackToEager) {
  ScopedConfigSnapshot config_snapshot;
  ExecutionConfig::get_instance().max_tokens_for_graph_mode(33);
  options_.is_draft_engine(false);
  options_.max_seqs_per_batch(33);
  rebuild_impl();

  const int32_t batch_size = 33;
  const uint64_t seed = 79;
  auto forward_input = prepare_inputs(batch_size, seed);
  const int32_t start_cnt = model_->forward_cnt();

  auto first_output = impl_
                          ->run({forward_input.token_ids},
                                {forward_input.positions},
                                kv_caches_,
                                {forward_input.input_params})
                          .hidden_states;
  auto second_output = impl_
                           ->run({forward_input.token_ids},
                                 {forward_input.positions},
                                 kv_caches_,
                                 {forward_input.input_params})
                           .hidden_states;

  torch_mlu::synchronize();
  EXPECT_TRUE(torch::allclose(first_output, second_output, 1e-5, 1e-6));
  EXPECT_EQ(model_->forward_cnt(), start_cnt + 2);
  EXPECT_EQ(model_->last_tokens_size(), batch_size);
}

TEST_F(MluGraphExecutorTest, PersistentTensorBytesIncludeLazyAux) {
  ::xllm::mlu::GraphPersistentParam param(
      model_args_, tensor_options_.device(), options_);
  const std::size_t output_bytes =
      static_cast<std::size_t>(param.output_.numel()) *
      param.output_.element_size();
  const std::size_t base_bytes = param.get_persistent_tensor_bytes();

  param.aux_hidden_states_ = torch::zeros_like(param.output_);

  EXPECT_GT(base_bytes, output_bytes);
  EXPECT_EQ(param.get_persistent_tensor_bytes(), base_bytes + output_bytes);
}

TEST_F(MluGraphExecutorTest, LinearStatePaddingTailUsesPaddingId) {
  ::xllm::mlu::GraphPersistentParam param(
      model_args_, tensor_options_.device(), options_);

  ForwardInput first_input = prepare_inputs(/*batch_size=*/4, /*seed=*/83);
  first_input.input_params.embedding.linear_state_ids = {10, 20, 30, 40};
  first_input.input_params.embedding.linear_state_indices =
      torch::tensor(first_input.input_params.embedding.linear_state_ids,
                    tensor_options_.dtype(torch::kInt32));

  param.init_params(first_input.input_params,
                    /*padding_num_tokens=*/4,
                    /*padding_needed=*/0);
  param.update_input_buffer(first_input.token_ids,
                            first_input.positions,
                            first_input.input_params,
                            /*padding_needed=*/0);

  ForwardInput second_input = prepare_inputs(/*batch_size=*/3, /*seed=*/89);
  second_input.input_params.embedding.linear_state_ids = {11, 21, 31};
  second_input.input_params.embedding.linear_state_indices =
      torch::tensor(second_input.input_params.embedding.linear_state_ids,
                    tensor_options_.dtype(torch::kInt32));

  param.update_input_buffer(second_input.token_ids,
                            second_input.positions,
                            second_input.input_params,
                            /*padding_needed=*/1);

  torch_mlu::synchronize();
  torch::Tensor linear_state_indices =
      param.params_.embedding.linear_state_indices.cpu();
  EXPECT_TRUE(torch::equal(linear_state_indices,
                           torch::tensor({11, 21, 31, kPaddingLinearStateId},
                                         torch::dtype(torch::kInt32))));
}

TEST_F(MluGraphExecutorTest, PrefillThenDecodeCapturesAndReplays) {
  options_.is_draft_engine(false);
  rebuild_impl();

  const int32_t batch_size = 5;
  const uint64_t prefill_seed = 23;
  auto prefill_input = prepare_inputs(batch_size, prefill_seed);
  prefill_input.input_params.meta.batch_forward_type =
      BatchForwardType::PREFILL;

  ModelOutput prefill_output = impl_->run({prefill_input.token_ids},
                                          {prefill_input.positions},
                                          kv_caches_,
                                          {prefill_input.input_params});

  const uint64_t decode_seed = 29;
  auto decode_input = prepare_inputs(batch_size, decode_seed);
  auto first_decode_output = impl_
                                 ->run({decode_input.token_ids},
                                       {decode_input.positions},
                                       kv_caches_,
                                       {decode_input.input_params})
                                 .hidden_states;
  auto second_decode_output = impl_
                                  ->run({decode_input.token_ids},
                                        {decode_input.positions},
                                        kv_caches_,
                                        {decode_input.input_params})
                                  .hidden_states;

  torch_mlu::synchronize();
  EXPECT_TRUE(prefill_output.hidden_states.defined());
  EXPECT_TRUE(
      torch::allclose(first_decode_output, second_decode_output, 1e-5, 1e-6));
  EXPECT_EQ(model_->forward_cnt(), 2);
}

TEST_F(MluGraphExecutorTest, EqualDpDecodePadsToTpGraphSize) {
  options_.is_draft_engine(false);
  options_.world_size(8);
  options_.dp_size(2);
  rebuild_impl();

  auto forward_input = prepare_inputs(/*batch_size=*/2, /*seed=*/61);
  forward_input.input_params.parallel.dp_global_token_nums = {2, 2};
  forward_input.input_params.parallel.dp_is_decode = {1, 1};

  auto first_output = impl_
                          ->run({forward_input.token_ids},
                                {forward_input.positions},
                                kv_caches_,
                                {forward_input.input_params})
                          .hidden_states;
  auto second_output = impl_
                           ->run({forward_input.token_ids},
                                 {forward_input.positions},
                                 kv_caches_,
                                 {forward_input.input_params})
                           .hidden_states;

  torch_mlu::synchronize();
  EXPECT_TRUE(torch::allclose(first_output, second_output, 1e-5, 1e-6));
  EXPECT_EQ(model_->forward_cnt(), 1);
  EXPECT_EQ(model_->last_tokens_size(), 4);
  EXPECT_EQ(model_->last_dp_token_nums(), std::vector<int32_t>({4, 4}));
}

TEST_F(MluGraphExecutorTest, UnevenDpDecodePadsToTpGraphSize) {
  options_.is_draft_engine(false);
  options_.world_size(8);
  options_.dp_size(2);
  rebuild_impl();

  auto forward_input = prepare_inputs(/*batch_size=*/2, /*seed=*/67);
  forward_input.input_params.parallel.dp_global_token_nums = {1, 2};
  forward_input.input_params.parallel.dp_is_decode = {1, 1};

  auto first_output = impl_
                          ->run({forward_input.token_ids},
                                {forward_input.positions},
                                kv_caches_,
                                {forward_input.input_params})
                          .hidden_states;
  auto second_output = impl_
                           ->run({forward_input.token_ids},
                                 {forward_input.positions},
                                 kv_caches_,
                                 {forward_input.input_params})
                           .hidden_states;

  torch_mlu::synchronize();
  EXPECT_TRUE(torch::allclose(first_output, second_output, 1e-5, 1e-6));
  EXPECT_EQ(model_->forward_cnt(), 1);
  EXPECT_EQ(model_->last_tokens_size(), 4);
  EXPECT_EQ(model_->last_dp_token_nums(), std::vector<int32_t>({4, 4}));
}

TEST_F(MluGraphExecutorTest, MtpSeqLensCapacityUsesSpecFactor) {
  options_.is_draft_engine(false);
  options_.num_speculative_tokens(1);
  options_.num_decoding_tokens(options_.num_speculative_tokens() + 1);
  options_.enable_speculative_decode(true);
  options_.max_seqs_per_batch(2);
  rebuild_impl();

  auto forward_input = prepare_inputs(/*batch_size=*/4, /*seed=*/71);
  forward_input.input_params.parallel.dp_global_token_nums = {4, 4};
  forward_input.input_params.parallel.dp_is_decode = {1, 1};

  auto first_output = impl_
                          ->run({forward_input.token_ids},
                                {forward_input.positions},
                                kv_caches_,
                                {forward_input.input_params})
                          .hidden_states;
  auto second_output = impl_
                           ->run({forward_input.token_ids},
                                 {forward_input.positions},
                                 kv_caches_,
                                 {forward_input.input_params})
                           .hidden_states;

  torch_mlu::synchronize();
  EXPECT_TRUE(torch::allclose(first_output, second_output, 1e-5, 1e-6));
}

TEST_F(MluGraphExecutorTest, MtpSeqLensCapacityIncludesGraphPadding) {
  options_.is_draft_engine(false);
  options_.num_speculative_tokens(2);
  options_.num_decoding_tokens(options_.num_speculative_tokens() + 1);
  options_.enable_speculative_decode(true);
  options_.max_seqs_per_batch(8);
  rebuild_impl();

  // Eight MTP validation requests produce 24 token rows. The MLU graph
  // executor pads that input to the 32-token bucket, so cumulative sequence
  // lengths need 33 entries.
  auto forward_input = prepare_inputs(/*batch_size=*/24, /*seed=*/73);

  EXPECT_NO_THROW({
    ModelOutput first_output = impl_->run({forward_input.token_ids},
                                          {forward_input.positions},
                                          kv_caches_,
                                          {forward_input.input_params});
    ModelOutput second_output = impl_->run({forward_input.token_ids},
                                           {forward_input.positions},
                                           kv_caches_,
                                           {forward_input.input_params});

    torch_mlu::synchronize();
    EXPECT_TRUE(torch::allclose(
        first_output.hidden_states, second_output.hidden_states, 1e-5, 1e-6));
  });
}

TEST_F(MluGraphExecutorTest, DpDummyFallsBackToEager) {
  options_.is_draft_engine(false);
  rebuild_impl();

  const int32_t batch_size = 5;
  const uint64_t seed = 31;
  auto forward_input = prepare_inputs(batch_size, seed);
  forward_input.input_params.parallel.dp_global_token_nums = {batch_size, 0};
  forward_input.input_params.parallel.dp_is_decode = {1, 0};

  const int32_t start_cnt = model_->forward_cnt();
  auto first_output = impl_
                          ->run({forward_input.token_ids},
                                {forward_input.positions},
                                kv_caches_,
                                {forward_input.input_params})
                          .hidden_states;
  auto second_output = impl_
                           ->run({forward_input.token_ids},
                                 {forward_input.positions},
                                 kv_caches_,
                                 {forward_input.input_params})
                           .hidden_states;

  torch_mlu::synchronize();
  EXPECT_TRUE(torch::allclose(first_output, second_output, 1e-5, 1e-6));
  EXPECT_EQ(model_->forward_cnt(), start_cnt + 2);
}

TEST_F(MluGraphExecutorTest, DpUnevenDecodeFallsBackToEager) {
  options_.is_draft_engine(false);
  rebuild_impl();

  const int32_t batch_size = 5;
  auto forward_input = prepare_inputs(batch_size, 43);
  forward_input.input_params.parallel.dp_global_token_nums = {batch_size,
                                                              batch_size - 1};
  forward_input.input_params.parallel.dp_is_decode = {1, 1};
  forward_input.input_params.meta.q_max_seq_len = 2;

  const int32_t start_cnt = model_->forward_cnt();
  auto first_output = impl_
                          ->run({forward_input.token_ids},
                                {forward_input.positions},
                                kv_caches_,
                                {forward_input.input_params})
                          .hidden_states;
  auto second_output = impl_
                           ->run({forward_input.token_ids},
                                 {forward_input.positions},
                                 kv_caches_,
                                 {forward_input.input_params})
                           .hidden_states;

  torch_mlu::synchronize();
  EXPECT_TRUE(torch::allclose(first_output, second_output, 1e-5, 1e-6));
  EXPECT_EQ(model_->forward_cnt(), start_cnt + 2);
}

TEST_F(MluGraphExecutorTest, DpDummyDoesNotPoisonGraphCache) {
  options_.is_draft_engine(false);
  rebuild_impl();

  const int32_t batch_size = 5;
  auto dummy_input = prepare_inputs(batch_size, 37);
  dummy_input.input_params.parallel.dp_global_token_nums = {batch_size, 0};
  dummy_input.input_params.parallel.dp_is_decode = {1, 0};

  const int32_t start_cnt = model_->forward_cnt();
  impl_->run({dummy_input.token_ids},
             {dummy_input.positions},
             kv_caches_,
             {dummy_input.input_params});

  auto decode_input = prepare_inputs(batch_size, 41);
  decode_input.input_params.parallel.dp_global_token_nums = {batch_size,
                                                             batch_size};
  decode_input.input_params.parallel.dp_is_decode = {1, 1};

  auto first_decode = impl_
                          ->run({decode_input.token_ids},
                                {decode_input.positions},
                                kv_caches_,
                                {decode_input.input_params})
                          .hidden_states;
  auto second_decode = impl_
                           ->run({decode_input.token_ids},
                                 {decode_input.positions},
                                 kv_caches_,
                                 {decode_input.input_params})
                           .hidden_states;

  torch_mlu::synchronize();
  EXPECT_TRUE(torch::allclose(first_decode, second_decode, 1e-5, 1e-6));
  EXPECT_EQ(model_->forward_cnt(), start_cnt + 2);
}

TEST_F(MluGraphExecutorTest, DpUnevenDecodeDoesNotPoisonGraphCache) {
  options_.is_draft_engine(false);
  rebuild_impl();

  const int32_t batch_size = 5;
  auto uneven_input = prepare_inputs(batch_size, 47);
  uneven_input.input_params.parallel.dp_global_token_nums = {batch_size,
                                                             batch_size - 1};
  uneven_input.input_params.parallel.dp_is_decode = {1, 1};
  uneven_input.input_params.meta.q_max_seq_len = 2;

  const int32_t start_cnt = model_->forward_cnt();
  impl_->run({uneven_input.token_ids},
             {uneven_input.positions},
             kv_caches_,
             {uneven_input.input_params});

  auto decode_input = prepare_inputs(batch_size, 53);
  decode_input.input_params.parallel.dp_global_token_nums = {batch_size,
                                                             batch_size};
  decode_input.input_params.parallel.dp_is_decode = {1, 1};

  auto first_decode = impl_
                          ->run({decode_input.token_ids},
                                {decode_input.positions},
                                kv_caches_,
                                {decode_input.input_params})
                          .hidden_states;
  auto second_decode = impl_
                           ->run({decode_input.token_ids},
                                 {decode_input.positions},
                                 kv_caches_,
                                 {decode_input.input_params})
                           .hidden_states;

  torch_mlu::synchronize();
  EXPECT_TRUE(torch::allclose(first_decode, second_decode, 1e-5, 1e-6));
  EXPECT_EQ(model_->forward_cnt(), start_cnt + 2);
}

}  // namespace xllm
