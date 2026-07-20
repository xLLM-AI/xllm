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

#include <acl/acl.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <torch_npu/torch_npu.h>

#include <cstdlib>
#include <memory>
#include <optional>
#include <vector>

#include "common/metrics.h"
#include "core/framework/batch/batch.h"
#include "core/framework/block/block.h"
#include "core/framework/block/block_manager_impl.h"
#include "core/framework/config/execution_config.h"
#include "core/framework/config/speculative_config.h"
#include "core/framework/kv_cache/kv_cache.h"
#include "core/framework/kv_cache/kv_cache_utils.h"
#include "core/framework/kv_cache/linear_state_restore.h"
#include "core/framework/model/model_args.h"
#include "core/framework/model/model_output.h"
#include "core/framework/model_context.h"
#include "core/framework/model_loader.h"
#include "core/framework/request/sequence.h"
#include "core/framework/request/stopping_checker.h"
#include "core/framework/sampling/sampling_params.h"
#include "core/layers/common/attention_metadata.h"
#include "core/layers/npu/npu_lm_head_impl.h"
#include "core/layers/npu/npu_word_embedding_impl.h"
#include "core/layers/npu_torch/tests_utils.h"
#include "core/runtime/acl_graph_executor_impl.h"
#include "core/runtime/acl_graph_persistent_param.h"
#include "core/runtime/base_executor_impl.h"
#include "core/runtime/options.h"
#include "models/model_registry.h"

// Global test environment for ACL graph executor tests
class AclGraphExecutorTestEnvironment : public ::testing::Environment {
 public:
  void SetUp() override {
    // Initialize glog
    google::InitGoogleLogging("acl_graph_executor_test");
    google::SetStderrLogging(google::INFO);

    // Add any other global initialization here
    std::cout << "Global test environment setup completed" << std::endl;
    int ret = aclrtSetDevice(0);
    if (ret != 0) {
      LOG(ERROR) << "ACL set device id: 0 failed, ret:" << ret;
    }
    torch_npu::init_npu("npu:0");
  }

  void TearDown() override {
    // Cleanup if needed
    google::ShutdownGoogleLogging();
    torch_npu::finalize_npu();
    aclrtResetDevice(0);
    aclFinalize();
    LOG(INFO) << "AclGraphExecutorTestEnvironment TearDown completed.";
  }
};

// Register the global test environment
::testing::Environment* const test_env =
    ::testing::AddGlobalTestEnvironment(new AclGraphExecutorTestEnvironment);

namespace xllm {

namespace {
const KVCache& first_full_attention_cache(
    const std::vector<KVCache>& kv_caches) {
  for (const auto& kv_cache : kv_caches) {
    if (!kv_cache.empty()) {
      auto k_cache = kv_cache.get_k_cache();
      if (k_cache.defined() && k_cache.numel() > 0) {
        return kv_cache;
      }
    }
  }
  LOG(FATAL) << "No full-attention KV cache found";
  std::abort();
}
}  // namespace

// Initialize glog for testing - use a function to ensure proper initialization
// order
void InitializeGlog() {
  static bool initialized = false;
  if (!initialized) {
    google::InitGoogleLogging("acl_graph_executor_test");
    google::SetStderrLogging(google::INFO);
    initialized = true;
  }
}

// Simple CausalLM implementation for testing ACL graph executor
// Uses basic operations to verify graph capture and replay functionality
class SimpleCausalLM : public CausalLM {
 public:
  SimpleCausalLM(const ModelArgs& args, const torch::Device& device)
      : args_(args), device_(device) {
    // Initialize a simple linear layer for testing
    linear_ = register_module("linear",
                              torch::nn::Linear(torch::nn::LinearOptions(
                                  args.hidden_size(), args.hidden_size())));

    // Initialize token embedding table
    const int64_t vocab_size = std::max(args.vocab_size(), 1000L);
    token_embedding_table_ = register_parameter(
        "token_embedding",
        torch::randn({vocab_size, args.hidden_size()},
                     torch::dtype(torch::kFloat32).device(device)));

    // Initialize position embedding table
    const int64_t max_pos = args.max_position_embeddings();
    pos_embedding_table_ = register_parameter(
        "pos_embedding",
        torch::randn({max_pos, args.hidden_size()},
                     torch::dtype(torch::kFloat32).device(device)));
    // Initialize block-related tensors for Rec multi-round computation
    block_size_ = torch::tensor(4L, torch::dtype(torch::kInt64).device(device));
    scalar_one_ = torch::tensor(1L, torch::dtype(torch::kInt64).device(device));

    // Initialize scalar tensors for computation
    // const tensors
    kv_scale_ =
        torch::tensor(0.01f, torch::dtype(torch::kFloat32).device(device));
    q_scale_ =
        torch::tensor(0.01f, torch::dtype(torch::kFloat32).device(device));
    cache_scale_ =
        torch::tensor(0.005f, torch::dtype(torch::kFloat32).device(device));
    block_scale_ =
        torch::tensor(0.001f, torch::dtype(torch::kFloat32).device(device));

    // Move to device
    this->to(device);
  }

  torch::Tensor forward_impl(const torch::Tensor& tokens,
                             const torch::Tensor& positions,
                             std::vector<KVCache>& kv_caches,
                             const ModelInputParams& params) {
    // Simple computation: token embedding + position embedding + linear layer
    // This creates temporary tensors that NPUGraph mempool will manage
    LOG(INFO) << "SimpleCausalLM forward_impl, tokens: " << tokens.sizes()
              << ", positions: " << positions.sizes()
              << ", kv_caches: " << kv_caches.size()
              << ", params: " << params.meta.num_sequences;
    const int64_t num_tokens = tokens.size(0);
    const int64_t hidden_size = args_.hidden_size();

    // Create token embeddings using standard embedding lookup
    auto token_embeddings = torch::embedding(token_embedding_table_, tokens);

    // Create position embeddings using standard embedding lookup
    auto position_embeddings =
        torch::embedding(pos_embedding_table_, positions);

    // Combine embeddings
    auto combined = token_embeddings + position_embeddings;

    // Apply linear layer
    auto output = linear_->forward(combined);

    // Add some computation using other params to make it more realistic
    // if (params.attention.device.kv_seq_lens.defined()) {
    //   // Use kv_seq_lens in computation
    //   auto kv_lens_sum = torch::sum(params.attention.device.kv_seq_lens);
    //   output = output + kv_lens_sum * kv_scale_;
    // }

    // if (params.attention.device.q_seq_lens.defined()) {
    //   // Use q_seq_lens in computation
    //   auto q_lens_sum = torch::sum(params.attention.device.q_seq_lens);
    //   output = output + q_lens_sum * q_scale_;
    // }

    if (params.attention.device.new_cache_slots.defined()) {
      // Use new_cache_slots in computation
      auto cache_slots_sum =
          torch::sum(params.attention.device.new_cache_slots);
      output = output + cache_slots_sum * cache_scale_;
    }

    if (params.attention.device.block_tables.defined() && !kv_caches.empty()) {
      // Use block_tables to do embedding lookup from kv_cache - Rec multi-round
      // computation Calculate max_seq_len from actual seq_len tensor
      auto max_seq_len = torch::max(params.attention.device.kv_seq_lens);

      // Calculate max_block_nums_per_seq
      auto max_block_nums_per_seq = torch::ceil(max_seq_len / block_size_);

      // Get kv_cache tensor from KVCache
      const auto& kv_cache_tensor =
          first_full_attention_cache(kv_caches).get_k_cache();

      // Create col_indices and mask
      int64_t block_table_len = params.attention.device.block_tables.size(1);
      auto col_indices = torch::arange(
          block_table_len, torch::dtype(torch::kInt64).device(device_));
      auto mask = col_indices < (max_block_nums_per_seq - scalar_one_);

      // Directly compute embedding
      auto kv_embeddings = torch::embedding(
          kv_cache_tensor, params.attention.device.block_tables);

      // Apply mask and sum
      auto kv_embeddings_masked = kv_embeddings * mask.view({1, -1, 1});
      auto kv_embeddings_sum = torch::sum(kv_embeddings_masked);
      output = output + kv_embeddings_sum * block_scale_;
    }

    return output;
  }

  // Adapter method to match CausalLM base class interface
  ModelOutput forward(const torch::Tensor& tokens,
                      const torch::Tensor& positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelInputParams& parameters) override {
    auto hidden_states = forward_impl(tokens, positions, kv_caches, parameters);
    ModelOutput model_output(hidden_states);
    if (return_aux_hidden_states_) {
      model_output.aux_hidden_states =
          torch::cat({hidden_states, hidden_states}, /*dim=*/-1);
    }
    return model_output;
  }

  void set_return_aux_hidden_states(bool value) {
    return_aux_hidden_states_ = value;
  }

  const torch::TensorOptions& options() const override {
    static torch::TensorOptions opts =
        torch::dtype(torch::kFloat32).device(device_);
    return opts;
  }

  const ModelArgs& args() const { return args_; }

  // Implement required virtual functions
  torch::Tensor logits(const torch::Tensor& hidden_states,
                       const torch::Tensor& selected_idxes) override {
    // Simple logits computation
    const int64_t vocab_size = std::max(args_.vocab_size(), 1000L);
    return torch::randn({hidden_states.size(0), vocab_size},
                        torch::dtype(torch::kFloat32).device(device_));
  }

  void load_model(std::unique_ptr<ModelLoader> loader) override {
    // Simple implementation for testing
  }

  torch::Device device() const override { return device_; }

  void prepare_expert_weight(int32_t layer_id,
                             const std::vector<int32_t>& expert_ids) override {
    // Simple implementation for testing
  }

  void update_expert_weight(int32_t layer_id) override {
    // Simple implementation for testing
  }

  layer::NpuLmHead get_npu_lm_head() override {
    // Simple implementation for testing
    return layer::NpuLmHead(nullptr);
  }

  void set_npu_lm_head(layer::NpuLmHead& head) override {
    // Simple implementation for testing
  }

  layer::NpuWordEmbedding get_npu_word_embedding() override {
    // Simple implementation for testing
    return layer::NpuWordEmbedding(nullptr);
  }

  void set_npu_word_embedding(layer::NpuWordEmbedding& embedding) override {
    // Simple implementation for testing
  }

 private:
  ModelArgs args_;
  torch::Device device_;
  torch::nn::Linear linear_{nullptr};
  torch::Tensor token_embedding_table_;
  torch::Tensor pos_embedding_table_;

  // Pre-allocated constant scalar tensors for computation
  torch::Tensor kv_scale_;
  torch::Tensor q_scale_;
  torch::Tensor cache_scale_;
  torch::Tensor block_scale_;
  torch::Tensor block_size_;
  torch::Tensor scalar_one_;
  bool return_aux_hidden_states_ = false;
};

class AclGraphExecutorTest : public ::testing::Test {
 protected:
  AclGraphExecutorTest() = default;

  void SetUp() override {
    if (initialized_) {
      return;
    }
    initialized_ = true;
    sequences_.reserve(100);

    // Set up model args
    model_args_.model_type("test_model");
    model_args_.dtype("float32");
    model_args_.hidden_size(128);
    model_args_.max_position_embeddings(2048);
    model_args_.vocab_size(1000);  // Set a reasonable vocab size

    // Set up device
    device_ = std::make_unique<torch::Device>("npu:0");

    // Set up runtime options
    options_.num_decoding_tokens(1);
    options_.block_size(4);

    // Create simple model
    model_ = std::make_unique<SimpleCausalLM>(model_args_, *device_);

    // Initialize block manager
    const uint32_t n_blocks = 1000;
    const uint32_t block_size = 4;
    BlockManager::Options block_options;
    block_options.num_blocks(n_blocks).block_size(block_size);
    block_manager_ = std::make_unique<BlockManagerImpl>(block_options);

    // Initialize sampling and stopping parameters
    sampling_param_.frequency_penalty = 0.1;
    stopping_checker_.set_max_generated_tokens(20);

    // Initialize sequence parameters
    seq_params_.seq_capacity = 100;
    seq_params_.stopping_checker = &stopping_checker_;
    seq_params_.sampling_param = &sampling_param_;
    seq_params_.skip_special_tokens = true;
    seq_params_.echo = false;
    seq_params_.logprobs = false;
    seq_params_.enable_schedule_overlap = false;

    // Initialize input embedding and mm_data
    input_embedding_ =
        torch::zeros({1, model_args_.hidden_size()},
                     torch::dtype(torch::kFloat32).device(*device_));
    mm_data_ = MMData();  // Default constructor creates empty MMData

    // Initialize KV caches
    kv_caches_.clear();
    const int64_t hidden_size = model_args_.hidden_size();

    // Create KV cache with shape [n_blocks, block_size, hidden_size]
    torch::Tensor kv_cache =
        torch::randn({n_blocks, block_size * hidden_size},
                     torch::dtype(torch::kFloat32).device(*device_));
    kv_caches_.emplace_back(KVCacheTensors{kv_cache, kv_cache});
  }

  void TearDown() override { return; }

  void reset() {
    for (auto& sequence : sequences_) {
      auto blocks = sequence.kv_state().blocks(BlockType::KV);
      if (!blocks.empty()) {
        block_manager_->deallocate(blocks);
      }
    }
  }

  // Helper function to create a simple batch
  std::unique_ptr<Batch> CreateTestBatch() {
    sequences_.emplace_back(0,
                            std::vector<int32_t>{1, 3, 5, 7, 5, 4, 3, 2, 1},
                            input_embedding_,
                            mm_data_,
                            fake_decoder_,
                            seq_params_);
    auto& sequence = sequences_.back();

    // Allocate blocks and configure sequence
    sequence.add_blocks(BlockType::KV, block_manager_->allocate(3));
    // Set kv_cache_tokens_num to be >= num_prompt_tokens to move to decode
    // stage
    sequence.kv_state().incr_kv_cache_tokens_num(
        /*size=*/9);  // 9 prompt tokens
    sequence.append_token(100);

    // Create batch with pointer to sequence (batch doesn't own sequence)
    auto batch = std::make_unique<Batch>();
    batch->add(&sequence);

    return batch;
  }
  bool initialized_ = false;
  ModelArgs model_args_;
  std::unique_ptr<torch::Device> device_;
  runtime::Options options_;
  std::unique_ptr<CausalLM> model_;

  // Shared resources for all tests
  std::unique_ptr<BlockManagerImpl> block_manager_;
  RequestSamplingParam sampling_param_;
  StoppingChecker stopping_checker_;
  SequenceParams seq_params_;
  torch::Tensor input_embedding_;
  MMData mm_data_;
  std::vector<KVCache> kv_caches_;

  // Sequences managed by test class
  std::vector<Sequence> sequences_;

  // Create a sequence in decode phase
  IncrementalDecoder fake_decoder_ = IncrementalDecoder("", 1, false, false);
};

// Test that ACL graph executor produces same results as eager execution
TEST_F(AclGraphExecutorTest, GraphExecutorVsEagerExecution) {
  // Create test batch
  auto batch = CreateTestBatch();
  ASSERT_FALSE(batch->empty());

  // Prepare forward input
  auto forward_input = batch->prepare_forward_input(
      options_.num_decoding_tokens(), 0, model_args_);
  forward_input = forward_input.to(*device_, torch::kFloat32);

  std::cout << "forward_input.token_ids: " << forward_input.token_ids
            << std::endl;
  std::cout << "forward_input.positions: " << forward_input.positions
            << std::endl;
  std::cout << "forward_input.input_params.attention.device.q_seq_lens: "
            << forward_input.input_params.attention.device.q_seq_lens
            << std::endl;
  std::cout << "forward_input.input_params.attention.device.kv_seq_lens: "
            << forward_input.input_params.attention.device.kv_seq_lens
            << std::endl;
  std::cout << "forward_input.input_params.attention.device.new_cache_slots: "
            << forward_input.input_params.attention.device.new_cache_slots
            << std::endl;
  std::cout << "forward_input.input_params.attention.device.block_tables: "
            << forward_input.input_params.attention.device.block_tables
            << std::endl;
  // Test eager execution (direct model forward)
  auto eager_model_output = model_->forward({forward_input.token_ids},
                                            {forward_input.positions},
                                            kv_caches_,
                                            {forward_input.input_params});
  auto eager_output = eager_model_output.hidden_states;
  // Create ACL graph executor
  auto graph_executor = std::make_unique<::xllm::npu::AclGraphExecutorImpl>(
      model_.get(), model_args_, *device_, options_);

  // Test graph execution with NPUGraph mempool optimization
  auto graph_model_output = graph_executor->run({forward_input.token_ids},
                                                {forward_input.positions},
                                                kv_caches_,
                                                {forward_input.input_params});
  auto graph_output = graph_model_output.hidden_states;
  // Compare outputs - should be identical
  EXPECT_TRUE(
      torch::allclose(eager_output, graph_output, /*rtol=*/1e-5, /*atol=*/1e-6))
      << "Eager output:\n"
      << eager_output << "\nGraph output:\n"
      << graph_output;
}

// Test that graph replay produces consistent results across multiple runs
TEST_F(AclGraphExecutorTest, GraphReplayConsistency) {
  // Create test batch
  auto batch = CreateTestBatch();
  ASSERT_FALSE(batch->empty());

  // Prepare forward input
  auto forward_input = batch->prepare_forward_input(
      options_.num_decoding_tokens(), 0, model_args_);
  forward_input = forward_input.to(*device_, torch::kFloat32);

  // Create ACL graph executor
  auto graph_executor = std::make_unique<::xllm::npu::AclGraphExecutorImpl>(
      model_.get(), model_args_, *device_, options_);

  // First execution (should create graph with NPUGraph mempool)
  auto output1 = graph_executor->run({forward_input.token_ids},
                                     {forward_input.positions},
                                     kv_caches_,
                                     {forward_input.input_params});

  // Second execution (should replay graph using mempool-managed tensors)
  auto output2 = graph_executor->run({forward_input.token_ids},
                                     {forward_input.positions},
                                     kv_caches_,
                                     {forward_input.input_params});

  // Compare outputs - should be identical
  EXPECT_TRUE(torch::allclose(output1.hidden_states,
                              output2.hidden_states,
                              /*rtol=*/1e-5,
                              /*atol=*/1e-6))
      << "First output:\n"
      << output1.hidden_states << "\nSecond output:\n"
      << output2.hidden_states;
}

TEST_F(AclGraphExecutorTest, PreservesAuxHiddenStatesAcrossGraphReplay) {
  auto batch = CreateTestBatch();
  ASSERT_FALSE(batch->empty());

  auto forward_input =
      batch->prepare_forward_input(options_.num_decoding_tokens(),
                                   /*min_decoding_batch_size=*/0,
                                   model_args_);
  forward_input = forward_input.to(*device_, /*dtype=*/torch::kFloat32);

  SimpleCausalLM* simple_model = dynamic_cast<SimpleCausalLM*>(model_.get());
  ASSERT_NE(simple_model, nullptr);
  simple_model->set_return_aux_hidden_states(/*value=*/true);
  options_.enable_graph_aux_hidden_states(/*value=*/true);

  ModelOutput eager_output = model_->forward({forward_input.token_ids},
                                             {forward_input.positions},
                                             kv_caches_,
                                             {forward_input.input_params});
  auto graph_executor = std::make_unique<::xllm::npu::AclGraphExecutorImpl>(
      model_.get(), model_args_, *device_, options_);
  const double eager_fallbacks_before =
      COUNTER_num_model_execution_total_eager.get_value();
  auto run_graph = [&]() {
    return graph_executor->run({forward_input.token_ids},
                               {forward_input.positions},
                               kv_caches_,
                               {forward_input.input_params});
  };
  ModelOutput capture_output = run_graph();
  for (int32_t slot_idx = 1;
       slot_idx < graph_executor->graph_slot_count_for_test();
       ++slot_idx) {
    run_graph();
  }
  ModelOutput replay_output = run_graph();
  EXPECT_EQ(COUNTER_num_model_execution_total_eager.get_value(),
            eager_fallbacks_before)
      << "ACL graph aux-hidden test unexpectedly fell back to eager";

  ASSERT_TRUE(eager_output.aux_hidden_states.defined());
  ASSERT_TRUE(capture_output.aux_hidden_states.defined());
  ASSERT_TRUE(replay_output.aux_hidden_states.defined());
  EXPECT_EQ(capture_output.aux_hidden_states.size(/*dim=*/-1),
            model_args_.hidden_size() * 2);
  EXPECT_TRUE(torch::allclose(eager_output.aux_hidden_states,
                              capture_output.aux_hidden_states,
                              /*rtol=*/1e-5,
                              /*atol=*/1e-6));
  EXPECT_TRUE(torch::allclose(eager_output.aux_hidden_states,
                              replay_output.aux_hidden_states,
                              /*rtol=*/1e-5,
                              /*atol=*/1e-6));
}

TEST(DeepseekV4ModelTest, ReturnsPreHcHiddenStatesForMtp) {
  ModelArgs model_args;
  model_args.model_type("deepseek_v4");
  model_args.dtype("float32");
  model_args.vocab_size(8);
  model_args.hidden_size(2);
  model_args.n_layers(0);
  model_args.n_heads(1);
  model_args.hc_mult(2);
  model_args.hc_eps(0.1f);
  model_args.rms_norm_eps(1e-5f);
  model_args.window_size(128);
  model_args.max_position_embeddings(0);
  model_args.index_head_dim(0);
  model_args.qk_rope_head_dim(0);
  model_args.o_lora_rank(0);
  model_args.num_speculative_tokens(1);

  const torch::Device device("npu:0");
  const torch::TensorOptions tensor_options =
      torch::dtype(torch::kFloat32).device(device);
  layer::test::MockProcessGroup process_group(
      device, /*rank=*/0, /*world_size=*/1);
  ParallelArgs parallel_args(
      /*rank=*/0, /*world_size=*/1, &process_group);
  parallel_args.tp_group_ = &process_group;
  const ModelContext context(
      parallel_args, model_args, QuantArgs(), tensor_options);
  const CausalLMFactory factory =
      ModelRegistry::get_causallm_factory("deepseek_v4");
  ASSERT_TRUE(static_cast<bool>(factory));
  std::unique_ptr<CausalLM> model = factory(context);
  ASSERT_NE(model, nullptr);

  const torch::Tensor pre_hc_hidden =
      torch::tensor({1.0f, 2.0f, 3.0f, 4.0f}, tensor_options).view({1, 2, 2});
  const torch::Tensor tokens =
      torch::zeros({1}, torch::dtype(torch::kInt).device(device));
  const torch::Tensor positions =
      torch::zeros({1}, torch::dtype(torch::kInt).device(device));
  ModelInputParams input_params;
  input_params.meta.num_sequences = 1;
  input_params.embedding.input_embedding = pre_hc_hidden;
  input_params.attn_metadata = std::make_shared<layer::AttentionMetadata>();
  std::vector<KVCache> kv_caches;

  const ModelOutput output =
      model->forward(tokens, positions, kv_caches, input_params);

  ASSERT_TRUE(output.aux_hidden_states.defined());
  EXPECT_TRUE(torch::equal(output.aux_hidden_states,
                           pre_hc_hidden.flatten(/*start_dim=*/1)));
  ASSERT_TRUE(output.hidden_states.defined());
  EXPECT_EQ(output.hidden_states.sizes(), torch::IntArrayRef({1, 2}));
  EXPECT_EQ(output.aux_hidden_states.sizes(), torch::IntArrayRef({1, 4}));
}

// Test graph creation and execution with different batch sizes
TEST_F(AclGraphExecutorTest, DifferentBatchSizes) {
  // Test with different batch sizes to ensure graph creation works
  const std::vector<uint32_t> batch_sizes = {1, 2, 4};

  for (auto batch_size : batch_sizes) {
    // Clear sequences from previous iteration to avoid block exhaustion
    sequences_.clear();

    // Create multiple sequences for larger batch sizes
    auto batch = std::make_unique<Batch>();

    for (uint32_t i = 0; i < batch_size; ++i) {
      sequences_.emplace_back(i,
                              std::vector<int32_t>{static_cast<int32_t>(1 + i),
                                                   static_cast<int32_t>(3 + i),
                                                   static_cast<int32_t>(5 + i),
                                                   static_cast<int32_t>(7 + i)},
                              input_embedding_,
                              mm_data_,
                              fake_decoder_,
                              seq_params_);
      auto& sequence = sequences_.back();
      sequence.add_blocks(BlockType::KV, block_manager_->allocate(2));
      std::cout << "batch_size: " << batch_size << " i: " << i
                << " sequence.kv_state().current_max_tokens_capacity(): "
                << sequence.kv_state().current_max_tokens_capacity()
                << std::endl;
      // Set kv_cache_tokens_num to be >= num_prompt_tokens to move to decode
      // stage
      sequence.kv_state().incr_kv_cache_tokens_num(
          /*size=*/4);  // 4 prompt tokens

      sequence.append_token(100 + i);
      // Add sequence pointer to batch (batch doesn't own sequence)
      batch->add(&sequence);
    }

    // Prepare forward input
    auto forward_input = batch->prepare_forward_input(
        options_.num_decoding_tokens(), 0, model_args_);
    forward_input = forward_input.to(*device_, torch::kFloat32);
    // Create ACL graph executor
    auto graph_executor = new ::xllm::npu::AclGraphExecutorImpl(
        model_.get(), model_args_, *device_, options_);

    // Test graph execution
    auto output = graph_executor->run({forward_input.token_ids},
                                      {forward_input.positions},
                                      kv_caches_,
                                      {forward_input.input_params});

    // Verify output shape
    EXPECT_EQ(output.hidden_states.size(0),
              batch_size * options_.num_decoding_tokens())
        << "Batch size: " << batch_size;
    EXPECT_EQ(output.hidden_states.size(1), model_args_.hidden_size())
        << "Batch size: " << batch_size;
  }
}

// Test decode batch-size threshold fallback: ACL graph should fall back to
// eager when batch_size exceeds the configured limit (default: 16).
TEST_F(AclGraphExecutorTest, DecodeBatchSizeThresholdFallsBackToEager) {
  constexpr uint32_t batch_size = 17;
  sequences_.clear();
  auto batch = std::make_unique<Batch>();

  for (uint32_t i = 0; i < batch_size; ++i) {
    sequences_.emplace_back(i,
                            std::vector<int32_t>{static_cast<int32_t>(1 + i),
                                                 static_cast<int32_t>(3 + i),
                                                 static_cast<int32_t>(5 + i),
                                                 static_cast<int32_t>(7 + i)},
                            input_embedding_,
                            mm_data_,
                            fake_decoder_,
                            seq_params_);
    auto& sequence = sequences_.back();
    sequence.add_blocks(BlockType::KV, block_manager_->allocate(2));
    sequence.kv_state().incr_kv_cache_tokens_num(/*size=*/4);
    sequence.append_token(100 + i);
    batch->add(&sequence);
  }

  auto forward_input = batch->prepare_forward_input(
      options_.num_decoding_tokens(), 0, model_args_);
  forward_input = forward_input.to(*device_, torch::kFloat32);

  auto npu_executor = std::make_unique<BaseExecutorImpl>(
      model_.get(), model_args_, *device_, options_);
  auto graph_executor = std::make_unique<::xllm::npu::AclGraphExecutorImpl>(
      model_.get(), model_args_, *device_, options_);

  auto eager_out = npu_executor->run({forward_input.token_ids},
                                     {forward_input.positions},
                                     kv_caches_,
                                     {forward_input.input_params});
  auto graph_out = graph_executor->run({forward_input.token_ids},
                                       {forward_input.positions},
                                       kv_caches_,
                                       {forward_input.input_params});

  EXPECT_EQ(graph_out.hidden_states.size(0),
            batch_size * options_.num_decoding_tokens());
  EXPECT_EQ(graph_out.hidden_states.size(1), model_args_.hidden_size());
  EXPECT_TRUE(torch::allclose(eager_out.hidden_states,
                              graph_out.hidden_states,
                              /*rtol=*/1e-5,
                              /*atol=*/1e-6));
}

// Test ACL graph executor against original NPU executor implementation
TEST_F(AclGraphExecutorTest, AclGraphExecutorVsBaseExecutorImpl) {
  // Create test batch
  auto batch = CreateTestBatch();
  ASSERT_FALSE(batch->empty());

  // Prepare forward input
  auto forward_input = batch->prepare_forward_input(
      options_.num_decoding_tokens(), 0, model_args_);
  forward_input = forward_input.to(*device_, torch::kFloat32);
  // Test NPU Executor Impl (original implementation)
  auto npu_executor = std::make_unique<BaseExecutorImpl>(
      model_.get(), model_args_, *device_, options_);

  auto npu_model_output = npu_executor->run({forward_input.token_ids},
                                            {forward_input.positions},
                                            kv_caches_,
                                            {forward_input.input_params});
  auto npu_output = npu_model_output.hidden_states;

  // Test ACL Graph Executor with NPUGraph mempool optimization
  auto graph_executor = std::make_unique<::xllm::npu::AclGraphExecutorImpl>(
      model_.get(), model_args_, *device_, options_);

  auto graph_model_output = graph_executor->run({forward_input.token_ids},
                                                {forward_input.positions},
                                                kv_caches_,
                                                {forward_input.input_params});
  auto graph_output = graph_model_output.hidden_states;

  // Compare outputs - should be identical
  EXPECT_TRUE(
      torch::allclose(npu_output, graph_output, /*rtol=*/1e-5, /*atol=*/1e-6))
      << "NPU Executor output:\n"
      << npu_output << "\nACL Graph Executor output:\n"
      << graph_output;

  // Verify output shapes are the same
  EXPECT_EQ(npu_output.sizes(), graph_output.sizes())
      << "Output shape mismatch: NPU=" << npu_output.sizes()
      << ", Graph=" << graph_output.sizes();
}

// Test multiple runs to verify consistency across different execution modes
TEST_F(AclGraphExecutorTest, AclGraphExecutorVsBaseExecutorImplMultipleRuns) {
  // Create test batch
  auto batch = CreateTestBatch();
  ASSERT_FALSE(batch->empty());

  // Prepare forward input
  auto forward_input = batch->prepare_forward_input(
      options_.num_decoding_tokens(), 0, model_args_);
  forward_input = forward_input.to(*device_, torch::kFloat32);
  // Create both executors
  auto npu_executor = std::make_unique<BaseExecutorImpl>(
      model_.get(), model_args_, *device_, options_);
  auto graph_executor = std::make_unique<::xllm::npu::AclGraphExecutorImpl>(
      model_.get(), model_args_, *device_, options_);

  // Run multiple times and compare results
  const int num_runs = 3;
  for (int i = 0; i < num_runs; ++i) {
    // Direct model forward call (baseline)
    auto direct_model_output = model_->forward({forward_input.token_ids},
                                               {forward_input.positions},
                                               kv_caches_,
                                               {forward_input.input_params});
    auto direct_output = direct_model_output.hidden_states;

    // NPU Executor run
    auto npu_model_output = npu_executor->run({forward_input.token_ids},
                                              {forward_input.positions},
                                              kv_caches_,
                                              {forward_input.input_params});
    auto npu_output = npu_model_output.hidden_states;

    // ACL Graph Executor run with NPUGraph mempool
    auto graph_model_output = graph_executor->run({forward_input.token_ids},
                                                  {forward_input.positions},
                                                  kv_caches_,
                                                  {forward_input.input_params});
    auto graph_output = graph_model_output.hidden_states;

    // Compare direct model output with NPU Executor output
    EXPECT_TRUE(torch::allclose(
        direct_output, npu_output, /*rtol=*/1e-5, /*atol=*/1e-6))
        << "Run " << i << " - Direct model vs NPU Executor mismatch:\n"
        << "Direct model output:\n"
        << direct_output << "\nNPU Executor output:\n"
        << npu_output;

    // Compare direct model output with ACL Graph Executor output
    EXPECT_TRUE(torch::allclose(
        direct_output, graph_output, /*rtol=*/1e-5, /*atol=*/1e-6))
        << "Run " << i << " - Direct model vs ACL Graph Executor mismatch:\n"
        << "Direct model output:\n"
        << direct_output << "\nACL Graph Executor output:\n"
        << graph_output;

    // Compare NPU Executor output with ACL Graph Executor output
    EXPECT_TRUE(
        torch::allclose(npu_output, graph_output, /*rtol=*/1e-5, /*atol=*/1e-6))
        << "Run " << i << " - NPU Executor vs ACL Graph Executor mismatch:\n"
        << "NPU Executor output:\n"
        << npu_output << "\nACL Graph Executor output:\n"
        << graph_output;
  }
}

TEST_F(AclGraphExecutorTest, BatchInputCarriesLinearStateIds) {
  model_args_.layer_types({"linear_attention", "full_attention"});
  auto batch = CreateTestBatch();
  ASSERT_FALSE(batch->empty());
  ASSERT_FALSE(sequences_.empty());

  // embedding_ids come from the single_block slot, while linear_state_ids come
  // from the dedicated linear_state_slot slot; the two are decoupled and carry
  // independent ids through transport.
  auto& seq = sequences_.back();
  auto embedding_block = block_manager_->allocate(1);
  ASSERT_EQ(embedding_block.size(), 1);
  const int32_t expected_embedding_id = embedding_block[0].id();
  seq.add_blocks(BlockType::SINGLE, embedding_block);

  auto linear_state_slot = block_manager_->allocate(1);
  ASSERT_EQ(linear_state_slot.size(), 1);
  const int32_t expected_linear_state_id = linear_state_slot[0].id();
  seq.add_blocks(BlockType::LINEAR, linear_state_slot);
  ASSERT_NE(expected_embedding_id, expected_linear_state_id);

  auto forward_input = batch->prepare_forward_input(
      options_.num_decoding_tokens(), 0, model_args_);
  ASSERT_EQ(forward_input.input_params.meta.num_sequences, 1);
  ASSERT_EQ(forward_input.input_params.embedding.linear_state_ids.size(), 1);
  EXPECT_EQ(forward_input.input_params.embedding.linear_state_ids[0],
            expected_linear_state_id);
  ASSERT_EQ(forward_input.input_params.embedding.embedding_ids.size(), 1);
  EXPECT_EQ(forward_input.input_params.embedding.embedding_ids[0],
            expected_embedding_id);

  forward_input.input_params.parallel.has_initial_state = {0};
  forward_input = forward_input.to(*device_, torch::kFloat32);
  npu::GraphPersistentParam persistent_param(model_args_, *device_, options_);
  std::optional<ModelInputParams> params_for_capture = persistent_param.update(
      forward_input.token_ids,
      first_full_attention_cache(kv_caches_).get_k_cache(),
      first_full_attention_cache(kv_caches_).get_v_cache(),
      forward_input.positions,
      forward_input.input_params,
      /*padded_num_tokens=*/2,
      /*return_capture_params=*/true);
  ASSERT_TRUE(params_for_capture.has_value());
  EXPECT_EQ(params_for_capture->meta.num_sequences, 2);
  EXPECT_EQ(
      params_for_capture->embedding.linear_state_ids,
      std::vector<int32_t>({expected_linear_state_id, kPaddingLinearStateId}));
  ASSERT_EQ(params_for_capture->parallel.has_initial_state.size(), 2);
  EXPECT_EQ(params_for_capture->parallel.has_initial_state[0], 0);
  EXPECT_EQ(params_for_capture->parallel.has_initial_state[1], 0);
}

TEST(LinearStateRestoreTest, RestoreCopiesCheckpointSlot) {
  std::vector<KVCache> kv_caches;
  torch::Tensor conv_cache = torch::zeros({3, 2, 2}, torch::kFloat32);
  torch::Tensor ssm_cache = torch::zeros({3, 2, 2}, torch::kFloat32);
  conv_cache.select(0, 1).fill_(7.0);
  ssm_cache.select(0, 1).fill_(11.0);
  kv_caches.emplace_back(LinearAttentionKVCacheTensors{conv_cache, ssm_cache});

  LinearStateCacheOp restore;
  restore.linear_state_id = 2;
  restore.restore_requested = true;
  restore.restore_src_slot_id = 1;
  std::vector<int64_t> has_initial_state(1, 0);
  restore_linear_state_slots(kv_caches, {restore}, has_initial_state);

  EXPECT_EQ(has_initial_state[0], 1);
  EXPECT_TRUE(
      torch::allclose(conv_cache.select(0, 2), conv_cache.select(0, 1)));
  EXPECT_TRUE(torch::allclose(ssm_cache.select(0, 2), ssm_cache.select(0, 1)));
}

TEST(LinearStateRestoreTest, RestoreRequestedButUnresolvedColdStarts) {
  std::vector<KVCache> kv_caches;
  kv_caches.emplace_back(LinearAttentionKVCacheTensors{
      torch::zeros({2, 4, 3}, torch::dtype(torch::kFloat32)),
      torch::zeros({2, 4, 4}, torch::dtype(torch::kFloat32))});

  LinearStateCacheOp no_restore;
  no_restore.linear_state_id = 1;
  LinearStateCacheOp missing_restore;
  missing_restore.linear_state_id = 1;
  missing_restore.restore_requested = true;
  missing_restore.restore_src_slot_id = -1;

  const std::vector<LinearStateCacheOp> cache_ops = {no_restore,
                                                     missing_restore};
  // Pre-fill both entries with the kv-cache default of 1 so the test can
  // distinguish "SKIPPED leaves the default untouched" from "COLD_START
  // overrides it to 0".
  std::vector<int64_t> has_initial_state = {1, 1};
  restore_linear_state_slots(kv_caches, cache_ops, has_initial_state);
  EXPECT_EQ(has_initial_state[0], 1);
  EXPECT_EQ(has_initial_state[1], 0);
}

TEST_F(AclGraphExecutorTest, GraphDoubleBufferFlagControlsSlotCount) {
  ExecutionConfig& execution_config = ExecutionConfig::get_instance();
  const bool original_enable_graph_double_buffer =
      execution_config.enable_graph_double_buffer();

  execution_config.enable_graph_double_buffer(true);
  std::unique_ptr<::xllm::npu::AclGraphExecutorImpl> double_buffer_executor =
      std::make_unique<::xllm::npu::AclGraphExecutorImpl>(
          model_.get(), model_args_, *device_, options_);
  EXPECT_EQ(double_buffer_executor->graph_slot_count_for_test(), 2);

  execution_config.enable_graph_double_buffer(false);
  std::unique_ptr<::xllm::npu::AclGraphExecutorImpl> single_buffer_executor =
      std::make_unique<::xllm::npu::AclGraphExecutorImpl>(
          model_.get(), model_args_, *device_, options_);
  EXPECT_EQ(single_buffer_executor->graph_slot_count_for_test(), 1);

  execution_config.enable_graph_double_buffer(
      original_enable_graph_double_buffer);
}

TEST(AclGraphPersistentParamTest, SpecVerifyMetadataUsesTokenCapacity) {
  SpeculativeConfig& speculative_config = SpeculativeConfig::get_instance();
  const bool original_enable_atb_spec_kernel =
      speculative_config.enable_atb_spec_kernel();
  speculative_config.enable_atb_spec_kernel(false);

  ModelArgs args;
  args.model_type("deepseek_v4");
  args.dtype("float32");
  args.hidden_size(8);
  args.max_position_embeddings(32);

  runtime::Options options;
  options.block_size(4);
  options.max_seqs_per_batch(10);
  options.max_tokens_per_batch(64);
  options.num_decoding_tokens(3);
  options.enable_speculative_decode(true);
  options.is_draft_engine(false);

  const torch::Device device("npu:0");
  ::xllm::npu::GraphPersistentParam persistent_param(
      args,
      device,
      options,
      /*need_update_attn_mask=*/false,
      /*is_hybrid_linear_attention=*/false);
  EXPECT_EQ(persistent_param.q_seq_lens().size(0), 30);
  EXPECT_EQ(persistent_param.kv_seq_lens().size(0), 30);
  EXPECT_EQ(persistent_param.persistent_block_tables().size(0), 30);

  constexpr int64_t kValidateRows = 12;
  const torch::TensorOptions int_options =
      torch::dtype(torch::kInt).device(device);
  const torch::Tensor tokens = torch::arange(kValidateRows, int_options);
  const torch::Tensor positions = torch::arange(kValidateRows, int_options);
  ModelInputParams params;
  params.is_spec_verify = true;
  params.meta.batch_forward_type = BatchForwardType::DECODE;
  params.meta.num_sequences = kValidateRows;
  params.attention.host.q_seq_lens.assign(kValidateRows, 1);
  params.attention.host.kv_seq_lens.assign(kValidateRows, 8);
  params.attention.device.q_seq_lens =
      torch::ones({kValidateRows}, int_options);
  params.attention.device.kv_seq_lens =
      torch::full({kValidateRows}, 8, int_options);
  params.attention.device.new_cache_slots =
      torch::zeros({kValidateRows}, int_options);
  params.attention.device.block_tables =
      torch::zeros({kValidateRows, 2}, int_options);

  std::optional<ModelInputParams> capture_params;
  EXPECT_NO_THROW(capture_params = persistent_param.update(
                      tokens,
                      torch::Tensor(),
                      torch::Tensor(),
                      positions,
                      params,
                      /*padded_num_tokens=*/kValidateRows,
                      /*return_capture_params=*/true));
  EXPECT_TRUE(capture_params.has_value());
  if (capture_params.has_value()) {
    EXPECT_EQ(capture_params->attention.device.q_seq_lens.size(0),
              kValidateRows);
    EXPECT_EQ(capture_params->attention.device.block_tables.size(0),
              kValidateRows);
  }

  ::xllm::npu::GraphPersistentParam hybrid_persistent_param(
      args,
      device,
      options,
      /*need_update_attn_mask=*/false,
      /*is_hybrid_linear_attention=*/true);
  EXPECT_EQ(hybrid_persistent_param.q_seq_lens().size(0), 10);
  EXPECT_EQ(hybrid_persistent_param.persistent_block_tables().size(0), 10);

  speculative_config.enable_atb_spec_kernel(original_enable_atb_spec_kernel);
}

TEST(AclGraphPersistentParamTest, AuxHiddenStatesUseGraphTokenCapacity) {
  SpeculativeConfig& speculative_config = SpeculativeConfig::get_instance();
  const bool original_enable_atb_spec_kernel =
      speculative_config.enable_atb_spec_kernel();
  speculative_config.enable_atb_spec_kernel(false);

  ModelArgs args;
  args.model_type("deepseek_v4");
  args.dtype("float32");
  args.hidden_size(8);
  args.max_position_embeddings(32);

  runtime::Options options;
  options.block_size(4);
  options.max_seqs_per_batch(10);
  options.max_tokens_per_batch(64);
  options.num_decoding_tokens(3);
  options.enable_speculative_decode(true);
  options.is_draft_engine(false);

  const torch::Device device("npu:0");
  const torch::TensorOptions tensor_options =
      torch::dtype(torch::kFloat32).device(device);
  const torch::Tensor aux_hidden_states = torch::ones({12, 16}, tensor_options);

  ::xllm::npu::GraphPersistentParam target_param(args, device, options);
  target_param.set_aux_hidden_states(aux_hidden_states);
  EXPECT_EQ(target_param.aux_hidden_states().size(0), 30);

  options.is_draft_engine(true);
  ::xllm::npu::GraphPersistentParam draft_param(args, device, options);
  draft_param.set_aux_hidden_states(aux_hidden_states.slice(
      /*dim=*/0, /*start=*/0, /*end=*/options.max_seqs_per_batch()));
  EXPECT_EQ(draft_param.aux_hidden_states().size(0), 10);

  speculative_config.enable_atb_spec_kernel(original_enable_atb_spec_kernel);
}

TEST(SpeculativeConfigTest, MtpAlgorithmClassificationIsCaseInsensitive) {
  EXPECT_TRUE(SpeculativeConfig::is_mtp_algorithm("MTP"));
  EXPECT_TRUE(SpeculativeConfig::is_mtp_algorithm("mtp"));
  EXPECT_TRUE(SpeculativeConfig::is_mtp_algorithm("Mtp"));
  EXPECT_TRUE(SpeculativeConfig::is_mtp_algorithm("mTp"));

  EXPECT_FALSE(SpeculativeConfig::is_mtp_algorithm("Eagle3"));
  EXPECT_FALSE(SpeculativeConfig::is_mtp_algorithm("DFlash"));
  EXPECT_FALSE(SpeculativeConfig::is_mtp_algorithm("Suffix"));
  EXPECT_FALSE(SpeculativeConfig::is_mtp_algorithm("unknown"));
}

TEST(AclGraphExecutorHybridTest, KvCacheSupportsLinearOnlyLayers) {
  auto conv_cache = torch::zeros({4, 32, 3}, torch::dtype(torch::kFloat32));
  auto ssm_cache = torch::zeros({4, 8, 64, 64}, torch::dtype(torch::kFloat32));
  KVCache linear_only_cache(
      LinearAttentionKVCacheTensors{conv_cache, ssm_cache});

  EXPECT_FALSE(linear_only_cache.empty());
  EXPECT_FALSE(linear_only_cache.get_conv_cache().defined() == false);
  EXPECT_FALSE(linear_only_cache.get_ssm_cache().defined() == false);
  EXPECT_FALSE(linear_only_cache.get_k_cache().defined());
  EXPECT_FALSE(linear_only_cache.get_v_cache().defined());
}

TEST(AclGraphExecutorHybridTest, ModelArgsCountsHybridLayerTypes) {
  ModelArgs args;
  args.n_layers(4);
  args.layer_types(
      {"linear_attention", "full_attention", "linear_attention", "attention"});

  EXPECT_FALSE(is_full_attention_layer(args, 0));
  EXPECT_TRUE(is_full_attention_layer(args, 1));
  EXPECT_FALSE(is_full_attention_layer(args, 2));
  EXPECT_TRUE(is_full_attention_layer(args, 3));
  EXPECT_TRUE(has_linear_attention_layers(args));
}

TEST_F(AclGraphExecutorTest, GraphExecutorUsesFirstFullAttentionKvCache) {
  auto batch = CreateTestBatch();
  ASSERT_FALSE(batch->empty());

  auto forward_input = batch->prepare_forward_input(
      options_.num_decoding_tokens(), 0, model_args_);
  forward_input = forward_input.to(*device_, torch::kFloat32);

  auto conv_cache =
      torch::zeros({4, 32, 3}, torch::dtype(torch::kFloat32).device(*device_));
  auto ssm_cache = torch::zeros({4, 8, 64, 64},
                                torch::dtype(torch::kFloat32).device(*device_));
  auto full_k = torch::randn({1000, 4 * model_args_.hidden_size()},
                             torch::dtype(torch::kFloat32).device(*device_));
  auto full_v = full_k.clone();

  std::vector<KVCache> hybrid_kv_caches;
  hybrid_kv_caches.emplace_back(
      LinearAttentionKVCacheTensors{conv_cache, ssm_cache});
  hybrid_kv_caches.emplace_back(KVCacheTensors{full_k, full_v});

  auto eager_model_output = model_->forward({forward_input.token_ids},
                                            {forward_input.positions},
                                            hybrid_kv_caches,
                                            {forward_input.input_params});
  auto graph_executor = std::make_unique<::xllm::npu::AclGraphExecutorImpl>(
      model_.get(), model_args_, *device_, options_);
  auto graph_model_output = graph_executor->run({forward_input.token_ids},
                                                {forward_input.positions},
                                                hybrid_kv_caches,
                                                {forward_input.input_params});

  EXPECT_TRUE(torch::allclose(eager_model_output.hidden_states,
                              graph_model_output.hidden_states,
                              /*rtol=*/1e-5,
                              /*atol=*/1e-6));
}

}  // namespace xllm
