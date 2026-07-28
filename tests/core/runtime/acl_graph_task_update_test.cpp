/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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
#include <vector>

#include "core/framework/batch/batch.h"
#include "core/framework/block/block.h"
#include "core/framework/block/block_manager_impl.h"
#include "core/framework/kv_cache/kv_cache.h"
#include "core/framework/model/model_args.h"
#include "core/framework/model/model_output.h"
#include "core/framework/model_loader.h"
#include "core/framework/request/sequence.h"
#include "core/framework/request/stopping_checker.h"
#include "core/framework/sampling/sampling_params.h"
#include "core/kernels/ops_api.h"
#include "core/layers/npu/npu_lm_head_impl.h"
#include "core/layers/npu/npu_word_embedding_impl.h"
#include "core/platform/npu/acl_graph_task_update_context.h"
#include "core/runtime/acl_graph_executor_impl.h"
#include "core/runtime/base_executor_impl.h"
#include "core/runtime/options.h"
#include "tests/npu_test_environment.h"

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wattributes"
#endif

#include "torch_npu/csrc/core/npu/NPUEvent.h"
#include "torch_npu/csrc/core/npu/NPUGraph.h"

#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

class AclGraphTaskUpdateTestEnvironment : public ::testing::Environment {
 public:
  void SetUp() override {
    xllm::testing::init_npu_test_runtime();

    google::InitGoogleLogging("acl_graph_task_update_test");
    google::SetStderrLogging(google::INFO);
    int ret = aclrtSetDevice(0);
    if (ret != 0) {
      LOG(ERROR) << "ACL set device id: 0 failed, ret:" << ret;
    }
    torch_npu::init_npu("npu:0");
  }

  void TearDown() override {
    google::ShutdownGoogleLogging();
    torch_npu::finalize_npu();
    aclrtResetDevice(0);
    aclFinalize();

    xllm::testing::finalize_npu_test_runtime();
  }
};

::testing::Environment* const task_update_test_env =
    ::testing::AddGlobalTestEnvironment(new AclGraphTaskUpdateTestEnvironment);

namespace xllm {

namespace {

constexpr int64_t kConvKernelSize = 4;
constexpr int64_t kConvChannels = 2048;
constexpr int64_t kHiddenSize = 2048;
constexpr int64_t kMaxSeqLen = 256;
constexpr int64_t kVocabSize = 1000;
constexpr int64_t kNumBlocks = 100;
constexpr int64_t kBlockSize = 4;

constexpr torch::ScalarType kDtype = torch::kFloat16;

}  // namespace

class HybridConv1dMockLM final : public CausalLM {
 public:
  HybridConv1dMockLM(const ModelArgs& args, const torch::Device& device)
      : args_(args), device_(device) {
    linear_ = register_module(
        "linear",
        torch::nn::Linear(torch::nn::LinearOptions(kHiddenSize, kHiddenSize)));

    conv_weight_ =
        register_parameter("conv_weight",
                           torch::randn({kConvKernelSize, kConvChannels},
                                        torch::dtype(kDtype).device(device)));

    token_embedding_table_ =
        register_parameter("token_embedding",
                           torch::randn({kVocabSize, kHiddenSize},
                                        torch::dtype(kDtype).device(device)));

    pos_embedding_table_ =
        register_parameter("pos_embedding",
                           torch::randn({kMaxSeqLen, kHiddenSize},
                                        torch::dtype(kDtype).device(device)));

    this->to(device);
  }

  ModelOutput forward(const torch::Tensor& tokens,
                      const torch::Tensor& positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelInputParams& params) override {
    const int64_t num_tokens = tokens.size(0);
    auto token_emb = torch::embedding(token_embedding_table_, tokens);
    auto pos_emb = torch::embedding(pos_embedding_table_, positions);
    auto hidden = token_emb + pos_emb;

    auto graph_context = params.graph.acl_graph_task_update_context;
    const bool register_graph_task =
        graph_context != nullptr && graph_context->capturing;

    for (auto& kv_cache : kv_caches) {
      if (kv_cache.empty() || !kv_cache.get_conv_cache().defined()) {
        continue;
      }
      torch::Tensor conv_cache = kv_cache.get_conv_cache();
      torch::Tensor conv_input =
          hidden.slice(/*dim=*/1, 0, kConvChannels).contiguous();

      const bool is_spec_verify = params.is_spec_verify;
      const std::vector<int64_t> empty_host_args;
      const std::vector<int64_t> cache_indices(
          params.embedding.linear_state_ids.begin(),
          params.embedding.linear_state_ids.end());
      const auto& nat_ref =
          is_spec_verify ? params.num_accepted_tokens_host : empty_host_args;

      if (register_graph_task) {
        const auto branch = is_spec_verify
                                ? npu::CausalConv1dGraphBranch::kSpecVerify
                                : npu::CausalConv1dGraphBranch::kDecode;

        torch::Tensor conv_output = torch::empty_like(conv_input);
        c10_npu::NPUStream stream = c10_npu::getCurrentNPUStream();
        auto event = std::make_shared<c10_npu::NPUEvent>(ACL_EVENT_EXTERNAL);
        event->block(stream);
        event->reset(stream);

        c10_npu::graph_task_group_begin(stream);
        xllm::kernel::causal_conv1d_out(
            conv_output,
            conv_input,
            conv_weight_,
            conv_cache,
            std::optional<torch::Tensor>(),
            torch::IntArrayRef(params.parallel.query_start_loc),
            torch::IntArrayRef(cache_indices),
            torch::IntArrayRef(empty_host_args),
            torch::IntArrayRef(nat_ref),
            /*activation_mode=*/npu::kCausalConv1dActivationSilu,
            /*pad_slot_id=*/npu::kCausalConv1dGraphPadSlotId,
            /*run_mode=*/npu::kCausalConv1dRunModeUpdate);
        c10_npu::NPUTaskGroupHandle handle =
            c10_npu::graph_task_group_end(stream);

        npu::CausalConv1dGraphTask task;
        task.output = conv_output;
        task.x = conv_input;
        task.weight = conv_weight_;
        task.conv_state = conv_cache;
        task.bias = std::nullopt;
        task.activation_mode = npu::kCausalConv1dActivationSilu;
        task.pad_slot_id = npu::kCausalConv1dGraphPadSlotId;
        task.run_mode = npu::kCausalConv1dRunModeUpdate;
        task.branch = branch;
        task.handle = handle;
        task.event = std::move(event);
        graph_context->causal_conv1d_tasks.emplace_back(std::move(task));

        auto conv_proj =
            torch::zeros({num_tokens, kHiddenSize}, conv_input.options());
        conv_proj.slice(/*dim=*/1, 0, kConvChannels).copy_(conv_output);
        hidden = hidden + conv_proj;
      } else {
        torch::Tensor conv_output = torch::empty_like(conv_input);
        xllm::kernel::causal_conv1d_out(
            conv_output,
            conv_input,
            conv_weight_,
            conv_cache,
            std::optional<torch::Tensor>(),
            torch::IntArrayRef(params.parallel.query_start_loc),
            torch::IntArrayRef(cache_indices),
            torch::IntArrayRef(empty_host_args),
            torch::IntArrayRef(nat_ref),
            /*activation_mode=*/npu::kCausalConv1dActivationSilu,
            /*pad_slot_id=*/npu::kCausalConv1dGraphPadSlotId,
            /*run_mode=*/npu::kCausalConv1dRunModeUpdate);

        auto conv_proj =
            torch::zeros({num_tokens, kHiddenSize}, conv_input.options());
        conv_proj.slice(/*dim=*/1, 0, kConvChannels).copy_(conv_output);
        hidden = hidden + conv_proj;
      }
      break;
    }

    hidden = linear_->forward(hidden);

    return ModelOutput(hidden);
  }

  bool is_hybrid_linear_attention() override { return true; }

  const torch::TensorOptions& options() const override {
    static torch::TensorOptions opts = torch::dtype(kDtype).device(device_);
    return opts;
  }

  torch::Tensor logits(const torch::Tensor& hidden_states,
                       const torch::Tensor& selected_idxes) override {
    return torch::randn({hidden_states.size(0), kVocabSize},
                        torch::dtype(kDtype).device(device_));
  }

  void load_model(std::unique_ptr<ModelLoader> loader) override {}
  torch::Device device() const override { return device_; }
  void prepare_expert_weight(int32_t, const std::vector<int32_t>&) override {}
  void update_expert_weight(int32_t) override {}
  layer::NpuLmHead get_npu_lm_head() override {
    return layer::NpuLmHead(nullptr);
  }
  void set_npu_lm_head(layer::NpuLmHead&) override {}
  layer::NpuWordEmbedding get_npu_word_embedding() override {
    return layer::NpuWordEmbedding(nullptr);
  }
  void set_npu_word_embedding(layer::NpuWordEmbedding&) override {}

 private:
  ModelArgs args_;
  torch::Device device_;
  torch::nn::Linear linear_{nullptr};
  torch::Tensor conv_weight_;
  torch::Tensor token_embedding_table_;
  torch::Tensor pos_embedding_table_;
};

class AclGraphTaskUpdateTest : public ::testing::Test {
 protected:
  void SetUp() override {
    sequences_.reserve(100);

    model_args_.model_type("test_hybrid_model");
    model_args_.dtype("float16");
    model_args_.hidden_size(kHiddenSize);
    model_args_.max_position_embeddings(kMaxSeqLen);
    model_args_.vocab_size(kVocabSize);
    model_args_.n_layers(2);
    model_args_.layer_types({"linear_attention", "full_attention"});

    device_ = std::make_unique<torch::Device>("npu:0");
    options_.num_decoding_tokens(1);
    options_.block_size(kBlockSize);
    options_.max_seqs_per_batch(16);

    model_ = std::make_unique<HybridConv1dMockLM>(model_args_, *device_);

    BlockManager::Options block_options;
    block_options.num_blocks(kNumBlocks).block_size(kBlockSize);
    block_manager_ = std::make_unique<BlockManagerImpl>(block_options);

    sampling_param_.frequency_penalty = 0.0f;
    stopping_checker_.set_max_generated_tokens(20);

    seq_params_.seq_capacity = 100;
    seq_params_.stopping_checker = &stopping_checker_;
    seq_params_.sampling_param = &sampling_param_;
    seq_params_.skip_special_tokens = true;
    seq_params_.echo = false;
    seq_params_.logprobs = false;
    seq_params_.enable_schedule_overlap = false;

    input_embedding_ =
        torch::zeros({1, kHiddenSize}, torch::dtype(kDtype).device(*device_));
    mm_data_ = MMData();
  }

  void TearDown() override {}

  void reset_sequences() {
    for (auto& sequence : sequences_) {
      auto kv_blocks = sequence.kv_state().blocks(BlockType::KV);
      if (!kv_blocks.empty()) {
        block_manager_->deallocate(kv_blocks);
      }
      auto single_blocks = sequence.kv_state().blocks(BlockType::SINGLE);
      if (!single_blocks.empty()) {
        block_manager_->deallocate(single_blocks);
      }
    }
    sequences_.clear();
  }

  std::vector<KVCache> create_hybrid_kv_caches() {
    std::vector<KVCache> kv_caches;
    auto conv_cache =
        torch::zeros({kNumBlocks, kConvKernelSize - 1, kConvChannels},
                     torch::dtype(kDtype).device(*device_));
    auto ssm_cache = torch::zeros({kNumBlocks, 8, 64, 64},
                                  torch::dtype(kDtype).device(*device_));
    kv_caches.emplace_back(
        LinearAttentionKVCacheTensors{conv_cache, ssm_cache});

    auto k_cache = torch::randn({kNumBlocks, kBlockSize * kHiddenSize},
                                torch::dtype(kDtype).device(*device_));
    auto v_cache = k_cache.clone();
    kv_caches.emplace_back(KVCacheTensors{k_cache, v_cache});
    return kv_caches;
  }

  std::vector<KVCache> clone_kv_caches(const std::vector<KVCache>& src) {
    std::vector<KVCache> cloned;
    cloned.emplace_back(LinearAttentionKVCacheTensors{
        src[0].get_conv_cache().clone(), src[0].get_ssm_cache().clone()});
    cloned.emplace_back(KVCacheTensors{src[1].get_k_cache().clone(),
                                       src[1].get_v_cache().clone()});
    return cloned;
  }

  void populate_query_start_loc(ModelInputParams& params) {
    const auto& q_seq_lens = params.attention.host.q_seq_lens;
    params.parallel.query_start_loc.clear();
    params.parallel.query_start_loc.reserve(q_seq_lens.size() + 1);
    params.parallel.query_start_loc.push_back(0);
    for (auto len : q_seq_lens) {
      params.parallel.query_start_loc.push_back(
          params.parallel.query_start_loc.back() + len);
    }
  }

  std::unique_ptr<Batch> create_decode_batch(uint32_t batch_size,
                                             int32_t token_seed = 100) {
    auto batch = std::make_unique<Batch>();
    for (uint32_t i = 0; i < batch_size; ++i) {
      sequences_.emplace_back(i,
                              std::vector<int32_t>{1, 3, 5, 7},
                              input_embedding_,
                              mm_data_,
                              fake_decoder_,
                              seq_params_);
      auto& sequence = sequences_.back();

      auto linear_state_block = block_manager_->allocate(1);
      sequence.add_blocks(BlockType::SINGLE, linear_state_block);
      sequence.add_blocks(BlockType::KV, block_manager_->allocate(2));
      sequence.kv_state().incr_kv_cache_tokens_num(4);
      sequence.append_token(token_seed + static_cast<int32_t>(i));
      batch->add(&sequence);
    }
    return batch;
  }

  std::unique_ptr<Batch> create_decode_batch_with_prompts(
      const std::vector<std::vector<int32_t>>& prompts,
      int32_t token_seed = 100) {
    auto batch = std::make_unique<Batch>();
    for (size_t i = 0; i < prompts.size(); ++i) {
      const auto& prompt = prompts[i];
      int64_t prompt_len = static_cast<int64_t>(prompt.size());
      int64_t total_len = prompt_len + 1;
      int64_t num_kv_blocks = (total_len + kBlockSize - 1) / kBlockSize;

      sequences_.emplace_back(
          i, prompt, input_embedding_, mm_data_, fake_decoder_, seq_params_);
      auto& sequence = sequences_.back();

      auto linear_state_block = block_manager_->allocate(1);
      sequence.add_blocks(BlockType::SINGLE, linear_state_block);
      sequence.add_blocks(
          BlockType::KV,
          block_manager_->allocate(static_cast<size_t>(num_kv_blocks)));
      sequence.kv_state().incr_kv_cache_tokens_num(
          static_cast<size_t>(prompt_len));
      sequence.append_token(token_seed + static_cast<int32_t>(i));
      batch->add(&sequence);
    }
    return batch;
  }

  void setup_spec_verify_input(ForwardInput& fi,
                               int32_t num_sequences,
                               int32_t num_spec_tokens) {
    int32_t total_tokens = num_sequences * num_spec_tokens;

    fi.input_params.meta.batch_forward_type =
        BatchForwardType(BatchForwardType::CHUNKED_PREFILL);
    fi.input_params.meta.q_max_seq_len = num_spec_tokens;
    fi.input_params.meta.num_sequences = num_sequences;
    fi.input_params.is_spec_verify = true;

    fi.input_params.attention.host.q_seq_lens.assign(
        static_cast<size_t>(num_sequences), num_spec_tokens);

    fi.input_params.num_accepted_tokens_host.assign(
        static_cast<size_t>(num_sequences), 1);
    fi.input_params.num_accepted_tokens = torch::ones(
        {num_sequences}, torch::dtype(torch::kInt32).device(*device_));

    std::vector<int32_t> token_ids_vec;
    std::vector<int32_t> positions_vec;
    token_ids_vec.reserve(static_cast<size_t>(total_tokens));
    positions_vec.reserve(static_cast<size_t>(total_tokens));
    for (int32_t s = 0; s < num_sequences; ++s) {
      int32_t kv_len =
          fi.input_params.attention.host.kv_seq_lens[static_cast<size_t>(s)];
      for (int32_t t = 0; t < num_spec_tokens; ++t) {
        token_ids_vec.push_back(50 + s * num_spec_tokens + t);
        positions_vec.push_back(kv_len + t);
      }
    }
    fi.token_ids = torch::tensor(token_ids_vec, torch::kInt32).to(*device_);
    fi.positions = torch::tensor(positions_vec, torch::kInt32).to(*device_);

    fi.input_params.graph.use_expanded_decode_for_spec_verify_attention = true;
    std::vector<int32_t> expanded_kv_vec;
    expanded_kv_vec.reserve(static_cast<size_t>(total_tokens));
    for (int32_t s = 0; s < num_sequences; ++s) {
      int32_t kv_len =
          fi.input_params.attention.host.kv_seq_lens[static_cast<size_t>(s)];
      for (int32_t t = 0; t < num_spec_tokens; ++t) {
        expanded_kv_vec.push_back(kv_len + t + 1);
      }
    }
    fi.input_params.graph.expanded_kv_seq_lens_vec = expanded_kv_vec;
    fi.input_params.graph.expanded_kv_seq_lens =
        torch::tensor(expanded_kv_vec, torch::kInt32).to(*device_);

    auto block_tables = fi.input_params.attention.device.block_tables;
    int64_t block_table_stride = block_tables.size(1);
    auto expanded_bt =
        torch::zeros({static_cast<int64_t>(total_tokens), block_table_stride},
                     block_tables.options());
    for (int32_t s = 0; s < num_sequences; ++s) {
      for (int32_t t = 0; t < num_spec_tokens; ++t) {
        expanded_bt[s * num_spec_tokens + t] = block_tables[s];
      }
    }
    fi.input_params.graph.expanded_block_tables = expanded_bt;

    std::vector<int32_t> q_cu_vec;
    q_cu_vec.reserve(static_cast<size_t>(num_sequences + 1));
    q_cu_vec.push_back(0);
    for (int32_t s = 0; s < num_sequences; ++s) {
      q_cu_vec.push_back(q_cu_vec.back() + num_spec_tokens);
    }
    fi.input_params.attention.host.q_cu_seq_lens = q_cu_vec;

    populate_query_start_loc(fi.input_params);

    fi.input_params.attention.rebuild_device_buffer(*device_);
  }

  ModelArgs model_args_;
  std::unique_ptr<torch::Device> device_;
  runtime::Options options_;
  std::unique_ptr<HybridConv1dMockLM> model_;
  std::unique_ptr<BlockManagerImpl> block_manager_;
  RequestSamplingParam sampling_param_;
  StoppingChecker stopping_checker_;
  SequenceParams seq_params_;
  torch::Tensor input_embedding_;
  MMData mm_data_;
  std::vector<Sequence> sequences_;
  IncrementalDecoder fake_decoder_ = IncrementalDecoder("", 1, false, false);
};

TEST_F(AclGraphTaskUpdateTest, CaptureReplayVsEagerDecodeBranch) {
  auto batch = create_decode_batch(/*batch_size=*/2);
  ASSERT_FALSE(batch->empty());

  auto forward_input = batch->prepare_forward_input(
      options_.num_decoding_tokens(), 0, model_args_);
  forward_input = forward_input.to(*device_, kDtype);
  populate_query_start_loc(forward_input.input_params);

  auto kv_eager = create_hybrid_kv_caches();
  auto eager_out = model_->forward({forward_input.token_ids},
                                   {forward_input.positions},
                                   kv_eager,
                                   {forward_input.input_params});

  auto kv_graph = create_hybrid_kv_caches();
  auto graph_exec = std::make_unique<npu::AclGraphExecutorImpl>(
      model_.get(), model_args_, *device_, options_);
  auto graph_out = graph_exec->run({forward_input.token_ids},
                                   {forward_input.positions},
                                   kv_graph,
                                   {forward_input.input_params});

  EXPECT_EQ(eager_out.hidden_states.sizes(), graph_out.hidden_states.sizes());
  EXPECT_TRUE(torch::allclose(eager_out.hidden_states.to(torch::kFloat32),
                              graph_out.hidden_states.to(torch::kFloat32),
                              /*rtol=*/1e-2,
                              /*atol=*/1e-2))
      << "Decode branch: eager vs graph mismatch";
}

TEST_F(AclGraphTaskUpdateTest,
       ReplayWithDifferentParamsProducesDifferentOutputs) {
  std::vector<std::vector<int32_t>> prompts_run1 = {{1, 3, 5, 7}, {2, 4, 6, 8}};
  auto batch1 =
      create_decode_batch_with_prompts(prompts_run1, /*token_seed=*/100);
  auto fi1 = batch1->prepare_forward_input(
      options_.num_decoding_tokens(), 0, model_args_);
  fi1 = fi1.to(*device_, kDtype);
  populate_query_start_loc(fi1.input_params);

  auto kv_graph = create_hybrid_kv_caches();
  auto graph_exec = std::make_unique<npu::AclGraphExecutorImpl>(
      model_.get(), model_args_, *device_, options_);

  auto out1 = graph_exec->run(
      {fi1.token_ids}, {fi1.positions}, kv_graph, {fi1.input_params});

  auto kv_eager1 = create_hybrid_kv_caches();
  auto eager1 = model_->forward(
      {fi1.token_ids}, {fi1.positions}, kv_eager1, {fi1.input_params});
  EXPECT_TRUE(torch::allclose(out1.hidden_states.to(torch::kFloat32),
                              eager1.hidden_states.to(torch::kFloat32),
                              /*rtol=*/1e-2,
                              /*atol=*/1e-2))
      << "Run 1: graph vs eager mismatch";

  reset_sequences();
  std::vector<std::vector<int32_t>> prompts_run2 = {
      {10, 20, 30, 40, 50, 60, 70, 80}, {11, 22}};
  auto batch2 =
      create_decode_batch_with_prompts(prompts_run2, /*token_seed=*/200);
  auto fi2 = batch2->prepare_forward_input(
      options_.num_decoding_tokens(), 0, model_args_);
  fi2 = fi2.to(*device_, kDtype);
  populate_query_start_loc(fi2.input_params);

  EXPECT_NE(fi1.input_params.embedding.linear_state_ids,
            fi2.input_params.embedding.linear_state_ids)
      << "linear_state_ids (cache_indices) must differ between runs";

  auto out1_saved = out1.hidden_states.clone();
  auto out2 = graph_exec->run(
      {fi2.token_ids}, {fi2.positions}, kv_graph, {fi2.input_params});

  EXPECT_FALSE(torch::allclose(out1_saved.to(torch::kFloat32),
                               out2.hidden_states.to(torch::kFloat32),
                               /*rtol=*/1e-2,
                               /*atol=*/1e-2))
      << "Task update failed: different params produced identical outputs";
}

TEST_F(AclGraphTaskUpdateTest, PaddingBatchDoesNotPolluteRealSequences) {
  constexpr uint32_t kCaptureBatchSize = 4;
  constexpr uint32_t kReplayBatchSize = 3;

  auto capture_batch = create_decode_batch(kCaptureBatchSize);
  auto capture_fi = capture_batch->prepare_forward_input(
      options_.num_decoding_tokens(), 0, model_args_);
  capture_fi = capture_fi.to(*device_, kDtype);
  populate_query_start_loc(capture_fi.input_params);

  auto kv_graph = create_hybrid_kv_caches();
  auto graph_exec = std::make_unique<npu::AclGraphExecutorImpl>(
      model_.get(), model_args_, *device_, options_);
  graph_exec->run({capture_fi.token_ids},
                  {capture_fi.positions},
                  kv_graph,
                  {capture_fi.input_params});

  reset_sequences();

  auto replay_batch = create_decode_batch(kReplayBatchSize);
  auto replay_fi = replay_batch->prepare_forward_input(
      options_.num_decoding_tokens(), 0, model_args_);
  replay_fi = replay_fi.to(*device_, kDtype);
  populate_query_start_loc(replay_fi.input_params);

  auto kv_eager = clone_kv_caches(kv_graph);

  auto graph_out = graph_exec->run({replay_fi.token_ids},
                                   {replay_fi.positions},
                                   kv_graph,
                                   {replay_fi.input_params});
  auto eager_out = model_->forward({replay_fi.token_ids},
                                   {replay_fi.positions},
                                   kv_eager,
                                   {replay_fi.input_params});

  const int64_t real_tokens =
      static_cast<int64_t>(kReplayBatchSize) * options_.num_decoding_tokens();
  EXPECT_EQ(graph_out.hidden_states.size(0), real_tokens);

  auto eager_real =
      eager_out.hidden_states.slice(0, 0, real_tokens).to(torch::kFloat32);
  auto graph_real =
      graph_out.hidden_states.slice(0, 0, real_tokens).to(torch::kFloat32);

  auto diff = (eager_real - graph_real).abs();
  float max_diff = diff.max().item<float>();
  float mean_diff = diff.mean().item<float>();
  LOG(INFO) << "Padding test max_diff=" << max_diff
            << " mean_diff=" << mean_diff;

  EXPECT_TRUE(torch::allclose(eager_real,
                              graph_real,
                              /*rtol=*/1e-2,
                              /*atol=*/1e-2))
      << "Padding batch pollutes real sequence output, max_diff=" << max_diff;
}

TEST_F(AclGraphTaskUpdateTest, CaptureReplayVsEagerSpecVerifyBranch) {
  constexpr int32_t kNumSequences = 2;
  constexpr int32_t kNumSpecTokens = 4;

  auto batch = create_decode_batch(/*batch_size=*/kNumSequences);
  ASSERT_FALSE(batch->empty());

  auto fi = batch->prepare_forward_input(
      options_.num_decoding_tokens(), 0, model_args_);
  fi = fi.to(*device_, kDtype);
  setup_spec_verify_input(fi, kNumSequences, kNumSpecTokens);

  ASSERT_TRUE(fi.input_params.is_spec_verify);
  ASSERT_EQ(fi.input_params.num_accepted_tokens_host.size(),
            static_cast<size_t>(kNumSequences));

  auto kv_eager = create_hybrid_kv_caches();
  auto eager_out =
      model_->forward(fi.token_ids, fi.positions, kv_eager, fi.input_params);

  auto kv_graph = create_hybrid_kv_caches();
  auto graph_exec = std::make_unique<npu::AclGraphExecutorImpl>(
      model_.get(), model_args_, *device_, options_);
  auto graph_out =
      graph_exec->run(fi.token_ids, fi.positions, kv_graph, fi.input_params);

  EXPECT_EQ(eager_out.hidden_states.sizes(), graph_out.hidden_states.sizes());
  EXPECT_TRUE(torch::allclose(eager_out.hidden_states.to(torch::kFloat32),
                              graph_out.hidden_states.to(torch::kFloat32),
                              /*rtol=*/1e-2,
                              /*atol=*/1e-2))
      << "Spec-verify branch: eager vs graph mismatch";
}

}  // namespace xllm
