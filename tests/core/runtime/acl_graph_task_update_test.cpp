/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <torch_npu/torch_npu.h>

#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "core/framework/batch/batch.h"
#include "core/framework/block/block.h"
#include "core/framework/block/block_manager_impl.h"
#include "core/framework/config/execution_config.h"
#include "core/framework/kv_cache/kv_cache.h"
#include "core/framework/kv_cache/linear_state_restore.h"
#include "core/framework/model/model_args.h"
#include "core/framework/model/model_output.h"
#include "core/framework/model_loader.h"
#include "core/framework/request/sequence.h"
#include "core/framework/request/stopping_checker.h"
#include "core/framework/sampling/sampling_params.h"
#include "core/kernels/ops_api.h"
#include "core/layers/common/attention_metadata_builder.h"
#include "core/layers/common/expanded_decode_metadata_builder.h"
#include "core/layers/npu/npu_lm_head_impl.h"
#include "core/layers/npu/npu_word_embedding_impl.h"
#include "core/layers/npu_torch/attention.h"
#include "core/platform/npu/acl_graph_task_update_context.h"
#include "core/platform/stream.h"
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
constexpr int64_t kBlockSize = 128;
constexpr int64_t kAttentionNumHeads = 8;
constexpr int64_t kAttentionNumKvHeads = 1;
constexpr int64_t kAttentionHeadDim = 256;
constexpr double kAttentionScale = 1.0 / 16.0;

constexpr torch::ScalarType kDtype = torch::kFloat16;

}  // namespace

class HybridConv1dMockLM final : public CausalLM {
 public:
  HybridConv1dMockLM(const ModelArgs& args,
                     const torch::Device& device,
                     bool enable_fia_decode = true,
                     int32_t attention_repetitions = 1)
      : args_(args),
        device_(device),
        attention_repetitions_(attention_repetitions) {
    CHECK_GT(attention_repetitions_, 0);
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

    if (enable_fia_decode) {
      attention_ =
          register_module("attention",
                          layer::Attention(kAttentionNumHeads,
                                           kAttentionHeadDim,
                                           kAttentionScale,
                                           kAttentionNumKvHeads,
                                           /*sliding_window=*/-1,
                                           /*enable_fia_decode=*/true));
    } else {
      attention_ = register_module("attention",
                                   layer::Attention(kAttentionNumHeads,
                                                    kAttentionHeadDim,
                                                    kAttentionScale,
                                                    kAttentionNumKvHeads,
                                                    /*sliding_window=*/-1));
    }

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

    layer::AttentionMetadataBuildOptions metadata_build_options;
    metadata_build_options.materialize_linear_state_validity =
        !params.enable_graph;
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
        task.capture_order = graph_context->next_capture_order++;
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

    for (auto& kv_cache : kv_caches) {
      if (kv_cache.empty() || !kv_cache.get_k_cache().defined()) {
        continue;
      }

      for (int32_t attention_index = 0;
           attention_index < attention_repetitions_;
           ++attention_index) {
        layer::AttentionMetadata attn_metadata =
            layer::AttentionMetadataBuilder::build(params,
                                                   /*enable_mla=*/false,
                                                   /*attn_mask=*/std::nullopt,
                                                   device_,
                                                   metadata_build_options);
        torch::Tensor query = hidden.to(torch::kBFloat16).contiguous();
        torch::Tensor key = query
                                .slice(/*dim=*/1,
                                       /*start=*/0,
                                       kAttentionNumKvHeads * kAttentionHeadDim)
                                .contiguous();
        torch::Tensor value = key.clone();
        torch::Tensor attention_output = std::get<0>(
            attention_->forward(attn_metadata, query, key, value, kv_cache));
        hidden = hidden + attention_output.to(hidden.scalar_type());
      }

      if (register_graph_task) {
        saw_causal_conv_graph_task_ |=
            !graph_context->causal_conv1d_tasks.empty();
        saw_fia_graph_task_ |=
            !graph_context->fused_infer_attention_tasks.empty();
        fia_graph_task_count_ =
            graph_context->fused_infer_attention_tasks.size();
        if (!graph_context->fused_infer_attention_tasks.empty()) {
          captured_fia_batch_size_ = static_cast<uint32_t>(
              graph_context->fused_infer_attention_tasks.front().query.size(0));
        }
        all_fia_graph_tasks_share_workspace_ = fia_graph_task_count_ > 1;
        for (size_t task_index = 1; task_index < fia_graph_task_count_;
             ++task_index) {
          all_fia_graph_tasks_share_workspace_ &=
              graph_context->fused_infer_attention_tasks[task_index]
                  .workspace.data_ptr() ==
              graph_context->fused_infer_attention_tasks.front()
                  .workspace.data_ptr();
        }
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
  bool saw_causal_conv_and_fia_graph_tasks() const {
    return saw_causal_conv_graph_task_ && saw_fia_graph_task_;
  }
  bool saw_causal_conv_graph_task() const {
    return saw_causal_conv_graph_task_;
  }
  bool saw_fia_graph_task() const { return saw_fia_graph_task_; }
  uint32_t captured_fia_batch_size() const { return captured_fia_batch_size_; }
  size_t fia_graph_task_count() const { return fia_graph_task_count_; }
  bool all_fia_graph_tasks_share_workspace() const {
    return all_fia_graph_tasks_share_workspace_;
  }
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
  layer::Attention attention_{nullptr};
  torch::Tensor conv_weight_;
  torch::Tensor token_embedding_table_;
  torch::Tensor pos_embedding_table_;
  int32_t attention_repetitions_ = 1;
  bool saw_causal_conv_graph_task_ = false;
  bool saw_fia_graph_task_ = false;
  uint32_t captured_fia_batch_size_ = 0;
  size_t fia_graph_task_count_ = 0;
  bool all_fia_graph_tasks_share_workspace_ = false;
};

class AclGraphTaskUpdateTest : public ::testing::Test {
 protected:
  void SetUp() override {
    sequences_.reserve(100);

    auto& execution_config = ExecutionConfig::get_instance();
    original_enable_graph_ = execution_config.enable_graph();
    original_enable_graph_double_buffer_ =
        execution_config.enable_graph_double_buffer();
    original_enable_graph_mode_decode_no_padding_ =
        execution_config.enable_graph_mode_decode_no_padding();
    original_acl_graph_decode_batch_size_limit_ =
        execution_config.acl_graph_decode_batch_size_limit();
    execution_config.enable_graph(true);
    execution_config.enable_graph_double_buffer(false);
    execution_config.enable_graph_mode_decode_no_padding(false);
    execution_config.acl_graph_decode_batch_size_limit(32);

    model_args_.model_type("test_hybrid_model");
    model_args_.dtype("float16");
    model_args_.hidden_size(kHiddenSize);
    model_args_.max_position_embeddings(kMaxSeqLen);
    model_args_.vocab_size(kVocabSize);
    model_args_.n_layers(2);
    model_args_.n_heads(kAttentionNumHeads);
    model_args_.n_kv_heads(kAttentionNumKvHeads);
    model_args_.head_dim(kAttentionHeadDim);
    model_args_.layer_types({"linear_attention", "full_attention"});

    device_ = std::make_unique<torch::Device>("npu:0");
    options_.num_decoding_tokens(1);
    options_.block_size(kBlockSize);
    options_.max_seqs_per_batch(32);

    model_ = std::make_unique<HybridConv1dMockLM>(model_args_, *device_);

    BlockManager::Options block_options;
    block_options.num_blocks(kNumBlocks).block_size(kBlockSize);
    block_manager_ = std::make_unique<BlockManagerImpl>(block_options);

    sampling_param_.frequency_penalty = 0.0f;
    stopping_checker_.set_max_generated_tokens(20);

    seq_params_.seq_capacity = kMaxSeqLen;
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

  void TearDown() override {
    reset_sequences();
    auto& execution_config = ExecutionConfig::get_instance();
    execution_config.enable_graph(original_enable_graph_);
    execution_config.enable_graph_double_buffer(
        original_enable_graph_double_buffer_);
    execution_config.enable_graph_mode_decode_no_padding(
        original_enable_graph_mode_decode_no_padding_);
    execution_config.acl_graph_decode_batch_size_limit(
        original_acl_graph_decode_batch_size_limit_);
  }

  void reset_sequences() {
    for (auto& sequence : sequences_) {
      auto kv_blocks = sequence.kv_state().blocks(BlockType::KV);
      if (!kv_blocks.empty()) {
        block_manager_->deallocate(kv_blocks);
      }
      auto linear_blocks = sequence.kv_state().blocks(BlockType::LINEAR);
      if (!linear_blocks.empty()) {
        block_manager_->deallocate(linear_blocks);
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

    auto k_cache = torch::zeros(
        {kNumBlocks, kBlockSize, kAttentionNumKvHeads, kAttentionHeadDim},
        torch::dtype(torch::kBFloat16).device(*device_));
    auto v_cache = torch::zeros_like(k_cache);
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
      sequence.add_blocks(BlockType::LINEAR, linear_state_block);
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
      sequence.add_blocks(BlockType::LINEAR, linear_state_block);
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

  std::vector<std::vector<int32_t>> create_mixed_boundary_prompts(
      uint32_t batch_size,
      int32_t token_seed) {
    const std::vector<int64_t> prompt_lengths = {126, 127, 128};
    std::vector<std::vector<int32_t>> prompts;
    prompts.reserve(batch_size);
    for (uint32_t batch_index = 0; batch_index < batch_size; ++batch_index) {
      const int64_t prompt_length =
          prompt_lengths[batch_index % prompt_lengths.size()];
      std::vector<int32_t> prompt;
      prompt.reserve(static_cast<size_t>(prompt_length));
      for (int64_t token_index = 0; token_index < prompt_length;
           ++token_index) {
        prompt.emplace_back((token_seed + static_cast<int32_t>(batch_index) +
                             static_cast<int32_t>(token_index)) %
                            kVocabSize);
      }
      prompts.emplace_back(std::move(prompt));
    }
    return prompts;
  }

  void expect_fia_padding_replay_matches_eager(uint32_t capture_batch_size,
                                               uint32_t replay_batch_size,
                                               uint32_t expected_bucket) {
    auto capture_prompts =
        create_mixed_boundary_prompts(capture_batch_size, /*token_seed=*/10);
    auto capture_batch =
        create_decode_batch_with_prompts(capture_prompts, /*token_seed=*/100);
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
    ASSERT_TRUE(model_->saw_causal_conv_and_fia_graph_tasks());
    EXPECT_EQ(model_->captured_fia_batch_size(), expected_bucket);

    reset_sequences();
    auto replay_prompts =
        create_mixed_boundary_prompts(replay_batch_size, /*token_seed=*/200);
    auto replay_batch =
        create_decode_batch_with_prompts(replay_prompts, /*token_seed=*/300);
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

    const int64_t real_tokens = static_cast<int64_t>(replay_batch_size);
    ASSERT_EQ(graph_out.hidden_states.size(0), real_tokens);
    torch::Tensor graph_real =
        graph_out.hidden_states.slice(0, 0, real_tokens).to(torch::kFloat32);
    torch::Tensor eager_real =
        eager_out.hidden_states.slice(0, 0, real_tokens).to(torch::kFloat32);
    EXPECT_TRUE(torch::allclose(eager_real,
                                graph_real,
                                /*rtol=*/1e-2,
                                /*atol=*/1e-2))
        << "FIA padding replay mismatch for capture_bs=" << capture_batch_size
        << ", replay_bs=" << replay_batch_size << ", bucket=" << expected_bucket
        << ", max_abs_diff="
        << (eager_real - graph_real).abs().max().item<float>();
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
    fi.input_params.linear_state_validity_mask = build_linear_state_mask(
        fi.input_params.attention.host.kv_cache_tokens_nums, num_sequences);

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

    std::vector<int32_t> expanded_kv_vec;
    expanded_kv_vec.reserve(static_cast<size_t>(total_tokens));
    for (int32_t s = 0; s < num_sequences; ++s) {
      int32_t kv_len =
          fi.input_params.attention.host.kv_seq_lens[static_cast<size_t>(s)];
      for (int32_t t = 0; t < num_spec_tokens; ++t) {
        expanded_kv_vec.push_back(kv_len + t + 1);
      }
    }
    auto expanded_kv_seq_lens =
        torch::tensor(expanded_kv_vec, torch::kInt32).to(*device_);

    torch::Tensor host_block_tables =
        fi.input_params.attention.host.block_tables.contiguous();
    auto host_block_table_accessor = host_block_tables.accessor<int32_t, 2>();
    std::vector<int32_t> expanded_cache_slots;
    expanded_cache_slots.reserve(static_cast<size_t>(total_tokens));
    for (int32_t sequence_index = 0; sequence_index < num_sequences;
         ++sequence_index) {
      const int32_t kv_len =
          fi.input_params.attention.host
              .kv_seq_lens[static_cast<size_t>(sequence_index)];
      for (int32_t token_index = 0; token_index < num_spec_tokens;
           ++token_index) {
        const int32_t position = kv_len + token_index;
        const int32_t logical_block = position / kBlockSize;
        const int32_t block_offset = position % kBlockSize;
        const int32_t physical_block =
            host_block_table_accessor[sequence_index][logical_block];
        expanded_cache_slots.emplace_back(physical_block * kBlockSize +
                                          block_offset);
      }
    }
    fi.input_params.attention.host.new_cache_slots =
        std::move(expanded_cache_slots);

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
    layer::ExpandedDecodeMetadataBuilder::populate_expanded_layout(
        fi.input_params,
        expanded_kv_seq_lens,
        expanded_bt,
        expanded_kv_vec,
        kBlockSize);

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
  bool original_enable_graph_ = false;
  bool original_enable_graph_double_buffer_ = true;
  bool original_enable_graph_mode_decode_no_padding_ = false;
  int32_t original_acl_graph_decode_batch_size_limit_ = 16;
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
       MultipleFiaInvocationsShareWorkspaceWithinBucket) {
  auto shared_workspace_model =
      std::make_unique<HybridConv1dMockLM>(model_args_,
                                           *device_,
                                           /*enable_fia_decode=*/true,
                                           /*attention_repetitions=*/2);
  auto batch = create_decode_batch(/*batch_size=*/2);
  ASSERT_FALSE(batch->empty());

  auto forward_input = batch->prepare_forward_input(
      options_.num_decoding_tokens(), 0, model_args_);
  forward_input = forward_input.to(*device_, kDtype);
  populate_query_start_loc(forward_input.input_params);

  auto kv_eager = create_hybrid_kv_caches();
  auto eager_out =
      shared_workspace_model->forward({forward_input.token_ids},
                                      {forward_input.positions},
                                      kv_eager,
                                      {forward_input.input_params});

  auto kv_graph = create_hybrid_kv_caches();
  auto graph_exec = std::make_unique<npu::AclGraphExecutorImpl>(
      shared_workspace_model.get(), model_args_, *device_, options_);
  auto graph_out = graph_exec->run({forward_input.token_ids},
                                   {forward_input.positions},
                                   kv_graph,
                                   {forward_input.input_params});

  EXPECT_EQ(shared_workspace_model->fia_graph_task_count(), 2);
  EXPECT_TRUE(shared_workspace_model->all_fia_graph_tasks_share_workspace());
  EXPECT_TRUE(torch::allclose(eager_out.hidden_states.to(torch::kFloat32),
                              graph_out.hidden_states.to(torch::kFloat32),
                              /*rtol=*/1e-2,
                              /*atol=*/1e-2))
      << "Shared-workspace FIA capture/replay must match eager, max_abs_diff="
      << (eager_out.hidden_states.to(torch::kFloat32) -
          graph_out.hidden_states.to(torch::kFloat32))
             .abs()
             .max()
             .item<float>();
}

TEST_F(AclGraphTaskUpdateTest,
       NonQwenDefaultAttentionKeepsPagedAttentionInEagerAndGraph) {
  ModelArgs non_qwen_args = model_args_;
  non_qwen_args.model_type("minimax_m2");
  non_qwen_args.dtype("bfloat16");
  auto non_qwen_model = std::make_unique<HybridConv1dMockLM>(
      non_qwen_args, *device_, /*enable_fia_decode=*/false);

  auto batch = create_decode_batch(/*batch_size=*/2);
  ASSERT_FALSE(batch->empty());
  auto forward_input = batch->prepare_forward_input(
      options_.num_decoding_tokens(), 0, non_qwen_args);
  forward_input = forward_input.to(*device_, kDtype);
  populate_query_start_loc(forward_input.input_params);

  auto kv_eager = create_hybrid_kv_caches();
  auto eager_out = non_qwen_model->forward({forward_input.token_ids},
                                           {forward_input.positions},
                                           kv_eager,
                                           {forward_input.input_params});

  auto kv_graph = create_hybrid_kv_caches();
  auto graph_exec = std::make_unique<npu::AclGraphExecutorImpl>(
      non_qwen_model.get(), non_qwen_args, *device_, options_);
  auto graph_out = graph_exec->run({forward_input.token_ids},
                                   {forward_input.positions},
                                   kv_graph,
                                   {forward_input.input_params});

  EXPECT_TRUE(non_qwen_model->saw_causal_conv_graph_task());
  EXPECT_FALSE(non_qwen_model->saw_fia_graph_task());
  EXPECT_EQ(eager_out.hidden_states.sizes(), graph_out.hidden_states.sizes());
  EXPECT_TRUE(torch::allclose(eager_out.hidden_states.to(torch::kFloat32),
                              graph_out.hidden_states.to(torch::kFloat32),
                              /*rtol=*/1e-2,
                              /*atol=*/1e-2))
      << "Non-Qwen default Attention must keep PA eager/graph behavior, "
      << "max_abs_diff="
      << (eager_out.hidden_states.to(torch::kFloat32) -
          graph_out.hidden_states.to(torch::kFloat32))
             .abs()
             .max()
             .item<float>();
}

TEST_F(AclGraphTaskUpdateTest,
       Qwen35OptOutKeepsPagedAttentionForExpandedSpecVerify) {
  constexpr int32_t kNumSequences = 2;
  constexpr int32_t kNumSpecTokens = 4;
  auto pa_model = std::make_unique<HybridConv1dMockLM>(
      model_args_, *device_, /*enable_fia_decode=*/false);

  auto batch = create_decode_batch(/*batch_size=*/kNumSequences);
  ASSERT_FALSE(batch->empty());
  auto forward_input = batch->prepare_forward_input(
      options_.num_decoding_tokens(), 0, model_args_);
  forward_input = forward_input.to(*device_, kDtype);
  setup_spec_verify_input(forward_input, kNumSequences, kNumSpecTokens);

  auto kv_eager = create_hybrid_kv_caches();
  auto eager_out = pa_model->forward(forward_input.token_ids,
                                     forward_input.positions,
                                     kv_eager,
                                     forward_input.input_params);

  auto kv_graph = create_hybrid_kv_caches();
  auto graph_exec = std::make_unique<npu::AclGraphExecutorImpl>(
      pa_model.get(), model_args_, *device_, options_);
  auto graph_out = graph_exec->run(forward_input.token_ids,
                                   forward_input.positions,
                                   kv_graph,
                                   forward_input.input_params);

  EXPECT_TRUE(pa_model->saw_causal_conv_graph_task());
  EXPECT_FALSE(pa_model->saw_fia_graph_task());
  EXPECT_EQ(eager_out.hidden_states.sizes(), graph_out.hidden_states.sizes());
  EXPECT_TRUE(torch::isfinite(eager_out.hidden_states).all().item<bool>());
  EXPECT_TRUE(torch::isfinite(graph_out.hidden_states).all().item<bool>());
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
  expect_fia_padding_replay_matches_eager(
      /*capture_batch_size=*/4, /*replay_batch_size=*/3, /*expected_bucket=*/4);
}

TEST_F(AclGraphTaskUpdateTest, FiaPaddingReplaysAcrossBucketEight) {
  expect_fia_padding_replay_matches_eager(
      /*capture_batch_size=*/5, /*replay_batch_size=*/7, /*expected_bucket=*/8);
}

TEST_F(AclGraphTaskUpdateTest, FiaPaddingReplaysAcrossBucketThirtyTwo) {
  expect_fia_padding_replay_matches_eager(/*capture_batch_size=*/17,
                                          /*replay_batch_size=*/17,
                                          /*expected_bucket=*/32);
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

TEST_F(AclGraphTaskUpdateTest, StaticMtpPrepareReplaysFiaSpecVerifyWidths) {
  ExecutionConfig::get_instance().enable_graph_mode_decode_no_padding(true);

  for (const int32_t spec_width : std::vector<int32_t>{4, 5, 6}) {
    SCOPED_TRACE("spec_width=" + std::to_string(spec_width));
    reset_sequences();
    auto fia_model =
        std::make_unique<HybridConv1dMockLM>(model_args_, *device_);
    auto batch = create_decode_batch(/*batch_size=*/1);
    ASSERT_FALSE(batch->empty());

    auto forward_input = batch->prepare_forward_input(
        options_.num_decoding_tokens(), 0, model_args_);
    forward_input = forward_input.to(*device_, kDtype);
    setup_spec_verify_input(forward_input, /*num_sequences=*/1, spec_width);
    forward_input.input_params.graph.input_tokens_override =
        forward_input.token_ids;
    forward_input.input_params.graph.spec_verify_source_addresses_stable = true;
    forward_input.input_params.meta.is_graph_warmup = true;

    auto kv_graph = create_hybrid_kv_caches();
    auto graph_exec = std::make_unique<npu::AclGraphExecutorImpl>(
        fia_model.get(), model_args_, *device_, options_);
    graph_exec->run(forward_input.token_ids,
                    forward_input.positions,
                    kv_graph,
                    forward_input.input_params);
    ASSERT_TRUE(fia_model->saw_causal_conv_and_fia_graph_tasks());

    auto kv_eager = clone_kv_caches(kv_graph);
    forward_input.input_params.meta.is_graph_warmup = false;
    const auto& expanded_kv_seq_lens =
        forward_input.input_params.graph.expanded_kv_seq_lens_vec;
    ASSERT_EQ(expanded_kv_seq_lens.size(), static_cast<size_t>(spec_width));
    const SpecVerifyGraphTaskSignal signal{
        .linear_state_id =
            forward_input.input_params.embedding.linear_state_ids.front(),
        .num_accepted_tokens =
            forward_input.input_params.num_accepted_tokens_host.front(),
        .spec_width = spec_width,
        .block_table_width =
            forward_input.input_params.attention.device.block_tables.size(1),
        .base_kv_seq_len = expanded_kv_seq_lens.front(),
        .max_kv_seq_len = expanded_kv_seq_lens.back(),
    };

    Stream signal_stream(*device_);
    if (spec_width == 5) {
      SpecVerifyGraphTaskSignal mismatched_signal = signal;
      ++mismatched_signal.num_accepted_tokens;
      EXPECT_FALSE(graph_exec->prepare_static_mtp_graph_tasks(mismatched_signal,
                                                              signal_stream));

      mismatched_signal = signal;
      ++mismatched_signal.max_kv_seq_len;
      EXPECT_FALSE(graph_exec->prepare_static_mtp_graph_tasks(mismatched_signal,
                                                              signal_stream));
    }

    ASSERT_TRUE(
        graph_exec->prepare_static_mtp_graph_tasks(signal, signal_stream));
    forward_input.input_params.graph.spec_verify_static_graph_tasks_prepared =
        true;
    auto graph_out = graph_exec->run(forward_input.token_ids,
                                     forward_input.positions,
                                     kv_graph,
                                     forward_input.input_params);
    auto eager_out = fia_model->forward(forward_input.token_ids,
                                        forward_input.positions,
                                        kv_eager,
                                        forward_input.input_params);

    EXPECT_EQ(eager_out.hidden_states.sizes(), graph_out.hidden_states.sizes());
    EXPECT_TRUE(torch::allclose(eager_out.hidden_states.to(torch::kFloat32),
                                graph_out.hidden_states.to(torch::kFloat32),
                                /*rtol=*/1e-2,
                                /*atol=*/1e-2))
        << "Static MTP FIA prepare must match eager replay, max_abs_diff="
        << (eager_out.hidden_states.to(torch::kFloat32) -
            graph_out.hidden_states.to(torch::kFloat32))
               .abs()
               .max()
               .item<float>();
  }
}

TEST_F(AclGraphTaskUpdateTest, StaticMtpPrepareRejectsUnsupportedWidth) {
  const SpecVerifyGraphTaskSignal signal{
      .linear_state_id = 1,
      .num_accepted_tokens = 1,
      .spec_width = 3,
      .block_table_width = 1,
      .base_kv_seq_len = 4,
      .max_kv_seq_len = 6,
  };
  EXPECT_FALSE(std::make_unique<npu::AclGraphExecutorImpl>(
                   model_.get(), model_args_, *device_, options_)
                   ->prepare_static_mtp_graph_tasks(signal, Stream(*device_)));
}

}  // namespace xllm
