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
#include <torch/torch.h>

#include <cstdint>
#include <vector>

#include "core/framework/config/execution_config.h"
#include "core/framework/config/model_config.h"
#include "core/framework/parallel_state/parallel_args.h"
#include "core/platform/platform.h"
#include "core/runtime/forward_params.h"
#include "core/runtime/options.h"
#include "core/runtime/unified_mtp_executor.h"
#include "core/runtime/unified_mtp_worker_impl.h"
#include "core/runtime/worker.h"
#include "core/util/scope_guard.h"

namespace xllm {
namespace {

TEST(MtpWorkerSelectionTest, UnifiedFlagSelectsPythonMtpWorker) {
  if (Platform::device_count() < 1) {
    GTEST_SKIP() << "An NPU is required for worker construction.";
  }
  const ModelConfig model_config = ModelConfig::get_instance();
  const ExecutionConfig execution_config = ExecutionConfig::get_instance();
  ScopeGuard restore_config([&] {
    ModelConfig::get_instance() = model_config;
    ExecutionConfig::get_instance() = execution_config;
  });
  ModelConfig::get_instance().model_impl("python");
  const ParallelArgs parallel_args(
      /*rank=*/0, /*world_size=*/1, /*process_group=*/nullptr);
  const torch::Device device("npu:0");
  runtime::Options options;
  options.enable_speculative_decode(true).num_speculative_tokens(3);
  ExecutionConfig::get_instance() = ExecutionConfig();
  {
    Worker default_worker(parallel_args, device, options, WorkerType::LLM);
    EXPECT_FALSE(default_worker.task_pipeline_uses_worker_prepare());
  }
  for (const bool unified : {false, true}) {
    ExecutionConfig::get_instance().enable_unified_mtp_graph(unified);
    Worker worker(parallel_args, device, options, WorkerType::LLM);
    EXPECT_EQ(worker.task_pipeline_uses_worker_prepare(), unified);
  }
  options.enable_speculative_decode(false);
  Worker ordinary_worker(parallel_args, device, options, WorkerType::LLM);
  EXPECT_FALSE(ordinary_worker.task_pipeline_uses_worker_prepare());
}

TEST(MtpWorkerSelectionTest, RequestRoutingDoesNotRejectUnsupportedSampling) {
  LlmForwardInput input;
  input.sampling_params.sampling_mode = SamplingMode::GREEDY;
  EXPECT_TRUE(supports_unified_mtp_request(input, false));
  input.sampling_params.logprobs = true;
  input.sampling_params.max_top_logprobs = 5;
  EXPECT_TRUE(supports_unified_mtp_request(input, false));
  EXPECT_FALSE(supports_unified_mtp_request(input, true));
  input.sampling_params.sampling_mode = SamplingMode::MIXED;
  EXPECT_FALSE(supports_unified_mtp_request(input, false));
  input.sampling_params.sampling_mode = SamplingMode::GREEDY;
  input.sampling_params.use_beam_search = true;
  EXPECT_FALSE(supports_unified_mtp_request(input, false));
  input.sampling_params.use_beam_search = false;
  for (torch::Tensor* control : {&input.sampling_params.temperatures,
                                 &input.sampling_params.top_k,
                                 &input.sampling_params.top_p,
                                 &input.sampling_params.frequency_penalties,
                                 &input.sampling_params.presence_penalties,
                                 &input.sampling_params.repetition_penalties,
                                 &input.sampling_params.filter_mask,
                                 &input.sampling_params.filter_bitmask}) {
    *control = torch::ones({1});
    EXPECT_FALSE(supports_unified_mtp_request(input, false));
    *control = torch::Tensor();
  }
  input.json_object_states.emplace_back();
  EXPECT_FALSE(supports_unified_mtp_request(input, false));
  input.json_object_states.clear();
  input.input_params.multi_block_tables.emplace_back(
      torch::zeros({1, 1}, torch::kInt));
  EXPECT_FALSE(supports_unified_mtp_request(input, false));
}

class DecodeMetadataTestWorker final : public UnifiedMtpWorkerImpl {
 public:
  DecodeMetadataTestWorker(const ParallelArgs& parallel_args,
                           const torch::Device& device,
                           const runtime::Options& options)
      : UnifiedMtpWorkerImpl(parallel_args, device, options, WorkerType::LLM) {
    context_.set_model_impl("python");
    target_spec_verify_mode_ =
        mtp_async::TargetSpecVerifyMode::DEEPSEEK_V32_EXPANDED_VERIFY;
  }

  void initialize_target_cache() {
    embedding_cache_ = std::make_unique<EmbeddingCache>(8);
  }

  void stage_tokens(const LlmForwardInput& input, int64_t token) {
    c10::StreamGuard guard = compute_stream_->set_stream_guard();
    const int64_t batch_size =
        static_cast<int64_t>(input.input_params.embedding.embedding_ids.size());
    // Queue arithmetic ahead of D2H, so the Host cannot assume the submitted
    // copy has finished when snapshot_pending_target_tokens is called.
    torch::Tensor work =
        torch::ones({1024, 1024}, torch::dtype(torch::kFloat).device(device_));
    for (int32_t i = 0; i < 8; ++i) {
      work = torch::matmul(work, work) / 1024;
    }
    SampleOutput output;
    output.next_tokens = torch::full(
        {batch_size, 2}, token, torch::dtype(torch::kLong).device(device_));
    output.embeddings = torch::ones(
        {batch_size, 2, 4}, torch::dtype(torch::kFloat).device(device_));
    torch::Tensor host = torch::full(
        {batch_size, 2},
        -777,
        torch::dtype(torch::kLong).device(torch::kCPU).pinned_memory(true));
    host.copy_(output.next_tokens, /*non_blocking=*/true);
    stage_target_context_write(input,
                               output,
                               torch::zeros({batch_size}, torch::kInt),
                               torch::ones({batch_size}, torch::kInt),
                               compute_stream_->record_event(),
                               std::move(host),
                               {},
                               torch::Tensor());
  }

  bool matches_pending(const LlmForwardInput& input) const {
    return pending_target_context_matches(input);
  }

  size_t pending_count() const { return pending_target_context_queue_.size(); }

  torch::Tensor snapshot_tokens() { return snapshot_pending_target_tokens(); }

  torch::Tensor lease_host_tokens(const torch::Tensor& tokens) {
    return acquire_accepted_tokens_host_buffer(tokens);
  }

  const void* next_draft_workspace_identity(const LlmForwardInput& input) {
    const torch::Tensor embeddings =
        torch::empty({input.input_params.meta.num_sequences, 2, 4},
                     torch::dtype(torch::kBFloat16).device(device_));
    return &unified_executor().acquire_draft_workspace(
        input.input_params.meta.num_sequences, embeddings);
  }

  void resolve_decode_context(LlmForwardInput& input) const {
    update_decode_step_input(input,
                             std::vector<EmbeddingCache::DecodeState>(1));
  }

  LlmForwardInput build_verify(const LlmForwardInput& input, bool adaptive) {
    LlmForwardInput verify_input;
    if (adaptive) {
      prepare_validate_inputs(input, verify_input, std::vector<int32_t>{2});
    } else {
      prepare_validate_inputs(input, verify_input);
    }
    CHECK_EQ(prepare_stream_->synchronize(), 0);
    return verify_input;
  }

  LlmForwardInput build_verify(const LlmForwardInput& input,
                               const std::vector<int32_t>& verify_widths) {
    LlmForwardInput verify_input;
    prepare_validate_inputs(input, verify_input, verify_widths);
    CHECK_EQ(prepare_stream_->synchronize(), 0);
    return verify_input;
  }
};

class StateBoundaryTestWorker final : public UnifiedMtpWorkerImpl {
 public:
  using UnifiedMtpWorkerImpl::UnifiedMtpWorkerImpl;

  void seed_state() {
    unified_executor().remember({torch::tensor({41}, torch::kLong),
                                 torch::ones({1, 2, 4}),
                                 torch::tensor({10}, torch::kInt),
                                 torch::tensor({11}, torch::kInt),
                                 {0},
                                 {"request-a"}});
  }

  std::optional<ForwardOutput> dispatch(const LlmForwardInput& input) {
    return UnifiedMtpWorkerImpl::step(input);
  }

 protected:
  std::optional<ForwardOutput> step_decode(
      const LlmForwardInput& input) override {
    ForwardOutput output;
    const auto* state = unified_executor().continuation_for(input);
    if (state != nullptr) {
      output.sample_output.next_tokens = state->accepted_tokens;
    }
    return output;
  }

  std::optional<ForwardOutput> step_prefill(
      const LlmForwardInput& /*input*/) override {
    return ForwardOutput();
  }

  std::optional<ForwardOutput> step_empty(
      const LlmForwardInput& /*input*/) override {
    return std::nullopt;
  }
};

TEST(MtpDeviceStateTest,
     NonDecodeDispatchInvalidatesContinuationBeforeSameRequestReturns) {
  if (Platform::device_count() < 1) {
    GTEST_SKIP() << "An NPU is required to construct the worker.";
  }
  for (const bool overlap : {false, true}) {
    runtime::Options options;
    options.block_size(128).num_speculative_tokens(1).enable_schedule_overlap(
        overlap);
    ParallelArgs args(0, 1, nullptr);
    StateBoundaryTestWorker worker(
        args, torch::Device("npu:0"), options, WorkerType::LLM);
    LlmForwardInput decode;
    decode.token_ids = torch::tensor({41}, torch::kInt);
    decode.input_params.meta.num_sequences = 1;
    decode.input_params.meta.batch_forward_type = BatchForwardType::DECODE;
    decode.input_params.embedding.request_ids = {"request-a"};
    decode.input_params.embedding.embedding_ids = {0};
    for (const BatchForwardType::Value type :
         {BatchForwardType::PREFILL,
          BatchForwardType::CHUNKED_PREFILL,
          BatchForwardType::MIXED,
          BatchForwardType::EMPTY}) {
      worker.seed_state();
      ASSERT_TRUE(worker.dispatch(decode)->sample_output.next_tokens.defined());
      LlmForwardInput intervening = decode.clone();
      intervening.input_params.meta.batch_forward_type = type;
      if (type == BatchForwardType::EMPTY) {
        intervening.input_params.meta.num_sequences = 0;
      }
      worker.dispatch(intervening);
      EXPECT_FALSE(
          worker.dispatch(decode)->sample_output.next_tokens.defined());
    }
  }
}

class MtpDecodeMetadataTest : public ::testing::TestWithParam<int32_t> {
 protected:
  void SetUp() override {
    if (Platform::device_count() < 1) {
      GTEST_SKIP() << "An NPU is required for worker metadata preparation.";
    }
  }

  ParallelArgs parallel_args() const { return parallel_args(GetParam()); }

  ParallelArgs parallel_args(int32_t kv_split_size) const {
    ParallelArgs args(/*rank=*/0, /*world_size=*/2, /*process_group=*/nullptr);
    args.cp_size(1).kv_split_size(kv_split_size);
    return args;
  }

  runtime::Options options() const {
    runtime::Options options;
    options.model_id("glm_dcp_metadata_test")
        .block_size(128)
        .num_speculative_tokens(1)
        .max_seqs_per_batch(2)
        .world_size(2)
        .dp_size(1)
        .cp_size(1);
    return options;
  }

  LlmForwardInput make_input(int32_t position,
                             const torch::Tensor& block_tables) const {
    LlmForwardInput input;
    input.token_ids_host = torch::tensor({42}, torch::kInt);
    input.positions_host = torch::tensor({position}, torch::kInt);
    input.token_ids = input.token_ids_host.to(torch::Device("npu:0"));
    input.positions = input.positions_host.to(torch::Device("npu:0"));
    input.input_params.meta.num_sequences = 1;
    input.input_params.meta.batch_forward_type = BatchForwardType::DECODE;
    input.input_params.attention.host.q_seq_lens = {1};
    input.input_params.attention.host.kv_seq_lens = {position + 1};
    input.input_params.attention.host.block_tables = block_tables;
    return input;
  }

  void check_verify_across_logical_page(bool adaptive) {
    DecodeMetadataTestWorker worker(
        parallel_args(), torch::Device("npu:0"), options());
    const int32_t page_size = GetParam() == 1 ? 128 : 256;
    LlmForwardInput input =
        make_input(page_size - 1, torch::tensor({{10, 11}}, torch::kInt));
    worker.resolve_decode_context(input);
    ASSERT_EQ(input.positions_host.item<int32_t>(), page_size - 1);
    const LlmForwardInput verify_input = worker.build_verify(input, adaptive);
    const auto& params = verify_input.input_params;

    EXPECT_TRUE(
        torch::equal(verify_input.positions_host,
                     torch::tensor({page_size - 1, page_size}, torch::kInt)));
    EXPECT_TRUE(torch::equal(
        params.attention.device.new_cache_slots.cpu(),
        torch::tensor({11 * page_size - 1, 11 * page_size}, torch::kInt)));
    EXPECT_EQ(params.graph.expanded_kv_seq_lens_vec,
              (std::vector<int32_t>{page_size, page_size + 1}));
    EXPECT_TRUE(torch::equal(params.graph.expanded_paged_kv_indptr.cpu(),
                             torch::tensor({0, 1, 3}, torch::kInt)));
    EXPECT_TRUE(torch::equal(params.graph.expanded_paged_kv_indices.cpu(),
                             torch::tensor({10, 10, 11}, torch::kInt)));
    EXPECT_TRUE(torch::equal(params.graph.expanded_paged_kv_last_page_len.cpu(),
                             torch::tensor({page_size, 1}, torch::kInt)));
  }

  void check_linear_state_rows(bool adaptive) {
    DecodeMetadataTestWorker worker(
        parallel_args(), torch::Device("npu:0"), options());
    LlmForwardInput input;
    input.token_ids_host = torch::tensor({42, 43}, torch::kInt);
    input.positions_host = torch::tensor({5, 9}, torch::kInt);
    input.token_ids = input.token_ids_host.to(torch::Device("npu:0"));
    input.positions = input.positions_host.to(torch::Device("npu:0"));
    input.input_params.meta.num_sequences = 2;
    input.input_params.meta.batch_forward_type = BatchForwardType::DECODE;
    input.input_params.attention.host.q_seq_lens = {1, 1};
    input.input_params.attention.host.kv_seq_lens = {6, 10};
    input.input_params.attention.host.block_tables =
        torch::tensor({{10, 11}, {20, 21}}, torch::kInt);
    // Include the sentinel used by models without linear-attention layers.
    input.input_params.embedding.linear_state_ids = {7, -1};
    input.input_params.embedding.linear_state_indices =
        torch::tensor({7, -1}, torch::kInt).to(torch::Device("npu:0"));

    const LlmForwardInput verify_input =
        adaptive ? worker.build_verify(input, std::vector<int32_t>{1, 2})
                 : worker.build_verify(input, /*adaptive=*/false);
    const torch::Tensor expected =
        adaptive ? torch::tensor({7, -1, -1}, torch::kInt)
                 : torch::tensor({7, 7, -1, -1}, torch::kInt);
    const auto& embedding = verify_input.input_params.embedding;
    ASSERT_TRUE(embedding.linear_state_indices.defined());
    EXPECT_TRUE(torch::equal(embedding.linear_state_indices.cpu(), expected));
    EXPECT_EQ(embedding.linear_state_indices.numel(),
              verify_input.token_ids.numel());
    EXPECT_EQ(embedding.linear_state_ids, (std::vector<int32_t>{7, -1}));
    EXPECT_TRUE(
        torch::equal(input.input_params.embedding.linear_state_indices.cpu(),
                     torch::tensor({7, -1}, torch::kInt)));
  }
};

TEST_P(MtpDecodeMetadataTest, Keeps305TokenContextWithinAllocatedPages) {
  DecodeMetadataTestWorker worker(
      parallel_args(), torch::Device("npu:0"), options());
  const torch::Tensor block_tables =
      GetParam() == 1 ? torch::tensor({{10, 11, 12}}, torch::kInt)
                      : torch::tensor({{10, 11}}, torch::kInt);
  LlmForwardInput input = make_input(/*position=*/305, block_tables);

  worker.resolve_decode_context(input);

  EXPECT_EQ(input.positions_host.item<int32_t>(), 305);
  EXPECT_EQ(input.input_params.attention.host.kv_seq_lens,
            (std::vector<int32_t>{306}));
}

TEST_P(MtpDecodeMetadataTest, BuildsFixedVerifyAcrossLogicalPage) {
  check_verify_across_logical_page(/*adaptive=*/false);
}

TEST_P(MtpDecodeMetadataTest, BuildsAdaptiveVerifyAcrossLogicalPage) {
  check_verify_across_logical_page(/*adaptive=*/true);
}

TEST_P(MtpDecodeMetadataTest, ExpandsFixedVerifyLinearStateRows) {
  check_linear_state_rows(/*adaptive=*/false);
}

TEST_P(MtpDecodeMetadataTest, ExpandsVariableVerifyLinearStateRows) {
  check_linear_state_rows(/*adaptive=*/true);
}

TEST_F(MtpDecodeMetadataTest,
       SnapshotsCompletedD2HAndBoundsPendingGenerations) {
  DecodeMetadataTestWorker worker(
      parallel_args(/*kv_split_size=*/1), torch::Device("npu:0"), options());
  worker.initialize_target_cache();
  LlmForwardInput input;
  input.input_params.embedding.embedding_ids = {0};
  input.input_params.embedding.request_ids = {"request-a"};
  worker.stage_tokens(input, /*token=*/41);
  ASSERT_TRUE(worker.matches_pending(input));
  LlmForwardInput replaced = input.clone();
  replaced.input_params.embedding.request_ids = {"request-b"};
  EXPECT_FALSE(worker.matches_pending(replaced));
  LlmForwardInput changed_slot = input.clone();
  changed_slot.input_params.embedding.embedding_ids = {1};
  EXPECT_FALSE(worker.matches_pending(changed_slot));
  const torch::Tensor first = worker.snapshot_tokens();
  EXPECT_TRUE(torch::equal(first, torch::full({1, 2}, 41, torch::kLong)));
  for (int64_t token = 42; token <= 45; ++token) {
    worker.stage_tokens(input, token);
    EXPECT_LE(worker.pending_count(), 2u);
  }
  const torch::Tensor last = worker.snapshot_tokens();
  EXPECT_TRUE(torch::equal(last, torch::full({1, 2}, 45, torch::kLong)));
  EXPECT_TRUE(torch::equal(first, torch::full({1, 2}, 41, torch::kLong)));
  EXPECT_EQ(worker.pending_count(), 0u);
}

TEST_F(MtpDecodeMetadataTest, RejectsPendingStateAfterBatchResizeOrReorder) {
  DecodeMetadataTestWorker worker(
      parallel_args(/*kv_split_size=*/1), torch::Device("npu:0"), options());
  worker.initialize_target_cache();
  LlmForwardInput input;
  input.input_params.embedding.embedding_ids = {0, 1};
  input.input_params.embedding.request_ids = {"request-a", "request-b"};
  worker.stage_tokens(input, /*token=*/42);
  ASSERT_TRUE(worker.matches_pending(input));
  LlmForwardInput reordered = input.clone();
  reordered.input_params.embedding.embedding_ids = {1, 0};
  reordered.input_params.embedding.request_ids = {"request-b", "request-a"};
  EXPECT_FALSE(worker.matches_pending(reordered));
  LlmForwardInput shrunk = input.clone();
  shrunk.input_params.embedding.embedding_ids = {0};
  shrunk.input_params.embedding.request_ids = {"request-a"};
  EXPECT_FALSE(worker.matches_pending(shrunk));
  EXPECT_TRUE(torch::equal(worker.snapshot_tokens(),
                           torch::full({2, 2}, 42, torch::kLong)));
}

TEST_F(MtpDecodeMetadataTest, NextDraftWorkspaceDoesNotFollowTableCapacity) {
  DecodeMetadataTestWorker worker(
      parallel_args(/*kv_split_size=*/1), torch::Device("npu:0"), options());
  LlmForwardInput input;
  input.input_params.meta.num_sequences = 1;
  input.input_params.attention.device.block_tables =
      torch::zeros({1, 65}, torch::kInt);
  const void* first = worker.next_draft_workspace_identity(input);
  input.input_params.meta.num_sequences = 2;
  EXPECT_NE(worker.next_draft_workspace_identity(input), first);
  input.input_params.meta.num_sequences = 1;
  EXPECT_EQ(worker.next_draft_workspace_identity(input), first);
}

TEST_F(MtpDecodeMetadataTest, PinnedOutputPoolKeepsConsumerViewsIndependent) {
  runtime::Options overlap_options = options();
  overlap_options.enable_schedule_overlap(true);
  DecodeMetadataTestWorker worker(parallel_args(/*kv_split_size=*/1),
                                  torch::Device("npu:0"),
                                  overlap_options);
  const torch::Tensor source = torch::ones({1, 2}, torch::kLong);
  torch::Tensor first = worker.lease_host_tokens(source);
  first.fill_(41);
  const void* first_address = first.data_ptr();
  torch::Tensor consumer = first.view({2});
  first = torch::Tensor();
  torch::Tensor second = worker.lease_host_tokens(source);
  EXPECT_NE(second.data_ptr(), first_address);
  second.fill_(42);
  EXPECT_TRUE(torch::equal(consumer, torch::full({2}, 41, torch::kLong)));
  consumer = torch::Tensor();
  torch::Tensor reused = worker.lease_host_tokens(source);
  EXPECT_EQ(reused.data_ptr(), first_address);

  // Unified token/count views share a byte arena. A live count view must
  // retain the lease even after its token sibling and parent are released.
  const torch::Tensor packed_source = torch::empty({20}, torch::kUInt8);
  torch::Tensor packed = worker.lease_host_tokens(packed_source);
  const void* packed_address = packed.data_ptr();
  torch::Tensor counts = packed.narrow(0, 16, 4).view(torch::kInt);
  counts.fill_(3);
  packed = torch::Tensor();
  torch::Tensor newer = worker.lease_host_tokens(packed_source);
  EXPECT_NE(newer.data_ptr(), packed_address);
  newer.fill_(0);
  EXPECT_EQ(counts.item<int32_t>(), 3);
}

INSTANTIATE_TEST_SUITE_P(KvSplit,
                         MtpDecodeMetadataTest,
                         ::testing::Values(1, 2));

}  // namespace
}  // namespace xllm
