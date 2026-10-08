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

#include "core/framework/speculative/mtp_async_input_builder.h"

#include <gtest/gtest.h>
#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/kernels/xllm_torch_ops.h"
#include "core/layers/common/attention_metadata.h"
#include "core/layers/common/attention_metadata_builder.h"
#include "core/layers/common/expanded_decode_metadata_builder.h"
#include "core/runtime/forward_params.h"
#include "core/runtime/py_attention_metadata.h"
#include "models/llm/py_causal_lm.h"

namespace py = pybind11;

namespace xllm::mtp_async {
namespace {

constexpr int32_t kBlockSize = 4;

LlmForwardInput make_draft_input(int64_t batch_size, int64_t hidden_size) {
  LlmForwardInput input;
  input.token_ids = torch::zeros({batch_size * 2}, torch::kInt);
  input.positions = torch::zeros({batch_size * 2}, torch::kInt);
  input.input_params.embedding.input_embedding =
      torch::zeros({batch_size * 2, hidden_size}, torch::kFloat32);
  input.input_params.attention.device.kv_seq_lens =
      torch::zeros({batch_size * 2}, torch::kInt);
  return input;
}

LlmForwardInput make_block_table_source(const torch::Tensor& block_tables,
                                        std::vector<int32_t> kv_seq_lens) {
  LlmForwardInput input;
  input.input_params.attention.device.block_tables = block_tables;
  input.input_params.attention.host.block_tables = block_tables;
  input.input_params.attention.host.kv_seq_lens = std::move(kv_seq_lens);
  input.input_params.multi_block_tables.emplace_back(torch::zeros({1}));
  return input;
}

void prepare_single_sequence(LlmForwardInput& draft_input,
                             const LlmForwardInput& block_table_source,
                             int32_t base_kv_seq_len,
                             bool rebuild_expanded_decode_metadata = true) {
  const torch::Tensor accepted_tokens = torch::tensor({{42, -1}}, torch::kLong);
  const torch::Tensor accepted_embeddings =
      torch::tensor({{{1.0F, 2.0F}, {3.0F, 4.0F}}});
  const torch::Tensor embedding_placeholder = torch::zeros({2});
  const torch::Tensor base_positions =
      torch::tensor({base_kv_seq_len - 2}, torch::kInt);
  const torch::Tensor base_kv_seq_lens =
      torch::tensor({base_kv_seq_len - 1}, torch::kInt);

  prepare_next_draft_from_accepted_state(draft_input,
                                         block_table_source,
                                         accepted_tokens,
                                         accepted_embeddings,
                                         embedding_placeholder,
                                         base_positions,
                                         base_kv_seq_lens,
                                         /*use_chunked_prefill=*/false,
                                         rebuild_expanded_decode_metadata,
                                         kBlockSize);
}

void prepend_python_model_path() {
  std::filesystem::path repo_root(__FILE__);
  for (int32_t depth = 0; depth < 5; ++depth) {
    repo_root = repo_root.parent_path();
  }
  py::list sys_path = py::module_::import("sys").attr("path");
  sys_path.attr("insert")(0, repo_root.string());
}

TEST(MtpAsyncInputBuilderTest, BuildsExpandedMetadataAcrossBlockBoundary) {
  LlmForwardInput draft_input = make_draft_input(/*batch_size=*/1,
                                                 /*hidden_size=*/2);
  const torch::Tensor block_tables = torch::tensor({{10, 11}}, torch::kInt);
  LlmForwardInput block_table_source =
      make_block_table_source(block_tables, {5});

  prepare_single_sequence(
      draft_input, block_table_source, /*base_kv_seq_len=*/5);

  const auto& attention = draft_input.input_params.attention.device;
  EXPECT_TRUE(torch::equal(draft_input.input_params.graph.expanded_kv_seq_lens,
                           torch::tensor({4, 5}, torch::kInt)));
  EXPECT_EQ(draft_input.input_params.graph.expanded_kv_seq_lens_vec,
            (std::vector<int32_t>{4, 5}));
  EXPECT_TRUE(torch::equal(attention.paged_kv_indptr,
                           torch::tensor({0, 1, 3}, torch::kInt)));
  EXPECT_TRUE(torch::equal(attention.paged_kv_indices,
                           torch::tensor({10, 10, 11}, torch::kInt)));
  EXPECT_TRUE(torch::equal(attention.paged_kv_last_page_len,
                           torch::tensor({4, 1}, torch::kInt)));
}

TEST(MtpAsyncInputBuilderTest, CanSkipExpandedMetadataRebuild) {
  LlmForwardInput draft_input = make_draft_input(/*batch_size=*/1,
                                                 /*hidden_size=*/2);
  const torch::Tensor template_block_tables =
      torch::tensor({{90, 91}, {90, 91}}, torch::kInt);
  draft_input.input_params.attention.device.block_tables =
      template_block_tables;
  const torch::Tensor block_tables = torch::tensor({{10, 11}}, torch::kInt);
  LlmForwardInput block_table_source =
      make_block_table_source(block_tables, {5});

  prepare_single_sequence(draft_input,
                          block_table_source,
                          /*base_kv_seq_len=*/5,
                          /*rebuild_expanded_decode_metadata=*/false);

  EXPECT_TRUE(
      torch::equal(draft_input.input_params.attention.device.block_tables,
                   template_block_tables));
  EXPECT_FALSE(draft_input.input_params.graph.expanded_kv_seq_lens.defined());
}

TEST(MtpAsyncInputBuilderTest, RefreshesVerifyPagesAfterAcceptedPrefix) {
  const torch::Tensor original_positions =
      torch::tensor({{2, 3, 4, 5}, {1, 2, 3, 4}}, torch::kInt);
  const torch::Tensor expected_positions =
      torch::tensor({{4, 5, 6, 7}, {2, 3, 4, 5}}, torch::kInt);
  const torch::Tensor expected_slots =
      torch::tensor({{44, 45, 46, 47}, {82, 83, 84, 85}}, torch::kInt);
  const torch::Tensor block_tables =
      torch::tensor({{10, 11}, {20, 21}}, torch::kInt);
  for (bool chunked : {false, true}) {
    for (bool step_major : {false, true}) {
      SCOPED_TRACE(::testing::Message()
                   << "chunked=" << chunked << ", step_major=" << step_major);
      auto flatten_rows = [step_major](const torch::Tensor& rows) {
        return (step_major ? rows.transpose(0, 1) : rows).flatten();
      };
      LlmForwardInput input;
      input.token_ids = torch::zeros({8}, torch::kInt);
      input.positions = flatten_rows(original_positions).clone();
      input.positions_host = input.positions.clone();
      auto& params = input.input_params;
      const torch::Tensor original_kv = flatten_rows(original_positions + 1);
      const std::vector<int32_t> original_host_kv =
          step_major ? std::vector<int32_t>{3, 2, 4, 3, 5, 4, 6, 5}
                     : std::vector<int32_t>{3, 4, 5, 6, 2, 3, 4, 5};
      const torch::Tensor expanded_tables =
          step_major ? block_tables.repeat({4, 1})
                     : block_tables.repeat_interleave(4, 0);
      params.attention.device.block_tables =
          chunked ? block_tables : expanded_tables;
      params.attention.device.kv_seq_lens =
          chunked ? torch::tensor({6, 5}, torch::kInt) : original_kv;
      params.attention.host.kv_seq_lens =
          chunked ? std::vector<int32_t>{6, 5} : original_host_kv;
      layer::ExpandedDecodeMetadataBuilder::populate_expanded_layout(
          ModelInputParams(params),
          original_kv,
          expanded_tables,
          original_host_kv,
          kBlockSize);

      prepare_target_verify_from_accepted_state(
          input,
          torch::tensor({{42, 43, -1, -1}, {52, -1, -1, -1}}, torch::kLong),
          torch::tensor({2, 1}, torch::kInt),
          torch::tensor({3, 2}, torch::kInt),
          kBlockSize,
          chunked,
          step_major);

      const torch::Tensor expected_kv = flatten_rows(expected_positions + 1);
      const std::vector<int32_t> expected_host_kv =
          step_major ? std::vector<int32_t>{5, 3, 6, 4, 7, 5, 8, 6}
                     : std::vector<int32_t>{5, 6, 7, 8, 3, 4, 5, 6};
      EXPECT_TRUE(
          torch::equal(input.positions, flatten_rows(expected_positions)));
      EXPECT_TRUE(torch::equal(input.positions_host, input.positions));
      EXPECT_TRUE(torch::equal(params.attention.device.new_cache_slots,
                               flatten_rows(expected_slots)));
      EXPECT_TRUE(torch::equal(
          params.attention.device.kv_seq_lens,
          chunked ? torch::tensor({8, 6}, torch::kInt) : expected_kv));
      EXPECT_EQ(params.attention.host.kv_seq_lens,
                chunked ? (std::vector<int32_t>{8, 6}) : expected_host_kv);
      EXPECT_EQ(params.meta.kv_max_seq_len, 8);
      EXPECT_TRUE(torch::equal(params.graph.expanded_kv_seq_lens, expected_kv));
      EXPECT_EQ(params.graph.expanded_kv_seq_lens_vec, expected_host_kv);
      const torch::Tensor expected_indptr =
          step_major
              ? torch::tensor({0, 2, 3, 5, 6, 8, 10, 12, 14}, torch::kInt)
              : torch::tensor({0, 2, 4, 6, 8, 9, 10, 12, 14}, torch::kInt);
      const torch::Tensor expected_indices =
          step_major
              ? torch::tensor(
                    {10, 11, 20, 10, 11, 20, 10, 11, 20, 21, 10, 11, 20, 21},
                    torch::kInt)
              : torch::tensor(
                    {10, 11, 10, 11, 10, 11, 10, 11, 20, 20, 20, 21, 20, 21},
                    torch::kInt);
      EXPECT_TRUE(
          torch::equal(params.graph.expanded_paged_kv_indptr, expected_indptr));
      EXPECT_TRUE(torch::equal(params.graph.expanded_paged_kv_indices,
                               expected_indices));
      EXPECT_TRUE(torch::equal(params.graph.expanded_paged_kv_last_page_len,
                               (expected_kv - 1).remainder(kBlockSize) + 1));
      const torch::Tensor token_rows =
          step_major ? input.token_ids.view({4, 2}).transpose(0, 1)
                     : input.token_ids.view({2, 4});
      EXPECT_TRUE(torch::equal(token_rows.select(1, 0),
                               torch::tensor({43, 52}, torch::kInt)));
    }
  }
}

TEST(MtpAsyncInputBuilderTest, BuildsTokenwiseSpecVerifyKvLengths) {
  EXPECT_EQ(layer::ExpandedDecodeMetadataBuilder::build_tokenwise_kv_seq_lens(
                /*q_seq_lens=*/{2, 1}, /*kv_seq_lens=*/{4, 3}),
            (std::vector<int32_t>{3, 4, 3}));
}

TEST(MtpAsyncInputBuilderTest, KeepsGenericPagedMetadataSeparate) {
  ModelInputParams params = ModelInputSnapshot(LlmModelParams()).view();
  params.attention.device.paged_kv_indptr = torch::tensor({0, 1}, torch::kInt);
  params.attention.device.paged_kv_indices = torch::tensor({99}, torch::kInt);
  params.attention.device.paged_kv_last_page_len =
      torch::tensor({1}, torch::kInt);

  layer::ExpandedDecodeMetadataBuilder::populate_expanded_layout(
      params,
      torch::tensor({3, 4}, torch::kInt),
      torch::tensor({{10}, {10}}, torch::kInt),
      /*expanded_host_kv_seq_lens=*/{3, 4},
      kBlockSize);

  EXPECT_TRUE(torch::equal(params.attention.device.paged_kv_indptr,
                           torch::tensor({0, 1}, torch::kInt)));
  EXPECT_TRUE(torch::equal(params.attention.device.paged_kv_indices,
                           torch::tensor({99}, torch::kInt)));
  EXPECT_TRUE(torch::equal(params.graph.expanded_paged_kv_indptr,
                           torch::tensor({0, 1, 2}, torch::kInt)));
  EXPECT_TRUE(torch::equal(params.graph.expanded_paged_kv_indices,
                           torch::tensor({10, 10}, torch::kInt)));
}

TEST(MtpAsyncInputBuilderTest, SupportsMaximumBlockTableWidth) {
  LlmForwardInput draft_input = make_draft_input(/*batch_size=*/1,
                                                 /*hidden_size=*/2);
  const torch::Tensor block_tables = torch::tensor({{10, 11}}, torch::kInt);
  LlmForwardInput block_table_source =
      make_block_table_source(block_tables, {8});

  prepare_single_sequence(
      draft_input, block_table_source, /*base_kv_seq_len=*/8);

  const auto& attention = draft_input.input_params.attention.device;
  EXPECT_TRUE(torch::equal(attention.paged_kv_indptr,
                           torch::tensor({0, 2, 4}, torch::kInt)));
  EXPECT_TRUE(torch::equal(attention.paged_kv_indices,
                           torch::tensor({10, 11, 10, 11}, torch::kInt)));
  EXPECT_TRUE(torch::equal(attention.paged_kv_last_page_len,
                           torch::tensor({3, 4}, torch::kInt)));
}

TEST(MtpAsyncInputBuilderTest, RejectsPageCountBeyondBlockTableWidth) {
  LlmForwardInput draft_input = make_draft_input(/*batch_size=*/1,
                                                 /*hidden_size=*/2);
  const torch::Tensor block_tables = torch::tensor({{10, 11}}, torch::kInt);
  LlmForwardInput block_table_source =
      make_block_table_source(block_tables, {9});

  EXPECT_DEATH(prepare_single_sequence(
                   draft_input, block_table_source, /*base_kv_seq_len=*/9),
               "Expanded KV length exceeds block-table capacity");
}

TEST(MtpAsyncInputBuilderTest, PybindViewExposesLinearStateReadAndWriteSlots) {
  ensure_xllm_torch_ops_registered();
  if (!Py_IsInitialized()) {
    setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0", 1);
    Py_InitializeEx(0);
  }
  py::gil_scoped_acquire gil;
  prepend_python_model_path();
  py::module_::import("xllm.python._npu_bootstrap");
  py::module_::import("xllm.python").attr("initialize_runtime")();
  py::module_ main_module = py::module_::import("__main__");
  if (!py::hasattr(main_module, "AttentionMetadataView")) {
    register_attention_metadata_views(main_module);
  }

  auto metadata = std::make_shared<layer::AttentionMetadata>();
  metadata->is_prefill = true;
  metadata->is_dummy = true;

  ModelInputParams params = ModelInputSnapshot(LlmModelParams()).view();
  params.embedding.linear_state_ids = {3, 7};
  params.embedding.linear_state_indices = torch::tensor({3, 7}, torch::kInt);

  LinearStateCacheOp direct_read;
  direct_read.linear_state_id = 3;
  direct_read.restore_src_slot_id = 2;
  LinearStateCacheOp legacy_restore;
  legacy_restore.linear_state_id = 7;
  legacy_restore.restore_requested = true;
  legacy_restore.restore_src_slot_id = 6;
  params.linear_state_cache_ops = {direct_read, legacy_restore};

  py::object py_metadata = py::cast(PyAttentionMetadataView(metadata, params));
  EXPECT_TRUE(torch::equal(
      py_metadata.attr("linear_state_indices").cast<torch::Tensor>(),
      torch::tensor({3, 7}, torch::kInt)));
  EXPECT_TRUE(torch::equal(
      py_metadata.attr("linear_state_write_indices").cast<torch::Tensor>(),
      torch::tensor({3, 7}, torch::kInt)));
  EXPECT_TRUE(torch::equal(
      py_metadata.attr("linear_state_read_indices").cast<torch::Tensor>(),
      torch::tensor({2, 7}, torch::kInt)));
  EXPECT_TRUE(py_metadata.attr("is_dummy").cast<bool>());
}

TEST(MtpAsyncInputBuilderTest, PybindViewsPreserveGraphMetadataStorage) {
  ensure_xllm_torch_ops_registered();
  if (!Py_IsInitialized()) {
    setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0", 1);
    Py_InitializeEx(0);
  }
  py::gil_scoped_acquire gil;
  prepend_python_model_path();
  py::module_::import("xllm.python._npu_bootstrap");
  py::module_::import("xllm.python").attr("initialize_runtime")();
  py::module_ main_module = py::module_::import("__main__");
  if (!py::hasattr(main_module, "AttentionMetadataView")) {
    register_attention_metadata_views(main_module);
  }

  auto metadata = std::make_shared<layer::AttentionMetadata>();
  metadata->slot_mapping = torch::arange(4, torch::kInt);
  metadata->expanded_decode.enabled = true;
  metadata->expanded_decode.kv_seq_lens =
      torch::tensor({3, 4, 7, 8}, torch::kInt);
  metadata->expanded_decode.block_table =
      torch::tensor({{10, 11}, {10, 11}, {20, 21}, {20, 21}}, torch::kInt);
  metadata->expanded_decode.paged_kv_indptr =
      torch::tensor({0, 1, 2, 4, 6}, torch::kInt);
  metadata->expanded_decode.paged_kv_indices =
      torch::tensor({10, 10, 20, 21, 20, 21}, torch::kInt);
  metadata->expanded_decode.paged_kv_last_page_len =
      torch::tensor({3, 4, 3, 4}, torch::kInt);
  metadata->expanded_decode.kv_seq_lens_host_vec = {3, 4, 7, 8};
  metadata->expanded_decode.kv_seq_lens_host =
      torch::tensor({3, 4, 7, 8}, torch::kInt);

  py::module_ runner_module = py::module_::import(
      "xllm.python.model_executor.runners.decode_acl_graph");
  py::object runner_class = runner_module.attr("DecodeAclGraphRunner");
  py::object runner = runner_class.attr("__new__")(runner_class);
  py::module_ types = py::module_::import("types");
  runner.attr("attention_backend") = types.attr("SimpleNamespace")(
      py::arg("page_size") = kBlockSize, py::arg("is_mla") = false);

  ModelInputParams params = ModelInputSnapshot(LlmModelParams()).view();
  // MTP may rewrite num_sequences to execution rows. The Python graph limit
  // must use the original requests, including a truly empty peer rank.
  params.meta.num_sequences = 6;
  params.parallel.dp_global_token_nums = {6, 0};
  params.parallel.dp_global_sequence_nums = {3, 0};
  params.parallel = params.parallel.to(torch::Device(torch::kCPU));
  py::object py_metadata = py::cast(PyAttentionMetadataView(metadata, params));
  EXPECT_EQ(
      py_metadata.attr("dp_global_sequence_nums").cast<std::vector<int32_t>>(),
      (std::vector<int32_t>{3, 0}));
  EXPECT_EQ(py_metadata.attr("dp_execution_token_counts")
                .cast<std::vector<int32_t>>(),
            (std::vector<int32_t>{6, 1}));
  runner.attr("dp_size") = 2;
  runner.attr("num_decoding_tokens") = 1;
  runner.attr("dp_rank") = 0;
  EXPECT_EQ(runner.attr("_decode_batch_sizes")(torch::arange(6), py_metadata)
                .cast<std::vector<int32_t>>(),
            (std::vector<int32_t>{3, 3}));
  runner.attr("dp_rank") = 1;
  EXPECT_EQ(runner.attr("_decode_batch_sizes")(torch::arange(1), py_metadata)
                .cast<std::vector<int32_t>>(),
            (std::vector<int32_t>{0, 3}));
  py::tuple selected = runner.attr("_decode_metadata")(py_metadata);

  EXPECT_TRUE(torch::equal(selected[0].cast<torch::Tensor>(),
                           metadata->expanded_decode.block_table));
  EXPECT_TRUE(torch::equal(selected[1].cast<torch::Tensor>(),
                           metadata->expanded_decode.kv_seq_lens));
  EXPECT_EQ(selected[2].cast<std::vector<int32_t>>(),
            metadata->expanded_decode.kv_seq_lens_host_vec);
  EXPECT_TRUE(torch::equal(selected[3].cast<torch::Tensor>(),
                           metadata->expanded_decode.paged_kv_indptr));

  // Sparse graph views borrow the final arena tensors without Host/CSR data.
  const torch::Tensor table = torch::tensor({{10, 11}}, torch::kInt);
  const torch::Tensor lengths = torch::tensor({128}, torch::kInt);
  const torch::Tensor slots = torch::tensor({1407}, torch::kInt);
  auto sparse = std::make_shared<layer::AttentionMetadata>(
      layer::AttentionMetadataBuilder::build_mtp_sparse_decode(
          table, lengths, slots, /*block_size=*/128));
  py::object sparse_view = py::cast(PyAttentionMetadataView(sparse));
  const torch::Tensor view_table =
      sparse_view.attr("block_table").cast<torch::Tensor>();
  const torch::Tensor view_lengths =
      sparse_view.attr("kv_seq_lens").cast<torch::Tensor>();
  const torch::Tensor view_slots =
      sparse_view.attr("slot_mapping").cast<torch::Tensor>();
  EXPECT_EQ(view_table.data_ptr(), table.data_ptr());
  EXPECT_EQ(view_lengths.data_ptr(), lengths.data_ptr());
  EXPECT_EQ(view_slots.data_ptr(), slots.data_ptr());
  EXPECT_TRUE(sparse->kv_seq_lens_vec.empty());
  EXPECT_FALSE(sparse->kv_seq_lens_host.defined());
  EXPECT_FALSE(sparse->paged_kv_indptr.defined());
  EXPECT_FALSE(sparse->paged_kv_indices.defined());
  EXPECT_FALSE(sparse->expanded_decode.enabled);

  // The Python view retains the owner as the request crosses a page boundary.
  sparse.reset();
  lengths.fill_(129);
  slots.fill_(1408);
  EXPECT_EQ(sparse_view.attr("kv_seq_lens").cast<torch::Tensor>().data_ptr(),
            view_lengths.data_ptr());
  EXPECT_EQ(sparse_view.attr("slot_mapping").cast<torch::Tensor>().data_ptr(),
            view_slots.data_ptr());
  EXPECT_EQ(view_lengths.item<int32_t>(), 129);
  EXPECT_EQ(view_slots.item<int32_t>(), 1408);
}

TEST(MtpAsyncInputBuilderTest, SharedModulesPointToTargetModel) {
  py::gil_scoped_acquire gil;
  py::module_ types = py::module_::import("types");
  py::object target_lm_head = py::module_::import("builtins").attr("object")();
  py::object target_embedding =
      py::module_::import("builtins").attr("object")();
  py::object target_body =
      types.attr("SimpleNamespace")(py::arg("embed_tokens") = target_embedding);
  py::object target_model = types.attr("SimpleNamespace")(
      py::arg("lm_head") = target_lm_head, py::arg("model") = target_body);
  py::object draft_body =
      types.attr("SimpleNamespace")(py::arg("embed_tokens") = py::none());
  py::object draft_model = types.attr("SimpleNamespace")(
      py::arg("lm_head") = py::none(), py::arg("model") = draft_body);

  ::xllm::detail::share_python_model_weights(draft_model, target_model);

  py::object draft_lm_head = draft_model.attr("lm_head");
  py::object draft_embedding = draft_model.attr("model").attr("embed_tokens");
  EXPECT_TRUE(draft_lm_head.is(target_lm_head));
  EXPECT_TRUE(draft_embedding.is(target_embedding));
}

}  // namespace
}  // namespace xllm::mtp_async
