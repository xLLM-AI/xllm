/* Copyright 2026 The xLLM Authors.

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

#include "core/layers/common/attention_metadata.h"
#include "core/layers/common/expanded_decode_metadata_builder.h"
#include "core/runtime/forward_params.h"
#include "core/runtime/py_attention_metadata.h"
#include "models/llm/py_causal_lm.h"

namespace py = pybind11;

namespace xllm::mtp_async {
namespace {

constexpr int32_t kBlockSize = 4;

ForwardInput make_draft_input(int64_t batch_size, int64_t hidden_size) {
  ForwardInput input;
  input.token_ids = torch::zeros({batch_size * 2}, torch::kInt);
  input.positions = torch::zeros({batch_size * 2}, torch::kInt);
  input.input_params.embedding.input_embedding =
      torch::zeros({batch_size * 2, hidden_size}, torch::kFloat32);
  input.input_params.attention.device.kv_seq_lens =
      torch::zeros({batch_size * 2}, torch::kInt);
  return input;
}

ForwardInput make_block_table_source(const torch::Tensor& block_tables,
                                     std::vector<int32_t> kv_seq_lens) {
  ForwardInput input;
  input.input_params.attention.device.block_tables = block_tables;
  input.input_params.attention.host.block_tables = block_tables;
  input.input_params.attention.host.kv_seq_lens = std::move(kv_seq_lens);
  input.input_params.multi_block_tables.emplace_back(torch::zeros({1}));
  return input;
}

void prepare_single_sequence(ForwardInput& draft_input,
                             const ForwardInput& block_table_source,
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
  ForwardInput draft_input = make_draft_input(/*batch_size=*/1,
                                              /*hidden_size=*/2);
  const torch::Tensor block_tables = torch::tensor({{10, 11}}, torch::kInt);
  ForwardInput block_table_source = make_block_table_source(block_tables, {5});

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
  ForwardInput draft_input = make_draft_input(/*batch_size=*/1,
                                              /*hidden_size=*/2);
  const torch::Tensor template_block_tables =
      torch::tensor({{90, 91}, {90, 91}}, torch::kInt);
  draft_input.input_params.attention.device.block_tables =
      template_block_tables;
  const torch::Tensor block_tables = torch::tensor({{10, 11}}, torch::kInt);
  ForwardInput block_table_source = make_block_table_source(block_tables, {5});

  prepare_single_sequence(draft_input,
                          block_table_source,
                          /*base_kv_seq_len=*/5,
                          /*rebuild_expanded_decode_metadata=*/false);

  EXPECT_TRUE(
      torch::equal(draft_input.input_params.attention.device.block_tables,
                   template_block_tables));
  EXPECT_FALSE(draft_input.input_params.graph.expanded_kv_seq_lens.defined());
}

TEST(MtpAsyncInputBuilderTest, BuildsTokenwiseSpecVerifyKvLengths) {
  EXPECT_EQ(layer::ExpandedDecodeMetadataBuilder::build_tokenwise_kv_seq_lens(
                /*q_seq_lens=*/{2, 1}, /*kv_seq_lens=*/{4, 3}),
            (std::vector<int32_t>{3, 4, 3}));
}

TEST(MtpAsyncInputBuilderTest, KeepsGenericPagedMetadataSeparate) {
  ModelInputParams params;
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
  ForwardInput draft_input = make_draft_input(/*batch_size=*/1,
                                              /*hidden_size=*/2);
  const torch::Tensor block_tables = torch::tensor({{10, 11}}, torch::kInt);
  ForwardInput block_table_source = make_block_table_source(block_tables, {8});

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
  ForwardInput draft_input = make_draft_input(/*batch_size=*/1,
                                              /*hidden_size=*/2);
  const torch::Tensor block_tables = torch::tensor({{10, 11}}, torch::kInt);
  ForwardInput block_table_source = make_block_table_source(block_tables, {9});

  EXPECT_DEATH(prepare_single_sequence(
                   draft_input, block_table_source, /*base_kv_seq_len=*/9),
               "Expanded KV length exceeds block-table capacity");
}

TEST(MtpAsyncInputBuilderTest, PybindViewSelectsExpandedGraphMetadata) {
  if (!Py_IsInitialized()) {
    setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0", 1);
    Py_InitializeEx(0);
  }
  py::gil_scoped_acquire gil;
  prepend_python_model_path();
  py::module_ main_module = py::module_::import("__main__");
  register_attention_metadata_views(main_module);

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

  py::object py_metadata = py::cast(PyAttentionMetadataView(metadata));
  py::tuple selected = runner.attr("_decode_metadata")(py_metadata);

  EXPECT_TRUE(torch::equal(selected[0].cast<torch::Tensor>(),
                           metadata->expanded_decode.block_table));
  EXPECT_TRUE(torch::equal(selected[1].cast<torch::Tensor>(),
                           metadata->expanded_decode.kv_seq_lens));
  EXPECT_EQ(selected[2].cast<std::vector<int32_t>>(),
            metadata->expanded_decode.kv_seq_lens_host_vec);
  EXPECT_TRUE(torch::equal(selected[3].cast<torch::Tensor>(),
                           metadata->expanded_decode.paged_kv_indptr));
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
