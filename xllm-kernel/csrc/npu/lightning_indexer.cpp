// Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.

#include <c10/util/Logging.h>
#include <torch/library.h>
#include <torch/types.h>

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "op_plugin/utils/op_api_common_base.h"

namespace xllm_kernel {
namespace {

torch::Tensor lightning_indexer_out(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& weights,
    const std::optional<torch::Tensor>& query_seq_lengths,
    const std::optional<torch::Tensor>& key_seq_lengths,
    const std::optional<torch::Tensor>& block_table,
    std::string_view layout_query,
    std::string_view layout_key,
    int64_t selected_count,
    int64_t sparse_mode,
    int64_t pre_tokens,
    int64_t next_tokens,
    bool return_value,
    torch::Tensor& sparse_indices_out,
    torch::Tensor& sparse_values_out) {
  CHECK(sparse_indices_out.is_contiguous())
      << "sparse_indices_out must be contiguous";
  CHECK(sparse_values_out.is_contiguous())
      << "sparse_values_out must be contiguous";
  CHECK(sparse_indices_out.scalar_type() == torch::kInt)
      << "sparse_indices_out must be int32";
  CHECK(sparse_values_out.scalar_type() == torch::kBFloat16)
      << "sparse_values_out must be bfloat16";

  std::string query_layout_str = std::string(layout_query);
  std::string key_layout_str = std::string(layout_key);
  char* query_layout_ptr = query_layout_str.data();
  char* key_layout_ptr = key_layout_str.data();

  const torch::DeviceGuard guard(query.device());
  EXEC_NPU_CMD_EXT(aclnnLightningIndexer,
                   query,
                   key,
                   weights,
                   query_seq_lengths,
                   key_seq_lengths,
                   block_table,
                   query_layout_ptr,
                   key_layout_ptr,
                   selected_count,
                   sparse_mode,
                   pre_tokens,
                   next_tokens,
                   return_value,
                   sparse_indices_out,
                   sparse_values_out);
  return sparse_indices_out;
}

torch::Tensor lightning_indexer(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& weights,
    const std::optional<torch::Tensor>& query_seq_lengths,
    const std::optional<torch::Tensor>& key_seq_lengths,
    const std::optional<torch::Tensor>& block_table,
    std::string_view layout_query,
    std::string_view layout_key,
    int64_t selected_count,
    int64_t sparse_mode,
    int64_t pre_tokens,
    int64_t next_tokens,
    bool return_value) {
  CHECK(selected_count > 0);
  for (int64_t size : query.sizes()) {
    CHECK(size > 0);
  }
  for (int64_t size : key.sizes()) {
    CHECK(size > 0);
  }
  const int64_t key_heads = key.size(layout_key == "TND" ? 1 : 2);
  const std::vector<int64_t> shape =
      layout_query == "BSND"
          ? std::vector<int64_t>{query.size(0),
                                 query.size(1),
                                 key_heads,
                                 selected_count}
          : std::vector<int64_t>{query.size(0), key_heads, selected_count};
  auto indices = torch::zeros(shape, query.options().dtype(torch::kInt32));
  auto values = torch::zeros(shape, query.options().dtype(torch::kBFloat16));
  return lightning_indexer_out(query,
                               key,
                               weights,
                               query_seq_lengths,
                               key_seq_lengths,
                               block_table,
                               layout_query,
                               layout_key,
                               selected_count,
                               sparse_mode,
                               pre_tokens,
                               next_tokens,
                               return_value,
                               indices,
                               values);
}

TORCH_LIBRARY_FRAGMENT(xllm_kernel, m) {
  m.def(
      "lightning_indexer(Tensor query, Tensor key, Tensor weights, "
      "Tensor? query_seq_lengths, Tensor? key_seq_lengths, Tensor? "
      "block_table, str layout_query, str layout_key, int selected_count, int "
      "sparse_mode, int pre_tokens, int next_tokens, bool return_value) -> "
      "Tensor");
  m.def(
      "lightning_indexer_out(Tensor query, Tensor key, Tensor weights, "
      "Tensor? query_seq_lengths, Tensor? key_seq_lengths, Tensor? "
      "block_table, str layout_query, str layout_key, int selected_count, "
      "int sparse_mode, int pre_tokens, int next_tokens, bool return_value, "
      "Tensor(a!) sparse_indices_out, Tensor(b!) sparse_values_out) -> "
      "Tensor(a!)");
}

TORCH_LIBRARY_IMPL(xllm_kernel, PrivateUse1, m) {
  m.impl("lightning_indexer", TORCH_FN(lightning_indexer));
  m.impl("lightning_indexer_out", TORCH_FN(lightning_indexer_out));
}

}  // namespace
}  // namespace xllm_kernel
