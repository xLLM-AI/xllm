// Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.

#include <torch/library.h>
#include <torch/types.h>

#include <limits>
#include <optional>
#include <string>
#include <string_view>

#include "csrc/ascendc/quant_lightning_indexer/op_kernel/quant_lightning_indexer_metadata.h"
#include "csrc/npu/ascendc_api.h"
#include "op_plugin/utils/op_api_common_base.h"

namespace xllm_kernel {
namespace {

constexpr const char* kIndexer = "aclnnXllmKernelQuantLightningIndexer";
constexpr const char* kMetadata =
    "aclnnXllmKernelQuantLightningIndexerMetadata";
constexpr const char* kScatter = "aclnnXllmKernelScatterNdUpdateV2";
constexpr int64_t kQuantMode = 0;
constexpr int64_t kSparseMode = 3;
constexpr bool kReturnValues = false;
constexpr int64_t kUnlimitedWindow = std::numeric_limits<int64_t>::max();

const AscendcApi& indexer_api() {
  static const AscendcApi api = load_ascendc_api(
      kIndexer, "aclnnXllmKernelQuantLightningIndexerGetWorkspaceSize");
  return api;
}

const AscendcApi& metadata_api() {
  static const AscendcApi api = load_ascendc_api(
      kMetadata,
      "aclnnXllmKernelQuantLightningIndexerMetadataGetWorkspaceSize");
  return api;
}

const AscendcApi& scatter_api() {
  static const AscendcApi api = load_ascendc_api(
      kScatter, "aclnnXllmKernelScatterNdUpdateV2GetWorkspaceSize");
  return api;
}

void GetApiFunc(const char* name,
                const char* /*workspace_name*/,
                void*& launch,
                void*& workspace) {
  const AscendcApi* api = nullptr;
  if (std::string_view(name) == kIndexer) {
    api = &indexer_api();
  } else if (std::string_view(name) == kMetadata) {
    api = &metadata_api();
  } else if (std::string_view(name) == kScatter) {
    api = &scatter_api();
  }
  CHECK(api) << "Unknown sparse indexing ACLNN entry point: " << name;
  launch = api->launch;
  workspace = api->workspace;
}

void prepare_ascendc_sparse_indexing() {
  indexer_api();
  metadata_api();
  scatter_api();
}

torch::Tensor quant_lightning_indexer(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& weights,
    const torch::Tensor& query_dequant_scale,
    const torch::Tensor& key_dequant_scale,
    const torch::Tensor& metadata,
    const std::optional<torch::Tensor>& query_seq_lengths,
    const std::optional<torch::Tensor>& key_seq_lengths,
    const std::optional<torch::Tensor>& block_table,
    int64_t selected_count,
    int64_t cmp_ratio) {
  const torch::DeviceGuard guard(query.device());
  auto indices = torch::zeros({query.size(0), key.size(2), selected_count},
                              query.options().dtype(torch::kInt32));
  auto values = torch::empty({0}, query.options().dtype(torch::kFloat32));
  const int64_t key_stride = key.stride(0);
  const int64_t scale_stride = key_dequant_scale.stride(0);
  char query_layout[] = "TND";
  char key_layout[] = "PA_BSND";
  char* layout_query = query_layout;
  char* layout_key = key_layout;

  EXEC_NPU_CMD_EXT(aclnnXllmKernelQuantLightningIndexer,
                   query,
                   key,
                   weights,
                   query_dequant_scale,
                   key_dequant_scale,
                   query_seq_lengths,
                   key_seq_lengths,
                   block_table,
                   metadata,
                   kQuantMode,  // query_quant_mode
                   kQuantMode,  // key_quant_mode
                   layout_query,
                   layout_key,
                   selected_count,
                   kSparseMode,
                   kUnlimitedWindow,
                   kUnlimitedWindow,
                   cmp_ratio,
                   kReturnValues,
                   key_stride,
                   scale_stride,
                   indices,
                   values);
  return indices;
}

torch::Tensor quant_lightning_indexer_metadata(
    int64_t num_heads_q,
    int64_t num_heads_k,
    int64_t head_dim,
    const torch::Tensor& actual_seq_lengths_query,
    const torch::Tensor& actual_seq_lengths_key,
    int64_t max_seqlen_q,
    int64_t max_seqlen_k,
    int64_t sparse_count,
    int64_t cmp_ratio) {
  const torch::DeviceGuard guard(actual_seq_lengths_query.device());
  const int64_t batch_size = actual_seq_lengths_key.numel();
  auto metadata =
      torch::zeros({optiling::QLI_META_SIZE},
                   actual_seq_lengths_query.options().dtype(torch::kInt32));
  char query_layout[] = "TND";
  char key_layout[] = "PA_BSND";
  char* layout_query = query_layout;
  char* layout_key = key_layout;

  EXEC_NPU_CMD_EXT(aclnnXllmKernelQuantLightningIndexerMetadata,
                   actual_seq_lengths_query,
                   actual_seq_lengths_key,
                   num_heads_q,
                   num_heads_k,
                   head_dim,
                   kQuantMode,  // query_quant_mode
                   kQuantMode,  // key_quant_mode
                   batch_size,
                   max_seqlen_q,
                   max_seqlen_k,
                   layout_query,
                   layout_key,
                   sparse_count,
                   kSparseMode,
                   kUnlimitedWindow,
                   kUnlimitedWindow,
                   cmp_ratio,
                   metadata);
  return metadata;
}

void scatter_nd_update(torch::Tensor& value,
                       const torch::Tensor& indices,
                       const torch::Tensor& updates) {
  const torch::DeviceGuard guard(value.device());
  const torch::IntArrayRef strides = value.strides();
  EXEC_NPU_CMD_EXT(
      aclnnXllmKernelScatterNdUpdateV2, value, indices, updates, strides);
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(xllm_kernel, m) {
  m.def("prepare_ascendc_sparse_indexing() -> ()",
        &prepare_ascendc_sparse_indexing);
  m.def(
      "quant_lightning_indexer(Tensor query, Tensor key, Tensor weights, "
      "Tensor query_dequant_scale, Tensor key_dequant_scale, Tensor metadata, "
      "Tensor? query_seq_lengths, Tensor? key_seq_lengths, Tensor? "
      "block_table, "
      "int selected_count, int cmp_ratio) -> Tensor");
  m.def(
      "quant_lightning_indexer_metadata(int num_heads_q, int num_heads_k, "
      "int head_dim, Tensor actual_seq_lengths_query, Tensor "
      "actual_seq_lengths_key, "
      "int max_seqlen_q, int max_seqlen_k, int sparse_count, int cmp_ratio) -> "
      "Tensor");
  m.def(
      "scatter_nd_update(Tensor(a!) value, Tensor indices, Tensor updates) -> "
      "()");
}

TORCH_LIBRARY_IMPL(xllm_kernel, PrivateUse1, m) {
  m.impl("quant_lightning_indexer", TORCH_FN(quant_lightning_indexer));
  m.impl("quant_lightning_indexer_metadata",
         TORCH_FN(quant_lightning_indexer_metadata));
  m.impl("scatter_nd_update", TORCH_FN(scatter_nd_update));
}

}  // namespace xllm_kernel
