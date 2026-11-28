// Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.

#include <torch/library.h>
#include <torch/types.h>

#include <array>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>

#include "csrc/ascendc/sparse_attn_sharedkv/op_kernel/sparse_attn_sharedkv_metadata.h"
#include "csrc/npu/ascendc_api.h"
#include "op_plugin/utils/op_api_common_base.h"

namespace xllm_kernel {
namespace {

const auto& sparse_attention_apis() {
  static const std::array apis{
      std::pair{"aclnnXllmKernelCompressor",
                load_ascendc_api("aclnnXllmKernelCompressor",
                                 "aclnnXllmKernelCompressorGetWorkspaceSize")},
      std::pair{"aclnnXllmKernelSparseAttnSharedkv",
                load_ascendc_api(
                    "aclnnXllmKernelSparseAttnSharedkv",
                    "aclnnXllmKernelSparseAttnSharedkvGetWorkspaceSize")},
      std::pair{
          "aclnnXllmKernelSparseAttnSharedkvMetadata",
          load_ascendc_api(
              "aclnnXllmKernelSparseAttnSharedkvMetadata",
              "aclnnXllmKernelSparseAttnSharedkvMetadataGetWorkspaceSize")},
      std::pair{"aclnnXllmKernelSparseFlashAttentionLse",
                load_ascendc_api(
                    "aclnnXllmKernelSparseFlashAttentionLse",
                    "aclnnXllmKernelSparseFlashAttentionLseGetWorkspaceSize")},
  };
  return apis;
}

void GetApiFunc(const char* name,
                const char* /*workspace_name*/,
                void*& launch,
                void*& workspace) {
  for (const auto& [api_name, api] : sparse_attention_apis()) {
    if (std::string_view(name) == api_name) {
      launch = api.launch;
      workspace = api.workspace;
      return;
    }
  }
  LOG(FATAL) << "Unknown sparse attention ACLNN entry point: " << name;
}

void prepare_ascendc_sparse_attention() { sparse_attention_apis(); }

std::tuple<torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor>
construct_compressor_output_tensor(const torch::Tensor& x,
                                   const torch::Tensor& norm_weight,
                                   const torch::Tensor& rope_sin,
                                   int64_t cmp_ratio,
                                   int64_t coff,
                                   bool enable_grad) {
  constexpr int64_t kBatchedRank = 3;
  torch::SmallVector<int64_t, 8> cmp_kv_size;
  torch::SmallVector<int64_t, 8> wkv_proj_size;
  torch::SmallVector<int64_t, 8> softmax_res_size;
  torch::SmallVector<int64_t, 8> norm_x_size;
  torch::SmallVector<int64_t, 8> norm_rstd_size;
  torch::Tensor cmp_kv;
  torch::Tensor wkv_proj;
  torch::Tensor softmax_res;
  torch::Tensor norm_x;
  torch::Tensor norm_rstd;
  int64_t cmp_s = 0;
  if (x.dim() == kBatchedRank) {
    cmp_s = (x.size(1) + cmp_ratio - 1) / cmp_ratio;
    cmp_kv_size = {x.size(0), cmp_s, norm_weight.size(0)};
    if (enable_grad) {
      wkv_proj_size = {x.size(0), x.size(1), coff * norm_weight.size(0)};
      softmax_res_size = {
          x.size(0), cmp_s, coff * cmp_ratio, norm_weight.size(0)};
      norm_x_size = {x.size(0), cmp_s, norm_weight.size(0)};
      norm_rstd_size = {x.size(0), cmp_s};
    }
  } else {
    cmp_s = rope_sin.size(0);
    cmp_kv_size = {cmp_s, norm_weight.size(0)};
    if (enable_grad) {
      wkv_proj_size = {x.size(0), coff * norm_weight.size(0)};
      softmax_res_size = {cmp_s, coff * cmp_ratio, norm_weight.size(0)};
      norm_x_size = {cmp_s, norm_weight.size(0)};
      norm_rstd_size = {cmp_s};
    }
  }

  cmp_kv = torch::zeros(cmp_kv_size, x.options().dtype(x.dtype()));
  if (enable_grad) {
    wkv_proj = torch::zeros(wkv_proj_size, x.options().dtype(x.dtype()));
    softmax_res = torch::zeros(softmax_res_size, x.options().dtype(x.dtype()));
    norm_x = torch::zeros(norm_x_size, x.options().dtype(x.dtype()));
    norm_rstd = torch::zeros(norm_rstd_size, x.options().dtype(x.dtype()));
  } else {
    wkv_proj = torch::empty({0}, x.options().dtype(x.dtype()));
    softmax_res = torch::empty({0}, x.options().dtype(x.dtype()));
    norm_x = torch::empty({0}, x.options().dtype(x.dtype()));
    norm_rstd = torch::empty({0}, x.options().dtype(x.dtype()));
  }

  return std::tuple<torch::Tensor,
                    torch::Tensor,
                    torch::Tensor,
                    torch::Tensor,
                    torch::Tensor>(
      cmp_kv, wkv_proj, softmax_res, norm_x, norm_rstd);
}

std::tuple<torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor>
compressor(const torch::Tensor& x,
           const torch::Tensor& wkv,
           const torch::Tensor& wgate,
           torch::Tensor& kv_state,
           torch::Tensor& score_state,
           const torch::Tensor& ape,
           const torch::Tensor& norm_weight,
           const torch::Tensor& rope_sin,
           const torch::Tensor& rope_cos,
           const std::optional<torch::Tensor>& kv_block_table,
           const std::optional<torch::Tensor>& score_block_table,
           const std::optional<torch::Tensor>& cu_seqlens,
           const std::optional<torch::Tensor>& seqused,
           const std::optional<torch::Tensor>& start_pos,
           int64_t rope_head_dim,
           int64_t cmp_ratio,
           int64_t coff,
           double norm_eps,
           int64_t rotary_mode,
           bool enable_grad) {
  constexpr int32_t DIM_1 = 1;
  constexpr int32_t DIM_2 = 2;
  constexpr int32_t DIM_3 = 3;
  constexpr int32_t VALUE_0 = 0;
  const torch::DeviceGuard guard(x.device());
  int64_t x_dim = x.dim();
  CHECK(x_dim == DIM_2 || x_dim == DIM_3)
      << "x dim num[" << x_dim << "] should be 2 or 3";

  int64_t norm_weight_dim = norm_weight.dim();
  CHECK(norm_weight_dim == DIM_1)
      << "norm_weight dim num[" << norm_weight_dim << "] should be 1";

  int64_t rope_sin_dim = rope_sin.dim();
  CHECK(rope_sin_dim == x_dim)
      << "rope_sin dim num[" << rope_sin_dim
      << "] should be equal to x dim num[" << x_dim << "]";

  CHECK(cmp_ratio != VALUE_0) << "cmp_ratio should not be 0";

  std::tuple<torch::Tensor,
             torch::Tensor,
             torch::Tensor,
             torch::Tensor,
             torch::Tensor>
      output = construct_compressor_output_tensor(
          x, norm_weight, rope_sin, cmp_ratio, coff, enable_grad);
  torch::Tensor cmp_kv = std::get<0>(output);
  torch::Tensor wkv_proj = std::get<1>(output);
  torch::Tensor softmax_res = std::get<2>(output);
  torch::Tensor norm_x = std::get<3>(output);
  torch::Tensor norm_rstd = std::get<4>(output);

  EXEC_NPU_CMD_EXT(aclnnXllmKernelCompressor,
                   x,
                   wkv,
                   wgate,
                   kv_state,
                   score_state,
                   ape,
                   norm_weight,
                   rope_sin,
                   rope_cos,
                   kv_block_table,
                   score_block_table,
                   cu_seqlens,
                   seqused,
                   start_pos,
                   rope_head_dim,
                   cmp_ratio,
                   coff,
                   norm_eps,
                   rotary_mode,
                   enable_grad,
                   cmp_kv,
                   wkv_proj,
                   softmax_res,
                   norm_x,
                   norm_rstd);

  return std::tuple<torch::Tensor,
                    torch::Tensor,
                    torch::Tensor,
                    torch::Tensor,
                    torch::Tensor>(
      cmp_kv, wkv_proj, softmax_res, norm_x, norm_rstd);
}

void check_sparse_attn_sharedkv_shape_and_dtype(const torch::Tensor& q,
                                                std::string_view layout_q,
                                                std::string_view layout_kv) {
  TORCH_CHECK(q.dim() >= 1,
              "Input tensor q's dim num should be at least 1, actual ",
              q.dim(),
              ".");
  TORCH_CHECK(q.dtype() == torch::kHalf || q.dtype() == torch::kBFloat16,
              "q should be FLOAT16 or BFLOAT16.");
  TORCH_CHECK(!layout_q.empty(), "layout_q should not be empty.");
  TORCH_CHECK(!layout_kv.empty(), "layout_kv should not be empty.");
}

torch::Tensor construct_sparse_attn_sharedkv_attn_out_tensor(
    const torch::Tensor& q) {
  return torch::empty(q.sizes(), q.options().dtype(q.dtype()));
}

torch::Tensor construct_sparse_attn_sharedkv_softmax_lse_tensor(
    const torch::Tensor& q,
    bool return_softmax_lse) {
  if (!return_softmax_lse) {
    return torch::empty({0}, q.options().dtype(torch::kFloat));
  }
  auto softmax_lse_shape = q.sizes().vec();
  softmax_lse_shape.back() = 1;
  return torch::empty(softmax_lse_shape, q.options().dtype(torch::kFloat));
}

int64_t get_sparse_attn_sharedkv_kv_stride(
    const std::optional<torch::Tensor>& kv) {
  if (!kv.has_value() || !kv.value().defined() || kv.value().dim() == 0) {
    return 0;
  }
  return kv.value().stride(0);
}

std::tuple<torch::Tensor, torch::Tensor> sparse_attn_sharedkv(
    const torch::Tensor& q,
    const std::optional<torch::Tensor>& ori_kv,
    const std::optional<torch::Tensor>& cmp_kv,
    const std::optional<torch::Tensor>& ori_sparse_indices,
    const std::optional<torch::Tensor>& cmp_sparse_indices,
    const std::optional<torch::Tensor>& ori_block_table,
    const std::optional<torch::Tensor>& cmp_block_table,
    const std::optional<torch::Tensor>& cu_seqlens_q,
    const std::optional<torch::Tensor>& cu_seqlens_ori_kv,
    const std::optional<torch::Tensor>& cu_seqlens_cmp_kv,
    const std::optional<torch::Tensor>& seqused_q,
    const std::optional<torch::Tensor>& seqused_kv,
    const std::optional<torch::Tensor>& sinks,
    const std::optional<torch::Tensor>& metadata,
    double softmax_scale,
    int64_t cmp_ratio,
    int64_t ori_mask_mode,
    int64_t cmp_mask_mode,
    int64_t ori_win_left,
    int64_t ori_win_right,
    std::string_view layout_q,
    std::string_view layout_kv,
    bool return_softmax_lse) {
  const torch::DeviceGuard guard(q.device());
  check_sparse_attn_sharedkv_shape_and_dtype(q, layout_q, layout_kv);
  torch::Tensor attn_out = construct_sparse_attn_sharedkv_attn_out_tensor(q);
  torch::Tensor softmax_lse =
      construct_sparse_attn_sharedkv_softmax_lse_tensor(q, return_softmax_lse);

  std::string layout_q_str = std::string(layout_q);
  std::string layout_kv_str = std::string(layout_kv);
  auto layout_q_arg = const_cast<char*>(layout_q_str.c_str());
  auto layout_kv_arg = const_cast<char*>(layout_kv_str.c_str());
  const int64_t ori_kv_stride = get_sparse_attn_sharedkv_kv_stride(ori_kv);
  const int64_t cmp_kv_stride = get_sparse_attn_sharedkv_kv_stride(cmp_kv);

  EXEC_NPU_CMD_EXT(aclnnXllmKernelSparseAttnSharedkv,
                   q,
                   ori_kv,
                   cmp_kv,
                   ori_sparse_indices,
                   cmp_sparse_indices,
                   ori_block_table,
                   cmp_block_table,
                   cu_seqlens_q,
                   cu_seqlens_ori_kv,
                   cu_seqlens_cmp_kv,
                   seqused_q,
                   seqused_kv,
                   sinks,
                   metadata,
                   softmax_scale,
                   cmp_ratio,
                   ori_mask_mode,
                   cmp_mask_mode,
                   ori_kv_stride,
                   cmp_kv_stride,
                   ori_win_left,
                   ori_win_right,
                   layout_q_arg,
                   layout_kv_arg,
                   return_softmax_lse,
                   attn_out,
                   softmax_lse);
  return std::make_tuple(attn_out, softmax_lse);
}

auto get_valid_tensor = [](const std::optional<torch::Tensor>& tensor_opt,
                           torch::Device device) {
  return tensor_opt.has_value()
             ? tensor_opt
             : torch::empty({0}, torch::dtype(torch::kInt32).device(device));
};

torch::Tensor sparse_attn_sharedkv_metadata(
    int64_t num_heads_q,
    int64_t num_heads_kv,
    int64_t head_dim,
    const std::optional<torch::Tensor>& cu_seqlens_q,
    const std::optional<torch::Tensor>& cu_seqlens_ori_kv,
    const std::optional<torch::Tensor>& cu_seqlens_cmp_kv,
    const std::optional<torch::Tensor>& seqused_q,
    const std::optional<torch::Tensor>& seqused_kv,
    int64_t batch_size,
    int64_t max_seqlen_q,
    int64_t max_seqlen_kv,
    int64_t ori_topk,
    int64_t cmp_topk,
    int64_t cmp_ratio,
    int64_t ori_mask_mode,
    int64_t cmp_mask_mode,
    int64_t ori_win_left,
    int64_t ori_win_right,
    std::string_view layout_q,
    std::string_view layout_kv,
    bool has_ori_kv,
    bool has_cmp_kv) {
  torch::Device output_device(std::string("npu"));
  if (cu_seqlens_q.has_value()) {
    output_device = cu_seqlens_q.value().device();
  } else if (cu_seqlens_ori_kv.has_value()) {
    output_device = cu_seqlens_ori_kv.value().device();
  } else if (cu_seqlens_cmp_kv.has_value()) {
    output_device = cu_seqlens_cmp_kv.value().device();
  } else if (seqused_q.has_value()) {
    output_device = seqused_q.value().device();
  } else if (seqused_kv.has_value()) {
    output_device = seqused_kv.value().device();
  }

  const torch::DeviceGuard guard(output_device);
  torch::Tensor output =
      torch::zeros({optiling::SAS_META_SIZE},
                   torch::dtype(torch::kInt32).device(output_device));
  auto cu_seqlens_q_val = get_valid_tensor(cu_seqlens_q, output_device);
  auto cu_seqlens_ori_kv_val =
      get_valid_tensor(cu_seqlens_ori_kv, output_device);
  auto cu_seqlens_cmp_kv_val =
      get_valid_tensor(cu_seqlens_cmp_kv, output_device);
  auto seqused_q_val = get_valid_tensor(seqused_q, output_device);
  auto seqused_kv_val = get_valid_tensor(seqused_kv, output_device);

  // convert str
  std::string layout_q_str = std::string(layout_q);
  std::string layout_kv_str = std::string(layout_kv);
  char* layout_q_ptr = layout_q_str.data();
  char* layout_kv_ptr = layout_kv_str.data();

  EXEC_NPU_CMD_EXT(aclnnXllmKernelSparseAttnSharedkvMetadata,
                   cu_seqlens_q_val,
                   cu_seqlens_ori_kv_val,
                   cu_seqlens_cmp_kv_val,
                   seqused_q_val,
                   seqused_kv_val,
                   num_heads_q,
                   num_heads_kv,
                   head_dim,
                   batch_size,
                   max_seqlen_q,
                   max_seqlen_kv,
                   ori_topk,
                   cmp_topk,
                   cmp_ratio,
                   ori_mask_mode,
                   cmp_mask_mode,
                   ori_win_left,
                   ori_win_right,
                   layout_q_ptr,
                   layout_kv_ptr,
                   has_ori_kv,
                   has_cmp_kv,
                   output);

  return output;
}

constexpr int64_t kDim0 = 0;
constexpr int64_t kDim1 = 1;
constexpr int64_t kDim2 = 2;
constexpr int64_t kDim3 = 3;

void check_sparse_flash_attention_lse_inputs(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& sparse_indices,
    int64_t sparse_block_size,
    const std::string& layout_query,
    const std::string& layout_kv) {
  TORCH_CHECK(query.numel() > 0, "Tensor query is empty.");
  TORCH_CHECK(key.numel() > 0, "Tensor key is empty.");
  TORCH_CHECK(value.numel() > 0, "Tensor value is empty.");
  TORCH_CHECK(sparse_indices.numel() > 0, "Tensor sparse_indices is empty.");
  TORCH_CHECK(
      query.dtype() == torch::kFloat16 || query.dtype() == torch::kBFloat16,
      "query should be FLOAT16 or BFLOAT16.");
  TORCH_CHECK(key.dtype() == query.dtype(),
              "key's dtype should be equal to query's dtype.");
  TORCH_CHECK(value.dtype() == query.dtype(),
              "value's dtype should be equal to query's dtype.");
  TORCH_CHECK(sparse_indices.dtype() == torch::kInt32,
              "sparse_indices should be INT32.");
  TORCH_CHECK(sparse_block_size > 0,
              "sparse_block_size should be greater than 0, actual ",
              sparse_block_size,
              ".");
  TORCH_CHECK(layout_query == "BSND" || layout_query == "TND",
              "The layout of query only support BSND and TND, but got ",
              layout_query);
  TORCH_CHECK(!layout_kv.empty(), "layout_kv should not be empty.");
}

void check_sparse_flash_attention_lse_output(const torch::Tensor& query,
                                             const torch::Tensor& output,
                                             const std::string& layout_query) {
  CHECK(query.dim() == (layout_query == "TND" ? 3 : 4));
  CHECK(output.device() == query.device())
      << "attention output must use the query device.";
  CHECK(output.dtype() == query.dtype())
      << "attention output dtype must match query dtype.";
  CHECK(output.is_contiguous()) << "attention output must be contiguous.";
  CHECK(output.sizes() == query.sizes())
      << "attention output shape must match query shape for layout "
      << layout_query;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
construct_sparse_flash_attention_lse_outputs(const torch::Tensor& query,
                                             const torch::Tensor& key,
                                             const std::string& layout_query,
                                             const std::string& layout_kv,
                                             bool return_softmax_lse) {
  torch::SmallVector<int64_t, 8> output_size;
  if (layout_query == "TND") {
    TORCH_CHECK(query.dim() == 3,
                "When the layout of query is TND, the query dimension must be "
                "3, but got ",
                query.dim());
    output_size = {query.size(kDim0), query.size(kDim1), query.size(kDim2)};
  } else {
    TORCH_CHECK(query.dim() == 4,
                "When the layout of query is BSND, the query dimension must "
                "be 4, but got ",
                query.dim());
    output_size = {query.size(kDim0),
                   query.size(kDim1),
                   query.size(kDim2),
                   query.size(kDim3)};
  }

  torch::Tensor attention_output =
      torch::empty(output_size, query.options().dtype(query.dtype()));
  torch::SmallVector<int64_t, 8> softmax_size;
  if (return_softmax_lse) {
    if (query.dim() == 3) {
      const int64_t kv_head_num =
          layout_kv == "PA_BSND" ? key.size(kDim2) : key.size(kDim1);
      softmax_size = {
          kv_head_num,
          query.size(kDim0),
          query.size(kDim1) / kv_head_num,
      };
    } else {
      softmax_size = {
          query.size(kDim0),
          key.size(kDim2),
          query.size(kDim1),
          query.size(kDim2) / key.size(kDim2),
      };
    }
  } else {
    softmax_size = {0};
  }

  torch::Tensor softmax_max =
      torch::empty(softmax_size, query.options().dtype(torch::kFloat32));
  torch::Tensor softmax_sum =
      torch::empty(softmax_size, query.options().dtype(torch::kFloat32));
  return {attention_output, softmax_max, softmax_sum};
}

void run_sparse_flash_attention_lse(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& sparse_indices,
    const std::optional<torch::Tensor>& block_table,
    const std::optional<torch::Tensor>& actual_seq_lengths_query,
    const std::optional<torch::Tensor>& actual_seq_lengths_kv,
    const std::optional<torch::Tensor>& query_rope,
    const std::optional<torch::Tensor>& key_rope,
    double scale_value,
    int64_t sparse_block_size,
    const std::string& layout_query,
    const std::string& layout_kv,
    int64_t sparse_mode,
    int64_t pre_tokens,
    int64_t next_tokens,
    int64_t attention_mode,
    bool return_softmax_lse,
    const torch::Tensor& attention_output,
    const torch::Tensor& softmax_max,
    const torch::Tensor& softmax_sum) {
  const torch::DeviceGuard guard(query.device());
  char* query_layout_ptr = const_cast<char*>(layout_query.c_str());
  char* kv_layout_ptr = const_cast<char*>(layout_kv.c_str());
  EXEC_NPU_CMD_EXT(aclnnXllmKernelSparseFlashAttentionLse,
                   query,
                   key,
                   value,
                   sparse_indices,
                   block_table,
                   actual_seq_lengths_query,
                   actual_seq_lengths_kv,
                   query_rope,
                   key_rope,
                   scale_value,
                   sparse_block_size,
                   query_layout_ptr,
                   kv_layout_ptr,
                   sparse_mode,
                   pre_tokens,
                   next_tokens,
                   attention_mode,
                   return_softmax_lse,
                   attention_output,
                   softmax_max,
                   softmax_sum);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
sparse_flash_attention_lse(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& sparse_indices,
    const std::optional<torch::Tensor>& block_table,
    const std::optional<torch::Tensor>& actual_seq_lengths_query,
    const std::optional<torch::Tensor>& actual_seq_lengths_kv,
    const std::optional<torch::Tensor>& query_rope,
    const std::optional<torch::Tensor>& key_rope,
    double scale_value,
    int64_t sparse_block_size,
    std::string_view layout_query,
    std::string_view layout_kv,
    int64_t sparse_mode,
    int64_t pre_tokens,
    int64_t next_tokens,
    int64_t attention_mode,
    bool return_softmax_lse) {
  std::string layout_query_str = std::string(layout_query);
  std::string layout_kv_str = std::string(layout_kv);
  check_sparse_flash_attention_lse_inputs(query,
                                          key,
                                          value,
                                          sparse_indices,
                                          sparse_block_size,
                                          layout_query_str,
                                          layout_kv_str);
  auto outputs = construct_sparse_flash_attention_lse_outputs(
      query, key, layout_query_str, layout_kv_str, return_softmax_lse);
  torch::Tensor attention_output = std::get<0>(outputs);
  torch::Tensor softmax_max = std::get<1>(outputs);
  torch::Tensor softmax_sum = std::get<2>(outputs);
  run_sparse_flash_attention_lse(query,
                                 key,
                                 value,
                                 sparse_indices,
                                 block_table,
                                 actual_seq_lengths_query,
                                 actual_seq_lengths_kv,
                                 query_rope,
                                 key_rope,
                                 scale_value,
                                 sparse_block_size,
                                 layout_query_str,
                                 layout_kv_str,
                                 sparse_mode,
                                 pre_tokens,
                                 next_tokens,
                                 attention_mode,
                                 return_softmax_lse,
                                 attention_output,
                                 softmax_max,
                                 softmax_sum);
  return {attention_output, softmax_max, softmax_sum};
}

torch::Tensor sparse_flash_attention_lse_out(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& sparse_indices,
    const std::optional<torch::Tensor>& block_table,
    const std::optional<torch::Tensor>& actual_seq_lengths_query,
    const std::optional<torch::Tensor>& actual_seq_lengths_kv,
    const std::optional<torch::Tensor>& query_rope,
    const std::optional<torch::Tensor>& key_rope,
    double scale_value,
    int64_t sparse_block_size,
    std::string_view layout_query,
    std::string_view layout_kv,
    int64_t sparse_mode,
    int64_t pre_tokens,
    int64_t next_tokens,
    int64_t attention_mode,
    bool return_softmax_lse,
    torch::Tensor& attention_output) {
  const torch::DeviceGuard guard(query.device());
  CHECK(!return_softmax_lse)
      << "sparse_flash_attention_lse_out only supports output without LSE";
  std::string layout_query_str = std::string(layout_query);
  std::string layout_kv_str = std::string(layout_kv);
  check_sparse_flash_attention_lse_inputs(query,
                                          key,
                                          value,
                                          sparse_indices,
                                          sparse_block_size,
                                          layout_query_str,
                                          layout_kv_str);
  check_sparse_flash_attention_lse_output(
      query, attention_output, layout_query_str);
  torch::Tensor empty_lse_max =
      torch::empty({0}, query.options().dtype(torch::kFloat32));
  torch::Tensor empty_lse_sum =
      torch::empty({0}, query.options().dtype(torch::kFloat32));
  run_sparse_flash_attention_lse(query,
                                 key,
                                 value,
                                 sparse_indices,
                                 block_table,
                                 actual_seq_lengths_query,
                                 actual_seq_lengths_kv,
                                 query_rope,
                                 key_rope,
                                 scale_value,
                                 sparse_block_size,
                                 layout_query_str,
                                 layout_kv_str,
                                 sparse_mode,
                                 pre_tokens,
                                 next_tokens,
                                 attention_mode,
                                 false,
                                 attention_output,
                                 empty_lse_max,
                                 empty_lse_sum);
  return attention_output;
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(xllm_kernel, m) {
  m.def("prepare_ascendc_sparse_attention() -> ()",
        &prepare_ascendc_sparse_attention);
  m.def(
      "compressor(Tensor x, Tensor wkv, Tensor wgate, Tensor(a!) kv_state, "
      "Tensor(b!) score_state, Tensor ape, Tensor norm_weight, Tensor "
      "rope_sin, Tensor rope_cos, Tensor? kv_block_table, Tensor? "
      "score_block_table, Tensor? cu_seqlens, Tensor? seqused, Tensor? "
      "start_pos, int rope_head_dim, int cmp_ratio, int coff, float "
      "norm_eps, int rotary_mode, bool enable_grad) -> (Tensor, Tensor, "
      "Tensor, Tensor, Tensor)");
  m.def(
      "sparse_attn_sharedkv(Tensor q, Tensor? ori_kv, Tensor? cmp_kv, "
      "Tensor? ori_sparse_indices, Tensor? cmp_sparse_indices, Tensor? "
      "ori_block_table, Tensor? cmp_block_table, Tensor? cu_seqlens_q, "
      "Tensor? cu_seqlens_ori_kv, Tensor? cu_seqlens_cmp_kv, Tensor? "
      "seqused_q, Tensor? seqused_kv, Tensor? sinks, Tensor? metadata, "
      "float softmax_scale, int cmp_ratio, int ori_mask_mode, int "
      "cmp_mask_mode, int ori_win_left, int ori_win_right, str layout_q, "
      "str layout_kv, bool return_softmax_lse) -> (Tensor, Tensor)");
  m.def(
      "sparse_attn_sharedkv_metadata(int num_heads_q, int num_heads_kv, int "
      "head_dim, Tensor? cu_seqlens_q, Tensor? cu_seqlens_ori_kv, Tensor? "
      "cu_seqlens_cmp_kv, Tensor? seqused_q, Tensor? seqused_kv, int "
      "batch_size, int max_seqlen_q, int max_seqlen_kv, int ori_topk, int "
      "cmp_topk, int cmp_ratio, int ori_mask_mode, int cmp_mask_mode, int "
      "ori_win_left, int ori_win_right, str layout_q, str layout_kv, bool "
      "has_ori_kv, bool has_cmp_kv) -> Tensor",
      &sparse_attn_sharedkv_metadata);
  m.def(
      "sparse_flash_attention_lse(Tensor query, Tensor key, Tensor value, "
      "Tensor sparse_indices, Tensor? block_table, Tensor? "
      "actual_seq_lengths_query, Tensor? actual_seq_lengths_kv, Tensor? "
      "query_rope, Tensor? key_rope, float scale_value, int sparse_block_size, "
      "str layout_query, str layout_kv, int sparse_mode, int "
      "pre_tokens=9223372036854775807, int next_tokens=9223372036854775807, "
      "int attention_mode=2, bool return_softmax_lse=False) -> (Tensor, "
      "Tensor, Tensor)");
  m.def(
      "sparse_flash_attention_lse_out(Tensor query, Tensor key, Tensor value, "
      "Tensor sparse_indices, Tensor? block_table, Tensor? "
      "actual_seq_lengths_query, Tensor? actual_seq_lengths_kv, Tensor? "
      "query_rope, Tensor? key_rope, float scale_value, int sparse_block_size, "
      "str layout_query, str layout_kv, int sparse_mode, int pre_tokens, int "
      "next_tokens, int attention_mode, bool return_softmax_lse, Tensor(a!) "
      "attention_output) -> Tensor(a!)");
}

TORCH_LIBRARY_IMPL(xllm_kernel, PrivateUse1, m) {
  m.impl("compressor", TORCH_FN(compressor));
  m.impl("sparse_attn_sharedkv", TORCH_FN(sparse_attn_sharedkv));
  m.impl("sparse_flash_attention_lse", TORCH_FN(sparse_flash_attention_lse));
  m.impl("sparse_flash_attention_lse_out",
         TORCH_FN(sparse_flash_attention_lse_out));
}

}  // namespace xllm_kernel
