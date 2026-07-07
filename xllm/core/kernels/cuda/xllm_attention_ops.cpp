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

// Stateless attention kernel ops for the Python model executor.
// Each op is a pure function with all state passed as explicit arguments.
// plan_info is a 1D int64 CPU tensor (~10-15 elements).

#include <glog/logging.h>
#include <torch/library.h>
#include <torch/torch.h>

#include "core/kernels/cuda/cuda_ops_api.h"
#include "core/kernels/cuda/utils.h"
#include "core/layers/common/attention_metadata.h"
#include "core/layers/cuda/flashinfer_planinfo.h"
#include "core/layers/cuda/flashinfer_workspace.h"

namespace xllm {
namespace {

// ----------- plan_info <-> Tensor conversion -----------

torch::Tensor plan_info_to_tensor(const ffi::Array<int64_t>& arr) {
  auto t = torch::empty({static_cast<int64_t>(arr.size())}, torch::kInt64);
  auto* ptr = t.data_ptr<int64_t>();
  for (size_t i = 0; i < arr.size(); i++) {
    ptr[i] = arr[i];
  }
  return t;
}

ffi::Array<int64_t> tensor_to_plan_info(const torch::Tensor& t) {
  CHECK(t.dtype() == torch::kInt64 && t.dim() == 1);
  auto* ptr = t.data_ptr<int64_t>();
  std::vector<int64_t> vec(ptr, ptr + t.numel());
  return ffi::Array<int64_t>(vec.begin(), vec.end());
}

c10::ScalarType parse_dtype(const std::string& s) {
  if (s == "bfloat16") return torch::kBFloat16;
  if (s == "float16") return torch::kFloat16;
  if (s == "float32") return torch::kFloat32;
  LOG(FATAL) << "Unknown dtype string: " << s;
  return torch::kBFloat16;
}

// ----------- reshape_paged_cache -----------

torch::Tensor reshape_paged_cache_op(const torch::Tensor& slot_mapping,
                                     const torch::Tensor& k,
                                     const torch::Tensor& v,
                                     const torch::Tensor& k_cache,
                                     const torch::Tensor& v_cache) {
  kernel::cuda::reshape_paged_cache(slot_mapping, k, v, k_cache, v_cache);
  return k_cache;
}

// ----------- plan ops (workspace from singleton internally) -----------

std::tuple<torch::Tensor, std::string> update_prefill_plan_op(
    const torch::Tensor& q_cu_seq_lens,
    const torch::Tensor& kv_cu_seq_lens,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    const std::string& dtype_q,
    const std::string& dtype_kv,
    const std::string& dtype_o) {
  auto plan_info = std::make_shared<layer::PlanInfo>();
  plan_info->layer_id = 0;

  layer::AttentionMetadata meta;
  meta.q_cu_seq_lens = q_cu_seq_lens;
  meta.kv_cu_seq_lens = kv_cu_seq_lens;
  meta.enable_cuda_graph = false;

  std::string backend =
      kernel::cuda::determine_attention_backend(0, false, false);

  layer::flashinfer::update_prefill_plan_info(
      plan_info,
      backend,
      meta,
      parse_dtype(dtype_q),
      parse_dtype(dtype_kv),
      parse_dtype(dtype_o),
      static_cast<int32_t>(head_dim),
      static_cast<int32_t>(head_dim),
      static_cast<int32_t>(num_heads),
      static_cast<int32_t>(num_kv_heads),
      false);

  return {plan_info_to_tensor(plan_info->plan_info), plan_info->uri};
}

std::tuple<torch::Tensor, std::string> update_decode_plan_op(
    const torch::Tensor& paged_kv_indptr,
    const torch::Tensor& paged_kv_last_page_len,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t block_size,
    int64_t sliding_window,
    bool use_tensor_core,
    bool enable_cuda_graph,
    const std::string& dtype_q,
    const std::string& dtype_kv,
    const std::string& dtype_o) {
  auto plan_info = std::make_shared<layer::PlanInfo>();
  plan_info->layer_id = 0;

  layer::AttentionMetadata meta;
  meta.paged_kv_indptr = paged_kv_indptr;
  meta.paged_kv_last_page_len = paged_kv_last_page_len;
  meta.enable_cuda_graph = enable_cuda_graph;

  // enable_cuda_graph=true makes flashinfer emit a fixed-layout plan (constant
  // grid/tiling independent of the actual seqlens), which is required for the
  // Python-side decode full CUDA graph: a single captured graph must replay
  // correctly as sequence length grows. The per-step seqlen tiling is written
  // into the fixed-address int workspace buffer, refreshed by re-planning
  // before each replay.
  layer::flashinfer::update_decode_plan_info(
      plan_info,
      "fa2",
      meta,
      parse_dtype(dtype_q),
      parse_dtype(dtype_kv),
      parse_dtype(dtype_o),
      static_cast<int32_t>(head_dim),
      static_cast<int32_t>(head_dim),
      static_cast<int32_t>(num_heads),
      static_cast<int32_t>(num_kv_heads),
      static_cast<int32_t>(block_size),
      static_cast<int32_t>(sliding_window),
      enable_cuda_graph,
      use_tensor_core);

  return {plan_info_to_tensor(plan_info->plan_info), plan_info->uri};
}

std::tuple<torch::Tensor, std::string> update_chunked_prefill_plan_op(
    const torch::Tensor& paged_kv_indptr,
    const torch::Tensor& paged_kv_last_page_len,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t block_size,
    int64_t sliding_window,
    const std::string& dtype_q,
    const std::string& dtype_kv,
    const std::string& dtype_o) {
  auto plan_info = std::make_shared<layer::PlanInfo>();
  plan_info->layer_id = 0;

  layer::AttentionMetadata meta;
  meta.paged_kv_indptr = paged_kv_indptr;
  meta.paged_kv_last_page_len = paged_kv_last_page_len;
  meta.enable_cuda_graph = false;

  layer::flashinfer::update_chunked_prefill_plan_info(
      plan_info,
      "fa2",
      meta,
      parse_dtype(dtype_q),
      parse_dtype(dtype_kv),
      parse_dtype(dtype_o),
      static_cast<int32_t>(head_dim),
      static_cast<int32_t>(head_dim),
      static_cast<int32_t>(num_heads),
      static_cast<int32_t>(num_kv_heads),
      static_cast<int32_t>(block_size),
      static_cast<int32_t>(sliding_window),
      false);

  return {plan_info_to_tensor(plan_info->plan_info), plan_info->uri};
}

// ----------- kernel ops (workspace from singleton) -----------

torch::Tensor batch_prefill_op(const torch::Tensor& q,
                               const torch::Tensor& k,
                               const torch::Tensor& v,
                               const torch::Tensor& q_cu_seq_lens,
                               const torch::Tensor& kv_cu_seq_lens,
                               const torch::Tensor& plan_info_tensor,
                               const std::string& uri,
                               double scale,
                               int64_t sliding_window,
                               torch::Tensor output) {
  auto& ws = layer::flashinfer::FlashinferWorkspace::get_instance();
  std::optional<torch::Tensor> output_lse = std::nullopt;
  auto plan_info = tensor_to_plan_info(plan_info_tensor);

  auto float_ws = ws.get_float_workspace_buffer();
  auto int_ws = ws.get_int_workspace_buffer();
  auto page_locked_ws = ws.get_page_locked_int_workspace_buffer();
  auto q_mut = q;
  auto k_mut = k;
  auto v_mut = v;
  auto q_cu = q_cu_seq_lens;
  auto kv_cu = kv_cu_seq_lens;

  kernel::cuda::batch_prefill(uri,
                              plan_info,
                              float_ws,
                              int_ws,
                              page_locked_ws,
                              q_mut,
                              k_mut,
                              v_mut,
                              q_cu,
                              kv_cu,
                              sliding_window,
                              scale,
                              output,
                              output_lse);
  return output;
}

torch::Tensor batch_decode_op(const torch::Tensor& q,
                              const torch::Tensor& k_cache,
                              const torch::Tensor& v_cache,
                              const torch::Tensor& paged_kv_indptr,
                              const torch::Tensor& paged_kv_indices,
                              const torch::Tensor& paged_kv_last_page_len,
                              const torch::Tensor& plan_info_tensor,
                              const std::string& uri,
                              double scale,
                              int64_t sliding_window,
                              bool use_tensor_core,
                              const std::optional<torch::Tensor>& qo_indptr,
                              torch::Tensor output) {
  auto& ws = layer::flashinfer::FlashinferWorkspace::get_instance();
  std::optional<torch::Tensor> output_lse = std::nullopt;
  auto plan_info = tensor_to_plan_info(plan_info_tensor);

  auto float_ws = ws.get_float_workspace_buffer();
  auto int_ws = ws.get_int_workspace_buffer();
  auto page_locked_ws = ws.get_page_locked_int_workspace_buffer();
  auto q_mut = q;
  auto k_cache_mut = k_cache;
  auto v_cache_mut = v_cache;
  auto indptr = paged_kv_indptr;
  auto indices = paged_kv_indices;
  auto last_page_len = paged_kv_last_page_len;

  kernel::cuda::batch_decode(uri,
                             plan_info,
                             float_ws,
                             int_ws,
                             page_locked_ws,
                             q_mut,
                             k_cache_mut,
                             v_cache_mut,
                             indptr,
                             indices,
                             last_page_len,
                             sliding_window,
                             scale,
                             output,
                             output_lse,
                             use_tensor_core,
                             qo_indptr);
  return output;
}

torch::Tensor batch_chunked_prefill_op(
    const torch::Tensor& q,
    const torch::Tensor& k_cache,
    const torch::Tensor& v_cache,
    const torch::Tensor& paged_kv_indptr,
    const torch::Tensor& paged_kv_indices,
    const torch::Tensor& paged_kv_last_page_len,
    const torch::Tensor& plan_info_tensor,
    const std::string& uri,
    double scale,
    int64_t sliding_window,
    const std::optional<torch::Tensor>& qo_indptr,
    torch::Tensor output) {
  auto& ws = layer::flashinfer::FlashinferWorkspace::get_instance();
  std::optional<torch::Tensor> output_lse = std::nullopt;
  auto plan_info = tensor_to_plan_info(plan_info_tensor);

  auto float_ws = ws.get_float_workspace_buffer();
  auto int_ws = ws.get_int_workspace_buffer();
  auto page_locked_ws = ws.get_page_locked_int_workspace_buffer();
  auto q_mut = q;
  auto k_cache_mut = k_cache;
  auto v_cache_mut = v_cache;
  auto indptr = paged_kv_indptr;
  auto indices = paged_kv_indices;
  auto last_page_len = paged_kv_last_page_len;

  kernel::cuda::batch_chunked_prefill(uri,
                                      plan_info,
                                      float_ws,
                                      int_ws,
                                      page_locked_ws,
                                      q_mut,
                                      k_cache_mut,
                                      v_cache_mut,
                                      indptr,
                                      indices,
                                      last_page_len,
                                      sliding_window,
                                      scale,
                                      output,
                                      output_lse,
                                      qo_indptr);
  return output;
}

}  // namespace

// Force-link anchor (same pattern as xllm_ops_library.h).
void ensure_xllm_attention_ops_registered() {}

}  // namespace xllm

TORCH_LIBRARY_FRAGMENT(xllm_ops, m) {
  m.def(
      "reshape_paged_cache(Tensor slot_mapping, Tensor k, Tensor v, "
      "Tensor(a!) k_cache, Tensor(b!) v_cache) -> Tensor");
  m.def(
      "update_prefill_plan(Tensor q_cu_seq_lens, "
      "Tensor kv_cu_seq_lens, int num_heads, int num_kv_heads, int head_dim, "
      "str dtype_q, str dtype_kv, str dtype_o) -> (Tensor, str)");
  m.def(
      "update_decode_plan(Tensor paged_kv_indptr, "
      "Tensor paged_kv_last_page_len, int num_heads, int num_kv_heads, "
      "int head_dim, int block_size, int sliding_window, "
      "bool use_tensor_core, bool enable_cuda_graph, "
      "str dtype_q, str dtype_kv, str dtype_o) "
      "-> (Tensor, str)");
  m.def(
      "update_chunked_prefill_plan("
      "Tensor paged_kv_indptr, Tensor paged_kv_last_page_len, "
      "int num_heads, int num_kv_heads, int head_dim, int block_size, "
      "int sliding_window, str dtype_q, str dtype_kv, str dtype_o) "
      "-> (Tensor, str)");
  m.def(
      "batch_prefill(Tensor q, Tensor k, Tensor v, "
      "Tensor q_cu_seq_lens, Tensor kv_cu_seq_lens, "
      "Tensor plan_info, str uri, "
      "float scale, int sliding_window, Tensor(a!) output) -> Tensor");
  m.def(
      "batch_decode(Tensor q, Tensor k_cache, Tensor v_cache, "
      "Tensor paged_kv_indptr, Tensor paged_kv_indices, "
      "Tensor paged_kv_last_page_len, "
      "Tensor plan_info, str uri, float scale, int sliding_window, "
      "bool use_tensor_core, Tensor? qo_indptr, Tensor(a!) output) -> Tensor");
  m.def(
      "batch_chunked_prefill(Tensor q, Tensor k_cache, Tensor v_cache, "
      "Tensor paged_kv_indptr, Tensor paged_kv_indices, "
      "Tensor paged_kv_last_page_len, "
      "Tensor plan_info, str uri, float scale, int sliding_window, "
      "Tensor? qo_indptr, Tensor(a!) output) -> Tensor");
}

TORCH_LIBRARY_IMPL(xllm_ops, CUDA, m) {
  m.impl("reshape_paged_cache", TORCH_FN(xllm::reshape_paged_cache_op));
  m.impl("update_prefill_plan", TORCH_FN(xllm::update_prefill_plan_op));
  m.impl("update_decode_plan", TORCH_FN(xllm::update_decode_plan_op));
  m.impl("update_chunked_prefill_plan",
         TORCH_FN(xllm::update_chunked_prefill_plan_op));
  m.impl("batch_prefill", TORCH_FN(xllm::batch_prefill_op));
  m.impl("batch_decode", TORCH_FN(xllm::batch_decode_op));
  m.impl("batch_chunked_prefill", TORCH_FN(xllm::batch_chunked_prefill_op));
}
