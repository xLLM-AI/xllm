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

#include "flashinfer_planinfo.h"

#include <glog/logging.h>

#include <vector>

#include "core/common/global_flags.h"
#include "core/platform/device.h"
#include "core/platform/platform.h"
#include "core/util/utils.h"
#include "flashinfer_workspace.h"
#include "kernels/cuda/utils.h"

using namespace xllm::kernel::cuda;

namespace xllm::layer::flashinfer {

// Helper function to deep copy ffi::Array<int64_t> to avoid lifetime issues
// with TVM runtime memory management
// This function immediately copies all data to avoid any dependency on TVM
// runtime
static ffi::Array<int64_t> deep_copy_plan_info(const ffi::Array<int64_t>& src) {
  // Get size first - this might fail if Array is invalid
  if (!src.defined()) {
    LOG(FATAL) << "src is not defined";
    return ffi::Array<int64_t>();
  }

  size_t src_size = src.size();
  if (src_size == 0) {
    return ffi::Array<int64_t>();
  }

  // Immediately extract all data to a vector to avoid any dependency on TVM
  // runtime
  std::vector<int64_t> temp_vec;
  temp_vec.reserve(src_size);
  // Use range-based for loop which is safer and more efficient
  // This immediately copies all elements before any potential invalidation
  for (const auto& elem : src) {
    temp_vec.push_back(elem);
  }

  // Create a new Array from the vector, which will have independent memory
  // This Array will not depend on TVM runtime lifetime
  return ffi::Array<int64_t>(temp_vec.begin(), temp_vec.end());
}

static torch::Tensor get_kv_len_arr_host(const AttentionMetadata& attn_meta) {
  if (attn_meta.kv_seq_lens.defined()) {
    return attn_meta.kv_seq_lens.to(torch::kCPU);
  }

  CHECK(attn_meta.kv_cu_seq_lens.defined())
      << "kv_seq_lens or kv_cu_seq_lens must be defined.";
  torch::Tensor kv_cu_seq_lens_host = attn_meta.kv_cu_seq_lens.to(torch::kCPU);
  return kv_cu_seq_lens_host.slice(0, 1) - kv_cu_seq_lens_host.slice(0, 0, -1);
}

void update_prefill_plan_info(std::shared_ptr<PlanInfo> plan_info,
                              const std::string& backend,
                              const AttentionMetadata& attn_meta,
                              c10::ScalarType query_dtype,
                              c10::ScalarType key_dtype,
                              c10::ScalarType output_dtype,
                              int32_t head_dim_qk,
                              int32_t head_dim_vo,
                              int32_t num_qo_heads,
                              int32_t num_kv_heads,
                              bool enable_cuda_graph) {
  CHECK(plan_info->layer_id != -1) << "Need to set layer_id to PlanInfo.";
  if (plan_info->layer_id != 0) return;

  const auto device =
      FlashinferWorkspace::get_instance().get_float_workspace_buffer().device();
  bind_tvmffi_stream_to_current_torch_stream(device);

  VLOG(kGraphExecutorLogVerboseLevel)
      << "update_prefill_plan_info: layer_id=" << plan_info->layer_id
      << ", enable_cuda_graph=" << enable_cuda_graph;

  auto float_workspace_buffer = to_ffi_tensor(
      FlashinferWorkspace::get_instance().get_float_workspace_buffer());
  auto int_workspace_buffer = to_ffi_tensor(
      FlashinferWorkspace::get_instance().get_int_workspace_buffer());
  auto page_locked_int_workspace_buffer =
      to_ffi_tensor(FlashinferWorkspace::get_instance()
                        .get_page_locked_int_workspace_buffer());
  plan_info->uri = get_batch_prefill_uri(backend,
                                         query_dtype,
                                         key_dtype,
                                         output_dtype,
                                         attn_meta.q_cu_seq_lens.scalar_type(),
                                         head_dim_qk,
                                         head_dim_vo,
                                         /*pos_encoding_mode=*/0,
                                         /*use_sliding_window=*/false,
                                         /*use_logits_soft_cap=*/false,
                                         /*use_fp16_qk_reduction=*/false);

  torch::Tensor qo_indptr_host = attn_meta.q_cu_seq_lens.to(torch::kCPU);
  torch::Tensor kv_cu_seq_lens_host = attn_meta.kv_cu_seq_lens.to(torch::kCPU);
  torch::Tensor kv_len_arr_host =
      kv_cu_seq_lens_host.slice(0, 1) - kv_cu_seq_lens_host.slice(0, 0, -1);
  const int64_t total_num_rows = qo_indptr_host[-1].item<int64_t>();
  const int64_t batch_size = qo_indptr_host.size(0) - 1;

  auto plan_func = get_function(plan_info->uri, "plan");
  // For sm90 architecture, the plan function doesn't accept
  // fixed_split_size / disable_split_kv / num_colocated_ctas
  ffi::Array<int64_t> plan_result =
      Platform::is_support_sm90a()
          ? plan_func(float_workspace_buffer,
                      int_workspace_buffer,
                      page_locked_int_workspace_buffer,
                      to_ffi_tensor(qo_indptr_host),
                      to_ffi_tensor(kv_cu_seq_lens_host),
                      to_ffi_tensor(kv_len_arr_host),
                      total_num_rows,
                      batch_size,
                      num_qo_heads,
                      num_kv_heads,
                      /*page_size=*/1,
                      enable_cuda_graph,
                      head_dim_qk,
                      head_dim_vo,
                      /*causal=*/true,
                      /*window_size_left=*/-1)
                .cast<ffi::Array<int64_t>>()
          : plan_func(float_workspace_buffer,
                      int_workspace_buffer,
                      page_locked_int_workspace_buffer,
                      to_ffi_tensor(qo_indptr_host),
                      to_ffi_tensor(kv_cu_seq_lens_host),
                      to_ffi_tensor(kv_len_arr_host),
                      total_num_rows,
                      batch_size,
                      num_qo_heads,
                      num_kv_heads,
                      /*page_size=*/1,
                      enable_cuda_graph,
                      head_dim_qk,
                      head_dim_vo,
                      /*causal=*/true,
                      /*window_size_left=*/-1,
                      /*fixed_split_size=*/-1,
                      /*disable_split_kv=*/false,
                      /*num_colocated_ctas=*/0)
                .cast<ffi::Array<int64_t>>();
  plan_info->plan_info = deep_copy_plan_info(plan_result);
}

void update_chunked_prefill_plan_info(std::shared_ptr<PlanInfo> plan_info,
                                      const std::string& backend,
                                      const AttentionMetadata& attn_meta,
                                      c10::ScalarType query_dtype,
                                      c10::ScalarType key_dtype,
                                      c10::ScalarType output_dtype,
                                      int32_t head_dim_qk,
                                      int32_t head_dim_vo,
                                      int32_t num_qo_heads,
                                      int32_t num_kv_heads,
                                      int32_t block_size,
                                      int32_t window_size_left,
                                      bool enable_cuda_graph,
                                      bool causal) {
  CHECK(plan_info->layer_id != -1) << "Need to set layer_id to PlanInfo.";
  if (plan_info->layer_id != 0) return;

  const auto device =
      FlashinferWorkspace::get_instance().get_float_workspace_buffer().device();
  bind_tvmffi_stream_to_current_torch_stream(device);

  VLOG(kGraphExecutorLogVerboseLevel)
      << "update_chunked_prefill_plan_info: layer_id=" << plan_info->layer_id
      << ", enable_cuda_graph=" << enable_cuda_graph;

  auto float_workspace_buffer = to_ffi_tensor(
      FlashinferWorkspace::get_instance().get_float_workspace_buffer());
  auto int_workspace_buffer = to_ffi_tensor(
      FlashinferWorkspace::get_instance().get_int_workspace_buffer());
  auto page_locked_int_workspace_buffer =
      to_ffi_tensor(FlashinferWorkspace::get_instance()
                        .get_page_locked_int_workspace_buffer());
  plan_info->uri =
      get_batch_prefill_uri(backend,
                            query_dtype,
                            key_dtype,
                            output_dtype,
                            attn_meta.paged_kv_indptr.scalar_type(),
                            head_dim_qk,
                            head_dim_vo,
                            /*pos_encoding_mode=*/0,
                            /*use_sliding_window=*/false,
                            /*use_logits_soft_cap=*/false,
                            /*use_fp16_qk_reduction=*/false);
  const int64_t batch_size = attn_meta.paged_kv_last_page_len.size(0);
  torch::Tensor qo_indptr_host;
  if (causal) {
    qo_indptr_host = attn_meta.qo_indptr.value().to(torch::kCPU);
  } else {
    qo_indptr_host = get_cache_buffer(batch_size + 1, torch::kCPU);
  }

  torch::Tensor paged_kv_indptr_host =
      attn_meta.paged_kv_indptr.to(torch::kCPU);
  torch::Tensor kv_len_arr_host = get_kv_len_arr_host(attn_meta);

  const int64_t total_num_rows = qo_indptr_host[-1].item<int64_t>();

  plan_info->plan_info = deep_copy_plan_info(
      get_function(plan_info->uri, "plan")(float_workspace_buffer,
                                           int_workspace_buffer,
                                           page_locked_int_workspace_buffer,
                                           to_ffi_tensor(qo_indptr_host),
                                           to_ffi_tensor(paged_kv_indptr_host),
                                           to_ffi_tensor(kv_len_arr_host),
                                           causal ? total_num_rows : batch_size,
                                           batch_size,
                                           num_qo_heads,  // num_qo_heads
                                           num_kv_heads,  // num_kv_heads
                                           block_size,    // block_size
                                           enable_cuda_graph,
                                           head_dim_qk,  // head_dim_qk
                                           head_dim_vo,  // head_dim_vo
                                           causal,
                                           window_size_left,
                                           /*fixed_split_size=*/-1,
                                           /*disable_split_kv=*/false,
                                           /*num_colocated_ctas=*/0)
          .cast<ffi::Array<int64_t>>());
}

void update_decode_plan_info(std::shared_ptr<PlanInfo> plan_info,
                             const std::string& backend,
                             const AttentionMetadata& attn_meta,
                             c10::ScalarType query_dtype,
                             c10::ScalarType key_dtype,
                             c10::ScalarType output_dtype,
                             int32_t head_dim_qk,
                             int32_t head_dim_vo,
                             int32_t num_qo_heads,
                             int32_t num_kv_heads,
                             int32_t block_size,
                             int32_t window_size_left,
                             bool enable_cuda_graph,
                             bool use_tensor_core) {
  CHECK(plan_info->layer_id != -1) << "Need to set layer_id to PlanInfo.";
  if (plan_info->layer_id != 0) return;

  if (use_tensor_core) {
    update_chunked_prefill_plan_info(plan_info,
                                     backend,
                                     attn_meta,
                                     query_dtype,
                                     key_dtype,
                                     output_dtype,
                                     head_dim_qk,
                                     head_dim_vo,
                                     num_qo_heads,
                                     num_kv_heads,
                                     block_size,
                                     window_size_left,
                                     enable_cuda_graph,
                                     /*causal=*/false);
  } else {
    const auto device = FlashinferWorkspace::get_instance()
                            .get_float_workspace_buffer()
                            .device();
    bind_tvmffi_stream_to_current_torch_stream(device);

    VLOG(kGraphExecutorLogVerboseLevel)
        << "update_decode_plan_info: layer_id=" << plan_info->layer_id
        << ", enable_cuda_graph=" << enable_cuda_graph;

    auto float_workspace_buffer = to_ffi_tensor(
        FlashinferWorkspace::get_instance().get_float_workspace_buffer());
    auto int_workspace_buffer = to_ffi_tensor(
        FlashinferWorkspace::get_instance().get_int_workspace_buffer());
    auto page_locked_int_workspace_buffer =
        to_ffi_tensor(FlashinferWorkspace::get_instance()
                          .get_page_locked_int_workspace_buffer());

    plan_info->uri =
        get_batch_decode_uri(query_dtype,
                             key_dtype,
                             output_dtype,
                             attn_meta.paged_kv_indptr.scalar_type(),
                             head_dim_qk,
                             head_dim_vo,
                             /*pos_encoding_mode=*/0,
                             /*use_sliding_window=*/false,
                             /*use_logits_soft_cap=*/false);

    torch::Tensor paged_kv_indptr_host =
        attn_meta.paged_kv_indptr.to(torch::kCPU);
    const int64_t batch_size = attn_meta.paged_kv_last_page_len.size(0);
    torch::Tensor empty_q_data =
        torch::empty({0}, torch::TensorOptions().dtype(query_dtype));
    torch::Tensor empty_kv_data =
        torch::empty({0}, torch::TensorOptions().dtype(key_dtype));

    plan_info->plan_info =
        deep_copy_plan_info(get_function(plan_info->uri, "plan")(
                                float_workspace_buffer,
                                int_workspace_buffer,
                                page_locked_int_workspace_buffer,
                                to_ffi_tensor(paged_kv_indptr_host),
                                batch_size,
                                num_qo_heads,
                                num_kv_heads,
                                block_size,
                                enable_cuda_graph,
                                window_size_left,
                                /*logits_soft_cap=*/0.0,
                                head_dim_qk,
                                head_dim_vo,
                                to_ffi_tensor(empty_q_data),
                                to_ffi_tensor(empty_kv_data))
                                .cast<ffi::Array<int64_t>>());
  }
}

}  // namespace xllm::layer::flashinfer
