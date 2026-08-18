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

#include "layers/musa/flashinfer_planinfo.h"

#include <glog/logging.h>

#include <vector>

#include "core/platform/device.h"
#include "core/platform/platform.h"
#include "core/util/utils.h"
#include "kernels/musa/musa_tvmffi_stream.h"
#include "layers/cuda/flashinfer_workspace.h"

namespace xllm::layer::musa::flashinfer {

using ::xllm::kernel::musa::get_batch_decode_uri;
using ::xllm::kernel::musa::get_batch_prefill_uri;
using ::xllm::kernel::musa::get_cache_buffer;
using ::xllm::kernel::musa::get_function;
using ::xllm::kernel::musa::to_ffi_tensor;
using ::xllm::kernel::musa::TvmffiStreamGuard;
using ::xllm::layer::flashinfer::FlashinferWorkspace;

namespace {

ffi::Array<int64_t> deep_copy_plan_info(const ffi::Array<int64_t>& src) {
  if (!src.defined()) {
    LOG(FATAL) << "src is not defined";
    return ffi::Array<int64_t>();
  }

  size_t src_size = src.size();
  if (src_size == 0) {
    return ffi::Array<int64_t>();
  }

  std::vector<int64_t> temp_vec;
  temp_vec.reserve(src_size);
  for (const auto& elem : src) {
    temp_vec.push_back(elem);
  }

  return ffi::Array<int64_t>(temp_vec.begin(), temp_vec.end());
}

torch::Tensor get_kv_len_arr_host(
    const xllm::layer::AttentionMetadata& attn_meta) {
  if (attn_meta.kv_seq_lens.defined()) {
    return attn_meta.kv_seq_lens.to(torch::kCPU);
  }

  CHECK(attn_meta.kv_cu_seq_lens.defined())
      << "kv_seq_lens or kv_cu_seq_lens must be defined.";
  torch::Tensor kv_cu_seq_lens_host = attn_meta.kv_cu_seq_lens.to(torch::kCPU);
  return kv_cu_seq_lens_host.slice(/*dim=*/0, /*start=*/1) -
         kv_cu_seq_lens_host.slice(/*dim=*/0, /*start=*/0, /*end=*/-1);
}

}  // namespace

void update_prefill_plan_info(std::shared_ptr<PlanInfo> plan_info,
                              const std::string& backend,
                              const ::xllm::layer::AttentionMetadata& attn_meta,
                              torch::ScalarType query_dtype,
                              torch::ScalarType key_dtype,
                              torch::ScalarType output_dtype,
                              int32_t head_dim_qk,
                              int32_t head_dim_vo,
                              int32_t num_qo_heads,
                              int32_t num_kv_heads,
                              bool enable_cuda_graph) {
  CHECK(plan_info->layer_id != -1) << "Need to set layer_id to PlanInfo.";
  if (plan_info->plan_info.size() > 0) {
    return;
  }

  const auto device =
      FlashinferWorkspace::get_instance().get_float_workspace_buffer().device();
  TvmffiStreamGuard stream_guard(device);

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
      kv_cu_seq_lens_host.slice(/*dim=*/0, /*start=*/1) -
      kv_cu_seq_lens_host.slice(/*dim=*/0, /*start=*/0, /*end=*/-1);
  const int64_t total_num_rows = qo_indptr_host[-1].item<int64_t>();
  const int64_t batch_size = qo_indptr_host.size(0) - 1;

  auto plan_func = get_function(plan_info->uri, "plan");
  VLOG(kGraphExecutorLogVerboseLevel)
      << "[FFI-TRACE] prefill plan() uri=" << plan_info->uri
      << " layer_id=" << plan_info->layer_id
      << " sm90a=" << Platform::is_support_sm90a()
      << " enable_cuda_graph=" << enable_cuda_graph
      << " total_num_rows=" << total_num_rows << " batch_size=" << batch_size
      << " num_qo_heads=" << num_qo_heads << " num_kv_heads=" << num_kv_heads
      << " head_dim_qk=" << head_dim_qk << " head_dim_vo=" << head_dim_vo;
  ffi::Array<int64_t> plan_result;
  try {
    // For sm90 architecture, the plan function doesn't accept
    // fixed_split_size / disable_split_kv / num_colocated_ctas
    plan_result = Platform::is_support_sm90a()
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
  } catch (const std::exception& e) {
    LOG(FATAL) << "[FFI-TRACE] prefill plan() THREW: " << e.what()
               << " | uri=" << plan_info->uri
               << " layer_id=" << plan_info->layer_id;
  }
  VLOG(kGraphExecutorLogVerboseLevel)
      << "[FFI-TRACE] prefill plan() OK, result.size=" << plan_result.size();
  plan_info->plan_info = deep_copy_plan_info(plan_result);
}

void update_chunked_prefill_plan_info(
    std::shared_ptr<PlanInfo> plan_info,
    const std::string& backend,
    const ::xllm::layer::AttentionMetadata& attn_meta,
    torch::ScalarType query_dtype,
    torch::ScalarType key_dtype,
    torch::ScalarType output_dtype,
    int32_t head_dim_qk,
    int32_t head_dim_vo,
    int32_t num_qo_heads,
    int32_t num_kv_heads,
    int32_t block_size,
    int32_t window_size_left,
    bool enable_cuda_graph,
    bool causal,
    int32_t max_kv_blocks_per_seq) {
  CHECK(plan_info->layer_id != -1) << "Need to set layer_id to PlanInfo.";
  if (plan_info->plan_info.size() > 0) {
    return;
  }

  const auto device =
      FlashinferWorkspace::get_instance().get_float_workspace_buffer().device();
  TvmffiStreamGuard stream_guard(device);

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
  if (causal && attn_meta.qo_indptr.has_value() &&
      attn_meta.qo_indptr->defined()) {
    qo_indptr_host = attn_meta.qo_indptr.value().to(torch::kCPU);
  } else {
    qo_indptr_host = get_cache_buffer(batch_size + 1, torch::kCPU);
  }

  const Fa3AttentionMetadata& fa3_metadata = attn_meta.fa3_metadata;
  torch::Tensor paged_kv_indptr_host =
      fa3_metadata.paged_kv_indptr_host.defined()
          ? fa3_metadata.paged_kv_indptr_host
          : attn_meta.paged_kv_indptr.to(torch::kCPU);
  torch::Tensor kv_len_arr_host = get_kv_len_arr_host(attn_meta);

  // Capture a plan large enough for future KV block growth.
  if (enable_cuda_graph && max_kv_blocks_per_seq > 0 && batch_size > 0 &&
      paged_kv_indptr_host.defined()) {
    auto opts = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
    torch::Tensor synth_indptr_host = torch::empty({batch_size + 1}, opts);
    auto* p = synth_indptr_host.data_ptr<int32_t>();
    for (int64_t i = 0; i <= batch_size; ++i) {
      p[i] = static_cast<int32_t>(i * max_kv_blocks_per_seq);
    }
    paged_kv_indptr_host = synth_indptr_host;
  }

  const int64_t total_num_rows = qo_indptr_host[-1].item<int64_t>();

  VLOG(kGraphExecutorLogVerboseLevel)
      << "[FFI-TRACE] chunked_prefill plan() uri=" << plan_info->uri
      << " layer_id=" << plan_info->layer_id << " causal=" << causal
      << " enable_cuda_graph=" << enable_cuda_graph
      << " total_num_rows=" << total_num_rows << " batch_size=" << batch_size
      << " block_size=" << block_size
      << " window_size_left=" << window_size_left;
  ffi::Array<int64_t> chunked_plan_result;
  try {
    chunked_plan_result = get_function(plan_info->uri, "plan")(
                              float_workspace_buffer,
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
                              .cast<ffi::Array<int64_t>>();
  } catch (const std::exception& e) {
    LOG(FATAL) << "[FFI-TRACE] chunked_prefill plan() THREW: " << e.what()
               << " | uri=" << plan_info->uri
               << " layer_id=" << plan_info->layer_id;
  }
  VLOG(kGraphExecutorLogVerboseLevel)
      << "[FFI-TRACE] chunked_prefill plan() OK, result.size="
      << chunked_plan_result.size();
  plan_info->plan_info = deep_copy_plan_info(chunked_plan_result);
}

void update_decode_plan_info(std::shared_ptr<PlanInfo> plan_info,
                             const std::string& backend,
                             const ::xllm::layer::AttentionMetadata& attn_meta,
                             torch::ScalarType query_dtype,
                             torch::ScalarType key_dtype,
                             torch::ScalarType output_dtype,
                             int32_t head_dim_qk,
                             int32_t head_dim_vo,
                             int32_t num_qo_heads,
                             int32_t num_kv_heads,
                             int32_t block_size,
                             int32_t window_size_left,
                             bool enable_cuda_graph,
                             bool use_tensor_core,
                             int32_t max_kv_blocks_per_seq) {
  CHECK(plan_info->layer_id != -1) << "Need to set layer_id to PlanInfo.";
  if (plan_info->plan_info.size() > 0) {
    return;
  }

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
                                     /*causal=*/false,
                                     max_kv_blocks_per_seq);
  } else {
    const auto device = FlashinferWorkspace::get_instance()
                            .get_float_workspace_buffer()
                            .device();
    TvmffiStreamGuard stream_guard(device);

    VLOG(kGraphExecutorLogVerboseLevel)
        << "update_decode_plan_info: layer_id=" << plan_info->layer_id
        << ", enable_cuda_graph=" << enable_cuda_graph;

    auto float_workspace_buffer = to_ffi_tensor(
        FlashinferWorkspace::get_instance().get_float_workspace_buffer());
    auto int_workspace_buffer = to_ffi_tensor(
        FlashinferWorkspace::get_instance().get_int_workspace_buffer());

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

    const Fa3AttentionMetadata& fa3_metadata = attn_meta.fa3_metadata;
    torch::Tensor paged_kv_indptr_host =
        fa3_metadata.paged_kv_indptr_host.defined()
            ? fa3_metadata.paged_kv_indptr_host
            : attn_meta.paged_kv_indptr.to(torch::kCPU);
    const int64_t batch_size = attn_meta.paged_kv_last_page_len.size(0);

    // Capture a plan large enough for future KV block growth.
    if (enable_cuda_graph && max_kv_blocks_per_seq > 0 && batch_size > 0) {
      auto opts =
          torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
      torch::Tensor synth_indptr_host = torch::empty({batch_size + 1}, opts);
      auto* p = synth_indptr_host.data_ptr<int32_t>();
      for (int64_t i = 0; i <= batch_size; ++i) {
        p[i] = static_cast<int32_t>(i * max_kv_blocks_per_seq);
      }
      VLOG(kGraphExecutorLogVerboseLevel)
          << "[FFI-TRACE] decode plan(): overriding paged_kv_indptr_host with "
          << "worst-case max-block layout for CUDA graph capture. batch_size="
          << batch_size << " max_kv_blocks_per_seq=" << max_kv_blocks_per_seq
          << " original_indptr_host[bs]="
          << (paged_kv_indptr_host.defined() && paged_kv_indptr_host.numel() > 0
                  ? paged_kv_indptr_host[batch_size].item<int32_t>()
                  : -1)
          << " synth_indptr_host[bs]=" << p[batch_size];
      paged_kv_indptr_host = synth_indptr_host;
    }
    torch::Tensor empty_q_data =
        torch::empty({0}, torch::TensorOptions().dtype(query_dtype));
    torch::Tensor empty_kv_data =
        torch::empty({0}, torch::TensorOptions().dtype(key_dtype));

    VLOG(kGraphExecutorLogVerboseLevel)
        << "[FFI-TRACE] decode plan() uri=" << plan_info->uri
        << " layer_id=" << plan_info->layer_id
        << " enable_cuda_graph=" << enable_cuda_graph
        << " batch_size=" << batch_size << " block_size=" << block_size
        << " window_size_left=" << window_size_left;
    ffi::Array<int64_t> decode_plan_result;
    try {
      decode_plan_result = get_function(plan_info->uri, "plan")(
                               float_workspace_buffer,
                               int_workspace_buffer,
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
                               .cast<ffi::Array<int64_t>>();
    } catch (const std::exception& e) {
      LOG(FATAL) << "[FFI-TRACE] decode plan() THREW: " << e.what()
                 << " | uri=" << plan_info->uri
                 << " layer_id=" << plan_info->layer_id;
    }
    VLOG(kGraphExecutorLogVerboseLevel)
        << "[FFI-TRACE] decode plan() OK, result.size="
        << decode_plan_result.size();
    plan_info->plan_info = deep_copy_plan_info(decode_plan_result);
  }
}

}  // namespace xllm::layer::musa::flashinfer
