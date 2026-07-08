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

using namespace xllm::kernel::cuda;

namespace {

// Helper function to deep copy ffi::Array<int64_t> to avoid lifetime issues
// with TVM runtime memory management
// This function immediately copies all data to avoid any dependency on TVM
// runtime
ffi::Array<int64_t> deep_copy_plan_info(const ffi::Array<int64_t>& src) {
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

torch::Tensor get_kv_len_arr_host(const AttentionMetadata& attn_meta) {
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

namespace xllm::layer::flashinfer {

void update_prefill_plan_info(std::shared_ptr<PlanInfo> plan_info,
                              const std::string& backend,
                              const AttentionMetadata& attn_meta,
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
  MusaTvmffiStreamGuard stream_guard(device);

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
  // Demoted from LOG(INFO) -- this fires once per full-attention layer per
  // prefill step and adds non-trivial latency (glog mutex + sink write) on the
  // hot path. Re-enable with `--v=kGraphExecutorLogVerboseLevel` when
  // diagnosing FFI plan() shape/dtype mismatches.
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

void update_chunked_prefill_plan_info(std::shared_ptr<PlanInfo> plan_info,
                                      const std::string& backend,
                                      const AttentionMetadata& attn_meta,
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
  MusaTvmffiStreamGuard stream_guard(device);

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

  // CUDA-graph correctness: same rationale as update_decode_plan_info. The
  // plan is cached on PlanInfo and reused by every captured replay; if the
  // warmup-time layout has fewer KV blocks than future replays will need
  // (e.g., decode after a block crossover), the cached plan dispatches
  // insufficient work and the captured attention kernel silently underreads.
  // Override paged_kv_indptr_host with the worst-case [0, max, 2*max, ...,
  // bs*max] layout for the plan call so the cached plan covers any future
  // runtime block count <= max_kv_blocks_per_seq.
  if (enable_cuda_graph && max_kv_blocks_per_seq > 0 && batch_size > 0 &&
      paged_kv_indptr_host.defined()) {
    auto opts = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
    torch::Tensor synth_indptr_host = torch::empty({batch_size + 1}, opts);
    int32_t* p = synth_indptr_host.data_ptr<int32_t>();
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
                             const AttentionMetadata& attn_meta,
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
    MusaTvmffiStreamGuard stream_guard(device);

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

    // Prefer the host mirror pre-staged by AttentionMetadataBuilder (see Plan
    // v2 in attention_metadata.h / batch_input_builder.cpp). The Mate FFI
    // decode `plan` consumes a kDLCPU pointer; using the cache avoids one more
    // D2H per forward step (plan runs once at layer 0, gated by
    // plan_info->plan_info being empty). Critical for CUDA graph capture: any
    // .to(kCPU) inside the captured region would abort capture. Fall back to a
    // lazy D2H for callers that have not opted in (e.g. legacy input builders).
    torch::Tensor paged_kv_indptr_host =
        attn_meta.paged_kv_indptr_host.defined()
            ? attn_meta.paged_kv_indptr_host
            : attn_meta.paged_kv_indptr.to(torch::kCPU);
    const int64_t batch_size = attn_meta.paged_kv_last_page_len.size(0);

    // CUDA-graph correctness: the plan_info computed here is cached on
    // PlanInfo (see early-return at the top of this function) and reused by
    // every captured replay. The plan function statically dispatches attention
    // work based on `paged_kv_indptr_host[batch_size]` (the cumulative block
    // count across all sequences). If the warmup-time layout uses 1 KV block,
    // the plan only dispatches enough work for 1 block per sequence -- and the
    // captured Mate FFI batch_decode kernel then silently underreads when the
    // KV cache later crosses a block boundary at replay time (e.g., decode
    // step 38 of a question with prefill=27 and block_size=64). Override the
    // host indptr here with the worst-case layout for this graph instance so
    // the plan is valid for any runtime block count <= max_kv_blocks_per_seq.
    // The actual per-replay indptr/indices live in persistent device tensors
    // refreshed by update() each call, and the attention kernel iterates over
    // those at runtime.
    if (enable_cuda_graph && max_kv_blocks_per_seq > 0 && batch_size > 0) {
      auto opts =
          torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
      torch::Tensor synth_indptr_host = torch::empty({batch_size + 1}, opts);
      // Fill [0, max, 2*max, ..., bs*max]. Use int32 view since plan() and the
      // upstream pinned-host buffers all use int32 for indptr.
      int32_t* p = synth_indptr_host.data_ptr<int32_t>();
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

    // mate's MateFlashinferDecodePlan signature takes only 2 workspace tensors
    // (not 3): float_workspace_buffer, int_workspace_buffer,
    // paged_kv_indptr_host, ...  Drop page_locked_int_workspace_buffer here so
    // the FFI arg count matches mate's 14-arg `plan` export.
    //
    // Demoted from LOG(INFO) -- this fires once per full-attention layer per
    // decode step (36 layers x every output token for Qwen3.5-27B), so the
    // glog mutex + sink write adds measurable TPOT. Re-enable with
    // `--v=kGraphExecutorLogVerboseLevel` when diagnosing FFI plan()
    // shape/dtype mismatches.
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

}  // namespace xllm::layer::flashinfer
