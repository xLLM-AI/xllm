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

#include "core/kernels/musa/attention_runner.h"

#include <glog/logging.h>

#include "core/common/global_flags.h"
#include "core/framework/config/execution_config.h"
#include "core/kernels/musa/global_capture_instance.h"
#include "core/kernels/musa/musa_ops_api.h"

namespace xllm {
namespace kernel {
namespace cuda {

void AttentionRunner::run_capture(
    const std::string& uri,
    ffi::Array<int64_t> plan_info,
    torch::Tensor float_workspace_buffer,
    torch::Tensor int_workspace_buffer,
    torch::Tensor page_locked_int_workspace_buffer,
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor q_cu_seq_lens,
    torch::Tensor kv_cu_seq_lens,
    int64_t window_left,
    double sm_scale,
    torch::Tensor output,
    std::optional<torch::Tensor>& output_lse,
    uint32_t padded_num_tokens) {
  // plan_info is supplied per replay via AttentionReplayParams; not stored
  // here.
  (void)plan_info;

  ::xllm::runtime::cuda::GlobalCaptureInstance::get_instance()
      .temporarily_end_graph();

  uri_ = uri;

  float_workspace_buffer_ = float_workspace_buffer;
  int_workspace_buffer_ = int_workspace_buffer;
  page_locked_int_workspace_buffer_ = page_locked_int_workspace_buffer;
  query_ = query;
  key_ = key;
  value_ = value;
  output_ = output;
  window_size_left_ = window_left;
  scale_ = sm_scale;
  padded_num_tokens_ = padded_num_tokens;
  runner_type_ = RunnerType::PREFILL;

  ::xllm::runtime::cuda::GlobalCaptureInstance::get_instance()
      .temporarily_begin_graph();
}

void AttentionRunner::run_chunked_prefill_capture(
    const std::string& uri,
    ffi::Array<int64_t> plan_info,
    torch::Tensor float_workspace_buffer,
    torch::Tensor int_workspace_buffer,
    torch::Tensor page_locked_int_workspace_buffer,
    torch::Tensor query,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor paged_kv_indptr,
    torch::Tensor paged_kv_indices,
    torch::Tensor paged_kv_last_page_len,
    int64_t window_left,
    double sm_scale,
    torch::Tensor output,
    std::optional<torch::Tensor>& output_lse,
    std::optional<torch::Tensor> qo_indptr,
    bool causal,
    const torch::Tensor& paged_kv_indptr_host,
    const torch::Tensor& paged_kv_indices_host,
    const torch::Tensor& paged_kv_last_page_len_host,
    uint32_t padded_num_tokens) {
  (void)plan_info;
  (void)paged_kv_indptr;
  (void)paged_kv_indices;
  (void)paged_kv_last_page_len;
  (void)output_lse;
  (void)qo_indptr;
  (void)paged_kv_indptr_host;
  (void)paged_kv_indices_host;
  (void)paged_kv_last_page_len_host;

  ::xllm::runtime::cuda::GlobalCaptureInstance::get_instance()
      .temporarily_end_graph();

  uri_ = uri;
  float_workspace_buffer_ = float_workspace_buffer;
  int_workspace_buffer_ = int_workspace_buffer;
  page_locked_int_workspace_buffer_ = page_locked_int_workspace_buffer;
  query_ = query;
  k_cache_ = k_cache;
  v_cache_ = v_cache;
  output_ = output;
  window_size_left_ = window_left;
  scale_ = sm_scale;
  padded_num_tokens_ = padded_num_tokens;
  runner_type_ = RunnerType::CHUNKED_PREFILL;
  causal_ = causal;

  ::xllm::runtime::cuda::GlobalCaptureInstance::get_instance()
      .temporarily_begin_graph();
}

void AttentionRunner::run_fa3_prefill_capture(torch::Tensor query,
                                              torch::Tensor key,
                                              torch::Tensor value,
                                              int64_t max_seqlen_q,
                                              int64_t max_seqlen_k,
                                              int64_t window_left,
                                              int64_t window_right,
                                              double sm_scale,
                                              torch::Tensor output,
                                              torch::Tensor output_lse) {
  // FA3 is launched outside the graph during replay. End the current graph
  // segment before storing its tensors, then resume capture for downstream
  // layers; this is the same sequencing used by the FlashInfer runner.
  ::xllm::runtime::cuda::GlobalCaptureInstance::get_instance()
      .temporarily_end_graph();

  query_ = std::move(query);
  key_ = std::move(key);
  value_ = std::move(value);
  output_ = std::move(output);
  output_lse_ = std::move(output_lse);
  window_size_left_ = window_left;
  window_size_right_ = window_right;
  scale_ = sm_scale;
  max_seqlen_q_ = max_seqlen_q;
  max_seqlen_k_ = max_seqlen_k;
  runner_type_ = RunnerType::FA3_PREFILL;

  ::xllm::runtime::cuda::GlobalCaptureInstance::get_instance()
      .temporarily_begin_graph();
}

void AttentionRunner::run_replay(const AttentionReplayParams& params) {
  torch::Tensor query_slice =
      query_.slice(/*dim=*/0, /*start=*/0, /*end=*/params.actual_num_tokens);
  torch::Tensor output_slice =
      output_.slice(/*dim=*/0, /*start=*/0, /*end=*/params.actual_num_tokens);

  if (runner_type_ == RunnerType::FA3_PREFILL) {
    torch::Tensor key_slice =
        key_.slice(/*dim=*/0, /*start=*/0, /*end=*/params.actual_num_tokens);
    torch::Tensor value_slice =
        value_.slice(/*dim=*/0, /*start=*/0, /*end=*/params.actual_num_tokens);
    CHECK(output_lse_.defined()) << "FA3 prefill replay requires output LSE";
    CHECK_EQ(output_lse_.dim(), 2)
        << "FA3 prefill replay requires a 2D output LSE buffer";
    CHECK(output_lse_.is_contiguous())
        << "FA3 prefill replay requires a contiguous output LSE buffer";
    const int64_t num_heads = output_lse_.size(0);
    const int64_t required_lse_elements = num_heads * params.actual_num_tokens;
    CHECK_GE(output_lse_.numel(), required_lse_elements);
    torch::Tensor output_lse = output_lse_.view({-1})
                                   .narrow(/*dim=*/0,
                                           /*start=*/0,
                                           /*length=*/required_lse_elements)
                                   .view({num_heads, params.actual_num_tokens});

    const int64_t max_seqlen_q =
        params.max_seqlen_q > 0 ? params.max_seqlen_q : max_seqlen_q_;
    const int64_t max_seqlen_k =
        params.max_seqlen_k > 0 ? params.max_seqlen_k : max_seqlen_k_;
    CHECK_GT(max_seqlen_q, 0);
    CHECK_GT(max_seqlen_k, 0);

    // The model normally produces contiguous projection views. Preserve the
    // existing eager FA3 behavior for the uncommon non-contiguous case.
    torch::Tensor query_contiguous = query_slice.contiguous();
    torch::Tensor key_contiguous = key_slice.contiguous();
    torch::Tensor value_contiguous = value_slice.contiguous();
    torch::Tensor q_cu_seq_lens = params.q_cu_seq_lens.contiguous();
    torch::Tensor kv_cu_seq_lens = params.kv_cu_seq_lens.contiguous();
    fa3_prefill(query_contiguous,
                key_contiguous,
                value_contiguous,
                q_cu_seq_lens,
                kv_cu_seq_lens,
                max_seqlen_q,
                max_seqlen_k,
                window_size_left_,
                window_size_right_,
                scale_,
                output_slice,
                output_lse);
    return;
  }

  // TODO: support output_lse for replay
  std::optional<torch::Tensor> output_lse = std::nullopt;
  if (runner_type_ == RunnerType::CHUNKED_PREFILL) {
    batch_chunked_prefill(uri_,
                          params.plan_info,
                          float_workspace_buffer_,
                          int_workspace_buffer_,
                          page_locked_int_workspace_buffer_,
                          query_slice,
                          k_cache_,
                          v_cache_,
                          params.paged_kv_indptr,
                          params.paged_kv_indices,
                          params.paged_kv_last_page_len,
                          window_size_left_,
                          scale_,
                          output_slice,
                          output_lse,
                          params.qo_indptr,
                          causal_,
                          params.paged_kv_indptr_host,
                          params.paged_kv_indices_host,
                          params.paged_kv_last_page_len_host);
    return;
  }

  torch::Tensor key_slice =
      key_.slice(/*dim=*/0, /*start=*/0, /*end=*/params.actual_num_tokens);
  torch::Tensor value_slice =
      value_.slice(/*dim=*/0, /*start=*/0, /*end=*/params.actual_num_tokens);
  batch_prefill(uri_,
                params.plan_info,
                float_workspace_buffer_,
                int_workspace_buffer_,
                page_locked_int_workspace_buffer_,
                query_slice,
                key_slice,
                value_slice,
                params.q_cu_seq_lens,
                params.kv_cu_seq_lens,
                window_size_left_,
                scale_,
                output_slice,
                output_lse);
}

void batch_chunked_prefill_with_optional_piecewise_capture(
    const std::string& uri,
    ffi::Array<int64_t> plan_info,
    torch::Tensor float_workspace_buffer,
    torch::Tensor int_workspace_buffer,
    torch::Tensor page_locked_int_workspace_buffer,
    torch::Tensor query,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor paged_kv_indptr,
    torch::Tensor paged_kv_indices,
    torch::Tensor paged_kv_last_page_len,
    int64_t window_left,
    double sm_scale,
    torch::Tensor output,
    std::optional<torch::Tensor>& output_lse,
    std::optional<torch::Tensor> qo_indptr,
    bool causal,
    const torch::Tensor& paged_kv_indptr_host,
    const torch::Tensor& paged_kv_indices_host,
    const torch::Tensor& paged_kv_last_page_len_host) {
  if (::xllm::ExecutionConfig::get_instance().enable_graph() &&
      ::xllm::ExecutionConfig::get_instance()
          .enable_prefill_piecewise_graph() &&
      ::xllm::runtime::cuda::GlobalCaptureInstance::get_instance()
          .is_capturing()) {
    AttentionRunner runner;
    const uint32_t padded_num_tokens =
        static_cast<uint32_t>(query.size(/*dim=*/0));
    runner.run_chunked_prefill_capture(uri,
                                       plan_info,
                                       float_workspace_buffer,
                                       int_workspace_buffer,
                                       page_locked_int_workspace_buffer,
                                       query,
                                       k_cache,
                                       v_cache,
                                       paged_kv_indptr,
                                       paged_kv_indices,
                                       paged_kv_last_page_len,
                                       window_left,
                                       sm_scale,
                                       output,
                                       output_lse,
                                       qo_indptr,
                                       causal,
                                       paged_kv_indptr_host,
                                       paged_kv_indices_host,
                                       paged_kv_last_page_len_host,
                                       padded_num_tokens);
    ::xllm::runtime::cuda::GlobalCaptureInstance::get_instance()
        .register_attention_runner(std::move(runner));
    return;
  }

  batch_chunked_prefill(uri,
                        plan_info,
                        float_workspace_buffer,
                        int_workspace_buffer,
                        page_locked_int_workspace_buffer,
                        query,
                        k_cache,
                        v_cache,
                        paged_kv_indptr,
                        paged_kv_indices,
                        paged_kv_last_page_len,
                        window_left,
                        sm_scale,
                        output,
                        output_lse,
                        qo_indptr,
                        causal,
                        paged_kv_indptr_host,
                        paged_kv_indices_host,
                        paged_kv_last_page_len_host);
}

void fa3_prefill_with_optional_piecewise_capture(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& cu_seqlens_q,
    const torch::Tensor& cu_seqlens_k,
    int64_t max_seqlen_q,
    int64_t max_seqlen_k,
    int64_t window_left,
    int64_t window_right,
    double sm_scale,
    torch::Tensor& output,
    torch::Tensor& output_lse) {
  if (::xllm::ExecutionConfig::get_instance().enable_graph() &&
      ::xllm::ExecutionConfig::get_instance()
          .enable_prefill_piecewise_graph() &&
      ::xllm::runtime::cuda::GlobalCaptureInstance::get_instance()
          .is_capturing()) {
    AttentionRunner runner;
    runner.run_fa3_prefill_capture(query,
                                   key,
                                   value,
                                   max_seqlen_q,
                                   max_seqlen_k,
                                   window_left,
                                   window_right,
                                   sm_scale,
                                   output,
                                   output_lse);
    ::xllm::runtime::cuda::GlobalCaptureInstance::get_instance()
        .register_attention_runner(std::move(runner));
    return;
  }

  torch::Tensor query_contiguous = query.contiguous();
  torch::Tensor key_contiguous = key.contiguous();
  torch::Tensor value_contiguous = value.contiguous();
  torch::Tensor q_cu_seq_lens = cu_seqlens_q.contiguous();
  torch::Tensor kv_cu_seq_lens = cu_seqlens_k.contiguous();
  fa3_prefill(query_contiguous,
              key_contiguous,
              value_contiguous,
              q_cu_seq_lens,
              kv_cu_seq_lens,
              max_seqlen_q,
              max_seqlen_k,
              window_left,
              window_right,
              sm_scale,
              output,
              output_lse);
}

void batch_prefill_with_optional_piecewise_capture(
    const std::string& uri,
    ffi::Array<int64_t> plan_info,
    torch::Tensor float_workspace_buffer,
    torch::Tensor int_workspace_buffer,
    torch::Tensor page_locked_int_workspace_buffer,
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor q_cu_seq_lens,
    torch::Tensor kv_cu_seq_lens,
    int64_t window_left,
    double sm_scale,
    torch::Tensor output,
    std::optional<torch::Tensor>& output_lse) {
  if (::xllm::ExecutionConfig::get_instance().enable_graph() &&
      ::xllm::ExecutionConfig::get_instance()
          .enable_prefill_piecewise_graph() &&
      ::xllm::runtime::cuda::GlobalCaptureInstance::get_instance()
          .is_capturing()) {
    AttentionRunner runner;

    uint32_t padded_num_tokens = static_cast<uint32_t>(query.size(0));

    runner.run_capture(uri,
                       plan_info,
                       float_workspace_buffer,
                       int_workspace_buffer,
                       page_locked_int_workspace_buffer,
                       query,
                       key,
                       value,
                       q_cu_seq_lens,
                       kv_cu_seq_lens,
                       window_left,
                       sm_scale,
                       output,
                       output_lse,
                       padded_num_tokens);

    ::xllm::runtime::cuda::GlobalCaptureInstance::get_instance()
        .register_attention_runner(std::move(runner));
    return;
  }
  batch_prefill(uri,
                plan_info,
                float_workspace_buffer,
                int_workspace_buffer,
                page_locked_int_workspace_buffer,
                query,
                key,
                value,
                q_cu_seq_lens,
                kv_cu_seq_lens,
                window_left,
                sm_scale,
                output,
                output_lse);
}

}  // namespace cuda
}  // namespace kernel
}  // namespace xllm
