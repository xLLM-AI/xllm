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

#include <glog/logging.h>

#include <optional>
#include <string>

#include "core/common/global_flags.h"
#include "core/kernels/musa/musa_ops_api.h"
#include "core/kernels/musa/musa_tvmffi_stream.h"

namespace xllm::kernel::cuda {
namespace {

void batch_prefill_impl(const std::string& uri,
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
                        bool is_causal,
                        const std::optional<torch::Tensor>& mask) {
  std::optional<torch::Tensor> processed_mask;
  std::optional<torch::Tensor> mask_indptr_opt;
  if (mask.has_value()) {
    auto m = mask.value();
    if (m.defined() && m.numel() > 0) {
      auto device = query.device();
      if (m.device() != device) {
        m = m.to(device);
      }
      if (!m.is_floating_point()) {
        m = m.to(torch::kFloat32);
      }

      int64_t seq_len = m.size(0);
      auto causal_mask = torch::tril(torch::ones(
          {seq_len, seq_len},
          torch::TensorOptions().dtype(torch::kFloat32).device(device)));
      auto combined_mask =
          causal_mask * m.unsqueeze(0).expand({seq_len, seq_len});

      const int64_t n = seq_len * seq_len;
      const int64_t num_bytes = (n + 7) / 8;
      auto flat = combined_mask.contiguous().view({-1});
      if (flat.device().type() != torch::kCPU) {
        flat = flat.cpu();
      }
      auto packed = torch::zeros(
          {num_bytes},
          torch::TensorOptions().dtype(torch::kUInt8).device(flat.device()));
      auto flat_acc = flat.accessor<float, 1>();
      auto packed_acc = packed.accessor<uint8_t, 1>();
      for (int64_t i = 0; i < n; ++i) {
        if (flat_acc[i] > 0.5f) {
          packed_acc[i / 8] |= static_cast<uint8_t>(1u << (i % 8));
        }
      }

      if (packed.device() != device) {
        packed = packed.to(device);
      }
      processed_mask = packed.contiguous();

      auto mask_indptr = torch::zeros(
          {2}, torch::TensorOptions().dtype(torch::kInt32).device(device));
      mask_indptr[0] = 0;
      mask_indptr[1] = static_cast<int32_t>(num_bytes);
      mask_indptr_opt = mask_indptr;
    }
  }

  bool use_custom_mask = processed_mask.has_value();
  std::string backend =
      determine_attention_backend(/*pos_encoding_mode=*/0,
                                  /*use_fp16_qk_reduction=*/false,
                                  use_custom_mask);

  if (backend == "fa2") {
    MusaTvmffiStreamGuard stream_guard(query.device());
    get_function(uri, "ragged_run")(
        to_ffi_tensor(float_workspace_buffer),
        to_ffi_tensor(int_workspace_buffer),
        plan_info,
        to_ffi_tensor(query),
        to_ffi_tensor(key),
        to_ffi_tensor(value),
        to_ffi_tensor(q_cu_seq_lens),
        to_ffi_tensor(kv_cu_seq_lens),
        to_ffi_tensor(output),
        output_lse.has_value() ? to_ffi_tensor(output_lse.value())
                               : ffi::Optional<ffi::Tensor>(),
        /*mask_mode_code=*/is_causal ? 1 : 0,
        /*kv_layout_code=*/0,
        window_left,
        support_pdl(),
        processed_mask.has_value() ? to_ffi_tensor(processed_mask.value())
                                   : ffi::Optional<ffi::Tensor>(),
        mask_indptr_opt.has_value() ? to_ffi_tensor(mask_indptr_opt.value())
                                    : ffi::Optional<ffi::Tensor>(),
        /*maybe_alibi_slopes=*/ffi::Optional<ffi::Tensor>(),
        /*maybe_prefix_len_ptr=*/ffi::Optional<ffi::Tensor>(),
        /*maybe_token_pos_in_items_ptr=*/ffi::Optional<ffi::Tensor>(),
        /*maybe_max_item_len_ptr=*/ffi::Optional<ffi::Tensor>(),
        /*logits_soft_cap=*/0.0,
        sm_scale,
        /*rope_rcp_scale=*/1.0,
        /*rope_rcp_theta=*/1.0 / 10000.0,
        /*token_pos_in_items_len=*/0);
  } else if (backend == "fa3") {
    torch::Tensor v_scale = torch::Tensor();

    auto [scale_v_tensor, scale_v_scalar] = split_scale_param(v_scale);

    MusaTvmffiStreamGuard stream_guard(query.device());
    get_function(uri, "ragged_run")(
        to_ffi_tensor(float_workspace_buffer),
        to_ffi_tensor(int_workspace_buffer),
        plan_info,
        to_ffi_tensor(query),
        to_ffi_tensor(key),
        to_ffi_tensor(value),
        to_ffi_tensor(q_cu_seq_lens),
        to_ffi_tensor(kv_cu_seq_lens),
        to_ffi_tensor(output),
        output_lse.has_value() ? to_ffi_tensor(output_lse.value())
                               : ffi::Optional<ffi::Tensor>(),
        /*mask_mode_code=*/is_causal ? 1 : 0,
        /*kv_layout_code=*/0,
        window_left,
        support_pdl(),
        /*maybe_prefix_len_ptr=*/ffi::Optional<ffi::Tensor>(),
        /*maybe_token_pos_in_items_ptr=*/ffi::Optional<ffi::Tensor>(),
        /*maybe_max_item_len_ptr=*/ffi::Optional<ffi::Tensor>(),
        scale_v_tensor.defined() ? to_ffi_tensor(scale_v_tensor)
                                 : ffi::Optional<ffi::Tensor>(),
        /*logits_soft_cap=*/0.0,
        sm_scale,
        scale_v_scalar,
        /*token_pos_in_items_len=*/0);
  }
}

}  // namespace

void batch_prefill(const std::string& uri,
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
                   const std::optional<torch::Tensor>& mask) {
  batch_prefill_impl(uri,
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
                     /*is_causal=*/true,
                     mask);
}

void batch_prefill_non_causal(const std::string& uri,
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
                              const std::optional<torch::Tensor>& mask) {
  batch_prefill_impl(uri,
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
                     /*is_causal=*/false,
                     mask);
}

void batch_chunked_prefill(const std::string& uri,
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
                           bool causal) {
  VLOG(kGraphExecutorLogVerboseLevel) << "plan_info: " << plan_info;

  torch::Tensor qo_indptr_to_use;
  if (qo_indptr.has_value()) {
    qo_indptr_to_use = qo_indptr.value();
    VLOG(kGraphExecutorLogVerboseLevel)
        << "use provided qo_indptr in CUDA graph execution";
  } else {
    const int64_t batch_size = paged_kv_last_page_len.size(0);
    torch::Tensor qo_indptr_host =
        get_cache_buffer(static_cast<int32_t>(batch_size + 1), torch::kCPU);
    qo_indptr_to_use = qo_indptr_host.to(torch::kCUDA);
  }

  torch::Tensor v_scale = torch::Tensor();
  auto [scale_v_tensor, scale_v_scalar] = split_scale_param(v_scale);

  MusaTvmffiStreamGuard stream_guard(query.device());
  get_function(uri, "paged_run")(
      to_ffi_tensor(float_workspace_buffer),
      to_ffi_tensor(int_workspace_buffer),
      plan_info,
      to_ffi_tensor(query),
      to_ffi_tensor(k_cache),
      to_ffi_tensor(v_cache),
      to_ffi_tensor(qo_indptr_to_use),
      to_ffi_tensor(paged_kv_indptr),
      to_ffi_tensor(paged_kv_indices),
      to_ffi_tensor(paged_kv_last_page_len),
      to_ffi_tensor(output),
      output_lse.has_value() ? to_ffi_tensor(output_lse.value())
                             : ffi::Optional<ffi::Tensor>(),
      /*mask_mode_code=*/causal ? 1 : 0,
      /*kv_layout_code=*/0,
      window_left,
      support_pdl(),
      /*maybe_custom_mask=*/ffi::Optional<ffi::Tensor>(),
      /*maybe_mask_indptr=*/ffi::Optional<ffi::Tensor>(),
      /*maybe_alibi_slopes=*/ffi::Optional<ffi::Tensor>(),
      /*maybe_prefix_len_ptr=*/ffi::Optional<ffi::Tensor>(),
      /*maybe_token_pos_in_items_ptr=*/ffi::Optional<ffi::Tensor>(),
      /*maybe_max_item_len_ptr=*/ffi::Optional<ffi::Tensor>(),
      /*logits_soft_cap=*/0.0,
      sm_scale,
      /*rope_rcp_scale=*/1.0,
      /*rope_rcp_theta=*/1.0 / 10000.0,
      /*token_pos_in_items_len=*/0);
}

}  // namespace xllm::kernel::cuda
