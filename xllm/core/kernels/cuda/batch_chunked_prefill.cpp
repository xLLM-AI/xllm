/* Copyright 2025-2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <glog/logging.h>

#include <unordered_map>

#include "core/common/global_flags.h"
#include "cuda_ops_api.h"
#include "utils.h"

namespace xllm::kernel::cuda {

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
    // Use provided qo_indptr from attn_metadata
    // TODO: construct qo_indptr in CUDA graph execution
    qo_indptr_to_use = qo_indptr.value();
    VLOG(kGraphExecutorLogVerboseLevel)
        << "use provided qo_indptr in CUDA graph execution";
  } else {
    // Create qo_indptr if not provided (backward compatibility)
    const int64_t batch_size = paged_kv_last_page_len.size(0);
    torch::Tensor qo_indptr_host =
        get_cache_buffer(batch_size + 1, torch::kCPU);
    qo_indptr_to_use = qo_indptr_host.to(torch::kCUDA);
  }

  torch::Tensor v_scale = torch::Tensor();
  auto [scale_v_tensor, scale_v_scalar] = split_scale_param(v_scale);

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
      /*mask_mode_code=*/causal ? 1 : 0,  // CAUSAL or NON_CAUSAL
      /*kv_layout_code=*/0,               // NHD layout
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
