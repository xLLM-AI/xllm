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

void batch_decode(const std::string& uri,
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
                  bool use_tensor_core,
                  std::optional<torch::Tensor> qo_indptr) {
  if (use_tensor_core) {
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
                          /*causal=*/false);
  } else {
    VLOG(kGraphExecutorLogVerboseLevel) << "plan_info: " << plan_info;

    get_function(uri, "run")(
        to_ffi_tensor(float_workspace_buffer),
        to_ffi_tensor(int_workspace_buffer),
        plan_info,
        to_ffi_tensor(query),
        to_ffi_tensor(k_cache),
        to_ffi_tensor(v_cache),
        to_ffi_tensor(paged_kv_indptr),
        to_ffi_tensor(paged_kv_indices),
        to_ffi_tensor(paged_kv_last_page_len),
        to_ffi_tensor(output),
        output_lse.has_value() ? to_ffi_tensor(output_lse.value())
                               : ffi::Optional<ffi::Tensor>(),
        /*kv_layout_code=*/0,  // NHD layout
        window_left,
        support_pdl(),
        /*maybe_alibi_slopes=*/ffi::Optional<ffi::Tensor>(),
        /*logits_soft_cap=*/0.0,
        sm_scale,
        /*rope_rcp_scale=*/1.0,
        /*rope_rcp_theta=*/1.0 / 10000.0);
  }
}

}  // namespace xllm::kernel::cuda
