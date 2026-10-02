/* Copyright 2026 The xLLM Authors.

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

#include <cstdint>
#include <mutex>
#include <tuple>
#include <unordered_map>

#include "core/kernels/npu/aclnn/pytorch_npu_helper.hpp"
#include "core/kernels/npu/xllm_ops/mega_gdn_constants.h"
#include "core/kernels/npu/xllm_ops/xllm_ops_api.h"
#include "core/platform/platform.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"

// libruntime.so exports this ABI; its declaration is outside the public CANN
// headers.
extern "C" int32_t rtGetC2cCtrlAddr(uint64_t* addr, uint32_t* len);

namespace xllm::kernel::npu {

namespace {

// A2/A3 require a device-local FFTS address for AIC/AIV synchronization; A5
// requires 0.
std::unordered_map<int32_t, int64_t> g_ffts_addr_cache;
std::mutex g_ffts_addr_cache_mutex;

int64_t get_ffts_addr(const torch::Device& device) {
  const int32_t device_index = static_cast<int32_t>(device.index());
  std::lock_guard<std::mutex> lock(g_ffts_addr_cache_mutex);
  auto it = g_ffts_addr_cache.find(device_index);
  if (it != g_ffts_addr_cache.end()) {
    return it->second;
  }

  const c10_npu::NPUGuard device_guard(device);
  if (Platform::is_ascend950()) {
    g_ffts_addr_cache.emplace(device_index, 0);
    return 0;
  }

  uint64_t ffts_addr = 0;
  uint32_t ffts_len = 0;
  const int32_t ret = rtGetC2cCtrlAddr(&ffts_addr, &ffts_len);
  CHECK(ret == 0)
      << "rtGetC2cCtrlAddr failed for the GDN prefill operator on device "
      << device_index << ", ret=" << ret;
  CHECK_GT(ffts_len, 0)
      << "rtGetC2cCtrlAddr returned an empty FFTS region on device "
      << device_index;
  CHECK_NE(ffts_addr, 0)
      << "rtGetC2cCtrlAddr returned a null FFTS address on device "
      << device_index;
  const int64_t resolved_addr = static_cast<int64_t>(ffts_addr);
  g_ffts_addr_cache.emplace(device_index, resolved_addr);
  return resolved_addr;
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> npu_mega_gdn_prefill(
    const torch::Tensor& mixed_qkv,
    const torch::Tensor& b,
    const torch::Tensor& a,
    const torch::Tensor& z,
    const torch::Tensor& conv_weight,
    torch::Tensor& conv_state,
    const torch::Tensor& a_log,
    const torch::Tensor& dt_bias,
    const torch::Tensor& conv_state_read_indices,
    const torch::Tensor& conv_state_write_indices,
    const torch::Tensor& ssm_state_read_indices,
    const torch::Tensor& ssm_state_write_indices,
    torch::Tensor& ssm_cache,
    const torch::Tensor& cu_seqlens,
    const torch::Tensor& norm_weight,
    int64_t num_matrices) {
  const MegaGdnMasks consts = get_or_create_mega_gdn_masks(mixed_qkv.device());

  const int64_t total_tokens = z.size(0);
  const int64_t num_value_heads = z.size(1);
  const int64_t head_dim = z.size(2);

  auto norm_output =
      torch::empty({total_tokens, num_value_heads, head_dim},
                   torch::TensorOptions(z.device()).dtype(torch::kBFloat16));
  torch::Tensor& conv_state_out = conv_state;
  torch::Tensor& ssm_cache_out = ssm_cache;

  const int64_t ffts_addr = get_ffts_addr(mixed_qkv.device());

  EXEC_NPU_CMD(aclnnMegaGdnPrefillOp,
               mixed_qkv,
               b,
               a,
               z,
               conv_weight,
               conv_state,
               a_log,
               dt_bias,
               conv_state_read_indices,
               conv_state_write_indices,
               ssm_state_read_indices,
               ssm_state_write_indices,
               ssm_cache,
               consts.mask_lower,
               consts.mask_full,
               consts.minus_identity_bf16,
               cu_seqlens,
               norm_weight,
               ffts_addr,
               num_matrices,
               norm_output,
               conv_state_out,
               ssm_cache_out);

  return {norm_output, conv_state_out, ssm_cache_out};
}

}  // namespace xllm::kernel::npu
