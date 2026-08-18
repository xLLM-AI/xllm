/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include <framework/core/MLUStream.h>
#include <framework/core/device.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <algorithm>
#include <cstdint>
#include <utility>

#include "kernels/mlu/mlu_ops_api.h"
#include "triton_jit/include/jit_kernel.h"

namespace xllm::kernel::mlu {

using xllm::triton_jit::JITKernel;

std::pair<torch::Tensor, torch::Tensor>
fused_recurrent_gated_delta_rule_packed_decode(
    const torch::Tensor& mixed_qkv,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& A_log,
    const torch::Tensor& dt_bias,
    double scale,
    torch::Tensor& ssm_cache,
    const torch::Tensor& ssm_state_indices,
    bool use_qk_l2norm_in_kernel) {
  torch::Tensor mixed_qkv_contig = mixed_qkv.contiguous();
  torch::Tensor a_contig = a.contiguous();
  torch::Tensor b_contig = b.contiguous();

  int32_t B = static_cast<int32_t>(mixed_qkv_contig.size(0));
  int32_t qkv_dim = static_cast<int32_t>(mixed_qkv_contig.size(1));
  int32_t HV = static_cast<int32_t>(ssm_cache.size(1));
  int32_t V = static_cast<int32_t>(ssm_cache.size(2));
  int32_t K = static_cast<int32_t>(ssm_cache.size(3));
  int32_t qk_dim = qkv_dim - HV * V;
  int32_t H = qk_dim / (2 * K);

  torch::Tensor out =
      torch::empty({B, 1, HV, V},
                   mixed_qkv_contig.options().dtype(mixed_qkv_contig.dtype()));

  int32_t stride_mixed_qkv_tok =
      static_cast<int32_t>(mixed_qkv_contig.stride(0));
  int32_t stride_a_tok = static_cast<int32_t>(a_contig.stride(0));
  int32_t stride_b_tok = static_cast<int32_t>(b_contig.stride(0));
  int32_t stride_init_state_token = static_cast<int32_t>(ssm_cache.stride(0));
  int32_t stride_final_state_token = static_cast<int32_t>(ssm_cache.stride(0));
  int32_t stride_indices_seq =
      static_cast<int32_t>(ssm_state_indices.stride(0));

  torch_mlu::DeviceProp* prop =
      torch_mlu::getDeviceProperties(torch_mlu::current_device());
  CHECK(prop != nullptr);
  int32_t core_count = prop->cluster_count * prop->core_num_per_cluster;

  // Pick tunables from the device ISA. arch6 (isa_version >= 600) has a larger
  // SRAM, so it tolerates bigger HV blocks and more pipeline stages; older
  // archs cap MAX_HV lower and use fewer stages. split_hv flips on for small B
  // so the HV-block dimension is fanned across cores to fill them.
  bool is_arch6 = prop->isa_version >= 600;
  int32_t max_hv = is_arch6 ? 32 : 8;
  int32_t num_stage = is_arch6 ? 4 : 3;
  int32_t split_hv = is_arch6 ? (B < 8 ? 1 : 0) : (B < 26 ? 1 : 0);

  int32_t heads_per_q = HV / H;  // V-heads per Q/K-head
  // block_hv (non-split path) must satisfy three conditions: (1) <= max_hv;
  // (2) divides HV (the kernel iterates HV in whole blocks, no tail mask);
  // (3) is a multiple of heads_per_q (BH = BLOCK_HV / heads_per_q exact, the
  // Q/K-head load misses no head). Take the largest such value.
  int32_t upper = std::min(max_hv, HV);
  int32_t block_hv =
      heads_per_q;  // min legal value (divides HV, multiple of heads_per_q)
  for (int32_t cand = upper; cand >= heads_per_q; --cand) {
    if (cand % heads_per_q == 0 && HV % cand == 0) {
      block_hv = cand;
      break;
    }
  }

  // Small B: also split HV across cores (SPLIT_HV=1) to fill cores: force
  // block_hv=1 → num_hv=HV, 1 HV block per core.
  if (split_hv) {
    block_hv = 1;
  }
  int32_t block_v = (split_hv && B * HV < core_count) ? 64 : 128;

  // Validate the actually-launched block_hv (after the split_hv override). The
  // two paths constrain BLOCK_HV differently:
  //   - non-split: must divide HV and be a multiple of heads_per_q.
  //   - split (BLOCK_HV=1): the kernel guards the tail with mask_hvb + gi_hv<
  //   HV,
  //     so it need not divide HV or be a multiple of heads_per_q.
  CHECK(HV % H == 0) << "HV must be divisible by H";
  if (split_hv) {
    CHECK(block_hv >= 1 && block_hv <= HV)
        << "BLOCK_HV out of range for split path";
  } else {
    CHECK(block_hv % heads_per_q == 0)
        << "block_hv must be divisible by (HV / H)";
    CHECK(HV % block_hv == 0) << "block_hv must divide HV";
  }

  int32_t num_v_blocks = (V + block_v - 1) / block_v;
  int32_t num_hv_blocks = (HV + block_hv - 1) / block_hv;
  int32_t total_blocks =
      split_hv ? num_hv_blocks * num_v_blocks * B : num_v_blocks * B;

  cnrtQueue_t queue = torch_mlu::getCurMLUStream();

  JITKernel& f = JITKernel::get(
      /*py_path=*/
      "xllm.core.kernels.mlu.triton_kernel.fused_recurrent_gated_delta_rule_"
      "packed_decode",
      /*fn_name=*/"tmo_fused_recurrent_gated_delta_rule_packed_decode_kernel");

  f.launch(static_cast<void*>(queue),
           /*grid=*/
           {static_cast<uint32_t>(std::min(total_blocks, core_count)), 1, 1},
           /*cfg=*/{/*num_warps=*/1, /*num_stages=*/num_stage},
           mixed_qkv_contig,
           a_contig,
           b_contig,
           A_log,
           dt_bias,
           out,
           ssm_cache,
           ssm_cache,
           ssm_state_indices,
           static_cast<float>(scale),
           stride_mixed_qkv_tok,
           stride_a_tok,
           stride_b_tok,
           stride_init_state_token,
           stride_final_state_token,
           stride_indices_seq,
           B,
           H,
           HV,
           K,
           V,
           /*BLOCK_N=*/4,
           /*BLOCK_HV=*/block_hv,
           /*BLOCK_V=*/block_v,
           /*BLOCK_K=*/K,
           /*SOFTPLUS_THRESHOLD=*/20.0f,
           /*USE_QK_L2NORM_IN_KERNEL=*/use_qk_l2norm_in_kernel ? 1 : 0,
           /*SPLIT_HV=*/split_hv);

  return std::make_pair(out, ssm_cache);
}

}  // namespace xllm::kernel::mlu
