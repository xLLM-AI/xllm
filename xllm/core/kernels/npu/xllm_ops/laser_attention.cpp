/* Copyright 2026 The xLLM Authors.

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

#include <c10/core/Device.h>
#include <glog/logging.h>
#include <torch/torch.h>
#include <torch_npu/csrc/libs/init_npu.h>
#include <torch_npu/torch_npu.h>

#include <cmath>
#ifdef TORCH_HIGHER_THAN_PTA6
#include <torch_npu/csrc/framework/OpCommand.h>
#else
#include <torch_npu/csrc/aten/NPUNativeFunctions.h>
#include <torch_npu/csrc/framework/utils/OpPreparation.h>
#endif

#include "acl/acl.h"
#include "aclnn_laser_attention.h"
#include "core/common/macros.h"
#include "core/kernels/npu/aclnn/pytorch_npu_helper.hpp"
#include "core/kernels/npu/utils.h"
#include "xllm_ops_api.h"

namespace xllm::kernel::npu {

// Laser attention kernel tuned for Wan2.2. q/k/v and output are in BNSD
// layout; output is cast back to the input dtype (kernel emits fp32).
torch::Tensor laser_attention(const torch::Tensor& q_bnsd,
                              const torch::Tensor& k_bnsd,
                              const torch::Tensor& v_bnsd,
                              double scale_value,
                              int64_t head_num) {
  check_tensor(q_bnsd, "query", "laser_attention");
  check_tensor(k_bnsd, "key", "laser_attention");
  check_tensor(v_bnsd, "value", "laser_attention");

  const auto input_dtype = q_bnsd.scalar_type();
  // Pad seq to a multiple of 256 and head_dim to 128, masking padded keys via
  // pre_tokens; slice the padding back off the output afterwards.
  namespace F = torch::nn::functional;
  const int64_t kSeqBase = 256;
  const int64_t kDimBase = 128;
  const int64_t kMaxToken = 2147483647;  // MAX_TOKEN = 2^31 - 1

  const int64_t q_seqlen = q_bnsd.size(2);   // BNSD: real query seq len
  const int64_t kv_seqlen = k_bnsd.size(2);  // real key/value seq len
  const int64_t head_dim = q_bnsd.size(3);   // real head dim

  // 1. cast to fp16 (kernel is fp16-only).
  auto q_f16 = q_bnsd.to(torch::kHalf).contiguous();
  auto k_f16 = k_bnsd.to(torch::kHalf).contiguous();
  auto v_f16 = v_bnsd.to(torch::kHalf).contiguous();

  // 2. zero-pad seq (dim=2) to a multiple of 256 and head_dim (dim=3) to 128.
  auto pad_to = [](const torch::Tensor& t, int64_t base, int64_t dim) {
    int64_t sz = t.size(dim);
    if (sz % base == 0) return t;
    int64_t pad = ((sz / base) + 1) * base - sz;
    // PadFuncOptions pads from the last dim backwards.
    std::vector<int64_t> spec = (dim == 3) ? std::vector<int64_t>{0, pad}
                                           : std::vector<int64_t>{0, 0, 0, pad};
    return F::pad(t, F::PadFuncOptions(spec).mode(torch::kConstant).value(0))
        .contiguous();
  };
  q_f16 = pad_to(pad_to(q_f16, kSeqBase, 2), kDimBase, 3);
  k_f16 = pad_to(pad_to(k_f16, kSeqBase, 2), kDimBase, 3);
  v_f16 = pad_to(pad_to(v_f16, kSeqBase, 2), kDimBase, 3);

  const auto q_c = q_f16;
  const auto k_c = k_f16;
  const auto v_c = v_f16;

  // 3. pre_tokens = number of trailing padded keys to exclude from softmax.
  int64_t pre_tokens = kMaxToken;
  if (kv_seqlen % kSeqBase != 0) {
    pre_tokens = (kv_seqlen / kSeqBase + 1) * kSeqBase - kv_seqlen;
  }
  const int64_t next_tokens = 1;

  // Kernel outputs are fp32.
  auto attn_out =
      torch::empty({q_c.size(0), q_c.size(1), q_c.size(2), q_c.size(3)},
                   q_c.options().dtype(torch::kFloat));
  auto softmax_lms = torch::empty({q_c.size(0), q_c.size(1), q_c.size(2)},
                                  q_c.options().dtype(torch::kFloat));

  char input_layout[] = "BNSD";
  const double keep_prob = 1.0;
  const bool is_high_precision = true;
  const c10::optional<at::Tensor> no_mask;  // atten/alibi/drop masks: unused

  // Two-stage aclnn call (GetWorkspaceSize + execute) handled by the macro.
  EXEC_NPU_CMD(aclnnLaserAttention,
               q_c,
               k_c,
               v_c,
               no_mask,
               no_mask,
               no_mask,
               scale_value,
               head_num,
               input_layout,
               keep_prob,
               pre_tokens,
               next_tokens,
               is_high_precision,
               softmax_lms,
               attn_out);

  // Slice off the seq/head_dim padding, then cast back to the caller's dtype.
  auto out_real =
      attn_out.slice(2, 0, q_seqlen).slice(3, 0, head_dim).contiguous();
  return out_real.to(input_dtype);
}

}  // namespace xllm::kernel::npu
