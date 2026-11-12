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
#include <torch/torch.h>
#include <torch_npu/csrc/aten/CustomFunctions.h>

#include <cmath>
#include <limits>
#include <optional>
#include <vector>

#include "aclnn_laser_attention.h"
#include "core/kernels/npu/aclnn/pytorch_npu_helper.hpp"
#include "core/kernels/npu/utils.h"
#include "core/kernels/npu/xllm_ops/xllm_ops_api.h"

namespace xllm::kernel::npu {

// Dense, non-causal attention in BNSD layout. Short and cross-attention
// use the existing dense NPU operator before any Laser preprocessing.
torch::Tensor laser_attention(const torch::Tensor& q_bnsd,
                              const torch::Tensor& k_bnsd,
                              const torch::Tensor& v_bnsd,
                              double scale_value,
                              int64_t head_num) {
  check_tensor(q_bnsd, "query", "laser_attention");
  check_tensor(k_bnsd, "key", "laser_attention");
  check_tensor(v_bnsd, "value", "laser_attention");
  CHECK_EQ(q_bnsd.dim(), 4) << "Laser attention expects BNSD tensors";
  CHECK_EQ(k_bnsd.dim(), 4);
  CHECK_EQ(v_bnsd.dim(), 4);
  CHECK(q_bnsd.device().is_privateuseone());
  CHECK(q_bnsd.device() == k_bnsd.device());
  CHECK(q_bnsd.device() == v_bnsd.device());
  const torch::ScalarType input_dtype = q_bnsd.scalar_type();
  CHECK(input_dtype == torch::kHalf || input_dtype == torch::kBFloat16);
  CHECK(input_dtype == k_bnsd.scalar_type());
  CHECK(input_dtype == v_bnsd.scalar_type());
  CHECK_EQ(q_bnsd.size(0), k_bnsd.size(0));
  CHECK_EQ(q_bnsd.size(1), head_num);
  CHECK_EQ(q_bnsd.size(1), k_bnsd.size(1));
  CHECK_EQ(q_bnsd.size(3), k_bnsd.size(3));
  CHECK(k_bnsd.sizes() == v_bnsd.sizes());
  CHECK_LE(q_bnsd.size(3), 128);
  CHECK(std::isfinite(scale_value));
  CHECK_GT(scale_value, 0.0);

  if (q_bnsd.size(2) < 2048 || q_bnsd.size(2) != k_bnsd.size(2)) {
    const auto results = at_npu::native::custom_ops::npu_fusion_attention(
        q_bnsd.contiguous(),
        k_bnsd.contiguous(),
        v_bnsd.contiguous(),
        head_num,
        "BNSD",
        /*pse=*/std::nullopt,
        /*padding_mask=*/std::nullopt,
        /*atten_mask=*/std::nullopt,
        scale_value,
        /*keep_prob=*/1.0,
        /*pre_tokens=*/std::numeric_limits<int64_t>::max(),
        /*next_tokens=*/std::numeric_limits<int64_t>::max());
    return std::get<0>(results);
  }

  // Pad seq to a multiple of 256 and head_dim to 128, masking padded keys via
  // pre_tokens; slice the padding back off the output afterwards.
  namespace F = torch::nn::functional;
  const int64_t kSeqBase = 256;
  const int64_t kDimBase = 128;
  const int64_t kMaxToken = 2147483647;  // MAX_TOKEN = 2^31 - 1

  const int64_t q_seqlen = q_bnsd.size(2);   // BNSD: real query seq len
  const int64_t kv_seqlen = k_bnsd.size(2);  // real key/value seq len
  const int64_t head_dim = q_bnsd.size(3);   // real head dim

  // 2. zero-pad seq (dim=2) to a multiple of 256 and head_dim (dim=3) to 128.
  auto pad_to = [](const torch::Tensor& t, int64_t base, int64_t dim) {
    int64_t sz = t.size(dim);
    if (sz % base == 0) {
      return t;
    }
    int64_t pad = ((sz / base) + 1) * base - sz;
    // PadFuncOptions pads from the last dim backwards.
    std::vector<int64_t> spec = (dim == 3) ? std::vector<int64_t>{0, pad}
                                           : std::vector<int64_t>{0, 0, 0, pad};
    return F::pad(t, F::PadFuncOptions(spec).mode(torch::kConstant).value(0))
        .contiguous();
  };
  // Preserve BF16 range and probability precision in the native kernel.
  const auto q_c =
      pad_to(pad_to(q_bnsd.contiguous(), kSeqBase, 2), kDimBase, 3);
  const auto k_c =
      pad_to(pad_to(k_bnsd.contiguous(), kSeqBase, 2), kDimBase, 3);
  const auto v_c =
      pad_to(pad_to(v_bnsd.contiguous(), kSeqBase, 2), kDimBase, 3);

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
  const std::optional<torch::Tensor> no_mask;  // atten/alibi/drop masks: unused

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
