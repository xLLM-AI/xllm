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

#include <framework/core/MLUStream.h>
#include <framework/core/device.h>
#include <glog/logging.h>

#include <algorithm>
#include <cstdint>

#include "compiler/triton/include/jit_kernel.h"
#include "kernels/mlu/mlu_ops_api.h"
#include "util/utils.h"

namespace xllm::kernel::mlu {
namespace {

constexpr char kModule[] =
    "xllm.core.kernels.mlu.triton_kernel.dcp.dcp_correct_attn_out";
constexpr int64_t kShardCount = 4;
constexpr int64_t kSmallHeads = 32;
constexpr int64_t kHeadTile = 64;
constexpr int64_t kValueDim = 512;
constexpr int64_t kDecodeMaxRows = 128;
constexpr int64_t kPrefillMinRows = 256;

}  // namespace

void dcp_correct_attn_out(torch::Tensor& output,
                          const torch::Tensor& lse,
                          const torch::Tensor& slots,
                          int64_t rank,
                          bool base_e) {
  CHECK_EQ(output.dim(), 3);
  CHECK_EQ(lse.dim(), 3);
  CHECK_EQ(slots.dim(), 1);
  CHECK(output.device().is_privateuseone());
  CHECK(output.device() == lse.device());
  CHECK(output.device() == slots.device());
  CHECK(output.scalar_type() == torch::kBFloat16 ||
        output.scalar_type() == torch::kFloat16 ||
        output.scalar_type() == torch::kFloat32);
  CHECK(lse.scalar_type() == torch::kBFloat16 ||
        lse.scalar_type() == torch::kFloat16 ||
        lse.scalar_type() == torch::kFloat32);
  CHECK_EQ(slots.scalar_type(), torch::kInt32);
  CHECK(slots.is_contiguous());
  const int64_t b = output.size(0);
  const int64_t h = output.size(1);
  const int64_t v = output.size(2);
  const int64_t n = lse.size(0);
  CHECK_GT(h, 0);
  CHECK_GT(v, 0);
  CHECK_GT(n, 0);
  CHECK_GE(rank, 0);
  CHECK_LT(rank, n);
  CHECK_EQ(lse.size(1), b);
  CHECK_EQ(lse.size(2), h);
  CHECK_EQ(slots.numel(), b);
  if (b == 0) {
    return;
  }
  const torch::DeviceGuard guard(output.device());
  cnrtQueue_t queue = torch_mlu::getCurMLUStream();
  const bool fast = base_e && n == kShardCount &&
                    (h == kSmallHeads || h == kHeadTile) && v == kValueDim &&
                    output.scalar_type() == torch::kBFloat16 &&
                    lse.scalar_type() == torch::kFloat32 &&
                    output.is_contiguous() && lse.is_contiguous();
  using xllm::triton_jit::JITKernel;
  const bool decode_launch = fast && h == kHeadTile && b <= kDecodeMaxRows;
  const bool prefill_launch = fast && b >= kPrefillMinRows;
  if (decode_launch || prefill_launch) {
    const torch_mlu::DeviceProp* prop =
        torch_mlu::getDeviceProperties(torch_mlu::current_device());
    CHECK(prop != nullptr);
    const int64_t cores = prop->cluster_count * prop->core_num_per_cluster;
    if (decode_launch) {
      JITKernel::get(kModule, "tmo_dcp_decode_kernel")
          .launch(static_cast<void*>(queue),
                  {static_cast<uint32_t>(std::min(b, cores)), 1, 1},
                  {/*num_warps=*/1, /*num_stages=*/4},
                  output,
                  output,
                  lse,
                  slots,
                  rank,
                  b,
                  false,
                  /*G=*/kHeadTile);
    } else {
      JITKernel::get(kModule, "tmo_dcp_prefill_kernel")
          .launch(static_cast<void*>(queue),
                  {static_cast<uint32_t>(cores), 1, 1},
                  {/*num_warps=*/1, /*num_stages=*/4},
                  output,
                  lse,
                  slots,
                  b,
                  h,
                  rank,
                  /*G=*/kHeadTile);
    }
    return;
  }
  const int64_t vn = util::ceil_pow2(static_cast<int32_t>(v));
  const int64_t nn = util::ceil_pow2(static_cast<int32_t>(n));
  JITKernel::get(kModule, "tmo_dcp_correct_attn_out_kernel")
      .launch(static_cast<void*>(queue),
              {static_cast<uint32_t>(b), static_cast<uint32_t>(h), 1},
              {/*num_warps=*/1, /*num_stages=*/1},
              output,
              output,
              lse,
              slots,
              rank,
              n,
              v,
              output.stride(0),
              output.stride(1),
              output.stride(2),
              output.stride(0),
              output.stride(1),
              output.stride(2),
              lse.stride(0),
              lse.stride(1),
              lse.stride(2),
              vn,
              nn,
              base_e);
}

void dcp_correct_attn_transpose(const torch::Tensor& output,
                                const torch::Tensor& lse,
                                const torch::Tensor& slots,
                                int64_t rank,
                                torch::Tensor& dst,
                                bool base_e) {
  CHECK_EQ(output.dim(), 3);
  CHECK_EQ(lse.dim(), 3);
  CHECK_EQ(slots.dim(), 1);
  CHECK(output.device().is_privateuseone());
  CHECK(output.device() == lse.device());
  CHECK(output.device() == slots.device());
  CHECK(output.device() == dst.device());
  CHECK(output.scalar_type() == torch::kBFloat16 ||
        output.scalar_type() == torch::kFloat16 ||
        output.scalar_type() == torch::kFloat32);
  CHECK(lse.scalar_type() == torch::kBFloat16 ||
        lse.scalar_type() == torch::kFloat16 ||
        lse.scalar_type() == torch::kFloat32);
  CHECK_EQ(output.scalar_type(), dst.scalar_type());
  CHECK_EQ(slots.scalar_type(), torch::kInt32);
  CHECK(slots.is_contiguous());
  const int64_t b = output.size(0);
  const int64_t h = output.size(1);
  const int64_t v = output.size(2);
  const int64_t n = lse.size(0);
  CHECK_GT(h, 0);
  CHECK_GT(v, 0);
  CHECK_GT(n, 0);
  CHECK_GE(rank, 0);
  CHECK_LT(rank, n);
  CHECK_EQ(lse.size(1), b);
  CHECK_EQ(lse.size(2), h);
  CHECK_EQ(slots.numel(), b);
  CHECK(dst.sizes() == torch::IntArrayRef({h, b, v}));
  CHECK(dst.is_contiguous());
  CHECK(!dst.is_alias_of(output));
  CHECK(!dst.is_alias_of(lse));
  CHECK(!dst.is_alias_of(slots));
  if (b == 0) {
    return;
  }
  const torch::DeviceGuard guard(output.device());
  cnrtQueue_t queue = torch_mlu::getCurMLUStream();
  const bool fast = base_e && n == kShardCount && h == kHeadTile &&
                    v == kValueDim &&
                    output.scalar_type() == torch::kBFloat16 &&
                    lse.scalar_type() == torch::kFloat32 &&
                    output.is_contiguous() && lse.is_contiguous();
  using xllm::triton_jit::JITKernel;
  if (fast && b <= kDecodeMaxRows) {
    const torch_mlu::DeviceProp* prop =
        torch_mlu::getDeviceProperties(torch_mlu::current_device());
    CHECK(prop != nullptr);
    const int64_t cores = prop->cluster_count * prop->core_num_per_cluster;
    JITKernel::get(kModule, "tmo_dcp_decode_kernel")
        .launch(static_cast<void*>(queue),
                {static_cast<uint32_t>(std::min(b, cores)), 1, 1},
                {/*num_warps=*/1, /*num_stages=*/4},
                output,
                dst,
                lse,
                slots,
                rank,
                b,
                true,
                /*G=*/kHeadTile);
    return;
  }
  const int64_t vn = util::ceil_pow2(static_cast<int32_t>(v));
  const int64_t nn = util::ceil_pow2(static_cast<int32_t>(n));
  JITKernel::get(kModule, "tmo_dcp_correct_attn_out_kernel")
      .launch(static_cast<void*>(queue),
              {static_cast<uint32_t>(b), static_cast<uint32_t>(h), 1},
              {/*num_warps=*/1, /*num_stages=*/1},
              output,
              dst,
              lse,
              slots,
              rank,
              n,
              v,
              output.stride(0),
              output.stride(1),
              output.stride(2),
              dst.stride(1),
              dst.stride(0),
              dst.stride(2),
              lse.stride(0),
              lse.stride(1),
              lse.stride(2),
              vn,
              nn,
              base_e);
}

}  // namespace xllm::kernel::mlu
