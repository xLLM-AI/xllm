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

#include <torch/library.h>

#include "core/kernels/npu/aclnn/pytorch_npu_helper.hpp"
#include "xllm_ops_api.h"

namespace xllm::kernel::npu {
namespace {

std::tuple<at::Tensor, at::Tensor, at::Tensor>
construct_hc_pre_sinkhorn_output_tensor(const at::Tensor& x, int64_t hc_mult) {
  at::SmallVector<int64_t, 8> y_size;
  at::SmallVector<int64_t, 8> post_size;
  at::SmallVector<int64_t, 8> comb_frag_size;

  if (x.dim() == 4) {
    y_size = {x.size(0), x.size(1), x.size(3)};
    post_size = {x.size(0), x.size(1), hc_mult};
    comb_frag_size = {x.size(0), x.size(1), hc_mult, hc_mult};
  } else {
    TORCH_CHECK(x.dim() == 3,
                "Input tensor x's dim num should be 3 or 4, actual ",
                x.dim(),
                ".");
    y_size = {x.size(0), x.size(2)};
    post_size = {x.size(0), hc_mult};
    comb_frag_size = {x.size(0), hc_mult, hc_mult};
  }

  at::Tensor y = at::empty(y_size, x.options().dtype(x.dtype()));
  at::Tensor post = at::empty(post_size, x.options().dtype(at::kFloat));
  at::Tensor comb_frag =
      at::empty(comb_frag_size, x.options().dtype(at::kFloat));
  return std::make_tuple(y, post, comb_frag);
}

void check_hc_pre_sinkhorn_shape_and_dtype(const at::Tensor& mixes,
                                           const at::Tensor& rsqrt,
                                           const at::Tensor& hc_scale,
                                           const at::Tensor& hc_base,
                                           const at::Tensor& x,
                                           int64_t hc_mult,
                                           int64_t hc_sinkhorn_iters,
                                           double hc_eps) {
  TORCH_CHECK(x.dim() == 3 || x.dim() == 4,
              "Input tensor x's dim num should be 3 or 4, actual ",
              x.dim(),
              ".");
  TORCH_CHECK(mixes.dim() == x.dim() - 1,
              "Input tensor mixes' dim num should be x.dim()-1, actual ",
              mixes.dim(),
              ".");
  TORCH_CHECK(rsqrt.dim() == mixes.dim(),
              "Input tensor rsqrt's dim num should be equal to mixes, actual ",
              rsqrt.dim(),
              ".");

  for (int64_t i = 0; i < x.dim(); ++i) {
    TORCH_CHECK(x.size(i) > 0,
                "Input tensor x's shape should be positive, but x.shape[",
                i,
                "] is ",
                x.size(i),
                ".");
  }

  TORCH_CHECK(mixes.dtype() == at::kFloat,
              "mixes should be FLOAT32, actual ",
              mixes.dtype(),
              ".");
  TORCH_CHECK(rsqrt.dtype() == at::kFloat,
              "rsqrt should be FLOAT32, actual ",
              rsqrt.dtype(),
              ".");
  TORCH_CHECK(hc_scale.dtype() == at::kFloat,
              "hc_scale should be FLOAT32, actual ",
              hc_scale.dtype(),
              ".");
  TORCH_CHECK(hc_base.dtype() == at::kFloat,
              "hc_base should be FLOAT32, actual ",
              hc_base.dtype(),
              ".");
  TORCH_CHECK(x.dtype() == at::kBFloat16,
              "x should be BFLOAT16, actual ",
              x.dtype(),
              ".");

  TORCH_CHECK(
      hc_mult > 0, "hc_mult should be greater than 0, actual ", hc_mult, ".");
  TORCH_CHECK(hc_sinkhorn_iters >= 0,
              "hc_sinkhorn_iters should be greater than or equal to 0, actual ",
              hc_sinkhorn_iters,
              ".");
  TORCH_CHECK(
      hc_eps > 0.0, "hc_eps should be greater than 0, actual ", hc_eps, ".");
}

}  // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor> hc_pre_sinkhorn(
    const at::Tensor& mixes,
    const at::Tensor& rsqrt,
    const at::Tensor& hc_scale,
    const at::Tensor& hc_base,
    const at::Tensor& x,
    int64_t hc_mult,
    int64_t hc_sinkhorn_iters,
    double hc_eps) {
  check_hc_pre_sinkhorn_shape_and_dtype(
      mixes, rsqrt, hc_scale, hc_base, x, hc_mult, hc_sinkhorn_iters, hc_eps);

  auto output_tensors = construct_hc_pre_sinkhorn_output_tensor(x, hc_mult);
  at::Tensor y = std::get<0>(output_tensors);
  at::Tensor post = std::get<1>(output_tensors);
  at::Tensor comb_frag = std::get<2>(output_tensors);

  EXEC_NPU_CMD(aclnnHcPreSinkhorn,
               mixes,
               rsqrt,
               hc_scale,
               hc_base,
               x,
               hc_mult,
               hc_sinkhorn_iters,
               hc_eps,
               y,
               post,
               comb_frag);

  return std::make_tuple(y, post, comb_frag);
}

}  // namespace xllm::kernel::npu
