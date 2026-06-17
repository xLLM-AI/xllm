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

at::Tensor construct_hc_post_output_tensor(const at::Tensor& residual) {
  constexpr int64_t kDim0 = 0;
  constexpr int64_t kDim1 = 1;
  constexpr int64_t kDim2 = 2;
  constexpr int64_t kDim3 = 3;
  at::SmallVector<int64_t, 8> output_size = {residual.size(kDim0),
                                             residual.size(kDim1),
                                             residual.size(kDim2),
                                             residual.size(kDim3)};
  return at::empty(output_size, residual.options().dtype(residual.dtype()));
}

void check_hc_post_shape_and_dtype(const at::Tensor& x,
                                   const at::Tensor& residual,
                                   const at::Tensor& post,
                                   const at::Tensor& comb) {
  TORCH_CHECK(x.dim() == 3,
              "Input tensor x's dim num should be 3, actual ",
              x.dim(),
              ".");
  for (size_t i = 0; i < 3; ++i) {
    TORCH_CHECK(x.size(i) > 0,
                "Input tensor x's shape should be positive, but x.shape[",
                i,
                "] is ",
                x.size(i),
                ".");
  }

  auto batch = x.size(0);
  auto sequence = x.size(1);
  auto d = x.size(2);

  TORCH_CHECK(residual.dim() == 4,
              "Input tensor residual's dim num should be 4, actual ",
              residual.dim(),
              ".");
  auto hc = residual.size(2);
  TORCH_CHECK(
      hc > 0, "The hc of residual should be positive, actual ", hc, ".");
  TORCH_CHECK(residual.size(0) == batch,
              "The residual.shape[0] should be batch, actual residual.shape[0] "
              "is ",
              residual.size(0),
              ", batch is ",
              batch,
              ".");
  TORCH_CHECK(residual.size(1) == sequence,
              "The residual.shape[1] should be sequence, actual "
              "residual.shape[1] is ",
              residual.size(1),
              ", sequence is ",
              sequence,
              ".");
  TORCH_CHECK(residual.size(3) == d,
              "The residual.shape[3] should be d, actual residual.shape[3] is ",
              residual.size(3),
              ", d is ",
              d,
              ".");

  TORCH_CHECK(post.dim() == 3,
              "Input tensor post's dim num should be 3, actual ",
              post.dim(),
              ".");
  TORCH_CHECK(post.size(0) == batch,
              "The post.shape[0] should be batch, actual post.shape[0] is ",
              post.size(0),
              ", batch is ",
              batch,
              ".");
  TORCH_CHECK(post.size(1) == sequence,
              "The post.shape[1] should be sequence, actual post.shape[1] is ",
              post.size(1),
              ", sequence is ",
              sequence,
              ".");
  TORCH_CHECK(post.size(2) == hc,
              "The post.shape[2] should be hc, actual post.shape[2] is ",
              post.size(2),
              ", hc is ",
              hc,
              ".");

  TORCH_CHECK(comb.dim() == 4,
              "Input tensor comb's dim num should be 4, actual ",
              comb.dim(),
              ".");
  TORCH_CHECK(comb.size(0) == batch,
              "The comb.shape[0] should be batch, actual comb.shape[0] is ",
              comb.size(0),
              ", batch is ",
              batch,
              ".");
  TORCH_CHECK(comb.size(1) == sequence,
              "The comb.shape[1] should be sequence, actual comb.shape[1] is ",
              comb.size(1),
              ", sequence is ",
              sequence,
              ".");
  TORCH_CHECK(comb.size(2) == hc,
              "The comb.shape[2] should be hc, actual comb.shape[2] is ",
              comb.size(2),
              ", hc is ",
              hc,
              ".");
  TORCH_CHECK(comb.size(3) == hc,
              "The comb.shape[3] should be hc, actual comb.shape[3] is ",
              comb.size(3),
              ", hc is ",
              hc,
              ".");

  TORCH_CHECK(x.dtype() == at::kFloat || x.dtype() == at::kHalf ||
                  x.dtype() == at::kBFloat16,
              "x should be FLOAT16, BFLOAT16, or FLOAT32.");
  TORCH_CHECK(residual.dtype() == x.dtype(),
              "x's dtype should be equal to residual's dtype.");
  TORCH_CHECK(post.dtype() == at::kFloat || post.dtype() == at::kHalf ||
                  post.dtype() == at::kBFloat16,
              "post should be FLOAT16, BFLOAT16, or FLOAT32.");
  TORCH_CHECK(comb.dtype() == post.dtype(),
              "comb's dtype should be equal to post's dtype.");
}

}  // namespace

at::Tensor hc_post(const at::Tensor& x,
                   const at::Tensor& residual,
                   const at::Tensor& post,
                   const at::Tensor& comb) {
  check_hc_post_shape_and_dtype(x, residual, post, comb);
  at::Tensor out = construct_hc_post_output_tensor(residual);
  EXEC_NPU_CMD(aclnnHcPost, x, residual, post, comb, out);
  return out;
}

}  // namespace xllm::kernel::npu
