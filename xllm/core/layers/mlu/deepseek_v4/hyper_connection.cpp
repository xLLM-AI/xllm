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

#include "layers/mlu/deepseek_v4/hyper_connection.h"

#include <tuple>
#include <vector>

#include "kernels/mlu/mlu_ops_api.h"

namespace {

std::vector<int64_t> leading_shape(const torch::Tensor& tensor,
                                   int64_t tail_dims) {
  std::vector<int64_t> shape;
  const int64_t leading_dims = tensor.dim() - tail_dims;
  shape.reserve(static_cast<size_t>(leading_dims + 1));
  for (int64_t dim_idx = 0; dim_idx < leading_dims; ++dim_idx) {
    shape.emplace_back(tensor.size(dim_idx));
  }
  return shape;
}

torch::Tensor flat_hc(const torch::Tensor& x, int64_t hc_mult, int64_t dim) {
  return x.reshape({-1, hc_mult, dim}).contiguous();
}

torch::Tensor flat_hidden(const torch::Tensor& x, int64_t dim) {
  return x.reshape({-1, dim}).contiguous();
}

torch::Tensor flat_matrix(const torch::Tensor& x, int64_t rows, int64_t cols) {
  return x.reshape({-1, rows, cols}).contiguous();
}

}  // namespace

namespace xllm {
namespace layer {

DeepseekV4HCPreImpl::DeepseekV4HCPreImpl(int64_t hc_mult,
                                         int64_t dim,
                                         int64_t sinkhorn_iters,
                                         double hc_eps,
                                         double norm_eps,
                                         const torch::TensorOptions& options)
    : hc_mult_(hc_mult),
      dim_(dim),
      sinkhorn_iters_(sinkhorn_iters),
      hc_eps_(hc_eps),
      norm_eps_(norm_eps) {
  const int64_t mix_hc = (2 + hc_mult_) * hc_mult_;
  const int64_t hc_dim = hc_mult_ * dim_;
  torch::TensorOptions param_options = options.requires_grad(false);
  hc_fn_ = register_parameter("hc_fn",
                              torch::empty({mix_hc, hc_dim}, param_options),
                              /*requires_grad=*/false);
  param_options = param_options.dtype(torch::kFloat32);
  hc_base_ = register_parameter("hc_base",
                                torch::empty({mix_hc}, param_options),
                                /*requires_grad=*/false);
  hc_scale_ = register_parameter(
      "hc_scale", torch::empty({3}, param_options), /*requires_grad=*/false);
}

DeepseekV4HCPreOutput DeepseekV4HCPreImpl::forward(
    const torch::Tensor& x,
    const std::optional<torch::Tensor>& rsqrt) {
  torch::Tensor x_hc = flat_hc(x, hc_mult_, dim_);
  torch::Tensor x_flat = x_hc.reshape({x_hc.size(0), hc_mult_ * dim_});

  torch::Tensor pre_scale;
  if (rsqrt.has_value()) {
    pre_scale = rsqrt.value().reshape({-1}).contiguous();
  } else {
    pre_scale =
        torch::rsqrt(x_flat.to(torch::kFloat32).square().mean(-1, false) +
                     norm_eps_)
            .contiguous();
  }

  torch::Tensor mixes = torch::nn::functional::linear(x_flat, hc_fn_);
  torch::Tensor pre;
  torch::Tensor post;
  torch::Tensor comb;
  std::tie(pre, post, comb) = kernel::mlu::hc_split_sinkhorn(mixes,
                                                             hc_scale_,
                                                             hc_base_,
                                                             pre_scale,
                                                             hc_mult_,
                                                             sinkhorn_iters_,
                                                             hc_eps_);

  torch::Tensor output = kernel::mlu::fused_mul_reduce_sum(x_hc, pre);
  std::vector<int64_t> out_shape = leading_shape(x, /*tail_dims=*/2);
  out_shape.emplace_back(dim_);
  output = output.reshape(out_shape);

  out_shape.back() = hc_mult_;
  post = post.reshape(out_shape);

  out_shape.emplace_back(hc_mult_);
  comb = comb.reshape(out_shape);
  return {output, post, comb};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
DeepseekV4HCPreImpl::fused_post_pre_norm(const torch::Tensor& x,
                                         const torch::Tensor& residual,
                                         const torch::Tensor& post,
                                         const torch::Tensor& comb,
                                         const torch::Tensor& gamma) {
  CHECK(supports_fused_mhc())
      << "fused_mhc only supports hc_mult=4 and hidden_size=4096";
  const int64_t token_count = x.numel() / dim_;
  return kernel::mlu::fused_mhc(
      flat_hidden(x, dim_),
      flat_hc(residual, hc_mult_, dim_),
      hc_fn_,
      gamma,
      post.reshape({token_count, hc_mult_}).contiguous(),
      flat_matrix(comb, hc_mult_, hc_mult_),
      hc_scale_,
      hc_base_,
      sinkhorn_iters_,
      hc_eps_);
}

void DeepseekV4HCPreImpl::load_state_dict(const StateDict& state_dict) {
  if (state_dict.size() == 0) {
    return;
  }
  LOAD_WEIGHT(hc_fn);
  LOAD_WEIGHT(hc_base);
  LOAD_WEIGHT(hc_scale);
}

DeepseekV4HCPostImpl::DeepseekV4HCPostImpl(double norm_eps)
    : norm_eps_(norm_eps) {}

std::tuple<torch::Tensor, torch::Tensor> DeepseekV4HCPostImpl::forward(
    const torch::Tensor& x,
    const torch::Tensor& residual,
    const torch::Tensor& post,
    const torch::Tensor& comb,
    bool compute_rms) {
  const int64_t hc_mult = residual.size(-2);
  const int64_t dim = residual.size(-1);
  torch::Tensor x_flat = flat_hidden(x, dim);
  torch::Tensor residual_flat = flat_hc(residual, hc_mult, dim);
  torch::Tensor post_flat = post.reshape({-1, hc_mult}).contiguous();
  torch::Tensor comb_flat = flat_matrix(comb, hc_mult, hc_mult);

  torch::Tensor output;
  torch::Tensor output_rms;
  std::tie(output, output_rms) = kernel::mlu::fused_mhc_post(
      x_flat, residual_flat, post_flat, comb_flat, compute_rms, norm_eps_);

  std::vector<int64_t> out_shape = residual.sizes().vec();
  if (output_rms.defined()) {
    std::vector<int64_t> rms_shape = leading_shape(x, /*tail_dims=*/1);
    rms_shape.emplace_back(1);
    output_rms = output_rms.reshape(rms_shape);
  }
  return {output.reshape(out_shape), output_rms};
}

DeepseekV4HCHeadImpl::DeepseekV4HCHeadImpl(int64_t hc_mult,
                                           int64_t dim,
                                           double hc_eps,
                                           double norm_eps,
                                           const torch::TensorOptions& options)
    : hc_mult_(hc_mult), dim_(dim), hc_eps_(hc_eps), norm_eps_(norm_eps) {
  const int64_t hc_dim = hc_mult_ * dim_;
  torch::TensorOptions param_options =
      options.dtype(torch::kFloat32).requires_grad(false);
  hc_head_fn_ =
      register_parameter("hc_head_fn",
                         torch::empty({hc_mult_, hc_dim}, param_options),
                         /*requires_grad=*/false);
  hc_head_base_ = register_parameter("hc_head_base",
                                     torch::empty({hc_mult_}, param_options),
                                     /*requires_grad=*/false);
  hc_head_scale_ = register_parameter("hc_head_scale",
                                      torch::empty({1}, param_options),
                                      /*requires_grad=*/false);
}

torch::Tensor DeepseekV4HCHeadImpl::forward(const torch::Tensor& x) {
  torch::Tensor x_hc = flat_hc(x, hc_mult_, dim_);
  torch::Tensor x_flat = x_hc.reshape({x_hc.size(0), hc_mult_ * dim_});
  torch::Tensor x_fp32 = x_flat.to(torch::kFloat32);
  torch::Tensor rsqrt =
      torch::rsqrt(x_fp32.square().mean(-1, true) + norm_eps_);
  torch::Tensor mixes =
      torch::nn::functional::linear(x_fp32, hc_head_fn_) * rsqrt;
  torch::Tensor pre =
      torch::sigmoid(mixes * hc_head_scale_ + hc_head_base_) + hc_eps_;
  torch::Tensor output = kernel::mlu::fused_mul_reduce_sum(x_hc, pre);
  std::vector<int64_t> out_shape = leading_shape(x, /*tail_dims=*/2);
  out_shape.emplace_back(dim_);
  return output.reshape(out_shape);
}

void DeepseekV4HCHeadImpl::load_state_dict(const StateDict& state_dict) {
  if (state_dict.size() == 0) {
    return;
  }
  LOAD_WEIGHT(hc_head_fn);
  LOAD_WEIGHT(hc_head_base);
  LOAD_WEIGHT(hc_head_scale);
}

}  // namespace layer
}  // namespace xllm
