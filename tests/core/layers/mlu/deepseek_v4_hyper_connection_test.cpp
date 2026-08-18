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

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "framework/state_dict/state_dict.h"
#include "layers/mlu/deepseek_v4/hyper_connection.h"
#include "layers/mlu/tests_utils.h"
#include "platform/device.h"
#include "platform/platform.h"

namespace xllm {
namespace layer {
namespace {

struct HCConfig {
  int64_t hc_mult = 4;
  int64_t dim = 4096;
  int64_t sinkhorn_iters = 20;
  double hc_eps = 1e-6;
  double norm_eps = 1e-6;
};

struct HCPreRefOut {
  torch::Tensor output;
  torch::Tensor post;
  torch::Tensor comb;
};

torch::Tensor seeded(const std::string& key,
                     torch::IntArrayRef shape,
                     torch::ScalarType dtype,
                     const torch::Device& device) {
  return (test::seeded_tensor(key, shape, dtype, device) - 0.5) * 0.05;
}

torch::Tensor linear_ref(const torch::Tensor& input,
                         const torch::Tensor& weight) {
  return torch::nn::functional::linear(
      input.to(torch::kFloat32), weight.to(input.device()).to(torch::kFloat32));
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> split_ref(
    const torch::Tensor& mixes,
    const torch::Tensor& hc_scale,
    const torch::Tensor& hc_base,
    int64_t hc_mult,
    int64_t sinkhorn_iters,
    double eps) {
  torch::Tensor mixes_flat = mixes.reshape({-1, (2 + hc_mult) * hc_mult});
  torch::Tensor scale = hc_scale.to(mixes.device()).to(torch::kFloat32);
  torch::Tensor base = hc_base.to(mixes.device()).to(torch::kFloat32);

  torch::Tensor pre_logits = mixes_flat.slice(-1, 0, hc_mult);
  torch::Tensor post_logits = mixes_flat.slice(-1, hc_mult, 2 * hc_mult);
  torch::Tensor comb_logits =
      mixes_flat.slice(-1, 2 * hc_mult).reshape({-1, hc_mult, hc_mult});

  torch::Tensor pre =
      torch::sigmoid(pre_logits * scale[0] + base.slice(0, 0, hc_mult)) + eps;
  torch::Tensor post =
      2.0 * torch::sigmoid(post_logits * scale[1] +
                           base.slice(0, hc_mult, 2 * hc_mult));
  torch::Tensor comb = comb_logits * scale[2] +
                       base.slice(0, 2 * hc_mult).reshape({hc_mult, hc_mult});

  comb = torch::softmax(comb, -1) + eps;
  comb = comb / (comb.sum(-2, true) + eps);
  for (int64_t iter = 1; iter < sinkhorn_iters; ++iter) {
    comb = comb / (comb.sum(-1, true) + eps);
    comb = comb / (comb.sum(-2, true) + eps);
  }
  return {pre, post, comb};
}

HCPreRefOut hc_pre_ref(const torch::Tensor& x,
                       const torch::Tensor& hc_fn,
                       const torch::Tensor& hc_scale,
                       const torch::Tensor& hc_base,
                       const HCConfig& config) {
  std::vector<int64_t> leading;
  leading.reserve(x.dim() - 2);
  for (int64_t dim_idx = 0; dim_idx < x.dim() - 2; ++dim_idx) {
    leading.emplace_back(x.size(dim_idx));
  }
  torch::Tensor x_hc = x.reshape({-1, config.hc_mult, config.dim}).contiguous();
  torch::Tensor x_flat =
      x_hc.reshape({x_hc.size(0), config.hc_mult * config.dim});
  torch::Tensor rsqrt = torch::rsqrt(
      x_flat.to(torch::kFloat32).square().mean(-1, true) + config.norm_eps);
  torch::Tensor mixes = linear_ref(x_flat, hc_fn) * rsqrt;
  torch::Tensor pre;
  torch::Tensor post;
  torch::Tensor comb;
  std::tie(pre, post, comb) = split_ref(mixes,
                                        hc_scale,
                                        hc_base,
                                        config.hc_mult,
                                        config.sinkhorn_iters,
                                        config.hc_eps);
  torch::Tensor output =
      (pre.unsqueeze(-1) * x_hc.to(torch::kFloat32)).sum(1).to(x.scalar_type());

  std::vector<int64_t> out_shape = leading;
  out_shape.emplace_back(config.dim);
  std::vector<int64_t> post_shape = leading;
  post_shape.emplace_back(config.hc_mult);
  std::vector<int64_t> comb_shape = leading;
  comb_shape.emplace_back(config.hc_mult);
  comb_shape.emplace_back(config.hc_mult);
  return {output.reshape(out_shape),
          post.reshape(post_shape),
          comb.reshape(comb_shape)};
}

std::tuple<torch::Tensor, torch::Tensor> hc_post_ref(
    const torch::Tensor& x,
    const torch::Tensor& residual,
    const torch::Tensor& post,
    const torch::Tensor& comb,
    bool compute_rms,
    double norm_eps) {
  torch::Tensor output = post.unsqueeze(-1).to(torch::kFloat32) *
                             x.unsqueeze(-2).to(torch::kFloat32) +
                         (comb.unsqueeze(-1).to(torch::kFloat32) *
                          residual.unsqueeze(-2).to(torch::kFloat32))
                             .sum(-3);
  output = output.to(x.scalar_type());
  torch::Tensor rsqrt;
  if (compute_rms) {
    rsqrt = torch::rsqrt(
        output.flatten(-2).to(torch::kFloat32).square().mean(-1, true) +
        norm_eps);
  }
  return {output, rsqrt};
}

torch::Tensor hc_head_ref(const torch::Tensor& x,
                          const torch::Tensor& hc_fn,
                          const torch::Tensor& hc_scale,
                          const torch::Tensor& hc_base,
                          const HCConfig& config) {
  std::vector<int64_t> leading;
  leading.reserve(x.dim() - 2);
  for (int64_t dim_idx = 0; dim_idx < x.dim() - 2; ++dim_idx) {
    leading.emplace_back(x.size(dim_idx));
  }
  torch::Tensor x_hc = x.reshape({-1, config.hc_mult, config.dim}).contiguous();
  torch::Tensor x_flat =
      x_hc.reshape({x_hc.size(0), config.hc_mult * config.dim});
  torch::Tensor rsqrt = torch::rsqrt(
      x_flat.to(torch::kFloat32).square().mean(-1, true) + config.norm_eps);
  torch::Tensor mixes = linear_ref(x_flat, hc_fn) * rsqrt;
  torch::Tensor pre =
      torch::sigmoid(mixes * hc_scale.to(mixes.device()).to(torch::kFloat32) +
                     hc_base.to(mixes.device()).to(torch::kFloat32)) +
      config.hc_eps;
  torch::Tensor output =
      (pre.unsqueeze(-1) * x_hc.to(torch::kFloat32)).sum(1).to(x.scalar_type());
  std::vector<int64_t> out_shape = leading;
  out_shape.emplace_back(config.dim);
  return output.reshape(out_shape);
}

}  // namespace

class DeepseekV4HyperConnectionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    torch::Device torch_device(Platform::type_torch(), 0);
    Device device(torch_device);
    device.set_seed();
    options_ = torch::TensorOptions()
                   .dtype(torch::kBFloat16)
                   .device(torch_device)
                   .requires_grad(false);
  }

  std::unordered_map<std::string, torch::Tensor> pre_weights() {
    const int64_t mix_hc = (2 + config_.hc_mult) * config_.hc_mult;
    const int64_t hc_dim = config_.hc_mult * config_.dim;
    return {
        {"hc_fn",
         seeded("deepseek_v4_hc.pre.fn",
                {mix_hc, hc_dim},
                torch::kBFloat16,
                options_.device())},
        {"hc_base",
         seeded("deepseek_v4_hc.pre.base",
                {mix_hc},
                torch::kFloat32,
                options_.device())},
        {"hc_scale",
         torch::tensor({0.1f, 0.2f, 0.15f}, options_.dtype(torch::kFloat32))}};
  }

  std::unordered_map<std::string, torch::Tensor> head_weights() {
    const int64_t hc_dim = config_.hc_mult * config_.dim;
    return {{"hc_head_fn",
             seeded("deepseek_v4_hc.head.fn",
                    {config_.hc_mult, hc_dim},
                    torch::kFloat32,
                    options_.device())},
            {"hc_head_base",
             seeded("deepseek_v4_hc.head.base",
                    {config_.hc_mult},
                    torch::kFloat32,
                    options_.device())},
            {"hc_head_scale",
             torch::tensor({0.1f}, options_.dtype(torch::kFloat32))}};
  }

  HCConfig config_;
  torch::TensorOptions options_;
};

TEST_F(DeepseekV4HyperConnectionTest, HCPreMatchesOfficialReference) {
  const int64_t batch_size = 1;
  const int64_t seq_len = 2;
  torch::Tensor x = seeded("deepseek_v4_hc.pre.x",
                           {batch_size, seq_len, config_.hc_mult, config_.dim},
                           torch::kBFloat16,
                           options_.device());
  std::unordered_map<std::string, torch::Tensor> weights = pre_weights();

  DeepseekV4HCPre hc_pre(config_.hc_mult,
                         config_.dim,
                         config_.sinkhorn_iters,
                         config_.hc_eps,
                         config_.norm_eps,
                         options_);
  hc_pre->load_state_dict(StateDict(weights));
  DeepseekV4HCPreOutput actual = hc_pre->forward(x);
  HCPreRefOut expected = hc_pre_ref(x,
                                    weights.at("hc_fn"),
                                    weights.at("hc_scale"),
                                    weights.at("hc_base"),
                                    config_);

  EXPECT_EQ(actual.output.sizes(), expected.output.sizes());
  EXPECT_EQ(actual.post.sizes(), expected.post.sizes());
  EXPECT_EQ(actual.comb.sizes(), expected.comb.sizes());
  test::verify_tensor_close(
      actual.output.cpu(), expected.output.cpu(), 1e-2, 1e-2);
  test::verify_tensor_close(actual.post.cpu(), expected.post.cpu(), 1e-3, 1e-3);
  test::verify_tensor_close(actual.comb.cpu(), expected.comb.cpu(), 1e-3, 1e-3);
}

TEST_F(DeepseekV4HyperConnectionTest, HCPostMatchesOfficialReference) {
  const int64_t tokens = 2;
  torch::Tensor x = seeded("deepseek_v4_hc.post.x",
                           {tokens, config_.dim},
                           torch::kBFloat16,
                           options_.device());
  torch::Tensor residual = seeded("deepseek_v4_hc.post.residual",
                                  {tokens, config_.hc_mult, config_.dim},
                                  torch::kBFloat16,
                                  options_.device());
  torch::Tensor post = seeded("deepseek_v4_hc.post.post",
                              {tokens, config_.hc_mult},
                              torch::kFloat32,
                              options_.device()) +
                       0.5;
  torch::Tensor comb =
      torch::softmax(seeded("deepseek_v4_hc.post.comb",
                            {tokens, config_.hc_mult, config_.hc_mult},
                            torch::kFloat32,
                            options_.device()),
                     -1);

  DeepseekV4HCPost hc_post(config_.norm_eps);
  torch::Tensor actual;
  torch::Tensor actual_rsqrt;
  std::tie(actual, actual_rsqrt) =
      hc_post->forward(x, residual, post, comb, /*compute_rms=*/true);
  torch::Tensor expected;
  torch::Tensor expected_rsqrt;
  std::tie(expected, expected_rsqrt) = hc_post_ref(x,
                                                   residual,
                                                   post,
                                                   comb,
                                                   /*compute_rms=*/true,
                                                   config_.norm_eps);

  EXPECT_EQ(actual.sizes(), expected.sizes());
  EXPECT_EQ(actual_rsqrt.sizes(), expected_rsqrt.sizes());
  test::verify_tensor_close(actual.cpu(), expected.cpu(), 1e-2, 1e-2);
  test::verify_tensor_close(
      actual_rsqrt.cpu(), expected_rsqrt.cpu(), 1e-4, 1e-4);
}

TEST_F(DeepseekV4HyperConnectionTest, FusedPostPreNormMatchesComposition) {
  const int64_t tokens = 2;
  torch::Tensor x = seeded("deepseek_v4_hc.fused.x",
                           {tokens, config_.dim},
                           torch::kBFloat16,
                           options_.device());
  torch::Tensor residual = seeded("deepseek_v4_hc.fused.residual",
                                  {tokens, config_.hc_mult, config_.dim},
                                  torch::kBFloat16,
                                  options_.device());
  torch::Tensor post = seeded("deepseek_v4_hc.fused.post",
                              {tokens, config_.hc_mult},
                              torch::kFloat32,
                              options_.device()) +
                       0.5;
  torch::Tensor comb =
      torch::softmax(seeded("deepseek_v4_hc.fused.comb",
                            {tokens, config_.hc_mult, config_.hc_mult},
                            torch::kFloat32,
                            options_.device()),
                     -1);
  torch::Tensor gamma = seeded("deepseek_v4_hc.fused.gamma",
                               {config_.dim},
                               torch::kBFloat16,
                               options_.device()) +
                        1.0;
  std::unordered_map<std::string, torch::Tensor> weights = pre_weights();
  DeepseekV4HCPre hc_pre(config_.hc_mult,
                         config_.dim,
                         config_.sinkhorn_iters,
                         config_.hc_eps,
                         config_.norm_eps,
                         options_);
  hc_pre->load_state_dict(StateDict(weights));
  DeepseekV4HCPost hc_post(config_.norm_eps);

  torch::Tensor expected_residual;
  torch::Tensor rsqrt;
  std::tie(expected_residual, rsqrt) =
      hc_post->forward(x, residual, post, comb, /*compute_rms=*/true);
  DeepseekV4HCPreOutput expected_pre =
      hc_pre->forward(expected_residual, rsqrt);
  torch::Tensor expected_norm =
      expected_pre.output.to(torch::kFloat32) *
      torch::rsqrt(
          expected_pre.output.to(torch::kFloat32).square().mean(-1, true) +
          config_.norm_eps) *
      gamma.to(torch::kFloat32);

  torch::Tensor actual_norm;
  torch::Tensor actual_residual;
  torch::Tensor actual_post;
  torch::Tensor actual_comb;
  std::tie(actual_norm, actual_residual, actual_post, actual_comb) =
      hc_pre->fused_post_pre_norm(x, residual, post, comb, gamma);

  test::verify_tensor_close(
      actual_norm.cpu(), expected_norm.to(torch::kBFloat16).cpu(), 2e-2, 2e-2);
  test::verify_tensor_close(
      actual_residual.cpu(), expected_residual.cpu(), 2e-2, 2e-2);
  test::verify_tensor_close(
      actual_post.cpu(), expected_pre.post.cpu(), 2e-3, 2e-3);
  test::verify_tensor_close(
      actual_comb.cpu(), expected_pre.comb.cpu(), 2e-3, 2e-3);
}

TEST_F(DeepseekV4HyperConnectionTest, HCHeadMatchesOfficialReference) {
  const int64_t batch_size = 1;
  const int64_t seq_len = 2;
  torch::Tensor x = seeded("deepseek_v4_hc.head.x",
                           {batch_size, seq_len, config_.hc_mult, config_.dim},
                           torch::kBFloat16,
                           options_.device());
  std::unordered_map<std::string, torch::Tensor> weights = head_weights();

  DeepseekV4HCHead hc_head(
      config_.hc_mult, config_.dim, config_.hc_eps, config_.norm_eps, options_);
  hc_head->load_state_dict(StateDict(weights));
  torch::Tensor actual = hc_head->forward(x);
  torch::Tensor expected = hc_head_ref(x,
                                       weights.at("hc_head_fn"),
                                       weights.at("hc_head_scale"),
                                       weights.at("hc_head_base"),
                                       config_);

  EXPECT_EQ(actual.sizes(), expected.sizes());
  test::verify_tensor_close(actual.cpu(), expected.cpu(), 1e-2, 1e-2);
}

}  // namespace layer
}  // namespace xllm
