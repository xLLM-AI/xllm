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

#include "core/layers/rwkv7_decoder_layer.h"

#include <gtest/gtest.h>
#include <torch/cuda.h>
#include <torch/torch.h>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_args.h"
#include "framework/model/model_input_params.h"
#include "framework/model_context.h"
#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"

namespace xllm {
namespace layer {
namespace {

constexpr int64_t kHiddenSize = 128;
constexpr int64_t kHeadSize = 64;
constexpr int64_t kNumHeads = 2;
constexpr int64_t kIntermediateSize = 512;
constexpr int64_t kLoraRank = 16;
constexpr int64_t kGateRank = 32;
constexpr int64_t kNumStateSlots = 4;

torch::Tensor seeded_tensor(const std::string& name,
                            const std::vector<int64_t>& shape,
                            const torch::TensorOptions& options) {
  const int64_t seed = static_cast<int64_t>(std::hash<std::string>{}(name));
  return (torch::randn(shape, options) * 0.02f) + static_cast<float>(seed % 7);
}

ModelArgs make_model_args() {
  ModelArgs args;
  args.model_type() = "rwkv7";
  args.hidden_size() = kHiddenSize;
  args.head_dim() = kHeadSize;
  args.n_heads() = kNumHeads;
  args.n_kv_heads() = 1;
  args.intermediate_size() = kIntermediateSize;
  args.layer_norm_eps() = 1e-5f;
  args.linear_num_key_heads() = kNumHeads;
  args.linear_num_value_heads() = static_cast<int32_t>(kNumHeads);
  args.linear_key_head_dim() = static_cast<int32_t>(kHeadSize);
  args.linear_value_head_dim() = static_cast<int32_t>(kHeadSize);
  args.linear_conv_kernel_dim() = 2;
  return args;
}

StateDict make_layer0_state_dict(const torch::TensorOptions& options) {
  std::unordered_map<std::string, torch::Tensor> tensors;
  const auto dev = options.device();
  const auto dtype = options.dtype();

  auto param = [&](const std::string& name) {
    return seeded_tensor(name, {1, 1, kHiddenSize}, options);
  };

  tensors["ln0.weight"] =
      torch::ones({kHiddenSize}, torch::kFloat32).to(dev).to(dtype);
  tensors["ln0.bias"] =
      torch::zeros({kHiddenSize}, torch::kFloat32).to(dev).to(dtype);
  tensors["ln1.weight"] = tensors["ln0.weight"];
  tensors["ln1.bias"] = tensors["ln0.bias"];
  tensors["ln2.weight"] = tensors["ln0.weight"];
  tensors["ln2.bias"] = tensors["ln0.bias"];

  tensors["att.x_r"] = param("att.x_r");
  tensors["att.x_w"] = param("att.x_w");
  tensors["att.x_k"] = param("att.x_k");
  tensors["att.x_v"] = param("att.x_v");
  tensors["att.x_a"] = param("att.x_a");
  tensors["att.x_g"] = param("att.x_g");
  tensors["att.w0"] = param("att.w0");
  tensors["att.a0"] = param("att.a0");
  tensors["att.v0"] = param("att.v0");
  tensors["att.k_k"] = torch::ones({1, 1, kHiddenSize}, options) * 0.5f;
  tensors["att.k_a"] = torch::ones({1, 1, kHiddenSize}, options) * 0.5f;
  tensors["att.r_k"] =
      seeded_tensor("att.r_k", {kNumHeads, kHeadSize}, options);

  tensors["att.w1"] =
      seeded_tensor("att.w1", {kHiddenSize, kLoraRank}, options);
  tensors["att.w2"] =
      seeded_tensor("att.w2", {kLoraRank, kHiddenSize}, options);
  tensors["att.a1"] =
      seeded_tensor("att.a1", {kHiddenSize, kLoraRank}, options);
  tensors["att.a2"] =
      seeded_tensor("att.a2", {kLoraRank, kHiddenSize}, options);
  tensors["att.g1"] =
      seeded_tensor("att.g1", {kHiddenSize, kGateRank}, options);
  tensors["att.g2"] =
      seeded_tensor("att.g2", {kGateRank, kHiddenSize}, options);

  tensors["att.receptance.weight"] = seeded_tensor(
      "att.receptance.weight", {kHiddenSize, kHiddenSize}, options);
  tensors["att.key.weight"] =
      seeded_tensor("att.key.weight", {kHiddenSize, kHiddenSize}, options);
  tensors["att.value.weight"] =
      seeded_tensor("att.value.weight", {kHiddenSize, kHiddenSize}, options);
  tensors["att.output.weight"] =
      seeded_tensor("att.output.weight", {kHiddenSize, kHiddenSize}, options);
  tensors["att.ln_x.weight"] =
      torch::ones({kHiddenSize}, torch::kFloat32).to(dev).to(dtype);
  tensors["att.ln_x.bias"] =
      torch::zeros({kHiddenSize}, torch::kFloat32).to(dev).to(dtype);

  tensors["ffn.x_k"] = param("ffn.x_k");
  tensors["ffn.key.weight"] = seeded_tensor(
      "ffn.key.weight", {kIntermediateSize, kHiddenSize}, options);
  tensors["ffn.value.weight"] = seeded_tensor(
      "ffn.value.weight", {kHiddenSize, kIntermediateSize}, options);

  return StateDict(std::move(tensors));
}

KVCache make_kv_cache(const torch::Device& device) {
  auto conv_cache = torch::zeros(
      {kNumStateSlots, 1, 3 * kHiddenSize},
      torch::TensorOptions().dtype(torch::kFloat16).device(device));
  auto ssm_cache = torch::zeros(
      {kNumStateSlots, kNumHeads, kHeadSize, kHeadSize},
      torch::TensorOptions().dtype(torch::kFloat32).device(device));
  return KVCache(LinearAttentionKVCacheTensors{conv_cache, ssm_cache});
}

ModelInputParams make_input_params(int32_t num_sequences,
                                   const std::vector<int32_t>& q_lens,
                                   const torch::Device& device) {
  ModelInputParams params;
  params.meta.num_sequences = num_sequences;

  std::vector<int32_t> q_cu_seq_lens;
  q_cu_seq_lens.reserve(static_cast<size_t>(q_lens.size()) + 1);
  q_cu_seq_lens.push_back(0);
  for (int32_t len : q_lens) {
    q_cu_seq_lens.push_back(q_cu_seq_lens.back() + len);
  }
  params.attention.host.q_seq_lens = q_cu_seq_lens;

  std::vector<int64_t> state_indices;
  state_indices.reserve(static_cast<size_t>(num_sequences));
  for (int32_t i = 0; i < num_sequences; ++i) {
    state_indices.push_back(i);
  }
  params.embedding.linear_state_indices = torch::tensor(
      state_indices, torch::TensorOptions().dtype(torch::kLong).device(device));
  return params;
}

}  // namespace

class RWKV7DecoderLayerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!torch::cuda::is_available()) {
      GTEST_SKIP() << "CUDA is not available.";
    }
    device_ = torch::Device(torch::kCUDA, 0);
    options_ = torch::TensorOptions().dtype(torch::kFloat16).device(device_);
    parallel_args_ = ParallelArgs(1, 1, nullptr);
    context_ =
        ModelContext(parallel_args_, make_model_args(), QuantArgs(), options_);
    layer_ = RWKV7DecoderLayer(context_, /*layer_id=*/0);
    layer_->load_state_dict(make_layer0_state_dict(options_));
    kv_cache_ = make_kv_cache(device_);
  }

  torch::Device device_{torch::kCPU};
  torch::TensorOptions options_;
  ParallelArgs parallel_args_{1, 1, nullptr};
  ModelContext context_{parallel_args_,
                        make_model_args(),
                        QuantArgs(),
                        torch::TensorOptions()};
  RWKV7DecoderLayer layer_{nullptr};
  KVCache kv_cache_;
};

TEST_F(RWKV7DecoderLayerTest, ForwardProducesFiniteOutput) {
  const int32_t seq_len = 4;
  auto input = seeded_tensor("input", {seq_len, kHiddenSize}, options_);
  auto params = make_input_params(/*num_sequences=*/1, {seq_len}, device_);

  torch::Tensor v_first;
  auto output =
      layer_->forward(input, kv_cache_, params, /*layer_id=*/0, v_first);

  EXPECT_EQ(output.sizes(), torch::IntArrayRef({seq_len, kHiddenSize}));
  EXPECT_TRUE(torch::isfinite(output).all().item<bool>());
  EXPECT_TRUE(v_first.defined());
  EXPECT_EQ(v_first.sizes(), torch::IntArrayRef({seq_len, kHiddenSize}));
}

TEST_F(RWKV7DecoderLayerTest, ForwardUpdatesLinearAttentionState) {
  const int32_t seq_len = 2;
  auto input = seeded_tensor("state_input", {seq_len, kHiddenSize}, options_);
  auto params = make_input_params(/*num_sequences=*/1, {seq_len}, device_);

  auto conv_before = kv_cache_.get_conv_cache().clone();
  auto ssm_before = kv_cache_.get_ssm_cache().clone();

  torch::Tensor v_first;
  auto output =
      layer_->forward(input, kv_cache_, params, /*layer_id=*/0, v_first);
  ASSERT_TRUE(output.defined());

  auto conv_after = kv_cache_.get_conv_cache();
  auto ssm_after = kv_cache_.get_ssm_cache();
  EXPECT_FALSE(torch::allclose(conv_before, conv_after));
  EXPECT_FALSE(torch::allclose(ssm_before, ssm_after));
}

TEST_F(RWKV7DecoderLayerTest, ForwardSupportsMultipleSequences) {
  const std::vector<int32_t> q_lens = {2, 3};
  const int32_t total_tokens = 5;
  auto input =
      seeded_tensor("multi_seq_input", {total_tokens, kHiddenSize}, options_);
  auto params = make_input_params(/*num_sequences=*/2, q_lens, device_);

  torch::Tensor v_first;
  auto output =
      layer_->forward(input, kv_cache_, params, /*layer_id=*/0, v_first);

  EXPECT_EQ(output.sizes(), torch::IntArrayRef({total_tokens, kHiddenSize}));
  EXPECT_TRUE(torch::isfinite(output).all().item<bool>());
}

}  // namespace layer
}  // namespace xllm
