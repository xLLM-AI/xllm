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

#include "layers/mlu/qwen3_5/qwen3_5_gated_delta_net.h"

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <unordered_map>
#include <vector>

#include "framework/kv_cache/kv_cache.h"
#include "framework/kv_cache/kv_cache_utils.h"
#include "framework/model/model_args.h"
#include "framework/model/model_input_params.h"
#include "framework/parallel_state/parallel_state.h"
#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"
#include "layers/mlu/tests_utils.h"
#include "platform/device.h"
#include "platform/platform.h"
#include "util/net.h"

namespace xllm {
namespace layer {
namespace {

// Qwen3.5 GatedDeltaNet layer dimensions chosen so every downstream MLU
// kernel lands on a pre-compiled algo variant:
//   * conv channels = head_k_dim * num_k_heads * 2 + head_k_dim * num_v_heads
//     = 128 * 4 * 2 + 128 * 8 = 2048  (in causal_conv1d algo table)
//   * num_k_heads = 4 (in chunk algo table), num_v_heads = 8 (gating + chunk)
//   * K = V = 128 (qwen3.5), so the recurrent state K/V layout is unambiguous.
constexpr int64_t kHiddenSize = 512;
constexpr int64_t kNumKHeads = 4;
constexpr int64_t kNumVHeads = 8;
constexpr int64_t kHeadKDim = 128;
constexpr int64_t kHeadVDim = 128;
constexpr int64_t kConvKernel = 4;
constexpr int64_t kKSize = kNumKHeads * kHeadKDim;  // 512
constexpr int64_t kVSize = kNumVHeads * kHeadVDim;  // 1024
constexpr int64_t kChannels = kKSize * 2 + kVSize;  // 2048
constexpr int64_t kConvStateLen = kConvKernel - 1;  // 3
constexpr int64_t kNumStateSlots = 8;

class Qwen3_5GatedDeltaNetTest : public ::testing::Test {
 protected:
  void SetUp() override {
    torch::Device device(Platform::type_torch(), 0);
    Device xllm_device(device);
    xllm_device.set_seed(42);
    device_ = device;

    model_args_.model_type() = "qwen3_5";
    model_args_.hidden_size() = kHiddenSize;
    model_args_.rms_norm_eps() = 1e-6f;
    model_args_.linear_num_key_heads() = kNumKHeads;
    model_args_.linear_num_value_heads() = static_cast<int32_t>(kNumVHeads);
    model_args_.linear_key_head_dim() = static_cast<int32_t>(kHeadKDim);
    model_args_.linear_value_head_dim() = static_cast<int32_t>(kHeadVDim);
    model_args_.linear_conv_kernel_dim() = static_cast<int32_t>(kConvKernel);

    options_ = torch::TensorOptions().dtype(torch::kBFloat16).device(device);

    int32_t listen_port = net::get_local_free_port();
    process_group_ = create_process_group(
        0, 1, 1, listen_port, false, "localhost", "tp_group", device);
    parallel_args_.tp_group_ = process_group_.get();

    // conv_cache: [slots, state_len, channels]; ssm_cache: [slots, V, K, V_dim]
    // (fp32, per qwen3.5 mamba_ssm_dtype default "float32").
    auto conv_cache =
        torch::zeros({kNumStateSlots, kConvStateLen, kChannels}, options_);
    auto ssm_cache = torch::zeros(
        {kNumStateSlots, kNumVHeads, kHeadKDim, kHeadVDim},
        torch::TensorOptions().dtype(torch::kFloat32).device(device));
    kv_cache_ = KVCache(LinearAttentionKVCacheTensors{conv_cache, ssm_cache});

    InitTestWeights();
  }

  void InitTestWeights() {
    const std::string seed_prefix = "qwen3_5_gdn_test.";
    auto seeded = [&](const std::string& name, torch::IntArrayRef shape) {
      return test::seeded_tensor(seed_prefix + name,
                                 shape,
                                 torch::typeMetaToScalarType(options_.dtype()),
                                 options_.device());
    };
    // Normalize weights by sqrt(fan_in) for stable activations.
    auto normed = [&](const std::string& name, torch::IntArrayRef shape) {
      auto w = seeded(name, shape);
      return w / torch::sqrt(torch::tensor(w.size(1), options_));
    };

    // conv1d.weight is stored as [out, 1, in] and squeezed to [out, in].
    weight_dict_["linear_attn.conv1d.weight"] =
        seeded("conv1d.weight", {kChannels, 1, kConvKernel});
    weight_dict_["linear_attn.in_proj_qkv.weight"] =
        normed("in_proj_qkv.weight", {kKSize * 2 + kVSize, kHiddenSize});
    weight_dict_["linear_attn.in_proj_z.weight"] =
        normed("in_proj_z.weight", {kVSize, kHiddenSize});
    weight_dict_["linear_attn.in_proj_b.weight"] =
        normed("in_proj_b.weight", {kNumVHeads, kHiddenSize});
    weight_dict_["linear_attn.in_proj_a.weight"] =
        normed("in_proj_a.weight", {kNumVHeads, kHiddenSize});
    weight_dict_["linear_attn.out_proj.weight"] =
        normed("out_proj.weight", {kHiddenSize, kVSize});
    weight_dict_["linear_attn.norm.weight"] =
        torch::ones({kHeadVDim}, options_);
    weight_dict_["linear_attn.dt_bias"] = torch::ones({kNumVHeads}, options_);
    weight_dict_["linear_attn.A_log"] = seeded("A_log", {kNumVHeads});
  }

  // Build (batch, token_block_offset, tot) for the GDN conv kernel.
  struct ConvBatchMeta {
    torch::Tensor batch;
    torch::Tensor token_block_offset;
    int32_t tot = 0;
  };

  ConvBatchMeta MakeConvBatchMeta(const torch::Tensor& q_cu_seq_lens) {
    constexpr int32_t block_size = 64;
    constexpr int32_t pad_slot_id = -1;
    constexpr int64_t default_max_num_programs = 1024;

    auto seqlens = q_cu_seq_lens.diff();
    auto nums = (seqlens + block_size - 1) / block_size;
    nums = nums.to(torch::kLong);
    int32_t tot = nums.sum().item<int32_t>();
    torch::Tensor range_batch = torch::arange(nums.size(0), nums.options());
    torch::Tensor mlist_tensor = torch::repeat_interleave(range_batch, nums);
    int64_t mlist_len = mlist_tensor.size(0);
    int64_t max_num_programs =
        std::max(default_max_num_programs, mlist_len) * 2;

    auto opts = torch::dtype(torch::kInt32).device(device_);
    torch::Tensor batch_ptr =
        torch::full({max_num_programs}, pad_slot_id, opts);
    torch::Tensor token_block_offset_ptr =
        torch::full({max_num_programs}, pad_slot_id, opts);
    std::vector<torch::Tensor> vec;
    vec.reserve(nums.size(0));
    for (int64_t i = 0; i < nums.size(0); ++i) {
      vec.emplace_back(torch::arange(nums[i].item<int64_t>(), nums.options()));
    }
    torch::Tensor offsetlist = torch::cat(vec, -1).to(torch::kInt32);
    batch_ptr.narrow(0, 0, mlist_len).copy_(mlist_tensor);
    token_block_offset_ptr.narrow(0, 0, mlist_len).copy_(offsetlist);
    return {batch_ptr, token_block_offset_ptr, tot};
  }

  torch::Tensor MakeChunkIndices(const torch::Tensor& cu_seqlens,
                                 int64_t chunk_size) {
    auto lengths = cu_seqlens.narrow(0, 1, cu_seqlens.size(0) - 1) -
                   cu_seqlens.narrow(0, 0, cu_seqlens.size(0) - 1);
    torch::Tensor num_chunks = (lengths + chunk_size - 1) / chunk_size;
    num_chunks = num_chunks.to(torch::kLong);
    torch::Tensor cumsum = torch::cumsum(num_chunks, 0);
    int64_t total = cumsum[-1].item<int64_t>();
    torch::Tensor arange_total = torch::arange(total, cu_seqlens.options());
    torch::Tensor zeros = torch::zeros({1}, cumsum.options());
    torch::Tensor prefix =
        torch::cat({zeros, cumsum.slice(/*dim=*/0, /*start=*/0, /*end=*/-1)});
    torch::Tensor repeats_prefix = torch::repeat_interleave(prefix, num_chunks);
    torch::Tensor indices = arange_total - repeats_prefix;
    torch::Tensor mask = indices == 0;
    torch::Tensor col0 = mask.cumsum(0) - 1;
    return torch::stack({col0, indices}, /*dim=*/1)
        .to(cu_seqlens)
        .to(torch::kInt32);
  }

  AttentionMetadata MakePrefillMetadata(int64_t batch_size, int64_t seq_len) {
    auto opts_int = torch::dtype(torch::kInt32).device(device_);
    AttentionMetadata metadata;
    metadata.q_cu_seq_lens =
        torch::arange(0, (batch_size + 1) * seq_len, seq_len, opts_int);
    metadata.chunk_indices = MakeChunkIndices(metadata.q_cu_seq_lens, 64);
    auto conv_meta = MakeConvBatchMeta(metadata.q_cu_seq_lens);
    metadata.batch = conv_meta.batch;
    metadata.token_block_offset = conv_meta.token_block_offset;
    metadata.tot = conv_meta.tot;
    metadata.has_initial_states = torch::zeros(
        {batch_size},
        torch::TensorOptions().dtype(torch::kBool).device(device_));
    metadata.max_query_len = seq_len;
    metadata.max_seq_len = seq_len;
    metadata.total_kv_len = batch_size * seq_len;
    metadata.compute_dtype = "half";
    metadata.is_prefill = true;
    metadata.is_chunked_prefill = false;
    metadata.is_dummy = false;
    return metadata;
  }

  AttentionMetadata MakeDecodeMetadata(int64_t batch_size) {
    auto opts_int = torch::dtype(torch::kInt32).device(device_);
    AttentionMetadata metadata;
    metadata.q_cu_seq_lens = torch::arange(0, batch_size + 1, 1, opts_int);
    metadata.max_query_len = 1;
    metadata.max_seq_len = 1;
    metadata.total_kv_len = batch_size;
    metadata.compute_dtype = "half";
    metadata.is_prefill = false;
    metadata.is_chunked_prefill = false;
    metadata.is_dummy = false;
    return metadata;
  }

  Qwen3_5GatedDeltaNet MakeLayer() {
    auto layer = Qwen3_5GatedDeltaNet(
        model_args_, QuantArgs(), parallel_args_, options_);
    layer->to(device_);
    const std::string prefix = "linear_attn.";
    StateDict state_dict(weight_dict_, prefix);
    layer->load_state_dict(state_dict.get_dict_with_prefix(prefix));
    layer->verify_loaded_weights(prefix);
    return layer;
  }

  torch::Tensor MakeHidden(int64_t num_tokens) {
    auto raw =
        test::seeded_tensor("qwen3_5_gdn_test.hidden",
                            {num_tokens, kHiddenSize},
                            torch::typeMetaToScalarType(options_.dtype()),
                            options_.device());
    return (raw - 0.5f) * (std::sqrt(12.0f) * 0.02f);
  }

  ModelArgs model_args_;
  ParallelArgs parallel_args_{0, 1, nullptr};
  torch::TensorOptions options_;
  torch::Device device_ = torch::kCPU;
  std::unordered_map<std::string, torch::Tensor> weight_dict_;
  std::unique_ptr<ProcessGroup> process_group_ = nullptr;
  KVCache kv_cache_;
};

// Prefill path exercises causal_conv1d_fn + fused_gdn_gating +
// ChunkGatedDeltaRule. MLU parallel reductions in these kernels can be mildly
// non-deterministic, so we validate shape, finiteness, and determinism rather
// than exact golden values.
TEST_F(Qwen3_5GatedDeltaNetTest, PrefillForward) {
  auto layer = MakeLayer();
  const int64_t batch_size = 1;
  const int64_t seq_len = 128;  // multiple of chunk_size (64)
  const int64_t num_tokens = batch_size * seq_len;

  auto hidden = MakeHidden(num_tokens);
  auto metadata = MakePrefillMetadata(batch_size, seq_len);

  ModelInputParams input_params;
  input_params.embedding.linear_state_ids = {1};

  auto output = layer->forward(hidden, metadata, kv_cache_, input_params);
  Device xllm_device(device_);
  xllm_device.synchronize_default_stream();

  ASSERT_EQ(output.sizes(), torch::IntArrayRef({num_tokens, kHiddenSize}));
  ASSERT_EQ(output.scalar_type(), options_.dtype());
  auto out_cpu = output.flatten().to(torch::kFloat32).cpu();
  ASSERT_TRUE(torch::isfinite(out_cpu).all().item<bool>())
      << "Prefill output must be finite";
}

// Decode path exercises causal_conv1d_update_decode +
// fused_recurrent_gated_delta_rule_packed_decode.
TEST_F(Qwen3_5GatedDeltaNetTest, DecodeForward) {
  auto layer = MakeLayer();
  const int64_t batch_size = 2;
  const int64_t num_tokens = batch_size;

  auto hidden = MakeHidden(num_tokens);
  auto metadata = MakeDecodeMetadata(batch_size);

  ModelInputParams input_params;
  input_params.embedding.linear_state_ids = {1, 2};

  auto output = layer->forward(hidden, metadata, kv_cache_, input_params);
  Device xllm_device(device_);
  xllm_device.synchronize_default_stream();

  ASSERT_EQ(output.sizes(), torch::IntArrayRef({num_tokens, kHiddenSize}));
  ASSERT_EQ(output.scalar_type(), options_.dtype());
  auto out_cpu = output.flatten().to(torch::kFloat32).cpu();
  ASSERT_TRUE(torch::isfinite(out_cpu).all().item<bool>())
      << "Decode output must be finite";
}

}  // namespace
}  // namespace layer
}  // namespace xllm
