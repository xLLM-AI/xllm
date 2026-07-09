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

#include "layers/mlu/qwen3_5/qwen3_5_attention.h"

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <algorithm>
#include <unordered_map>
#include <vector>

#include "framework/kv_cache/kv_cache.h"
#include "framework/kv_cache/kv_cache_utils.h"
#include "framework/model/model_args.h"
#include "framework/parallel_state/parallel_state.h"
#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"
#include "layers/common/rotary_embedding_util.h"
#include "layers/mlu/tests_utils.h"
#include "platform/device.h"
#include "platform/platform.h"
#include "util/net.h"

namespace xllm {
namespace layer {
namespace {

constexpr int64_t kHiddenSize = 512;
constexpr int64_t kHeadDim = 128;
constexpr int64_t kNumHeads = 8;
constexpr int64_t kNumKvHeads = 2;
constexpr int64_t kBlockSize = 16;

class Qwen3_5AttentionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    torch::Device device(Platform::type_torch(), 0);
    Device xllm_device(device);
    xllm_device.set_seed(42);
    device_ = device;

    options_ = torch::TensorOptions().dtype(torch::kBFloat16).device(device);

    int32_t listen_port = net::get_local_free_port();
    process_group_ = create_process_group(
        0, 1, 1, listen_port, false, "localhost", "tp_group", device);
    parallel_args_.tp_group_ = process_group_.get();

    InitKvCache();
  }

  // Configure model args for either the gated (attn_output_gate=true) or the
  // plain attention variant.
  ModelArgs MakeModelArgs(bool attn_output_gate) {
    ModelArgs args;
    args.model_type() = "qwen3_5";
    args.hidden_size() = kHiddenSize;
    args.head_dim() = kHeadDim;
    args.n_heads() = kNumHeads;
    args.n_kv_heads() = kNumKvHeads;
    args.max_position_embeddings() = 2048;
    args.rope_theta() = 1000000.0f;
    args.rms_norm_eps() = 1e-6f;
    args.partial_rotary_factor() = 0.25f;
    args.attn_output_gate() = attn_output_gate;
    args.sliding_window() = 0;
    // mrope section: [temporal, height, width] summing to rotary_dim/2 = 16.
    // With height/width = 0, mrope degenerates to standard rotary on temporal
    // positions, which is sufficient for exercising the mrope data path.
    args.rope_scaling_mrope_section() = {16, 0, 0};
    return args;
  }

  void InitKvCache() {
    int64_t block_num = 256;
    auto k_cache = MakeNoise("qwen3_5_attn_test.k_cache",
                             {block_num, kNumKvHeads, kBlockSize, kHeadDim},
                             0.01f);
    auto v_cache = MakeNoise("qwen3_5_attn_test.v_cache",
                             {block_num, kNumKvHeads, kBlockSize, kHeadDim},
                             0.01f);
    kv_cache_ = KVCache(KVCacheTensors{k_cache, v_cache});
  }

  // Build the mrope (cos, sin) tables for 2D positions [3, num_tokens],
  // mirroring Qwen3_5ModelImpl::apply_mrope.
  std::pair<torch::Tensor, torch::Tensor> ApplyMrope(
      const torch::Tensor& positions) {
    CHECK_EQ(positions.dim(), 2);
    auto target_cos_sin = cos_sin_.index({positions});
    auto chunks = target_cos_sin.chunk(/*chunks=*/2, /*dim=*/-1);
    auto cos_pos = chunks[0].contiguous();
    auto sin_pos = chunks[1].contiguous();
    auto long_opts = positions.options().dtype(torch::kLong);
    auto apply = [&](torch::Tensor x) {
      auto freqs_t = x[0].clone();
      int64_t mrop_length = static_cast<int64_t>(freqs_t.size(-1) / 2);
      for (int32_t dim_idx = 1; dim_idx <= 2; ++dim_idx) {
        int64_t offset = dim_idx;
        int64_t section_len = mrope_section_[dim_idx];
        int64_t length = section_len * 3;
        // Production mrope never has zero height/width sections; guard the
        // arange so the degenerate text-only section [t, 0, 0] does not trip
        // MLU's "inconsistent step sign" check on arange(offset, 0, 3).
        if (section_len <= 0) {
          continue;
        }
        auto idx_first_half = torch::arange(offset, length, 3, long_opts);
        auto idx_second_half = torch::arange(
            offset + mrop_length, length + mrop_length, 3, long_opts);
        auto idx_tensor =
            torch::cat({idx_first_half, idx_second_half}, 0).to(x.device());
        auto src = x[dim_idx].index_select(-1, idx_tensor);
        freqs_t.index_copy_(-1, idx_tensor, src);
      }
      return freqs_t;
    };
    cos_pos = apply(cos_pos.reshape({positions.size(0), -1, cos_pos.size(-1)}));
    sin_pos = apply(sin_pos.reshape({positions.size(0), -1, sin_pos.size(-1)}));
    return std::make_pair(cos_pos, sin_pos);
  }

  Qwen3_5Attention MakeLayer(bool attn_output_gate) {
    auto args = MakeModelArgs(attn_output_gate);
    // Precompute the concat cos/sin table used by apply_mrope.
    int64_t rotary_dim =
        static_cast<int64_t>(args.head_dim() * args.partial_rotary_factor());
    mrope_section_ = args.rope_scaling_mrope_section();
    cos_sin_ = xllm::layer::rotary::get_concat_rotary_embedding(
        rotary_dim,
        args.max_position_embeddings(),
        args.rope_theta(),
        options_);

    auto layer = Qwen3_5Attention(
        args, QuantArgs(), parallel_args_, options_, /*layer_id=*/0);
    layer->to(device_);
    const std::string prefix = "self_attn.";
    StateDict state_dict(WeightsFor(attn_output_gate), prefix);
    layer->load_state_dict(state_dict.get_dict_with_prefix(prefix));
    return layer;
  }

  std::unordered_map<std::string, torch::Tensor> WeightsFor(bool gate) {
    const std::string seed_prefix = "qwen3_5_attn_test.";
    auto seeded = [&](const std::string& name, torch::IntArrayRef shape) {
      return test::seeded_tensor(seed_prefix + name,
                                 shape,
                                 torch::typeMetaToScalarType(options_.dtype()),
                                 options_.device());
    };
    auto normed = [&](const std::string& name, torch::IntArrayRef shape) {
      auto w = seeded(name, shape);
      return w / torch::sqrt(torch::tensor(w.size(1), options_));
    };

    int64_t q_size = kNumHeads * kHeadDim;
    int64_t kv_size = kNumKvHeads * kHeadDim;
    int64_t q_out = gate ? q_size * 2 : q_size;
    std::unordered_map<std::string, torch::Tensor> weights;
    weights["self_attn.q_proj.weight"] = normed("q_proj", {q_out, kHiddenSize});
    weights["self_attn.k_proj.weight"] =
        normed("k_proj", {kv_size, kHiddenSize});
    weights["self_attn.v_proj.weight"] =
        normed("v_proj", {kv_size, kHiddenSize});
    weights["self_attn.o_proj.weight"] =
        normed("o_proj", {kHiddenSize, q_size});
    weights["self_attn.q_norm.weight"] = torch::ones({kHeadDim}, options_);
    weights["self_attn.k_norm.weight"] = torch::ones({kHeadDim}, options_);
    return weights;
  }

  torch::Tensor MakeNoise(const std::string& key,
                          torch::IntArrayRef shape,
                          float stddev) {
    auto raw =
        test::seeded_tensor(key,
                            shape,
                            torch::typeMetaToScalarType(options_.dtype()),
                            options_.device());
    return (raw - 0.5f) * (std::sqrt(12.0f) * stddev);
  }

  int64_t GetBlockNum(int64_t seq_len) const {
    return (seq_len + kBlockSize - 1) / kBlockSize + 1;
  }

  torch::Tensor MakeBlockTable(int64_t batch_size, int64_t seq_len) const {
    auto opts_int = options_.dtype(torch::kInt32);
    const int64_t blocks_per_req = GetBlockNum(seq_len);
    std::vector<int32_t> vec;
    vec.reserve(batch_size * blocks_per_req);
    for (int64_t b = 0; b < batch_size; ++b) {
      for (int64_t i = 0; i < blocks_per_req; ++i) {
        vec.push_back(static_cast<int32_t>(b * blocks_per_req + i));
      }
    }
    return torch::tensor(vec, opts_int).reshape({batch_size, blocks_per_req});
  }

  torch::Tensor MakeSlotMap(int64_t batch_size,
                            int64_t token_len,
                            int64_t kv_seq_len) const {
    auto opts_int = options_.dtype(torch::kInt32);
    const int64_t blocks_per_req = GetBlockNum(kv_seq_len);
    const int64_t slots_per_req = blocks_per_req * kBlockSize;
    const int64_t start_pos = kv_seq_len - token_len;
    std::vector<int32_t> vec;
    vec.reserve(batch_size * token_len);
    for (int64_t b = 0; b < batch_size; ++b) {
      for (int64_t i = 0; i < token_len; ++i) {
        vec.push_back(static_cast<int32_t>(b * slots_per_req + start_pos + i));
      }
    }
    return torch::tensor(vec, opts_int);
  }

  AttentionMetadata MakePrefillMetadata(int64_t batch_size, int64_t seq_len) {
    auto opts_int = options_.dtype(torch::kInt32);
    AttentionMetadata metadata;
    metadata.q_cu_seq_lens =
        torch::arange(0, (batch_size + 1) * seq_len, seq_len, opts_int);
    metadata.kv_cu_seq_lens = metadata.q_cu_seq_lens;
    metadata.slot_mapping = MakeSlotMap(batch_size, seq_len, seq_len);
    metadata.kv_seq_lens = torch::full({batch_size}, seq_len, opts_int);
    metadata.block_table = MakeBlockTable(batch_size, seq_len);
    metadata.max_query_len = seq_len;
    metadata.max_seq_len = seq_len;
    metadata.total_kv_len = batch_size * seq_len;
    metadata.compute_dtype = "half";
    metadata.is_prefill = true;
    metadata.is_chunked_prefill = false;
    metadata.is_dummy = false;
    return metadata;
  }

  AttentionMetadata MakeDecodeMetadata(int64_t batch_size, int64_t seq_len) {
    auto opts_int = options_.dtype(torch::kInt32);
    AttentionMetadata metadata;
    metadata.q_cu_seq_lens = torch::arange(0, batch_size + 1, 1, opts_int);
    metadata.kv_cu_seq_lens =
        torch::arange(0, (batch_size + 1) * seq_len, seq_len, opts_int);
    metadata.slot_mapping = MakeSlotMap(batch_size, 1, seq_len);
    metadata.kv_seq_lens = torch::full({batch_size}, seq_len, opts_int);
    metadata.block_table = MakeBlockTable(batch_size, seq_len);
    metadata.max_query_len = 1;
    metadata.max_seq_len = seq_len;
    metadata.total_kv_len = batch_size * seq_len;
    metadata.compute_dtype = "half";
    metadata.is_prefill = false;
    metadata.is_chunked_prefill = false;
    metadata.is_dummy = false;
    return metadata;
  }

  // 2D mrope positions [3, num_tokens]: temporal = arange per batch, h/w = 0.
  torch::Tensor MakePositions(int64_t batch_size,
                              int64_t seq_len,
                              bool prefill) {
    auto opts_int = options_.dtype(torch::kInt32);
    torch::Tensor temporal;
    if (prefill) {
      temporal = torch::arange(0, seq_len, opts_int).repeat({batch_size});
    } else {
      temporal = torch::full({batch_size}, seq_len, opts_int);
    }
    torch::Tensor zero = torch::zeros({temporal.size(0)}, opts_int);
    return torch::stack({temporal, zero, zero}, /*dim=*/0);
  }

  torch::Tensor MakeHidden(int64_t num_tokens) {
    auto raw =
        test::seeded_tensor("qwen3_5_attn_test.hidden",
                            {num_tokens, kHiddenSize},
                            torch::typeMetaToScalarType(options_.dtype()),
                            options_.device());
    return (raw - 0.5f) * (std::sqrt(12.0f) * 0.02f);
  }

  ParallelArgs parallel_args_{0, 1, nullptr};
  torch::TensorOptions options_;
  torch::Device device_ = torch::kCPU;
  std::unique_ptr<ProcessGroup> process_group_ = nullptr;
  KVCache kv_cache_;
  torch::Tensor cos_sin_;
  std::vector<int64_t> mrope_section_;
};

// Prefill with attn_output_gate=true (the qwen3.5 default).
TEST_F(Qwen3_5AttentionTest, PrefillWithGate) {
  auto layer = MakeLayer(/*attn_output_gate=*/true);
  const int64_t batch_size = 1;
  const int64_t seq_len = 128;
  const int64_t num_tokens = batch_size * seq_len;

  auto hidden = MakeHidden(num_tokens);
  auto positions = MakePositions(batch_size, seq_len, /*prefill=*/true);
  auto metadata = MakePrefillMetadata(batch_size, seq_len);
  std::tie(metadata.mrope_cos, metadata.mrope_sin) = ApplyMrope(positions);

  auto output = layer->forward(positions, hidden, metadata, kv_cache_);
  Device xllm_device(device_);
  xllm_device.synchronize_default_stream();

  ASSERT_EQ(output.sizes(), torch::IntArrayRef({num_tokens, kHiddenSize}));
  ASSERT_EQ(output.scalar_type(), options_.dtype());
  ASSERT_TRUE(torch::isfinite(output.flatten().to(torch::kFloat32).cpu())
                  .all()
                  .item<bool>())
      << "Prefill (gate) output must be finite";
}

// Prefill with attn_output_gate=false (plain q/k/v path).
TEST_F(Qwen3_5AttentionTest, PrefillWithoutGate) {
  auto layer = MakeLayer(/*attn_output_gate=*/false);
  const int64_t batch_size = 2;
  const int64_t seq_len = 64;
  const int64_t num_tokens = batch_size * seq_len;

  auto hidden = MakeHidden(num_tokens);
  auto positions = MakePositions(batch_size, seq_len, /*prefill=*/true);
  auto metadata = MakePrefillMetadata(batch_size, seq_len);
  std::tie(metadata.mrope_cos, metadata.mrope_sin) = ApplyMrope(positions);

  auto output = layer->forward(positions, hidden, metadata, kv_cache_);
  Device xllm_device(device_);
  xllm_device.synchronize_default_stream();

  ASSERT_EQ(output.sizes(), torch::IntArrayRef({num_tokens, kHiddenSize}));
  ASSERT_TRUE(torch::isfinite(output.flatten().to(torch::kFloat32).cpu())
                  .all()
                  .item<bool>())
      << "Prefill (no gate) output must be finite";
}

// Decode path (single token per request).
TEST_F(Qwen3_5AttentionTest, DecodeWithGate) {
  auto layer = MakeLayer(/*attn_output_gate=*/true);
  const int64_t batch_size = 4;
  const int64_t seq_len = 64;
  const int64_t num_tokens = batch_size;

  auto hidden = MakeHidden(num_tokens);
  auto positions = MakePositions(batch_size, seq_len, /*prefill=*/false);
  auto metadata = MakeDecodeMetadata(batch_size, seq_len);
  std::tie(metadata.mrope_cos, metadata.mrope_sin) = ApplyMrope(positions);

  auto output = layer->forward(positions, hidden, metadata, kv_cache_);
  Device xllm_device(device_);
  xllm_device.synchronize_default_stream();

  ASSERT_EQ(output.sizes(), torch::IntArrayRef({num_tokens, kHiddenSize}));
  ASSERT_TRUE(torch::isfinite(output.flatten().to(torch::kFloat32).cpu())
                  .all()
                  .item<bool>())
      << "Decode output must be finite";
}

}  // namespace
}  // namespace layer
}  // namespace xllm
