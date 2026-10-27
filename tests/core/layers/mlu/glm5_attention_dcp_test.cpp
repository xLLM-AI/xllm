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
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/framework/config/kv_cache_config.h"
#include "core/framework/model_context/model_context.h"
#include "core/layers/mlu/multi_device_test_utils.h"
#include "core/layers/mlu/tests_utils.h"
#include "framework/parallel_state/process_group.h"
#include "layers/mlu/dcp_decode_context.h"
#include "layers/mlu/deepseek_v2_attention.h"
#include "layers/mlu/dsa_topk_relay.h"
#include "platform/device.h"

namespace xllm::layer {
namespace {

constexpr int32_t kRanks = 4;
constexpr std::chrono::seconds kTimeout{300};
constexpr std::chrono::milliseconds kTerminationGrace{100};
constexpr int32_t kBatch = 8;
constexpr int32_t kBlock = 16;

struct Config {
  ModelArgs args;
  QuantArgs quant;
};

Config make_config() {
  Config result;
  ModelArgs& args = result.args;
  args.model_type() = "glm_moe_dsa";
  args.hidden_size() = 6144;
  args.n_heads() = 64;
  args.q_lora_rank() = 2048;
  args.kv_lora_rank() = 512;
  args.qk_nope_head_dim() = 192;
  args.qk_rope_head_dim() = 64;
  args.v_head_dim() = 256;
  args.index_n_heads() = 32;
  args.index_head_dim() = 128;
  args.index_topk() = 2048;
  args.max_position_embeddings() = 1048576;
  args.rms_norm_eps() = 1e-5f;
  args.rope_theta() = 8000000.0f;
  args.rope_scaling_rope_type() = "default";
  args.indexer_rope_interleave() = true;
  args.enable_mla() = true;
  result.quant.quant_method() = "smoothquant";
  result.quant.bits() = 8;
  result.quant.only_expert_per_group() = true;
  result.quant.activation_dynamic() = true;
  return result;
}

torch::Tensor make_tensor(const std::string& key,
                          torch::IntArrayRef shape,
                          torch::ScalarType dtype,
                          const torch::Device& device) {
  if (dtype == torch::kInt8) {
    return (test::seeded_tensor(key, shape, torch::kInt8, torch::kCPU)
                .remainder(17) -
            8)
        .to(device);
  }
  return ((test::seeded_tensor(key, shape, torch::kFloat32, torch::kCPU) -
           0.5f) *
          0.02f)
      .to(device, dtype);
}

StateDict make_weights(const ModelArgs& args,
                       const torch::TensorOptions& options) {
  std::unordered_map<std::string, torch::Tensor> weights;
  auto add = [&](const std::string& name,
                 torch::IntArrayRef shape,
                 torch::ScalarType dtype) {
    weights.emplace(
        name, make_tensor("glm5/" + name, shape, dtype, options.device()));
  };
  auto norm = [&](const std::string& name, int64_t size) {
    weights.emplace(name, torch::ones({size}, options.dtype(torch::kFloat32)));
  };
  auto quant = [&](const std::string& name, int64_t output, int64_t input) {
    add(name + ".qweight", {output, input}, torch::kInt8);
    weights.emplace(
        name + ".per_channel_scale",
        torch::full({output}, 0.01, options.dtype(torch::kFloat32)));
    weights.emplace(name + ".smooth",
                    torch::ones({input}, options.dtype(torch::kFloat32)));
  };
  add("q_a_proj.weight",
      {args.q_lora_rank(), args.hidden_size()},
      torch::kBFloat16);
  norm("q_a_layernorm.weight", args.q_lora_rank());
  quant("q_b_proj",
        args.n_heads() * (args.qk_nope_head_dim() + args.qk_rope_head_dim()),
        args.q_lora_rank());
  add("kv_a_proj_with_mqa.weight",
      {args.kv_lora_rank() + args.qk_rope_head_dim(), args.hidden_size()},
      torch::kBFloat16);
  norm("kv_a_layernorm.weight", args.kv_lora_rank());
  add("kv_b_proj.weight",
      {args.n_heads() * (args.qk_nope_head_dim() + args.v_head_dim()),
       args.kv_lora_rank()},
      torch::kBFloat16);
  quant("o_proj", args.hidden_size(), args.n_heads() * args.v_head_dim());
  norm("indexer.k_norm.weight", args.index_head_dim());
  weights.emplace(
      "indexer.k_norm.bias",
      torch::zeros({args.index_head_dim()}, options.dtype(torch::kFloat32)));
  add("indexer.weights_proj.weight",
      {args.index_n_heads(), args.hidden_size()},
      torch::kBFloat16);
  add("indexer.wk.weight",
      {args.index_head_dim(), args.hidden_size()},
      torch::kBFloat16);
  add("indexer.wq_b.weight",
      {args.index_n_heads() * args.index_head_dim(), args.q_lora_rank()},
      torch::kBFloat16);
  return StateDict(std::move(weights));
}

void check_close(const torch::Tensor& actual,
                 const torch::Tensor& expected,
                 const std::string& label) {
  CHECK(actual.sizes() == expected.sizes()) << label << " shape mismatch";
  CHECK(actual.scalar_type() == expected.scalar_type())
      << label << " dtype mismatch";
  const torch::Tensor a = actual.to(torch::kFloat32);
  const torch::Tensor b = expected.to(torch::kFloat32);
  CHECK(torch::isfinite(a).all().item<bool>()) << label;
  CHECK(torch::isfinite(b).all().item<bool>()) << label;
  const torch::Tensor error = (a - b).abs();
  // A zero-output implementation must fail the same elementwise contract.
  CHECK(!torch::allclose(torch::zeros_like(b), b, /*rtol=*/1e-2, /*atol=*/1e-2))
      << label << " reference is too small for a meaningful absolute tolerance";
  CHECK(torch::allclose(a, b, /*rtol=*/1e-2, /*atol=*/1e-2))
      << label << " max error=" << error.max().item<float>()
      << " mean error=" << error.mean().item<float>();
}

// The MTP decode view expands each sequence to one row per validation token.
// Rows share the sequence's physical cache, but have distinct causal lengths.
class Batch final {
 public:
  Batch(int32_t width, int32_t history, const Config& config)
      : width_(width),
        history_(history),
        rows_(kBatch * width),
        blocks_((history + width + kBlock * kRanks - 1) / (kBlock * kRanks)) {
    const int64_t capacity = kBatch * blocks_ * kRanks * kBlock;
    hidden = make_tensor("glm5/hidden",
                         {rows_, config.args.hidden_size()},
                         torch::kBFloat16,
                         torch::kCPU) *
             50;
    initial_kv = make_tensor("glm5/kv",
                             {capacity,
                              config.args.kv_lora_rank() +
                                  config.args.qk_rope_head_dim()},
                             torch::kBFloat16,
                             torch::kCPU) *
                 50;
    initial_index = make_tensor("glm5/index",
                                {capacity, config.args.index_head_dim()},
                                torch::kBFloat16,
                                torch::kCPU) *
                    50;
    positions = torch::empty({rows_}, torch::kInt32);
    auto pos = positions.accessor<int32_t, 1>();
    for (int32_t row = 0; row < rows_; ++row) {
      pos[row] = history_ + row % width_;
    }
  }

  AttentionMetadata metadata(int32_t shards,
                             const torch::Device& device) const {
    AttentionMetadata m;
    const int32_t pages = blocks_ * kRanks / shards;
    torch::Tensor table = torch::empty({rows_, pages}, torch::kInt32);
    torch::Tensor slots = torch::empty({rows_}, torch::kInt32);
    torch::Tensor lens = torch::empty({rows_}, torch::kInt32);
    auto t = table.accessor<int32_t, 2>();
    auto s = slots.accessor<int32_t, 1>();
    auto l = lens.accessor<int32_t, 1>();
    for (int32_t row = 0; row < rows_; ++row) {
      const int32_t seq = row / width_;
      l[row] = history_ + row % width_ + 1;
      s[row] = seq * blocks_ * kRanks * kBlock + l[row] - 1;
      for (int32_t page = 0; page < pages; ++page) {
        t[row][page] = seq * pages + page;
      }
    }
    m.q_cu_seq_lens = torch::arange(rows_ + 1, torch::kInt32).to(device);
    m.kv_cu_seq_lens = torch::cat({torch::zeros({1}, torch::kInt32),
                                   lens.cumsum(0).to(torch::kInt32)})
                           .to(device);
    m.kv_seq_lens = lens.to(device);
    m.block_table = table.to(device);
    m.slot_mapping = slots.to(device);
    m.max_query_len = 1;
    m.max_seq_len = history_ + width_;
    m.total_kv_len = lens.sum().item<int32_t>();
    m.compute_dtype = "bfloat16";
    m.is_prefill = false;
    m.is_chunked_prefill = false;
    m.is_dummy = false;
    return m;
  }

  torch::Tensor shard(const torch::Tensor& global,
                      int32_t shards,
                      int32_t rank) const {
    if (shards == 1) {
      return global.view({-1, 1, kBlock, global.size(-1)});
    }
    // Independent physical block selection; do not call production remappers.
    return global.view({-1, kRanks, kBlock, global.size(-1)})
        .select(1, rank)
        .contiguous()
        .view({-1, 1, kBlock, global.size(-1)});
  }

  KVCache cache(int32_t shards,
                int32_t rank,
                const torch::Device& device) const {
    return KVCache(IndexedKVCacheTensors{
        KVCacheTensors{shard(initial_kv, shards, rank).to(device),
                       torch::Tensor()},
        shard(initial_index, shards, rank).to(device)});
  }

  DsaTopkState fixed_topk(const torch::Device& device) const {
    torch::Tensor table = torch::empty({rows_, 2048}, torch::kInt32);
    auto t = table.accessor<int32_t, 2>();
    for (int32_t row = 0; row < rows_; ++row) {
      const int32_t base = row / width_ * blocks_ * kRanks * kBlock;
      // A sparse causal subset, rather than the first contiguous 2048 tokens.
      for (int32_t col = 0; col < 2048; ++col) {
        t[row][col] = base + 2 * col;
      }
    }
    return DsaTopkState(table.to(device),
                        torch::full({rows_}, 2048, torch::kInt32).to(device));
  }

  void check_topk(const DsaTopkState& state) const {
    const torch::Tensor table = state.block_tables().cpu().view({rows_, -1});
    const torch::Tensor lens = state.context_lens().cpu();
    const auto l = lens.accessor<int32_t, 1>();
    for (int32_t row = 0; row < rows_; ++row) {
      const int32_t length = history_ + row % width_ + 1;
      CHECK_EQ(l[row], length);
      const int32_t base = row / width_ * blocks_ * kRanks * kBlock;
      torch::Tensor valid = std::get<0>(table[row].slice(0, 0, l[row]).sort());
      CHECK(torch::equal(valid,
                         torch::arange(base, base + length, torch::kInt32)))
          << "Missing, duplicated, or noncausal top-k at row " << row;
    }
  }

  void check_local_topk(const DsaTopkState& global,
                        const DsaTopkState& local,
                        int32_t rank) const {
    const torch::Tensor table = global.block_tables().cpu().view({rows_, -1});
    const torch::Tensor lens = global.context_lens().cpu();
    const torch::Tensor actual = local.block_tables().cpu().view_as(table);
    const torch::Tensor actual_lens = local.context_lens().cpu();
    const auto input = table.accessor<int32_t, 2>();
    const auto counts = lens.accessor<int32_t, 1>();
    torch::Tensor expected = torch::zeros_like(table);
    torch::Tensor expected_lens = torch::zeros_like(lens);
    auto out = expected.accessor<int32_t, 2>();
    auto lengths = expected_lens.accessor<int32_t, 1>();
    for (int32_t row = 0; row < rows_; ++row) {
      for (int32_t col = 0; col < counts[row]; ++col) {
        const int32_t slot = input[row][col];
        if (slot < 0 || (slot / kBlock) % kRanks != rank) {
          continue;
        }
        out[row][lengths[row]++] =
            slot / (kBlock * kRanks) * kBlock + slot % kBlock;
      }
    }
    CHECK(torch::equal(actual_lens, expected_lens));
    CHECK(torch::equal(actual, expected))
        << "Rank-local top-k differs from CPU ownership reference";
  }

  void check_cache(const torch::Tensor& actual,
                   const torch::Tensor& reference,
                   int32_t rank) const {
    check_close(actual.cpu(),
                shard(reference.cpu().view_as(initial_kv), kRanks, rank),
                "DCP KV writes versus full KV");
    torch::Tensor unchanged = torch::ones({initial_kv.size(0)}, torch::kBool);
    auto mask = unchanged.accessor<bool, 1>();
    for (int32_t seq = 0; seq < kBatch; ++seq) {
      const int32_t start = seq * blocks_ * kRanks * kBlock + history_;
      for (int32_t j = 0; j < width_; ++j) {
        mask[start + j] = false;
      }
    }
    const torch::Tensor changed =
        (reference.cpu().view_as(initial_kv) != initial_kv).any(-1);
    CHECK(changed.index({~unchanged}).all().item<bool>())
        << "Every current token must update its KV slot";
    CHECK(torch::equal(reference.cpu().view_as(initial_kv).index({unchanged}),
                       initial_kv.index({unchanged})))
        << "Unwritten KV changed";
    const torch::Tensor local_mask =
        shard(unchanged.unsqueeze(1), kRanks, rank).flatten();
    CHECK(torch::equal(
        actual.cpu().view({-1, initial_kv.size(1)}).index({local_mask}),
        shard(initial_kv, kRanks, rank)
            .view({-1, initial_kv.size(1)})
            .index({local_mask})))
        << "Unowned or unwritten KV changed";
  }

  torch::Tensor hidden;
  torch::Tensor positions;
  torch::Tensor initial_kv;
  torch::Tensor initial_index;

 private:
  int32_t width_;
  int32_t history_;
  int32_t rows_;
  int32_t blocks_;
};

torch::Tensor forward(DeepseekV2Attention& attention,
                      const Batch& batch,
                      AttentionMetadata& metadata,
                      KVCache& cache,
                      const torch::Device& device,
                      DsaTopkTransfer& transfer,
                      ProcessGroup& group) {
  auto result = attention->forward(batch.positions.to(device),
                                   batch.hidden.to(device),
                                   metadata,
                                   cache,
                                   nullptr,
                                   &transfer);
  CHECK(result.layout == DeepseekV2AttentionImpl::PostAttnLayout::kTpShard);
  // Standalone attention returns the TP partial output; its decoder consumer
  // normally performs this reduction. Keep it outside the attention call.
  group.allreduce(result.output);
  return std::move(result.output);
}

void check_projection_shards(const DeepseekV2Attention& actual,
                             const StateDict& full_weights,
                             int32_t rank) {
  const auto local_params = actual->named_parameters();
  const std::array<std::pair<std::string, int64_t>, 7> partitions = {{
      {"q_b_proj.qweight", 0},
      {"q_b_proj.per_channel_scale", 0},
      {"q_b_proj.smooth", -1},
      {"kv_b_proj.weight", 0},
      {"o_proj.qweight", 1},
      {"o_proj.per_channel_scale", -1},
      {"o_proj.smooth", 0},
  }};
  for (const auto& [name, dim] : partitions) {
    const torch::Tensor& local = local_params[name];
    const torch::Tensor full = full_weights.get_tensor(name);
    CHECK(local.scalar_type() == full.scalar_type()) << name;
    if (dim < 0) {
      CHECK(torch::equal(local, full)) << name;
      continue;
    }
    CHECK_EQ(full.size(dim) % kRanks, 0) << name;
    const int64_t shard_size = full.size(dim) / kRanks;
    CHECK_EQ(local.size(dim), shard_size) << name;
    CHECK(torch::equal(local, full.narrow(dim, rank * shard_size, shard_size)))
        << name << " must contain this TP rank's complementary head slice";
  }
  CHECK(local_params["q_b_proj.qweight"].scalar_type() == torch::kInt8);
  CHECK(local_params["kv_b_proj.weight"].scalar_type() == torch::kBFloat16);
  CHECK(local_params["o_proj.qweight"].scalar_type() == torch::kInt8);
  CHECK_EQ(local_params["q_b_proj.qweight"].size(0), 4096);
  CHECK_EQ(local_params["kv_b_proj.weight"].size(0), 7168);
  CHECK_EQ(local_params["o_proj.qweight"].size(1), 4096);
}

void run_attention(int32_t rank,
                   int32_t port,
                   int32_t width,
                   bool shared,
                   bool sparse) {
  const Config config = make_config();
  torch::set_num_threads(1);
  torch::NoGradGuard no_grad;
  Device dev(rank);
  dev.set_device();
  const torch::Device device = dev.unwrap();
  const auto options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device);
  KVCacheConfig::get_instance().block_size(kBlock);
  auto group = create_process_group(rank,
                                    kRanks,
                                    kRanks,
                                    port,
                                    false,
                                    "127.0.0.1",
                                    "glm5_attention_dcp_test",
                                    device);
  CHECK(group);
  torch::Tensor ready = torch::ones({1}, options);
  group->allreduce(ready);
  CHECK_EQ(ready.item<float>(), kRanks);
  ParallelArgs base(rank, kRanks, group.get());
  base.tp_group_ = group.get();
  base.cp_group_ = group.get();
  base.cp_size() = 1;
  base.dp_size() = 1;
  base.kv_split_size() = 1;
  ParallelArgs sharded = base;
  sharded.kv_split_size() = kRanks;
  sharded.dcp_group_ = group.get();
  OptimizationConfig optimization;
  optimization.enable_fused_mla_kernel = true;
  optimization.enable_fused_indexer_qk = true;
  StateDict weights = make_weights(config.args, options);
  DeepseekV2Attention reference(
      config.args, config.quant, base, options, optimization, !sparse);
  DeepseekV2Attention actual(
      config.args, config.quant, sharded, options, optimization, !sparse);
  reference->load_state_dict(weights);
  actual->load_state_dict(weights);
  reference->verify_loaded_weights();
  actual->verify_loaded_weights();
  check_projection_shards(actual, weights, rank);
  Batch batch(width, sparse ? 4096 : 1152, config);
  AttentionMetadata ref_meta = batch.metadata(1, device);
  AttentionMetadata dcp_meta = batch.metadata(kRanks, device);
  KVCache ref_cache = batch.cache(1, rank, device);
  KVCache dcp_cache = batch.cache(kRanks, rank, device);
  DsaTopkTransfer ref_transfer =
      sparse ? DsaTopkTransfer::reuse_and_capture(batch.fixed_topk(device))
             : DsaTopkTransfer::capture_output();
  DsaTopkTransfer dcp_transfer =
      sparse ? DsaTopkTransfer::reuse_and_capture(batch.fixed_topk(device))
             : DsaTopkTransfer::capture_output();
  torch::Tensor expected = forward(
      reference, batch, ref_meta, ref_cache, device, ref_transfer, *group);
  torch::Tensor output =
      forward(actual, batch, dcp_meta, dcp_cache, device, dcp_transfer, *group);
  dev.synchronize_default_stream();
  check_close(output, expected, "Full attention");
  batch.check_cache(dcp_cache.get_k_cache(), ref_cache.get_k_cache(), rank);
  CHECK(ref_transfer.output());
  CHECK(dcp_transfer.output());
  CHECK(dcp_transfer.localized_output());
  batch.check_local_topk(
      *dcp_transfer.output(), *dcp_transfer.localized_output(), rank);
  if (!sparse) {
    batch.check_topk(*ref_transfer.output());
    batch.check_topk(*dcp_transfer.output());
  }
  if (shared) {
    DeepseekV2Attention ref_shared(
        config.args, config.quant, base, options, optimization, false);
    DeepseekV2Attention dcp_shared(
        config.args, config.quant, sharded, options, optimization, false);
    ref_shared->load_state_dict(weights);
    dcp_shared->load_state_dict(weights);
    DsaTopkTransfer ref_reuse = DsaTopkTransfer::reuse(*ref_transfer.output());
    DsaTopkTransfer dcp_reuse = DsaTopkTransfer::reuse(
        *dcp_transfer.output(), *dcp_transfer.localized_output());
    const torch::Tensor global_snapshot =
        dcp_reuse.input()->block_tables().clone();
    const torch::Tensor local_snapshot =
        dcp_reuse.localized_input()->block_tables().clone();
    const torch::Tensor global_lens = dcp_reuse.input()->context_lens().clone();
    const torch::Tensor local_lens =
        dcp_reuse.localized_input()->context_lens().clone();
    ref_cache = batch.cache(1, rank, device);
    dcp_cache = batch.cache(kRanks, rank, device);
    expected = forward(
        ref_shared, batch, ref_meta, ref_cache, device, ref_reuse, *group);
    output = forward(
        dcp_shared, batch, dcp_meta, dcp_cache, device, dcp_reuse, *group);
    dev.synchronize_default_stream();
    check_close(output, expected, "Shared attention");
    batch.check_cache(dcp_cache.get_k_cache(), ref_cache.get_k_cache(), rank);
    CHECK(torch::equal(global_snapshot, dcp_reuse.input()->block_tables()));
    CHECK(torch::equal(local_snapshot,
                       dcp_reuse.localized_input()->block_tables()));
    CHECK(torch::equal(global_lens, dcp_reuse.input()->context_lens()));
    CHECK(
        torch::equal(local_lens, dcp_reuse.localized_input()->context_lens()));
  }
  group->allreduce(ready);
  dev.synchronize_default_stream();
}

void check_merge(int32_t rank, int32_t port) {
  torch::NoGradGuard no_grad;
  Device dev(rank);
  dev.set_device();
  const torch::Device device = dev.unwrap();
  auto group = create_process_group(rank,
                                    kRanks,
                                    kRanks,
                                    port,
                                    false,
                                    "127.0.0.1",
                                    "glm5_dcp_merge_test",
                                    device);
  CHECK(group);
  const auto options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device);
  torch::Tensor all_outputs =
      make_tensor(
          "glm5/merge", {4, 32, 64, 512}, torch::kBFloat16, torch::kCPU) *
      50;
  torch::Tensor lse = torch::arange(4 * 32 * 64, torch::kFloat32)
                          .remainder(7)
                          .view({4, 32, 64}) *
                      0.25f;
  lse[0][0].fill_(-INFINITY);
  lse.select(1, 1).fill_(-INFINITY);
  torch::Tensor slots = torch::arange(32, torch::kInt32);
  slots[2] = -1;
  torch::Tensor factors = torch::softmax(lse, 0);
  factors = torch::where(
      torch::isfinite(factors), factors, torch::zeros_like(factors));
  torch::Tensor expected =
      (all_outputs.to(torch::kFloat32) * factors.unsqueeze(-1)).sum(0);
  expected[2].zero_();
  DcpDecodeContext context(KVShardLayout(kBlock, kRanks, rank), group.get());
  const torch::Tensor output =
      context.merge(all_outputs[rank].unsqueeze(1).to(device),
                    lse[rank].unsqueeze(-1).to(device),
                    slots.to(device),
                    /*head_sharded=*/true);
  dev.synchronize_default_stream();
  check_close(output.cpu(),
              expected.slice(1, rank * 16, (rank + 1) * 16)
                  .unsqueeze(1)
                  .to(torch::kBFloat16),
              "DCP merge FP32 reference");
  CHECK_EQ(output[1].abs().max().item<float>(), 0.0f);
  CHECK_EQ(output[2].abs().max().item<float>(), 0.0f);
  torch::Tensor ready = torch::ones({1}, options);
  group->allreduce(ready);
  dev.synchronize_default_stream();
}

TEST(Glm5AttentionDcpTest, FullVerify32Rows) {
  xllm::test::run_multi_device(
      kRanks,
      [](int32_t rank, int32_t port) {
        run_attention(
            rank, port, /*width=*/4, /*shared=*/false, /*sparse=*/false);
      },
      kTimeout,
      kTerminationGrace);
}
TEST(Glm5AttentionDcpTest, SharedVerify32Rows) {
  xllm::test::run_multi_device(
      kRanks,
      [](int32_t rank, int32_t port) {
        run_attention(
            rank, port, /*width=*/4, /*shared=*/true, /*sparse=*/false);
      },
      kTimeout,
      kTerminationGrace);
}
TEST(Glm5AttentionDcpTest, FullDecode8Rows) {
  xllm::test::run_multi_device(
      kRanks,
      [](int32_t rank, int32_t port) {
        run_attention(
            rank, port, /*width=*/1, /*shared=*/false, /*sparse=*/false);
      },
      kTimeout,
      kTerminationGrace);
}
TEST(Glm5AttentionDcpTest, FullExtend16Rows) {
  xllm::test::run_multi_device(
      kRanks,
      [](int32_t rank, int32_t port) {
        run_attention(
            rank, port, /*width=*/2, /*shared=*/false, /*sparse=*/false);
      },
      kTimeout,
      kTerminationGrace);
}
TEST(Glm5AttentionDcpTest, SparseContext4096) {
  xllm::test::run_multi_device(
      kRanks,
      [](int32_t rank, int32_t port) {
        run_attention(
            rank, port, /*width=*/4, /*shared=*/false, /*sparse=*/true);
      },
      kTimeout,
      kTerminationGrace);
}
TEST(Glm5AttentionDcpTest, MergeMatchesFp32) {
  xllm::test::run_multi_device(
      kRanks, check_merge, kTimeout, kTerminationGrace);
}

}  // namespace
}  // namespace xllm::layer
