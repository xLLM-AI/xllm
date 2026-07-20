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

#include "layers/mlu/deepseek_v4/deepseek_v4_decoder_layer.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_args.h"
#include "framework/model_context.h"
#include "framework/state_dict/state_dict.h"
#include "layers/common/attention_metadata.h"
#include "layers/common/dsa_metadata.h"
#include "layers/mlu/tests_utils.h"
#include "platform/device.h"
#include "platform/platform.h"

namespace xllm {
namespace layer {
namespace {

constexpr int64_t kBlockSize = 16;

torch::Tensor seeded(const std::string& key,
                     torch::IntArrayRef shape,
                     torch::ScalarType dtype,
                     const torch::Device& device) {
  return (test::seeded_tensor(key, shape, dtype, device) - 0.5) * 0.1;
}

std::vector<int64_t> offsets_from_lens(const std::vector<int64_t>& lens) {
  std::vector<int64_t> offsets{0};
  offsets.reserve(lens.size() + 1);
  for (int64_t len : lens) {
    offsets.emplace_back(offsets.back() + len);
  }
  return offsets;
}

torch::Tensor make_paged_table(int64_t batch_size,
                               int64_t block_count,
                               const torch::Device& device) {
  std::vector<int32_t> values;
  values.reserve(static_cast<size_t>(batch_size * block_count));
  for (int64_t seq_idx = 0; seq_idx < batch_size; ++seq_idx) {
    for (int64_t block_idx = 0; block_idx < block_count; ++block_idx) {
      values.emplace_back(
          static_cast<int32_t>(seq_idx * block_count + block_idx));
    }
  }
  return torch::tensor(
             values, torch::TensorOptions().dtype(torch::kInt32).device(device))
      .view({batch_size, block_count});
}

torch::Tensor make_slots(const std::vector<int64_t>& start_pos,
                         const std::vector<int64_t>& q_lens,
                         int64_t block_count,
                         const torch::Device& device) {
  std::vector<int32_t> slots;
  int64_t total_q_len = 0;
  for (int64_t len : q_lens) {
    total_q_len += len;
  }
  slots.reserve(static_cast<size_t>(total_q_len));
  for (int64_t seq_idx = 0; seq_idx < static_cast<int64_t>(q_lens.size());
       ++seq_idx) {
    for (int64_t token_idx = 0; token_idx < q_lens[seq_idx]; ++token_idx) {
      const int64_t pos = start_pos[seq_idx] + token_idx;
      const int64_t block_id = seq_idx * block_count + pos / kBlockSize;
      const int64_t block_offset = pos % kBlockSize;
      slots.emplace_back(
          static_cast<int32_t>(block_id * kBlockSize + block_offset));
    }
  }
  return torch::tensor(
      slots, torch::TensorOptions().dtype(torch::kInt32).device(device));
}

}  // namespace

class DeepseekV4DecoderLayerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    torch::Device torch_device(Platform::type_torch(), 0);
    Device device(torch_device);
    device.set_seed();
    device_ = torch_device;
    options_ = torch::TensorOptions()
                   .dtype(torch::kBFloat16)
                   .device(device_)
                   .requires_grad(false);
    parallel_args_ = test::create_default_parallel_args(process_group_);
  }

  ModelArgs make_args() const {
    ModelArgs args;
    args.model_type() = "deepseek_v4";
    args.hidden_size() = 256;
    args.head_dim() = 16;
    args.n_heads() = 4;
    args.n_kv_heads() = 1;
    args.q_lora_rank() = 16;
    args.rope_head_dim() = 8;
    args.o_groups() = 2;
    args.o_lora_rank() = 8;
    args.compress_ratios() = {1, 1};
    args.window_size() = 4;
    args.index_n_heads() = 4;
    args.index_head_dim() = 16;
    args.index_topk() = 2;
    args.rms_norm_eps() = 1e-6f;
    args.hc_mult() = 4;
    args.hc_sinkhorn_iters() = 20;
    args.hc_eps() = 1e-6f;
    args.n_hash_layers() = 1;
    args.n_routed_experts() = 4;
    args.num_experts_per_tok() = 2;
    args.n_group() = 2;
    args.topk_group() = 2;
    args.routed_scaling_factor() = 1.0f;
    args.norm_topk_prob() = true;
    args.moe_intermediate_size() = 128;
    args.n_shared_experts() = 0;
    args.hidden_act() = "silu";
    args.scoring_func() = "sqrtsoftplus";
    args.topk_method() = "noaux_tc";
    args.swiglu_limit() = 10.0f;
    args.vocab_size() = 128;
    return args;
  }

  StateDict make_state_dict(const ModelArgs& args) const {
    std::unordered_map<std::string, torch::Tensor> tensors;
    const int64_t hidden_size = args.hidden_size();
    const int64_t q_lora_rank = args.q_lora_rank();
    const int64_t head_dim = args.head_dim();
    const int64_t n_heads = args.n_heads();
    const int64_t o_groups = args.o_groups();
    const int64_t o_lora_rank = args.o_lora_rank();
    const int64_t hc_mult = args.hc_mult();
    const int64_t mix_hc = (2 + hc_mult) * hc_mult;
    const int64_t hc_dim = hc_mult * hidden_size;
    const int64_t num_experts = args.n_routed_experts();
    const int64_t topk = args.num_experts_per_tok();
    const int64_t inter_size = args.moe_intermediate_size();

    tensors["self_attn.wq_a.weight"] = seeded(
        "dsv4.decoder.wq_a", {q_lora_rank, hidden_size}, dtype_, device_);
    tensors["self_attn.q_norm.weight"] =
        seeded("dsv4.decoder.q_norm", {q_lora_rank}, torch::kFloat32, device_) +
        1.0f;
    tensors["self_attn.wq_b.weight"] = seeded("dsv4.decoder.wq_b",
                                              {n_heads * head_dim, q_lora_rank},
                                              dtype_,
                                              device_);
    tensors["self_attn.wkv.weight"] =
        seeded("dsv4.decoder.wkv", {head_dim, hidden_size}, dtype_, device_);
    tensors["self_attn.kv_norm.weight"] =
        seeded("dsv4.decoder.kv_norm", {head_dim}, torch::kFloat32, device_) +
        1.0f;
    tensors["self_attn.wo_a.weight"] =
        seeded("dsv4.decoder.wo_a",
               {o_groups * o_lora_rank, n_heads * head_dim / o_groups},
               dtype_,
               device_);
    tensors["self_attn.wo_b.weight"] =
        seeded("dsv4.decoder.wo_b",
               {hidden_size, o_groups * o_lora_rank},
               dtype_,
               device_);
    tensors["input_layernorm.weight"] =
        torch::ones({hidden_size}, options_.dtype(torch::kFloat32));
    tensors["post_attention_layernorm.weight"] =
        torch::ones({hidden_size}, options_.dtype(torch::kFloat32));

    tensors["hc_attn_fn"] = seeded(
        "dsv4.decoder.hc_attn_fn", {mix_hc, hc_dim}, torch::kFloat32, device_);
    tensors["hc_attn_base"] =
        seeded("dsv4.decoder.hc_attn_base", {mix_hc}, torch::kFloat32, device_);
    tensors["hc_attn_scale"] =
        torch::tensor({0.1f, 0.2f, 0.15f}, options_.dtype(torch::kFloat32));
    tensors["hc_ffn_fn"] = seeded(
        "dsv4.decoder.hc_ffn_fn", {mix_hc, hc_dim}, torch::kFloat32, device_);
    tensors["hc_ffn_base"] =
        seeded("dsv4.decoder.hc_ffn_base", {mix_hc}, torch::kFloat32, device_);
    tensors["hc_ffn_scale"] =
        torch::tensor({0.1f, 0.2f, 0.15f}, options_.dtype(torch::kFloat32));

    tensors["mlp.gate.weight"] = seeded(
        "dsv4.decoder.gate", {num_experts, hidden_size}, dtype_, device_);
    tensors["mlp.gate.bias"] = seeded(
        "dsv4.decoder.gate.bias", {num_experts}, torch::kFloat32, device_);
    tensors["mlp.gate.e_score_correction_bias"] =
        seeded("dsv4.decoder.gate.e_score_correction_bias",
               {num_experts},
               torch::kFloat32,
               device_);
    std::vector<int32_t> tid2eid;
    tid2eid.reserve(static_cast<size_t>(args.vocab_size() * topk));
    for (int64_t token_id = 0; token_id < args.vocab_size(); ++token_id) {
      for (int64_t topk_idx = 0; topk_idx < topk; ++topk_idx) {
        tid2eid.emplace_back(
            static_cast<int32_t>((token_id + topk_idx) % num_experts));
      }
    }
    tensors["mlp.gate.tid2eid"] =
        torch::tensor(
            tid2eid,
            torch::TensorOptions().dtype(torch::kInt32).device(device_))
            .view({args.vocab_size(), topk});

    for (int64_t expert_id = 0; expert_id < num_experts; ++expert_id) {
      const std::string prefix =
          "mlp.experts." + std::to_string(expert_id) + ".";
      tensors[prefix + "gate_proj.weight"] = seeded(
          prefix + "gate_proj", {inter_size, hidden_size}, dtype_, device_);
      tensors[prefix + "up_proj.weight"] = seeded(
          prefix + "up_proj", {inter_size, hidden_size}, dtype_, device_);
      tensors[prefix + "down_proj.weight"] = seeded(
          prefix + "down_proj", {hidden_size, inter_size}, dtype_, device_);
    }
    return StateDict(tensors);
  }

  StateDict make_ffn_alias_state_dict(const ModelArgs& args) const {
    std::unordered_map<std::string, torch::Tensor> tensors;
    StateDict state_dict = make_state_dict(args);
    for (const auto& item : state_dict) {
      const std::string& key = item.first;
      if (key.rfind("mlp.", 0) == 0) {
        tensors.emplace("ffn." + key.substr(4), item.second);
      } else {
        tensors.emplace(key, item.second);
      }
    }
    return StateDict(tensors);
  }

  KVCache make_kv_cache(const ModelArgs& args, int64_t batch_size) const {
    DeepSeekV4KVCacheTensors tensors;
    tensors.swa_cache =
        torch::zeros({batch_size, 1, kBlockSize, args.head_dim()}, options_);
    return KVCache(tensors);
  }

  AttentionMetadata make_metadata(const ModelArgs& args,
                                  int32_t layer_id,
                                  const std::vector<int64_t>& start_pos,
                                  const std::vector<int64_t>& q_lens) {
    const int64_t batch_size = static_cast<int64_t>(q_lens.size());
    const int64_t total_tokens =
        offsets_from_lens(q_lens)[static_cast<size_t>(batch_size)];
    const int64_t layer_count = layer_id + 1;
    std::shared_ptr<DSAMetadata> dsa = std::make_shared<DSAMetadata>();
    dsa->layer_id = layer_id;
    dsa->start_pos_vec = start_pos;
    dsa->query_start_offsets = offsets_from_lens(q_lens);

    std::vector<int32_t> q_cu{0};
    q_cu.reserve(static_cast<size_t>(batch_size + 1));
    std::vector<int32_t> kv_cu{0};
    kv_cu.reserve(static_cast<size_t>(batch_size + 1));
    std::vector<int32_t> q_seq;
    q_seq.reserve(static_cast<size_t>(batch_size));
    std::vector<int32_t> kv_seq;
    kv_seq.reserve(static_cast<size_t>(batch_size));
    std::vector<int32_t> input_positions;
    input_positions.reserve(static_cast<size_t>(total_tokens));
    for (int64_t seq_idx = 0; seq_idx < batch_size; ++seq_idx) {
      q_cu.emplace_back(q_cu.back() + static_cast<int32_t>(q_lens[seq_idx]));
      const int64_t kv_len = start_pos[seq_idx] + q_lens[seq_idx];
      kv_cu.emplace_back(kv_cu.back() + static_cast<int32_t>(kv_len));
      q_seq.emplace_back(static_cast<int32_t>(q_lens[seq_idx]));
      kv_seq.emplace_back(static_cast<int32_t>(kv_len));
      for (int64_t token_idx = 0; token_idx < q_lens[seq_idx]; ++token_idx) {
        input_positions.emplace_back(
            static_cast<int32_t>(start_pos[seq_idx] + token_idx));
      }
    }
    torch::TensorOptions int_options =
        torch::TensorOptions().dtype(torch::kInt32).device(device_);
    dsa->q_cu_seq_lens = torch::tensor(q_cu, int_options);
    dsa->kv_cu_seq_lens = torch::tensor(kv_cu, int_options);
    dsa->q_seq_lens = torch::tensor(q_seq, int_options);
    dsa->kv_seq_lens = torch::tensor(kv_seq, int_options);
    dsa->seq_lens_q = dsa->q_seq_lens;
    dsa->seq_lens = dsa->kv_seq_lens;
    dsa->input_positions = torch::tensor(input_positions, int_options);
    dsa->actual_seq_lengths_query = dsa->q_cu_seq_lens;
    dsa->actual_seq_lengths_kv = dsa->kv_seq_lens;
    dsa->max_seqlen_q = dsa->q_seq_lens.max();
    dsa->max_seqlen_kv = dsa->kv_seq_lens.max();
    dsa->swa_history_lens = torch::zeros({batch_size}, int_options);
    dsa->swa_context_lens = torch::arange(1, total_tokens + 1, int_options);
    dsa->swa_max_history_len = 0;
    dsa->swa_max_context_len = total_tokens;
    const int64_t block_count = 1;
    torch::Tensor table = make_paged_table(batch_size, block_count, device_);
    torch::Tensor slots = make_slots(start_pos, q_lens, block_count, device_);
    dsa->block_tables.resize(static_cast<size_t>(layer_count));
    dsa->slot_mappings.resize(static_cast<size_t>(layer_count));
    caches_info_.assign(static_cast<size_t>(layer_count),
                        {{0,
                          DSACacheType::SLIDING_WINDOW,
                          1,
                          static_cast<int32_t>(kBlockSize)}});
    for (int64_t idx = 0; idx < layer_count; ++idx) {
      dsa->block_tables[static_cast<size_t>(idx)] = {table};
      dsa->slot_mappings[static_cast<size_t>(idx)] = {slots};
    }
    dsa->caches_info = &caches_info_;

    AttentionMetadata metadata;
    metadata.is_prefill = true;
    metadata.is_chunked_prefill = false;
    metadata.is_dummy = false;
    metadata.q_cu_seq_lens = dsa->q_cu_seq_lens;
    metadata.kv_cu_seq_lens = dsa->kv_cu_seq_lens;
    metadata.q_seq_lens = dsa->q_seq_lens;
    metadata.kv_seq_lens = dsa->kv_seq_lens;
    metadata.max_query_len = dsa->q_seq_lens.max().item<int64_t>();
    metadata.max_seq_len = dsa->kv_seq_lens.max().item<int64_t>();
    metadata.total_kv_len = dsa->kv_seq_lens.sum().item<int64_t>();
    metadata.compute_dtype = "float";
    metadata.dsa_metadata = dsa;
    return metadata;
  }

  void use_dp_ep_parallel_args() {
    process_group_ = std::make_unique<test::MockProcessGroup>(
        device_, /*rank=*/0, /*world_size=*/2);
    dp_process_group_ = std::make_unique<test::MockProcessGroup>(
        device_, /*rank=*/0, /*world_size=*/2);
    tp_process_group_ = std::make_unique<test::MockProcessGroup>(
        device_, /*rank=*/0, /*world_size=*/1);
    ep_process_group_ = std::make_unique<test::MockProcessGroup>(
        device_, /*rank=*/0, /*world_size=*/2);
    single_rank_process_group_ = std::make_unique<test::MockProcessGroup>(
        device_, /*rank=*/0, /*world_size=*/1);

    parallel_args_ = ParallelArgs(
        /*rank=*/0, /*world_size=*/2, /*dp_size=*/2, process_group_.get());
    parallel_args_.process_group_ = process_group_.get();
    parallel_args_.dp_local_process_group_ = dp_process_group_.get();
    parallel_args_.tp_group_ = tp_process_group_.get();
    parallel_args_.single_rank_group_ = single_rank_process_group_.get();
    parallel_args_.cp_group_ = tp_process_group_.get();
    parallel_args_.ep_size_ = 2;
    parallel_args_.moe_ep_group_ = ep_process_group_.get();
    parallel_args_.moe_tp_group_ = tp_process_group_.get();
  }

  void run_case(int32_t layer_id, bool pass_input_ids, bool use_ffn_alias) {
    ModelArgs args = make_args();
    QuantArgs quant_args;
    ModelContext context(parallel_args_, args, quant_args, options_);
    torch::Tensor cos_sin =
        torch::cat({torch::ones({16, args.rope_head_dim()}, options_),
                    torch::zeros({16, args.rope_head_dim()}, options_)},
                   /*dim=*/-1);
    DeepseekV4DecoderLayer layer(context, layer_id);
    DSACacheMapping mapping;
    mapping.ori_cache_idx = 0;
    layer->set_cache_mapping(mapping);
    layer->load_state_dict(use_ffn_alias ? make_ffn_alias_state_dict(args)
                                         : make_state_dict(args));
    layer->verify_loaded_weights();

    const int64_t seq_len = 4;
    torch::Tensor hidden =
        seeded("dsv4.decoder.hidden." + std::to_string(layer_id),
               {seq_len, args.hc_mult(), args.hidden_size()},
               dtype_,
               device_);
    AttentionMetadata metadata =
        make_metadata(args, layer_id, /*start_pos=*/{0}, /*q_lens=*/{seq_len});
    if (cos_sin.defined() && metadata.dsa_metadata) {
      std::vector<torch::Tensor> chunks =
          cos_sin.chunk(/*chunks=*/2, /*dim=*/-1);
      metadata.dsa_metadata->cos_table = chunks[0].contiguous();
      metadata.dsa_metadata->sin_table = chunks[1].contiguous();
      metadata.dsa_metadata->inverse_sin_table =
          -metadata.dsa_metadata->sin_table;
    }
    KVCache kv_cache = make_kv_cache(args, /*batch_size=*/1);
    ModelInputParams input_params;
    std::optional<torch::Tensor> residual = std::nullopt;
    torch::Tensor positions = torch::arange(
        seq_len, torch::TensorOptions().dtype(torch::kInt64).device(device_));
    std::optional<torch::Tensor> input_ids = std::nullopt;
    if (pass_input_ids) {
      input_ids = torch::arange(
          seq_len, torch::TensorOptions().dtype(torch::kInt32).device(device_));
    }

    torch::Tensor output = layer->forward(hidden,
                                          residual,
                                          positions,
                                          metadata,
                                          kv_cache,
                                          input_params,
                                          input_ids);
    Device device(device_);
    device.synchronize_default_stream();

    ASSERT_TRUE(output.defined());
    ASSERT_EQ(output.sizes(), hidden.sizes());
    torch::Tensor output_cpu = output.cpu().to(torch::kFloat32);
    ASSERT_TRUE(torch::isfinite(output_cpu).all().item<bool>());
    const double abs_sum = torch::abs(output_cpu).sum().item<double>();
    ASSERT_GT(abs_sum, 0.0);
  }

  torch::Device device_{torch::kCPU};
  torch::TensorOptions options_;
  torch::ScalarType dtype_ = torch::kBFloat16;
  std::unique_ptr<ProcessGroup> process_group_;
  std::unique_ptr<ProcessGroup> dp_process_group_;
  std::unique_ptr<ProcessGroup> tp_process_group_;
  std::unique_ptr<ProcessGroup> ep_process_group_;
  std::unique_ptr<ProcessGroup> single_rank_process_group_;
  ParallelArgs parallel_args_{0, 1, nullptr};
  std::vector<std::vector<DSACacheInfo>> caches_info_;
};

TEST_F(DeepseekV4DecoderLayerTest, NonHashRoutingForwardSmoke) {
  run_case(/*layer_id=*/1, /*pass_input_ids=*/false, /*use_ffn_alias=*/false);
}

TEST_F(DeepseekV4DecoderLayerTest, HashRoutingForwardSmoke) {
  run_case(/*layer_id=*/0, /*pass_input_ids=*/true, /*use_ffn_alias=*/false);
}

TEST_F(DeepseekV4DecoderLayerTest, FfnAliasForwardSmoke) {
  run_case(/*layer_id=*/1, /*pass_input_ids=*/false, /*use_ffn_alias=*/true);
}

TEST_F(DeepseekV4DecoderLayerTest, DpEpMoeRequiresDpTokenNums) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  use_dp_ep_parallel_args();

  EXPECT_DEATH(run_case(/*layer_id=*/1,
                        /*pass_input_ids=*/false,
                        /*use_ffn_alias=*/false),
               "dp_global_token_nums is empty");
}

}  // namespace layer
}  // namespace xllm
