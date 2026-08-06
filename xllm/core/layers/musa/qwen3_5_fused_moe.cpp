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

#include "layers/musa/qwen3_5_fused_moe.h"

#include <glog/logging.h>

#include <algorithm>
#include <optional>
#include <tuple>
#include <utility>

#include "kernels/musa/musa_ops_api.h"
#include "kernels/ops_api.h"
#include "util/env_var.h"

namespace xllm {
namespace layer {
namespace {

torch::Tensor get_tensor_with_weight_suffix(const StateDict& state_dict,
                                            const std::string& name) {
  auto tensor = state_dict.get_tensor(name);
  if (!tensor.defined()) {
    tensor = state_dict.get_tensor(name + ".weight");
  }
  return tensor;
}

constexpr int64_t kMaxChunkTokens = 1024;
// Compact FP8 prefill path for large batches. Compact routing does not
// allocate the [experts, max_m, hidden]
// masked-GEMM buffer, so this size is safe for the 20k-token target while
// avoiding one route/quantize/GEMM/synchronize sequence per 1k tokens.
constexpr int64_t kMaxCompactPrefillTokens = 16384;
// The BF16 compact path has no activation-quantization buffers and can process
// the complete long-context prefill in one routed batch and avoids
// repeating routing per layer.
constexpr int64_t kMaxCompactBf16PrefillTokens = 65536;
constexpr int64_t kMaxMoeTopkTokens = 2048;
constexpr int64_t kMaskedGemmMAlignment = 256;
constexpr int64_t kCompactBf16MAlignment = 128;
constexpr int64_t kRaggedDecodeAlignment = 128;
// Mate Ragged changes the FP8 GEMM accumulation layout. B=1 passes the model
// correctness gate and captures the long-ISL decode win; B>1 remains on the
// established contiguous path until a wider ragged decode path is ready.
constexpr int64_t kMaxRaggedDecodeTokens = 1;
constexpr int64_t kMaxRaggedBf16DecodeTokens = 32;
// Separate large-token MoE reduction from the small-token path.
// The fused combine kernel amortizes launch cost on prefill-sized batches;
// eager Torch reduction is faster for decode-sized batches.
constexpr int64_t kFusedCombineMinTokens = 128;
constexpr int64_t kContiguousBf16PrefillMinTokens = 4096;

bool use_fused_moe_preprocess(int64_t num_tokens) {
  static const bool enabled =
      util::get_bool_env("XLLM_MUSA_FUSED_MOE_PREPROCESS", true);
  return enabled && num_tokens >= kFusedCombineMinTokens;
}

bool use_contiguous_bf16_prefill_gemm(int64_t num_tokens) {
  static const bool enabled =
      util::get_bool_env("XLLM_MUSA_CONTIGUOUS_BF16_PREFILL_GEMM", false);
  return enabled && num_tokens >= kContiguousBf16PrefillMinTokens;
}

bool use_ragged_moe_decode(int64_t num_tokens, bool is_decode) {
  static const bool enabled =
      util::get_bool_env("XLLM_MUSA_RAGGED_MOE_DECODE", false);
  return enabled && is_decode && num_tokens <= kMaxRaggedDecodeTokens;
}

bool use_ragged_bf16_moe(int64_t num_tokens, bool is_decode) {
  static const bool enabled =
      util::get_bool_env("XLLM_MUSA_RAGGED_BF16_MOE_DECODE", true);
  return enabled && is_decode && num_tokens <= kMaxRaggedBf16DecodeTokens;
}

bool use_bf16_moe_decode_token_loop(int64_t num_tokens, bool is_decode) {
  static const bool enabled =
      util::get_bool_env("XLLM_MUSA_BF16_MOE_DECODE_TOKEN_LOOP", false);
  return enabled && is_decode && num_tokens > 1;
}

bool use_fused_moe_aot(int64_t num_tokens, bool is_decode) {
  static const bool enabled =
      util::get_bool_env("XLLM_MUSA_FUSED_MOE_AOT", true);
  if (!enabled || !is_decode) {
    return false;
  }
  if (xllm::kernel::musa::fused_moe_aot_available(num_tokens)) {
    return true;
  }
  LOG_FIRST_N(WARNING, 1)
      << "MUSA fused MoE AOT artifacts are unavailable for decode batch "
      << num_tokens << "; using the contiguous fallback.";
  return false;
}

bool use_bf16_fused_moe_aot(int64_t num_tokens, bool is_decode) {
  static const bool enabled =
      util::get_bool_env("XLLM_MUSA_FUSED_MOE_BF16_AOT", true);
  if (!enabled || !is_decode) {
    return false;
  }
  if (xllm::kernel::musa::fused_moe_bf16_aot_available(num_tokens)) {
    return true;
  }
  LOG_FIRST_N(WARNING, 1)
      << "MUSA BF16 fused MoE AOT artifacts are unavailable for decode batch "
      << num_tokens << "; using the Ragged fallback.";
  return false;
}

bool use_moe_topk(int64_t num_tokens,
                  const torch::Tensor& router_logits,
                  int64_t num_experts,
                  int64_t topk) {
  static const bool enabled =
      util::get_bool_env("XLLM_MUSA_FUSED_MOE_TOPK_SMALL", true);
  static const int64_t max_tokens =
      std::clamp(util::get_int_env("XLLM_MUSA_FUSED_MOE_TOPK_MAX_TOKENS",
                                   kMaxMoeTopkTokens),
                 int64_t{1},
                 kMaxMoeTopkTokens);
  const bool shape_supported =
      num_tokens <= max_tokens && num_experts == 256 && topk == 8 &&
      router_logits.scalar_type() == torch::kBFloat16 &&
      router_logits.is_contiguous();
  if (!enabled || !shape_supported) {
    return false;
  }
  if (xllm::kernel::musa::moe_topk_softmax_available()) {
    return true;
  }
  LOG_FIRST_N(WARNING, 1)
      << "MUSA top-k artifact is unavailable; using Torch top-k.";
  return false;
}

bool use_fused_shared_expert_gate(int64_t num_tokens) {
  return num_tokens == 1;
}

}  // namespace

Qwen3_5FusedMoEImpl::Qwen3_5FusedMoEImpl(const ModelArgs& model_args,
                                         const FusedMoEArgs& moe_args,
                                         const QuantArgs& quant_args,
                                         const ParallelArgs& parallel_args,
                                         const torch::TensorOptions& options)
    : num_experts_(model_args.n_routed_experts()),
      topk_(model_args.num_experts_per_tok()),
      hidden_size_(model_args.hidden_size()),
      intermediate_size_(model_args.moe_intermediate_size()),
      use_fp8_(quant_args.quant_method() == kQuantMethodFp8),
      use_contiguous_bf16_moe_(
          !use_fp8_ &&
          util::get_bool_env("XLLM_MUSA_CONTIGUOUS_BF16_MOE", true)),
      use_contiguous_fp8_moe_(
          use_fp8_ && util::get_bool_env("XLLM_MUSA_CONTIGUOUS_FP8_MOE", true)),
      options_(options),
      tp_pg_(parallel_args.tp_group_) {
  CHECK(quant_args.quant_method().empty() ||
        quant_args.quant_method() == kQuantMethodFp8)
      << "Qwen3.5 MUSA MoE supports BF16 or blockwise FP8 only; got "
      << quant_args.quant_method();
  CHECK(moe_args.is_gated)
      << "Qwen3.5 MUSA MoE requires gated (SwiGLU) experts.";
  CHECK_GT(num_experts_, 0);
  CHECK_GT(topk_, 0);
  CHECK_LE(topk_, num_experts_);
  CHECK(tp_pg_ != nullptr)
      << "Qwen3.5 MUSA MoE requires a tensor-parallel process group.";
  world_size_ = tp_pg_->world_size();
  rank_ = tp_pg_->rank();
  CHECK_EQ(world_size_, 1)
      << "Qwen3.5 MUSA MoE currently supports TP1 only; got TP world size "
      << world_size_;
  CHECK_EQ(parallel_args.ep_size(), 1)
      << "Qwen3.5 MUSA MoE currently supports EP1 only.";
  CHECK_EQ(hidden_size_ % 128, 0)
      << "Qwen3.5 MUSA MoE hidden size must be divisible by 128.";
  CHECK_EQ(intermediate_size_ % 128, 0)
      << "Qwen3.5 MUSA MoE intermediate size must be divisible by 128.";

  num_experts_per_rank_ = num_experts_;
  start_expert_id_ = 0;

  gate_ = register_module(
      "gate",
      ReplicatedLinear(
          hidden_size_, num_experts_, /*bias=*/false, quant_args, options));
  shared_experts_ =
      register_module("shared_expert",
                      DenseMLP(hidden_size_,
                               model_args.shared_expert_intermediate_size(),
                               /*is_gated=*/true,
                               /*has_bias=*/false,
                               model_args.hidden_act(),
                               /*enable_result_reduction=*/true,
                               quant_args,
                               parallel_args.tp_group_,
                               options));
  shared_expert_gate_ = register_module(
      "shared_expert_gate",
      torch::nn::Linear(torch::nn::LinearOptions(hidden_size_, 1).bias(false)));
  shared_expert_gate_->weight.set_data(shared_expert_gate_->weight.to(options));
  activation_ = register_module("activation", Activation("silu", true));

  const auto expert_options =
      use_fp8_ ? options_.dtype(torch::kFloat8_e4m3fn) : options_;
  w13_ = register_parameter(
      "w13",
      torch::empty(
          {num_experts_per_rank_, intermediate_size_ * 2, hidden_size_},
          expert_options),
      false);
  w2_ = register_parameter(
      "w2",
      torch::empty({num_experts_per_rank_, hidden_size_, intermediate_size_},
                   expert_options),
      false);
  if (use_fp8_) {
    w13_scale_inv_ =
        register_parameter("w13_scale_inv",
                           torch::empty({num_experts_per_rank_,
                                         intermediate_size_ * 2 / 128,
                                         hidden_size_ / 128},
                                        options_.dtype(torch::kFloat32)),
                           false);
    w2_scale_inv_ =
        register_parameter("w2_scale_inv",
                           torch::empty({num_experts_per_rank_,
                                         hidden_size_ / 128,
                                         intermediate_size_ / 128},
                                        options_.dtype(torch::kFloat32)),
                           false);

    if (util::get_bool_env("XLLM_MUSA_FUSED_MOE_AOT", true) &&
        xllm::kernel::musa::fused_moe_aot_available(
            /*num_tokens=*/1)) {
      xllm::kernel::musa::prepare_fused_moe_aot(w13_.device());
    }
  } else if (util::get_bool_env("XLLM_MUSA_FUSED_MOE_BF16_AOT", true) &&
             xllm::kernel::musa::fused_moe_bf16_aot_available(
                 /*num_tokens=*/1)) {
    xllm::kernel::musa::prepare_fused_moe_bf16_aot(w13_.device());
  }
}

void Qwen3_5FusedMoEImpl::load_routed_weights(const StateDict& mlp_state_dict) {
  if (mlp_state_dict.size() == 0) {
    return;
  }

  // LOAD_MOE_* macros expect a local state_dict and sharding locals below.
  const auto state_dict = mlp_state_dict.get_dict_with_prefix("experts.");
  const int64_t rank = rank_;
  const int64_t world_size = world_size_;
  const int64_t start_expert_id = start_expert_id_;
  const int64_t num_experts_per_rank = num_experts_per_rank_;
  if (use_fp8_) {
    // FP8 Qwen3.5 checkpoints store one gate/up/down tensor per expert.  The
    // Mate layout is [gate, up] (SwiGLU convention).
    const std::vector<std::string> prefixes = {"gate_proj.", "up_proj."};
    LOAD_MOE_FUSED_WEIGHT("weight", w1, w3, w13);
    LOAD_MOE_FUSED_WEIGHT(
        "weight_scale_inv", w1_scale_inv, w3_scale_inv, w13_scale_inv);
    LOAD_MOE_WEIGHT("down_proj.", "weight", w2, 1);
    LOAD_MOE_WEIGHT("down_proj.", "weight_scale_inv", w2_scale_inv, 1);

    // Accept a pre-packed FP8 checkpoint as well.  This is useful for
    // converted checkpoints and does not change the per-expert fast path.
    if (!w13_is_loaded_) {
      auto packed = get_tensor_with_weight_suffix(state_dict, "gate_up_proj");
      auto packed_scale =
          state_dict.get_tensor("gate_up_proj.weight_scale_inv");
      if (!packed_scale.defined()) {
        packed_scale = state_dict.get_tensor("gate_up_proj_scale_inv");
      }
      if (packed.defined() && packed_scale.defined()) {
        CHECK_EQ(packed.sizes(), w13_.sizes());
        CHECK_EQ(packed_scale.sizes(), w13_scale_inv_.sizes());
        w13_.copy_(packed);
        w13_scale_inv_.copy_(packed_scale);
        w13_is_loaded_ = true;
        w13_scale_inv_is_loaded_ = true;
      }
    }
    if (!w2_is_loaded_) {
      auto packed = get_tensor_with_weight_suffix(state_dict, "down_proj");
      auto packed_scale = state_dict.get_tensor("down_proj.weight_scale_inv");
      if (!packed_scale.defined()) {
        packed_scale = state_dict.get_tensor("down_proj_scale_inv");
      }
      if (packed.defined() && packed_scale.defined()) {
        CHECK_EQ(packed.sizes(), w2_.sizes());
        CHECK_EQ(packed_scale.sizes(), w2_scale_inv_.sizes());
        w2_.copy_(packed);
        w2_scale_inv_.copy_(packed_scale);
        w2_is_loaded_ = true;
        w2_scale_inv_is_loaded_ = true;
      }
    }
    return;
  }

  // BF16 Qwen3.5 checkpoints pack gate_up_proj/down_proj; shards may land in
  // different safetensors files, so load flags are independent.
  auto packed_gate_up =
      get_tensor_with_weight_suffix(state_dict, "gate_up_proj");
  if (packed_gate_up.defined()) {
    CHECK_EQ(packed_gate_up.sizes(), w13_.sizes())
        << "Qwen3.5 packed gate_up_proj shape mismatch: "
        << packed_gate_up.sizes() << " vs " << w13_.sizes();
    w13_.copy_(packed_gate_up);
    w13_is_loaded_ = true;
  }
  auto packed_down = get_tensor_with_weight_suffix(state_dict, "down_proj");
  if (packed_down.defined()) {
    CHECK_EQ(packed_down.sizes(), w2_.sizes())
        << "Qwen3.5 packed down_proj shape mismatch: " << packed_down.sizes()
        << " vs " << w2_.sizes();
    w2_.copy_(packed_down);
    w2_is_loaded_ = true;
  }

  // Also accept the unpacked BF16 format used by older converters.
  if (!w13_is_loaded_) {
    const std::vector<std::string> prefixes = {"gate_proj.", "up_proj."};
    LOAD_MOE_FUSED_WEIGHT("weight", w1, w3, w13);
  }
  if (!w2_is_loaded_) {
    LOAD_MOE_WEIGHT("down_proj.", "weight", w2, 1);
  }
}

void Qwen3_5FusedMoEImpl::load_state_dict(const StateDict& state_dict) {
  if (state_dict.size() == 0) {
    return;
  }
  gate_->load_state_dict(state_dict.get_dict_with_prefix("gate."));
  shared_experts_->load_state_dict(
      state_dict.get_dict_with_prefix("shared_expert."));

  auto shared_gate = state_dict.get_tensor("shared_expert_gate.weight");
  if (shared_gate.defined()) {
    shared_gate = shared_gate.reshape({1, hidden_size_});
    CHECK_EQ(shared_expert_gate_->weight.sizes(), shared_gate.sizes())
        << "Qwen3.5 shared_expert_gate shape mismatch.";
    shared_expert_gate_->weight.data().copy_(shared_gate);
    shared_expert_gate_is_loaded_ = true;
  }
  load_routed_weights(state_dict);
}

torch::Tensor Qwen3_5FusedMoEImpl::forward_chunk(
    const torch::Tensor& hidden_states,
    bool is_decode) {
  const int64_t num_tokens = hidden_states.size(0);
  torch::Tensor topk_weights;
  torch::Tensor topk_ids;
  {
    auto router_logits = gate_->forward(hidden_states);
    if (use_moe_topk(num_tokens, router_logits, num_experts_, topk_)) {
      std::tie(topk_weights, topk_ids) =
          xllm::kernel::musa::moe_topk_softmax(router_logits, topk_);
    } else {
      auto router_probs = torch::softmax(router_logits.to(torch::kFloat32), -1);
      auto topk_result = torch::topk(router_probs, topk_, -1, true, true);
      topk_weights = std::get<0>(topk_result);
      topk_ids = std::get<1>(topk_result).to(torch::kInt32);
      topk_weights = topk_weights / topk_weights.sum(-1, true);
    }
  }

  const int64_t assignment_count = num_tokens * topk_;
  auto flat_ids = topk_ids.reshape({assignment_count});

  if (!use_fp8_ && use_bf16_fused_moe_aot(num_tokens, is_decode)) {
    return xllm::kernel::musa::fused_moe_aot_bf16(
        hidden_states, w13_, w2_, topk_weights, topk_ids);
  }

  if (use_fp8_ && use_fused_moe_aot(num_tokens, is_decode)) {
    return xllm::kernel::musa::fused_moe_aot_fp8(hidden_states,
                                                 w13_,
                                                 w13_scale_inv_,
                                                 w2_,
                                                 w2_scale_inv_,
                                                 topk_weights,
                                                 topk_ids);
  }

  if (use_fp8_ && use_contiguous_fp8_moe_ &&
      use_ragged_moe_decode(num_tokens, is_decode)) {
    auto preprocess = xllm::kernel::musa::fused_moe_ragged_preprocess_fp8(
        hidden_states.contiguous(),
        topk_ids,
        /*group_size=*/128,
        kRaggedDecodeAlignment);
    torch::Tensor gate_up =
        xllm::kernel::musa::ragged_moe_gemm_fp8(std::get<0>(preprocess),
                                                std::get<1>(preprocess),
                                                w13_,
                                                w13_scale_inv_,
                                                std::get<2>(preprocess),
                                                hidden_states.scalar_type(),
                                                kRaggedDecodeAlignment);
    auto activated = xllm::kernel::musa::fused_moe_ragged_swiglu_quant_fp8(
        gate_up,
        /*group_size=*/128,
        kRaggedDecodeAlignment);
    torch::Tensor down =
        xllm::kernel::musa::ragged_moe_gemm_fp8(std::get<0>(activated),
                                                std::get<1>(activated),
                                                w2_,
                                                w2_scale_inv_,
                                                std::get<2>(preprocess),
                                                hidden_states.scalar_type(),
                                                kRaggedDecodeAlignment);
    return xllm::kernel::musa::fused_moe_ragged_combine(
        down, topk_weights, num_tokens, kRaggedDecodeAlignment);
  }

  if (use_contiguous_bf16_moe_ && use_ragged_bf16_moe(num_tokens, is_decode)) {
    if (use_bf16_moe_decode_token_loop(num_tokens, is_decode)) {
      std::vector<torch::Tensor> token_outputs;
      token_outputs.reserve(num_tokens);
      for (int64_t token_idx = 0; token_idx < num_tokens; ++token_idx) {
        auto token_preprocess =
            xllm::kernel::musa::fused_moe_ragged_preprocess_bf16(
                hidden_states.narrow(0, token_idx, 1).contiguous(),
                topk_ids.narrow(0, token_idx, 1).contiguous(),
                kRaggedDecodeAlignment);
        torch::Tensor token_gate_up = xllm::kernel::musa::ragged_moe_gemm_bf16(
            std::get<0>(token_preprocess),
            w13_,
            std::get<1>(token_preprocess),
            hidden_states.scalar_type(),
            kRaggedDecodeAlignment);
        torch::Tensor token_activated =
            xllm::kernel::musa::fused_moe_ragged_swiglu_bf16(
                token_gate_up, kRaggedDecodeAlignment);
        torch::Tensor token_down = xllm::kernel::musa::ragged_moe_gemm_bf16(
            token_activated,
            w2_,
            std::get<1>(token_preprocess),
            hidden_states.scalar_type(),
            kRaggedDecodeAlignment);
        token_outputs.emplace_back(xllm::kernel::musa::fused_moe_ragged_combine(
            token_down,
            topk_weights.narrow(0, token_idx, 1).contiguous(),
            /*num_tokens=*/1,
            kRaggedDecodeAlignment));
      }
      return torch::cat(token_outputs, /*dim=*/0);
    }

    auto preprocess =
        num_tokens == 1
            ? std::tuple_cat(
                  xllm::kernel::musa::fused_moe_ragged_preprocess_bf16(
                      hidden_states.contiguous(),
                      topk_ids,
                      kRaggedDecodeAlignment),
                  std::make_tuple(torch::Tensor()))
            : xllm::kernel::musa::fused_moe_decode_preprocess_bf16(
                  hidden_states.contiguous(),
                  topk_ids,
                  num_experts_,
                  kRaggedDecodeAlignment);
    torch::Tensor gate_up =
        xllm::kernel::musa::ragged_moe_gemm_bf16(std::get<0>(preprocess),
                                                 w13_,
                                                 std::get<1>(preprocess),
                                                 hidden_states.scalar_type(),
                                                 kRaggedDecodeAlignment);
    torch::Tensor activated =
        num_tokens == 1 ? xllm::kernel::musa::fused_moe_ragged_swiglu_bf16(
                              gate_up, kRaggedDecodeAlignment)
                        : xllm::kernel::musa::fused_moe_indexed_swiglu_bf16(
                              gate_up, std::get<2>(preprocess));
    torch::Tensor down =
        xllm::kernel::musa::ragged_moe_gemm_bf16(activated,
                                                 w2_,
                                                 std::get<1>(preprocess),
                                                 hidden_states.scalar_type(),
                                                 kRaggedDecodeAlignment);
    if (num_tokens == 1) {
      return xllm::kernel::musa::fused_moe_ragged_combine(
          down, topk_weights, num_tokens, kRaggedDecodeAlignment);
    }
    auto valid_rows = std::get<2>(preprocess).to(torch::kLong);
    auto routed_assignments = down.index_select(0, valid_rows)
                                  .view({num_tokens, topk_, hidden_size_});
    return (routed_assignments * topk_weights.unsqueeze(-1))
        .sum(1)
        .to(hidden_states.scalar_type());
  }

  if (use_contiguous_bf16_moe_ && is_decode) {
    auto preprocess = xllm::kernel::musa::fused_moe_preprocess_bf16(
        hidden_states.contiguous(),
        topk_ids,
        num_experts_,
        kCompactBf16MAlignment);
    torch::Tensor gate_up =
        xllm::kernel::musa::ragged_moe_gemm_bf16(std::get<0>(preprocess),
                                                 w13_,
                                                 std::get<1>(preprocess),
                                                 hidden_states.scalar_type(),
                                                 kCompactBf16MAlignment);
    torch::Tensor activated;
    activation_->forward(gate_up, activated);
    torch::Tensor down =
        xllm::kernel::musa::ragged_moe_gemm_bf16(activated,
                                                 w2_,
                                                 std::get<1>(preprocess),
                                                 hidden_states.scalar_type(),
                                                 kCompactBf16MAlignment);

    auto valid_rows = std::get<2>(preprocess).to(torch::kLong);
    auto routed_assignments = down.index_select(0, valid_rows)
                                  .view({num_tokens, topk_, hidden_size_});
    return (routed_assignments * topk_weights.unsqueeze(-1))
        .sum(1)
        .to(hidden_states.scalar_type());
  }

  auto run_contiguous_fp8 =
      [&](const torch::Tensor& sorted_hidden_fp8,
          const torch::Tensor& sorted_hidden_scale,
          const torch::Tensor& src_to_dst,
          const torch::Tensor& token_counts_i32,
          const torch::Tensor& original_indices) -> torch::Tensor {
    auto gate_up = xllm::kernel::musa::contiguous_moe_gemm_fp8(
        sorted_hidden_fp8,
        sorted_hidden_scale,
        w13_,
        w13_scale_inv_,
        token_counts_i32,
        hidden_states.scalar_type());

    torch::Tensor activated;
    activation_->forward(gate_up, activated);
    auto [activated_fp8, activated_scale] =
        xllm::kernel::per_token_group_quant_fp8(activated, 128);
    auto down = xllm::kernel::musa::contiguous_moe_gemm_fp8(
        activated_fp8,
        activated_scale,
        w2_,
        w2_scale_inv_,
        token_counts_i32,
        hidden_states.scalar_type());

    if (num_tokens >= kFusedCombineMinTokens) {
      return xllm::kernel::musa::moe_combine_result_indexed(
          down,
          src_to_dst,
          topk_weights,
          num_tokens,
          static_cast<int32_t>(topk_));
    }

    CHECK(original_indices.defined());
    auto routed_assignments = torch::empty_like(down);
    routed_assignments.index_copy_(0, original_indices, down);
    routed_assignments =
        routed_assignments.view({num_tokens, topk_, hidden_size_});
    return (routed_assignments * topk_weights.unsqueeze(-1))
        .sum(1)
        .to(hidden_states.scalar_type());
  };

  if (use_contiguous_fp8_moe_ && use_fused_moe_preprocess(num_tokens)) {
    auto preprocess = xllm::kernel::musa::fused_moe_preprocess_fp8(
        hidden_states.contiguous(), topk_ids, num_experts_, 128);
    return run_contiguous_fp8(std::get<0>(preprocess),
                              std::get<1>(preprocess),
                              std::get<2>(preprocess),
                              std::get<3>(preprocess),
                              torch::Tensor());
  }

  if (use_contiguous_bf16_moe_ && !is_decode) {
    // Fuse histogram, aligned-prefix, hidden-state
    // fan-out, row expert ids, original-to-padded mapping, and contiguous
    // group sizes on device. Mate Ragged remains the small-prefill fallback;
    // large prefills use the contiguous grouped GEMM path.
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
        preprocess;
    {
      preprocess = xllm::kernel::musa::fused_moe_preprocess_bf16(
          hidden_states.contiguous(),
          topk_ids,
          num_experts_,
          kCompactBf16MAlignment);
    }
    torch::Tensor gate_up;
    {
      if (use_contiguous_bf16_prefill_gemm(num_tokens)) {
        gate_up = xllm::kernel::musa::contiguous_moe_gemm_bf16(
            std::get<0>(preprocess),
            w13_,
            std::get<3>(preprocess),
            hidden_states.scalar_type());
      } else {
        gate_up = xllm::kernel::musa::ragged_moe_gemm_bf16(
            std::get<0>(preprocess),
            w13_,
            std::get<1>(preprocess),
            hidden_states.scalar_type(),
            kCompactBf16MAlignment);
      }
    }

    torch::Tensor activated;
    {
      activated = xllm::kernel::musa::fused_moe_indexed_swiglu_bf16(
          gate_up, std::get<2>(preprocess));
    }
    torch::Tensor down;
    {
      if (use_contiguous_bf16_prefill_gemm(num_tokens)) {
        down = xllm::kernel::musa::contiguous_moe_gemm_bf16(
            activated,
            w2_,
            std::get<3>(preprocess),
            hidden_states.scalar_type());
      } else {
        down = xllm::kernel::musa::ragged_moe_gemm_bf16(
            activated,
            w2_,
            std::get<1>(preprocess),
            hidden_states.scalar_type(),
            kCompactBf16MAlignment);
      }
    }
    torch::Tensor combined;
    {
      combined = xllm::kernel::musa::moe_combine_result_indexed(
          down,
          std::get<2>(preprocess),
          topk_weights,
          num_tokens,
          static_cast<int32_t>(topk_));
    }
    return combined;
  }

  auto route_index =
      xllm::kernel::cuda::moe_compute_index(flat_ids, num_experts_);
  auto dst_src_i32 = std::get<1>(route_index);
  auto token_counts_i32 = std::get<2>(route_index);
  auto original_indices = dst_src_i32.to(torch::kLong);

  if (use_contiguous_fp8_moe_) {
    // The contiguous Mate kernel consumes only valid assignment rows, grouped
    // by expert. Gather in BF16 before quantization; Float8 index_copy is not
    // supported by torch_musa and byte-view scatter is not correctness-safe.
    auto sorted_token_indices = torch::floor_divide(original_indices, topk_);
    auto sorted_hidden =
        hidden_states.index_select(0, sorted_token_indices).contiguous();
    auto [sorted_hidden_fp8, sorted_hidden_scale] =
        xllm::kernel::per_token_group_quant_fp8(sorted_hidden, 128);
    return run_contiguous_fp8(sorted_hidden_fp8,
                              sorted_hidden_scale,
                              std::get<0>(route_index),
                              token_counts_i32,
                              original_indices);
  }

  // Mate's masked grouped GEMM dispatches full M tiles. Match its padding
  // contract and leave one complete 256-row tile so the final tile never
  // extends beyond the per-expert allocation.
  const int64_t max_m =
      (num_tokens / kMaskedGemmMAlignment + 1) * kMaskedGemmMAlignment;
  auto sorted_ids = flat_ids.index_select(0, original_indices);
  auto expert_offsets = token_counts_i32.cumsum(0) - token_counts_i32;
  auto sorted_positions = torch::arange(assignment_count,
                                        torch::TensorOptions()
                                            .dtype(torch::kLong)
                                            .device(hidden_states.device())) -
                          expert_offsets.index_select(0, sorted_ids);
  auto sorted_slots = sorted_ids * max_m + sorted_positions;
  auto slots = torch::empty_like(sorted_slots);
  slots.index_copy_(0, original_indices, sorted_slots);
  const int64_t expected_m =
      (assignment_count + num_experts_ - 1) / num_experts_;

  auto repeated_hidden = hidden_states.unsqueeze(1)
                             .expand({num_tokens, topk_, hidden_size_})
                             .reshape({assignment_count, hidden_size_})
                             .contiguous();
  auto expanded = torch::empty({num_experts_, max_m, hidden_size_},
                               hidden_states.options());
  expanded.view({num_experts_ * max_m, hidden_size_})
      .index_copy_(0, slots, repeated_hidden);
  torch::Tensor gate_up;
  if (use_fp8_) {
    auto [expanded_fp8, expanded_scale] =
        xllm::kernel::per_token_group_quant_fp8(expanded, 128);
    gate_up =
        xllm::kernel::musa::masked_moe_gemm_fp8(expanded_fp8,
                                                expanded_scale,
                                                w13_,
                                                w13_scale_inv_,
                                                token_counts_i32,
                                                hidden_states.scalar_type(),
                                                expected_m);
  } else {
    gate_up =
        xllm::kernel::musa::masked_moe_gemm_bf16(expanded,
                                                 w13_,
                                                 token_counts_i32,
                                                 hidden_states.scalar_type(),
                                                 expected_m);
  }

  torch::Tensor activated;
  activation_->forward(gate_up, activated);

  torch::Tensor down;
  if (use_fp8_) {
    auto [activated_fp8, activated_scale] =
        xllm::kernel::per_token_group_quant_fp8(activated, 128);
    down = xllm::kernel::musa::masked_moe_gemm_fp8(activated_fp8,
                                                   activated_scale,
                                                   w2_,
                                                   w2_scale_inv_,
                                                   token_counts_i32,
                                                   hidden_states.scalar_type(),
                                                   expected_m);
  } else {
    down = xllm::kernel::musa::masked_moe_gemm_bf16(activated,
                                                    w2_,
                                                    token_counts_i32,
                                                    hidden_states.scalar_type(),
                                                    expected_m);
  }

  auto routed_assignments = down.view({num_experts_ * max_m, hidden_size_})
                                .index_select(0, slots)
                                .view({num_tokens, topk_, hidden_size_});
  torch::Tensor masked_output =
      (routed_assignments * topk_weights.unsqueeze(-1))
          .sum(1)
          .to(hidden_states.scalar_type());
  return masked_output;
}

torch::Tensor Qwen3_5FusedMoEImpl::forward(
    const torch::Tensor& hidden_states,
    const ModelInputParams& input_params) {
  CHECK(hidden_states.dim() == 2)
      << "Qwen3.5 MUSA MoE expects [tokens, hidden] input.";
  CHECK(w13_is_loaded_ && w2_is_loaded_)
      << "Qwen3.5 MUSA MoE expert weights were not fully loaded.";
  if (use_fp8_) {
    CHECK(w13_scale_inv_is_loaded_ && w2_scale_inv_is_loaded_)
        << "Qwen3.5 MUSA MoE FP8 scale tensors were not fully loaded.";
  }

  const int64_t num_tokens = hidden_states.size(0);
  // MIXED contains both prefill and decode rows. It must use the general
  // compact route rather than a decode-only fixed-block specialization.
  const bool is_decode = input_params.meta.batch_forward_type.is_decode();
  const bool is_prefill = !is_decode;
  int64_t chunk_tokens = kMaxChunkTokens;
  if (is_prefill && use_contiguous_bf16_moe_) {
    chunk_tokens = kMaxCompactBf16PrefillTokens;
  } else if (is_prefill && use_contiguous_fp8_moe_) {
    chunk_tokens = kMaxCompactPrefillTokens;
  }
  std::vector<torch::Tensor> routed_chunks;
  routed_chunks.reserve((num_tokens + chunk_tokens - 1) / chunk_tokens);
  for (int64_t start = 0; start < num_tokens; start += chunk_tokens) {
    const int64_t length = std::min(chunk_tokens, num_tokens - start);
    routed_chunks.emplace_back(forward_chunk(
        hidden_states.narrow(/*dim=*/0, start, length), is_decode));
  }
  auto routed = routed_chunks.size() == 1
                    ? routed_chunks.front()
                    : torch::cat(routed_chunks, /*dim=*/0);

  torch::Tensor shared;
  {
    shared = shared_experts_->forward(hidden_states);
    if (use_fused_shared_expert_gate(hidden_states.size(0))) {
      xllm::kernel::musa::fused_shared_expert_gate_inplace(
          shared, hidden_states, shared_expert_gate_->weight);
    } else {
      auto shared_gate =
          torch::sigmoid(shared_expert_gate_->forward(hidden_states));
      shared = shared * shared_gate;
    }
  }
  return routed + shared;
}

void Qwen3_5FusedMoEImpl::verify_loaded_weights() const {
  CHECK(w13_is_loaded_ && w2_is_loaded_)
      << "Qwen3.5 MUSA MoE expert weights are missing.";
  if (use_fp8_) {
    CHECK(w13_scale_inv_is_loaded_ && w2_scale_inv_is_loaded_)
        << "Qwen3.5 MUSA MoE FP8 expert scales are missing.";
  }
  CHECK(shared_expert_gate_is_loaded_)
      << "Qwen3.5 MUSA MoE shared expert gate is missing.";
}

}  // namespace layer
}  // namespace xllm
