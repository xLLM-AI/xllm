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

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <algorithm>
#include <string>
#include <unordered_set>
#include <vector>

#include "core/framework/config/eplb_config.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_args.h"
#include "framework/model/model_input_params.h"
#include "framework/model_context.h"
#include "framework/parallel_state/parallel_args.h"
#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"
#if defined(USE_MLU)
#include "layers/mlu/attention.h"
#include "layers/mlu/deepseek_v2_decoder_layer_impl.h"
#elif defined(USE_CUDA)
#include "layers/cuda/attention.h"
#endif
#include "layers/common/attention_metadata_builder.h"
#include "layers/common/dsa_topk_share_plan.h"
#include "layers/mlu/tests_utils.h"
#include "platform/device.h"
#include "platform/platform.h"

namespace xllm {
namespace layer {

TEST(DeepseekV2DecoderLayerInterfaceTest, ForwardKeepsModelInputParamsConst) {
  using ForwardFn = torch::Tensor (DeepseekV2DecoderLayerImpl::*)(
      torch::Tensor&,
      std::optional<torch::Tensor>&,
      torch::Tensor&,
      const AttentionMetadata&,
      KVCache&,
      const ModelInputParams&,
      const std::optional<torch::Tensor>&,
      DsaTopkRelay*);

  ForwardFn forward = &DeepseekV2DecoderLayerImpl::forward;
  EXPECT_TRUE(forward != nullptr);
}

TEST(DeepseekV2DecoderLayerInterfaceTest, MtpForwardExposesTypedStateOutput) {
  using ForwardMtpFn = torch::Tensor (DeepseekV2DecoderLayerImpl::*)(
      torch::Tensor&,
      std::optional<torch::Tensor>&,
      torch::Tensor&,
      const AttentionMetadata&,
      KVCache&,
      const ModelInputParams&,
      const std::optional<torch::Tensor>&,
      const std::optional<DsaTopkState>&,
      std::optional<DsaTopkState>&);

  ForwardMtpFn forward_mtp = &DeepseekV2DecoderLayerImpl::forward_mtp;
  EXPECT_TRUE(forward_mtp != nullptr);
}

class DeepseekV2DecoderLayerTestPeer {
 public:
  using Carrier = DeepseekV2DecoderLayerImpl::PostAttnCarrier;
  using Mode = DeepseekV2DecoderLayerImpl::PostAttnMode;

  static Carrier make_carrier(Mode mode,
                              torch::Tensor skip_local = torch::Tensor(),
                              torch::Tensor ffn_in = torch::Tensor(),
                              PaddingInfo pad_info = {}) {
    Carrier carrier;
    carrier.ffn_in = std::move(ffn_in);
    carrier.skip_local = std::move(skip_local);
    carrier.pad_info = pad_info;
    carrier.mode = mode;
    return carrier;
  }

  static Carrier build_post_attn_carrier(
      DeepseekV2DecoderLayerImpl& decoder,
      torch::Tensor x,
      const torch::Tensor& residual,
      const ModelInputParams& input_params,
      DeepseekV2AttentionImpl::PostAttnLayout attn_layout,
      bool need_dp_gather,
      bool enable_moe_all2all) {
    (void)input_params;
    (void)need_dp_gather;
    (void)enable_moe_all2all;
    return decoder.build_post_attn_carrier(x, residual, attn_layout);
  }

  static torch::Tensor restore_ffn_output(
      DeepseekV2DecoderLayerImpl& decoder,
      torch::Tensor x,
      const Carrier& carrier,
      const ModelInputParams& input_params) {
    (void)input_params;
    return decoder.restore_ffn_output(x, carrier);
  }

  static bool can_keep_local_output(DeepseekV2DecoderLayerImpl& decoder,
                                    const Carrier& carrier,
                                    ProcessGroup* pg) {
    return decoder.can_keep_local_output(carrier, pg);
  }

  static torch::Tensor comm_out(DeepseekV2DecoderLayerImpl& decoder,
                                torch::Tensor x,
                                const Carrier& carrier,
                                ProcessGroup* pg) {
    return decoder.comm_out(x, carrier, pg);
  }

  static torch::Tensor post_norm(DeepseekV2DecoderLayerImpl& decoder,
                                 torch::Tensor x) {
    return std::get<0>(decoder.post_norm_->forward(x));
  }

  static DenseMLP mlp(DeepseekV2DecoderLayerImpl& decoder) {
    return decoder.mlp_;
  }

  static torch::Tensor reduce_out(DeepseekV2DecoderLayerImpl& decoder,
                                  torch::Tensor x,
                                  ProcessGroup* pg) {
    return decoder.reduce_out(x, pg);
  }

  static const DsaTopkShareDecision& topk_share_decision(
      const DeepseekV2DecoderLayerImpl& decoder) {
    return decoder.topk_share_decision_;
  }

  static bool mtp_topk_reuse(const DeepseekV2DecoderLayerImpl& decoder) {
    return decoder.mtp_topk_reuse_;
  }

  static bool attention_has_child(const DeepseekV2DecoderLayerImpl& decoder,
                                  const std::string& name) {
    return decoder.attention_->named_children().contains(name);
  }
};

class DeepseekV2DecoderLayerTest : public ::testing::Test {
 protected:
  using DecoderHolder = torch::nn::ModuleHolder<DeepseekV2DecoderLayerImpl>;

  void SetUp() override {
    // Base defaults from test helpers
    model_args_ = test::create_default_model_args();
    // test w8a8 only for now
    quant_args_ = test::create_default_quant_args();
    options_ = torch::TensorOptions()
                   .dtype(torch::kBFloat16)
                   .device(Platform::type_torch(), 0)
                   .requires_grad(false);
    parallel_args_ = test::create_default_parallel_args(mock_process_group_);
    // Use a small but structurally valid DeepSeek-style config so the tests
    // exercise Dense/MoE/MLA wiring without allocating near-production tensors.
    model_args_.model_type() = "deepseek_v3";
    model_args_.dtype() = "";  // default empty
    model_args_.vocab_size() = 4096;
    model_args_.hidden_size() = 512;
    model_args_.n_layers() = 8;
    model_args_.n_heads() = 8;
    model_args_.n_kv_heads() = 8;
    model_args_.intermediate_size() = 1024;
    model_args_.hidden_act() = "silu";
    model_args_.rms_norm_eps() = 1e-6f;
    model_args_.max_position_embeddings() = 4096;
    model_args_.eos_token_id() = 1;
    model_args_.bos_token_id() = 0;

    // Decoder layer specific routing between MoE and Dense
    model_args_.first_k_dense_replace() = 1;
    model_args_.moe_layer_freq() = 1;

    // MoE-related params
    model_args_.n_routed_experts() = 8;
    model_args_.n_shared_experts() = 1;
    model_args_.num_experts_per_tok() = 2;
    model_args_.moe_intermediate_size() = 128;
    model_args_.routed_scaling_factor() = 2.5f;
    model_args_.norm_topk_prob() = true;
    model_args_.n_group() = 2;
    model_args_.topk_group() = 1;
    model_args_.scoring_func() = "sigmoid";
    model_args_.topk_method() = "noaux_tc";

    // Q/K/V dims used by DeepseekV2Attention
    model_args_.qk_nope_head_dim() = 64;
    model_args_.qk_rope_head_dim() = 64;
    model_args_.v_head_dim() = 64;
    model_args_.head_dim() = 128;  // qk_nope_head_dim + qk_rope_head_dim
    model_args_.rotary_dim() = model_args_.qk_rope_head_dim();

    // Rope scaling related
    model_args_.rope_scaling_rope_type() = "deepseek_yarn";
    // The following values may be model/export dependent; set common defaults.
    model_args_.rope_scaling_beta_fast() = 32;
    model_args_.rope_scaling_beta_slow() = 1;
    model_args_.rope_scaling_factor() = 40;
    model_args_.rope_extrapolation_factor() = 1.0f;
    model_args_.rope_scaling_mscale() = 1.0f;
    model_args_.rope_scaling_mscale_all_dim() = 1.0f;
    model_args_.rope_scaling_original_max_position_embeddings() = 1024;
    model_args_.rope_scaling_attn_factor() = 1.0f;

    // Sliding window
    model_args_.use_sliding_window() = false;
    model_args_.sliding_window() = 512;
    model_args_.max_window_layers() = 8;

    // LORA ranks for DeepSeek-V3
    model_args_.q_lora_rank() = 128;
    model_args_.kv_lora_rank() = 64;  // qk_rope_head_dim + kv_lora_rank = 128

    // extra parameters for DeepSeek-V3.2-Exp
    model_args_.index_head_dim() = 128;
    model_args_.index_n_heads() = 0;
    model_args_.index_topk() = 0;
    model_args_.enable_mla() = true;

    // Build a ModelContext that the decoder requires
    context_ = ModelContext(parallel_args_, model_args_, quant_args_, options_);
  }

  // Collect registered child module names to verify module wiring
  static std::unordered_set<std::string> get_child_module_names(
      const torch::nn::Module& module) {
    std::unordered_set<std::string> names;
    for (const auto& named_child : module.named_children()) {
      names.insert(named_child.key());
    }
    return names;
  }

  // Create default test weights for decoder layer (w8a8 smoothquant format)
  std::unordered_map<std::string, torch::Tensor> create_default_test_weights(
      int32_t layer_id,
      int64_t hidden_size,
      int64_t intermediate_size,
      int64_t moe_intermediate_size = -1,
      int num_routed_experts = -1) {
    std::unordered_map<std::string, torch::Tensor> weight_dict;

    // Create input_layernorm weights (float32, not quantized)
    // Shape: [hidden_size]
    auto input_norm_weight = torch::full({hidden_size}, 1.0f, options_);
    weight_dict["input_layernorm.weight"] =
        input_norm_weight.to(torch::TensorOptions()
                                 .dtype(torch::kFloat32)
                                 .device(options_.device()));

    // Create post_attention_layernorm weights (float32, not quantized)
    // Shape: [hidden_size]
    auto post_norm_weight = torch::full({hidden_size}, 1.0f, options_);
    weight_dict["post_attention_layernorm.weight"] =
        post_norm_weight.to(torch::TensorOptions()
                                .dtype(torch::kFloat32)
                                .device(options_.device()));

    // Determine if this layer uses Dense MLP or MoE
    bool use_moe = (layer_id >= model_args_.first_k_dense_replace());

    if (use_moe) {
      // Create MoE weights
      int64_t test_moe_intermediate_size =
          (moe_intermediate_size > 0) ? moe_intermediate_size
                                      : model_args_.moe_intermediate_size();
      int test_num_routed_experts = (num_routed_experts > 0)
                                        ? num_routed_experts
                                        : model_args_.n_routed_experts();

      // Create gate weights (routing layer, not quantized)
      // Shape: [num_routed_experts, hidden_size]
      auto gate_weight =
          torch::full({test_num_routed_experts, hidden_size}, 0.8f, options_);
      weight_dict["mlp.gate.weight"] = gate_weight;

      // Create e_score_correction_bias if needed
      auto e_score_correction_bias =
          torch::full({test_num_routed_experts}, 0.1f, options_);
      weight_dict["mlp.gate.e_score_correction_bias"] = e_score_correction_bias;

      // Create shared experts weights if n_shared_experts > 0
      if (model_args_.n_shared_experts() > 0) {
        // gate_proj weights
        auto shared_gate_weight = torch::full(
            {test_moe_intermediate_size, hidden_size}, 0.3f, options_);
        auto shared_gate_qweight = shared_gate_weight.to(torch::kInt8);
        auto shared_gate_scale = torch::full({test_moe_intermediate_size},
                                             0.1f,
                                             torch::TensorOptions()
                                                 .dtype(torch::kFloat32)
                                                 .device(options_.device()));
        auto shared_gate_smooth = torch::full({hidden_size},
                                              0.05f,
                                              torch::TensorOptions()
                                                  .dtype(torch::kFloat32)
                                                  .device(options_.device()));

        // up_proj weights
        auto shared_up_weight = torch::full(
            {test_moe_intermediate_size, hidden_size}, 0.3f, options_);
        auto shared_up_qweight = shared_up_weight.to(torch::kInt8);
        auto shared_up_scale = torch::full({test_moe_intermediate_size},
                                           0.1f,
                                           torch::TensorOptions()
                                               .dtype(torch::kFloat32)
                                               .device(options_.device()));
        auto shared_up_smooth = torch::full({hidden_size},
                                            0.05f,
                                            torch::TensorOptions()
                                                .dtype(torch::kFloat32)
                                                .device(options_.device()));

        // down_proj weights
        auto shared_down_weight = torch::full(
            {hidden_size, test_moe_intermediate_size}, 0.2f, options_);
        auto shared_down_qweight = shared_down_weight.to(torch::kInt8);
        auto shared_down_scale = torch::full({hidden_size},
                                             0.1f,
                                             torch::TensorOptions()
                                                 .dtype(torch::kFloat32)
                                                 .device(options_.device()));
        auto shared_down_smooth = torch::full({test_moe_intermediate_size},
                                              0.05f,
                                              torch::TensorOptions()
                                                  .dtype(torch::kFloat32)
                                                  .device(options_.device()));

        weight_dict["mlp.shared_experts.gate_proj.qweight"] =
            shared_gate_qweight;
        weight_dict["mlp.shared_experts.gate_proj.per_channel_scale"] =
            shared_gate_scale;
        weight_dict["mlp.shared_experts.gate_proj.smooth"] = shared_gate_smooth;
        weight_dict["mlp.shared_experts.up_proj.qweight"] = shared_up_qweight;
        weight_dict["mlp.shared_experts.up_proj.per_channel_scale"] =
            shared_up_scale;
        weight_dict["mlp.shared_experts.up_proj.smooth"] = shared_up_smooth;
        weight_dict["mlp.shared_experts.down_proj.qweight"] =
            shared_down_qweight;
        weight_dict["mlp.shared_experts.down_proj.per_channel_scale"] =
            shared_down_scale;
        weight_dict["mlp.shared_experts.down_proj.smooth"] = shared_down_smooth;
      }

      // Create routed experts weights
      for (int expert_id = 0; expert_id < test_num_routed_experts;
           ++expert_id) {
        std::string expert_prefix =
            "mlp.experts." + std::to_string(expert_id) + ".";

        // gate_proj weights
        auto gate_proj_weight = torch::full(
            {test_moe_intermediate_size, hidden_size}, 0.5f, options_);
        auto gate_proj_qweight = gate_proj_weight.to(torch::kInt8);
        auto gate_proj_scale = torch::full({test_moe_intermediate_size},
                                           0.1f,
                                           torch::TensorOptions()
                                               .dtype(torch::kFloat32)
                                               .device(options_.device()));
        auto gate_proj_smooth = torch::full({hidden_size},
                                            0.05f,
                                            torch::TensorOptions()
                                                .dtype(torch::kFloat32)
                                                .device(options_.device()));

        // up_proj weights
        auto up_proj_weight = torch::full(
            {test_moe_intermediate_size, hidden_size}, 0.5f, options_);
        auto up_proj_qweight = up_proj_weight.to(torch::kInt8);
        auto up_proj_scale = torch::full({test_moe_intermediate_size},
                                         0.1f,
                                         torch::TensorOptions()
                                             .dtype(torch::kFloat32)
                                             .device(options_.device()));
        auto up_proj_smooth = torch::full({hidden_size},
                                          0.05f,
                                          torch::TensorOptions()
                                              .dtype(torch::kFloat32)
                                              .device(options_.device()));

        // down_proj weights
        auto down_proj_weight = torch::full(
            {hidden_size, test_moe_intermediate_size}, 0.3f, options_);
        auto down_proj_qweight = down_proj_weight.to(torch::kInt8);
        auto down_proj_scale = torch::full({hidden_size},
                                           0.1f,
                                           torch::TensorOptions()
                                               .dtype(torch::kFloat32)
                                               .device(options_.device()));
        auto down_proj_smooth = torch::full({test_moe_intermediate_size},
                                            0.05f,
                                            torch::TensorOptions()
                                                .dtype(torch::kFloat32)
                                                .device(options_.device()));

        weight_dict[expert_prefix + "gate_proj.qweight"] = gate_proj_qweight;
        weight_dict[expert_prefix + "gate_proj.per_channel_scale"] =
            gate_proj_scale;
        weight_dict[expert_prefix + "gate_proj.smooth"] = gate_proj_smooth;
        weight_dict[expert_prefix + "up_proj.qweight"] = up_proj_qweight;
        weight_dict[expert_prefix + "up_proj.per_channel_scale"] =
            up_proj_scale;
        weight_dict[expert_prefix + "up_proj.smooth"] = up_proj_smooth;
        weight_dict[expert_prefix + "down_proj.qweight"] = down_proj_qweight;
        weight_dict[expert_prefix + "down_proj.per_channel_scale"] =
            down_proj_scale;
        weight_dict[expert_prefix + "down_proj.smooth"] = down_proj_smooth;
      }
    } else {
      // Create Dense MLP weights
      // gate_proj weights (ColumnParallelLinear)
      // Shape: [intermediate_size, hidden_size]
      auto gate_weight =
          torch::full({intermediate_size, hidden_size}, 5.0f, options_);
      auto gate_qweight = gate_weight.to(torch::kInt8);
      auto gate_scale = torch::full({intermediate_size},
                                    0.1f,
                                    torch::TensorOptions()
                                        .dtype(torch::kFloat32)
                                        .device(options_.device()));
      auto gate_smooth = torch::full({hidden_size},
                                     0.05f,
                                     torch::TensorOptions()
                                         .dtype(torch::kFloat32)
                                         .device(options_.device()));

      // up_proj weights (ColumnParallelLinear)
      // Shape: [intermediate_size, hidden_size]
      auto up_weight =
          torch::full({intermediate_size, hidden_size}, 5.0f, options_);
      auto up_qweight = up_weight.to(torch::kInt8);
      auto up_scale = torch::full({intermediate_size},
                                  0.1f,
                                  torch::TensorOptions()
                                      .dtype(torch::kFloat32)
                                      .device(options_.device()));
      auto up_smooth = torch::full({hidden_size},
                                   0.05f,
                                   torch::TensorOptions()
                                       .dtype(torch::kFloat32)
                                       .device(options_.device()));

      // down_proj weights (RowParallelLinear)
      // Shape: [hidden_size, intermediate_size]
      auto down_weight =
          torch::full({hidden_size, intermediate_size}, 3.0f, options_);
      auto down_qweight = down_weight.to(torch::kInt8);
      auto down_scale = torch::full({hidden_size},
                                    0.1f,
                                    torch::TensorOptions()
                                        .dtype(torch::kFloat32)
                                        .device(options_.device()));
      auto down_smooth = torch::full({intermediate_size},
                                     0.05f,
                                     torch::TensorOptions()
                                         .dtype(torch::kFloat32)
                                         .device(options_.device()));

      weight_dict["mlp.gate_proj.qweight"] = gate_qweight;
      weight_dict["mlp.gate_proj.per_channel_scale"] = gate_scale;
      weight_dict["mlp.gate_proj.smooth"] = gate_smooth;
      weight_dict["mlp.up_proj.qweight"] = up_qweight;
      weight_dict["mlp.up_proj.per_channel_scale"] = up_scale;
      weight_dict["mlp.up_proj.smooth"] = up_smooth;
      weight_dict["mlp.down_proj.qweight"] = down_qweight;
      weight_dict["mlp.down_proj.per_channel_scale"] = down_scale;
      weight_dict["mlp.down_proj.smooth"] = down_smooth;
    }

    // Create attention weights for DeepSeek V2 (MLA)
    int64_t num_heads = model_args_.n_heads();
    int64_t q_lora_rank = model_args_.q_lora_rank();
    int64_t kv_lora_rank = model_args_.kv_lora_rank();
    int64_t qk_nope_head_dim = model_args_.qk_nope_head_dim();
    int64_t qk_rope_head_dim = model_args_.qk_rope_head_dim();
    int64_t qk_head_dim = qk_nope_head_dim + qk_rope_head_dim;
    int64_t v_head_dim = model_args_.v_head_dim();
    int64_t index_head_dim = model_args_.index_head_dim();
    int64_t index_n_heads = model_args_.index_n_heads();

    // Quantized weights (w8a8 smoothquant format)
    // o_proj weights
    auto o_proj_weight =
        torch::full({hidden_size, num_heads * v_head_dim}, 1.0f, options_);
    auto o_proj_qweight = o_proj_weight.to(torch::kInt8);
    auto o_proj_scale = torch::full({hidden_size},
                                    0.03f,
                                    torch::TensorOptions()
                                        .dtype(torch::kFloat32)
                                        .device(options_.device()));
    auto o_proj_smooth = torch::full({num_heads * v_head_dim},
                                     0.03f,
                                     torch::TensorOptions()
                                         .dtype(torch::kFloat32)
                                         .device(options_.device()));

    weight_dict["self_attn.o_proj.qweight"] = o_proj_qweight;
    weight_dict["self_attn.o_proj.per_channel_scale"] = o_proj_scale;
    weight_dict["self_attn.o_proj.smooth"] = o_proj_smooth;

    // q_b_proj weights
    auto q_b_proj_weight =
        torch::full({num_heads * qk_head_dim, q_lora_rank}, 1.0f, options_);
    auto q_b_proj_qweight = q_b_proj_weight.to(torch::kInt8);
    auto q_b_proj_scale = torch::full({num_heads * qk_head_dim},
                                      0.03f,
                                      torch::TensorOptions()
                                          .dtype(torch::kFloat32)
                                          .device(options_.device()));
    auto q_b_proj_smooth = torch::full({q_lora_rank},
                                       0.03f,
                                       torch::TensorOptions()
                                           .dtype(torch::kFloat32)
                                           .device(options_.device()));

    weight_dict["self_attn.q_b_proj.qweight"] = q_b_proj_qweight;
    weight_dict["self_attn.q_b_proj.per_channel_scale"] = q_b_proj_scale;
    weight_dict["self_attn.q_b_proj.smooth"] = q_b_proj_smooth;

    // Non-quantized weights (float32)
    // kv_b_proj.weight: [num_heads * (qk_nope_head_dim + v_head_dim),
    // kv_lora_rank]
    auto kv_b_proj_weight =
        torch::full({num_heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank},
                    0.02f,
                    options_);
    weight_dict["self_attn.kv_b_proj.weight"] =
        kv_b_proj_weight.to(torch::TensorOptions()
                                .dtype(torch::kFloat32)
                                .device(options_.device()));

    // kv_a_proj_with_mqa.weight: [kv_lora_rank + qk_rope_head_dim, hidden_size]
    auto kv_a_proj_with_mqa_weight = torch::full(
        {kv_lora_rank + qk_rope_head_dim, hidden_size}, 0.02f, options_);
    weight_dict["self_attn.kv_a_proj_with_mqa.weight"] =
        kv_a_proj_with_mqa_weight.to(torch::TensorOptions()
                                         .dtype(torch::kFloat32)
                                         .device(options_.device()));

    // q_a_proj.weight: [q_lora_rank, hidden_size]
    auto q_a_proj_weight =
        torch::full({q_lora_rank, hidden_size}, 0.02f, options_);
    weight_dict["self_attn.q_a_proj.weight"] =
        q_a_proj_weight.to(torch::TensorOptions()
                               .dtype(torch::kFloat32)
                               .device(options_.device()));

    // LayerNorm weights
    auto kv_a_layernorm_weight = torch::full({kv_lora_rank}, 1.0f, options_);
    weight_dict["self_attn.kv_a_layernorm.weight"] =
        kv_a_layernorm_weight.to(torch::TensorOptions()
                                     .dtype(torch::kFloat32)
                                     .device(options_.device()));

    auto q_a_layernorm_weight = torch::full({q_lora_rank}, 1.0f, options_);
    weight_dict["self_attn.q_a_layernorm.weight"] =
        q_a_layernorm_weight.to(torch::TensorOptions()
                                    .dtype(torch::kFloat32)
                                    .device(options_.device()));

    // Indexer weights (if enabled)
    if (model_args_.index_n_heads() > 0) {
      auto indexer_k_norm_bias = torch::full({index_head_dim}, 0.0f, options_);
      weight_dict["self_attn.indexer.k_norm.bias"] =
          indexer_k_norm_bias.to(torch::TensorOptions()
                                     .dtype(torch::kFloat32)
                                     .device(options_.device()));

      auto indexer_k_norm_weight =
          torch::full({index_head_dim}, 1.0f, options_);
      weight_dict["self_attn.indexer.k_norm.weight"] =
          indexer_k_norm_weight.to(torch::TensorOptions()
                                       .dtype(torch::kFloat32)
                                       .device(options_.device()));

      auto indexer_weights_proj_weight =
          torch::full({index_n_heads, hidden_size}, 0.02f, options_);
      weight_dict["self_attn.indexer.weights_proj.weight"] =
          indexer_weights_proj_weight.to(torch::TensorOptions()
                                             .dtype(torch::kFloat32)
                                             .device(options_.device()));

      auto indexer_wk_weight =
          torch::full({index_head_dim, hidden_size}, 0.02f, options_);
      weight_dict["self_attn.indexer.wk.weight"] =
          indexer_wk_weight.to(torch::TensorOptions()
                                   .dtype(torch::kFloat32)
                                   .device(options_.device()));

      auto indexer_wq_b_weight = torch::full(
          {index_n_heads * index_head_dim, q_lora_rank}, 0.02f, options_);
      weight_dict["self_attn.indexer.wq_b.weight"] =
          indexer_wq_b_weight.to(torch::TensorOptions()
                                     .dtype(torch::kFloat32)
                                     .device(options_.device()));
    }

    LOG(INFO) << "Test w8a8 smoothquant weights created successfully for layer "
              << layer_id << " (use_moe=" << use_moe << ")";

    return weight_dict;
  }

  // Helper function to create test weights with custom dimensions
  std::unordered_map<std::string, torch::Tensor> create_test_weights(
      int32_t layer_id,
      int64_t custom_hidden_size = -1,
      int64_t custom_intermediate_size = -1,
      int64_t custom_moe_intermediate_size = -1,
      int custom_num_routed_experts = -1) {
    int64_t test_hidden_size = (custom_hidden_size > 0)
                                   ? custom_hidden_size
                                   : model_args_.hidden_size();
    int64_t test_intermediate_size = (custom_intermediate_size > 0)
                                         ? custom_intermediate_size
                                         : model_args_.intermediate_size();

    return create_default_test_weights(layer_id,
                                       test_hidden_size,
                                       test_intermediate_size,
                                       custom_moe_intermediate_size,
                                       custom_num_routed_experts);
  }

  DecoderHolder make_decoder(int32_t layer_id) {
    const DsaTopkSharePlan topk_share_plan(model_args_);
    return DecoderHolder(
        DeepseekV2DecoderLayerImpl(context_, layer_id, topk_share_plan));
  }

  DecoderHolder make_loaded_decoder(int32_t layer_id) {
    auto decoder = make_decoder(layer_id);
    auto weight_dict = create_test_weights(layer_id);
    StateDict state_dict(weight_dict);
    decoder->load_state_dict(state_dict);
    return decoder;
  }

  torch::TensorOptions fp32_opts() const {
    return torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(options_.device());
  }

  torch::TensorOptions hidden_opts() const {
    return torch::TensorOptions()
        .dtype(options_.dtype())
        .device(options_.device());
  }

  void refresh_ctx() {
    context_ = ModelContext(parallel_args_, model_args_, quant_args_, options_);
  }

  void set_tp_dp_ctx(int64_t world_size,
                     int64_t dp_size,
                     int64_t tp_size,
                     int64_t ep_size) {
    global_pg_ = std::make_unique<test::MockProcessGroup>(
        options_.device(), /*rank=*/0, world_size);
    dp_pg_ = std::make_unique<test::MockProcessGroup>(
        options_.device(), /*rank=*/0, dp_size);
    tp_pg_ = std::make_unique<test::MockProcessGroup>(
        options_.device(), /*rank=*/0, tp_size);

    parallel_args_ =
        ParallelArgs(/*rank=*/0, world_size, dp_size, global_pg_.get());
    parallel_args_.ep_size_ = ep_size;
    parallel_args_.process_group_ = global_pg_.get();
    parallel_args_.dp_local_process_group_ = dp_pg_.get();
    parallel_args_.tp_group_ = tp_pg_.get();
    parallel_args_.single_rank_group_ =
        tp_size > 1 ? build_single_rank_pg() : tp_pg_.get();
    parallel_args_.cp_group_ = tp_pg_.get();
    refresh_ctx();
  }

  void set_tp_ctx(int64_t world_size, int64_t ep_size) {
    global_pg_ = std::make_unique<test::MockProcessGroup>(
        options_.device(), /*rank=*/0, world_size);
    tp_pg_ = std::make_unique<test::MockProcessGroup>(
        options_.device(), /*rank=*/0, world_size);
    dp_pg_.reset();

    parallel_args_ =
        ParallelArgs(/*rank=*/0, world_size, /*dp_size=*/1, global_pg_.get());
    parallel_args_.ep_size_ = ep_size;
    parallel_args_.process_group_ = global_pg_.get();
    parallel_args_.dp_local_process_group_ = nullptr;
    parallel_args_.tp_group_ = tp_pg_.get();
    parallel_args_.single_rank_group_ =
        world_size > 1 ? build_single_rank_pg() : tp_pg_.get();
    parallel_args_.cp_group_ = tp_pg_.get();
    refresh_ctx();
  }

  void set_mixed_dp_ctx() {
    global_pg_ = std::make_unique<test::MockProcessGroup>(
        options_.device(), /*rank=*/0, /*world_size=*/2);
    dp_pg_ = std::make_unique<test::MockProcessGroup>(
        options_.device(), /*rank=*/0, /*world_size=*/2);
    tp_pg_ = std::make_unique<test::MockProcessGroup>(
        options_.device(), /*rank=*/0, /*world_size=*/1);

    parallel_args_ = ParallelArgs(
        /*rank=*/0, /*world_size=*/2, /*dp_size=*/2, global_pg_.get());
    parallel_args_.ep_size_ = 2;
    parallel_args_.process_group_ = global_pg_.get();
    parallel_args_.dp_local_process_group_ = dp_pg_.get();
    parallel_args_.tp_group_ = tp_pg_.get();
    parallel_args_.single_rank_group_ = tp_pg_.get();
    parallel_args_.cp_group_ = tp_pg_.get();
    parallel_args_.moe_ep_group_ = global_pg_.get();
    parallel_args_.moe_tp_group_ = tp_pg_.get();
    refresh_ctx();
  }

  ProcessGroup* build_single_rank_pg() {
    single_rank_pg_ = std::make_unique<test::MockProcessGroup>(
        options_.device(), /*rank=*/0, /*world_size=*/1);
    return single_rank_pg_.get();
  }

  void set_sp_ctx(DecoderHolder& decoder,
                  ProcessGroup* process_group = nullptr,
                  std::vector<int32_t> tokens_per_rank = {2, 1},
                  std::vector<int32_t> padded_tokens_per_rank = {2, 2}) {
    if (process_group == nullptr) {
      sp_pg_ = std::make_unique<test::MockProcessGroup>(
          options_.device(),
          /*rank=*/0,
          static_cast<int64_t>(tokens_per_rank.size()));
      process_group = sp_pg_.get();
    } else {
      sp_pg_.reset();
    }
    sp_ctx_ = {};
    sp_ctx_.rank = 0;
    sp_ctx_.process_group = process_group;
    sp_ctx_.comm_plan.tokens_per_rank = std::move(tokens_per_rank);
    sp_ctx_.comm_plan.padded_tokens_per_rank =
        std::move(padded_tokens_per_rank);
    CHECK_EQ(process_group->world_size(),
             static_cast<int64_t>(sp_ctx_.comm_plan.tokens_per_rank.size()));
    sp_ctx_.comm_plan.token_num_offset = 0;
    sp_ctx_.comm_plan.ffn_can_rs =
        v32_sp::can_ffn_rs(sp_ctx_.comm_plan.tokens_per_rank);
    decoder->set_context_parallel_context(&sp_ctx_);
  }

  ModelInputParams build_prefill_params(int64_t batch_size, int64_t seq_len) {
    ModelInputParams input_params;
    input_params.meta.batch_forward_type = BatchForwardType::PREFILL;
    input_params.meta.num_sequences = batch_size;
    input_params.meta.q_max_seq_len = seq_len;
    input_params.meta.kv_max_seq_len = batch_size * seq_len;
    input_params.attention.device.q_seq_lens = torch::arange(
        0,
        (batch_size + 1) * seq_len,
        seq_len,
        torch::TensorOptions().dtype(torch::kInt32).device(options_.device()));
    input_params.attention.device.kv_seq_lens = torch::arange(
        0,
        (batch_size + 1) * seq_len,
        seq_len,
        torch::TensorOptions().dtype(torch::kInt32).device(options_.device()));
    input_params.attention.host.q_seq_lens.resize(batch_size, seq_len);
    input_params.attention.host.kv_seq_lens.resize(batch_size, seq_len);
    input_params.attention.device.new_cache_slots = torch::arange(
        1,
        batch_size * seq_len + 1,
        torch::TensorOptions().dtype(torch::kInt32).device(options_.device()));
    input_params.attention.device.block_tables = torch::zeros(
        {batch_size, batch_size * seq_len},
        torch::TensorOptions().dtype(torch::kInt32).device(options_.device()));

    for (int64_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
      const int64_t start = batch_idx * seq_len + 1;
      input_params.attention.device.block_tables[batch_idx].index_put_(
          {torch::indexing::Slice(0, seq_len)},
          torch::arange(start,
                        start + seq_len,
                        torch::TensorOptions()
                            .dtype(torch::kInt32)
                            .device(options_.device())));
    }

    return input_params.to(options_.device());
  }

  KVCache build_indexed_cache(torch::Tensor key_cache,
                              torch::Tensor index_cache) const {
    return KVCache(IndexedKVCacheTensors{
        KVCacheTensors{key_cache, torch::Tensor()}, index_cache});
  }

  KVCache build_cache(int64_t block_num, int64_t block_size) const {
    auto k_cache = torch::zeros(
        {block_num,
         1,
         block_size,
         model_args_.qk_rope_head_dim() + model_args_.kv_lora_rank()},
        options_);
    auto index_cache = torch::zeros(
        {block_num, 1, block_size, model_args_.index_head_dim()}, options_);
    return build_indexed_cache(k_cache, index_cache);
  }

  //   KVCache build_quant_cache(const std::string& seed_prefix,
  //                             int64_t block_num,
  //                             int64_t block_size) const {
  //     auto k_cache = test::seeded_tensor(
  //         seed_prefix + ".k_cache",
  //         {block_num,
  //          1,
  //          block_size,
  //          model_args_.qk_rope_head_dim() + model_args_.kv_lora_rank()},
  //         torch::kInt8,
  //         options_.device());
  //     auto k_cache_scale = test::seeded_tensor(seed_prefix + ".k_scale",
  //                                              {block_num, 1, block_size},
  //                                              torch::kFloat32,
  //                                              options_.device());
  //     auto index_cache = test::seeded_tensor(
  //         seed_prefix + ".index_cache",
  //         {block_num, 1, block_size, model_args_.index_head_dim()},
  //         torch::kBFloat16,
  //         options_.device());
  //     return KVCache(
  //         k_cache, torch::Tensor(), index_cache, k_cache_scale,
  //         torch::Tensor());
  //   }

  void verify_prefix(const torch::Tensor& output,
                     const std::vector<float>& expected_values) const {
    const auto output_prefix = output.flatten()
                                   .to(torch::kFloat32)
                                   .cpu()
                                   .slice(0, 0, expected_values.size());
    test::verify_tensor_close(
        output_prefix,
        torch::tensor(expected_values,
                      torch::TensorOptions().dtype(torch::kFloat32)),
        1e-3,
        1e-4);
  }

  void sync_dev() const {
    xllm::Device(options_.device()).synchronize_default_stream();
  }

  ModelArgs model_args_;
  QuantArgs quant_args_;
  ParallelArgs parallel_args_{0, 1, nullptr};
  torch::TensorOptions options_;
  std::unique_ptr<xllm::ProcessGroup> mock_process_group_;
  std::unique_ptr<test::MockProcessGroup> global_pg_;
  std::unique_ptr<test::MockProcessGroup> dp_pg_;
  std::unique_ptr<test::MockProcessGroup> tp_pg_;
  std::unique_ptr<test::MockProcessGroup> single_rank_pg_;
  std::unique_ptr<test::MockProcessGroup> sp_pg_;
  ModelContext context_{};
  v32_cp::DeepseekV32CPContext sp_ctx_{};
};

namespace {

class CountingProcessGroup : public test::MockProcessGroup {
 public:
  CountingProcessGroup(const torch::Device& device,
                       int64_t rank,
                       int64_t world_size)
      : test::MockProcessGroup(device, rank, world_size) {}

  void allreduce(torch::Tensor& input) override {
    ++allreduce_calls_;
    test::MockProcessGroup::allreduce(input);
  }

  void reduce_scatter(const torch::Tensor& input,
                      torch::Tensor& output) override {
    ++rs_calls_;
    test::MockProcessGroup::reduce_scatter(input, output);
  }

  int allreduce_calls() const { return allreduce_calls_; }
  int rs_calls() const { return rs_calls_; }

 private:
  int allreduce_calls_ = 0;
  int rs_calls_ = 0;
};

struct LayerCase {
  const char* name;
  int32_t layer_id;
  std::vector<float> expected_prefix;
};

enum class CarrierEnv {
  kBase,
  kCountingTp,
  kTpDp,
  kTpOnly,
};

struct CarrierCase {
  const char* name;
  CarrierEnv env;
  int64_t tp_world_size;
  int64_t ep_size;
  DeepseekV2AttentionImpl::PostAttnLayout attn_layout;
  bool need_dp_gather;
  bool enable_moe_all2all;
  int64_t rows;
  std::vector<float> attn_out;
  std::vector<float> residual;
  std::vector<int> dp_global_token_nums;
  std::vector<float> tp_full_tokens;
  DeepseekV2DecoderLayerTestPeer::Mode mode;
  bool pad_active;
  int64_t pad_org_tokens;
  int64_t pad_tokens;
  std::vector<float> expected_ffn_in;
  std::vector<float> expected_skip_local;
  int expected_allreduce_calls;
  int expected_rs_calls;
};

struct QuantCase {
  const char* name;
  const char* seed_prefix;
  bool check_stats;
};

class ExpertDegreeGuard {
 public:
  explicit ExpertDegreeGuard(int32_t value)
      : saved_(EPLBConfig::get_instance().expert_parallel_degree()) {
    EPLBConfig::get_instance().expert_parallel_degree(value);
  }

  ~ExpertDegreeGuard() {
    EPLBConfig::get_instance().expert_parallel_degree(saved_);
  }

 private:
  int32_t saved_;
};

std::string case_name(const ::testing::TestParamInfo<LayerCase>& info) {
  return info.param.name;
}

std::string carrier_name(const ::testing::TestParamInfo<CarrierCase>& info) {
  return info.param.name;
}

std::string quant_name(const ::testing::TestParamInfo<QuantCase>& info) {
  return info.param.name;
}

}  // namespace

class DeepseekV2DecoderLayerParamTest
    : public DeepseekV2DecoderLayerTest,
      public ::testing::WithParamInterface<LayerCase> {};

class DeepseekV2DecoderCarrierTest
    : public DeepseekV2DecoderLayerTest,
      public ::testing::WithParamInterface<CarrierCase> {
 protected:
  torch::Tensor mat(int64_t rows, const std::vector<float>& values) const {
    return torch::tensor(values, fp32_opts()).reshape({rows, 2});
  }

  CountingProcessGroup* set_counting_tp_ctx(int64_t world_size,
                                            int64_t ep_size) {
    global_pg_ = std::make_unique<test::MockProcessGroup>(
        options_.device(), /*rank=*/0, world_size);
    auto tp_pg = std::make_unique<CountingProcessGroup>(
        options_.device(), /*rank=*/0, world_size);
    auto* tp_pg_raw = tp_pg.get();
    tp_pg_ = std::move(tp_pg);
    dp_pg_.reset();

    parallel_args_ =
        ParallelArgs(/*rank=*/0, world_size, /*dp_size=*/1, global_pg_.get());
    parallel_args_.ep_size_ = ep_size;
    parallel_args_.process_group_ = global_pg_.get();
    parallel_args_.dp_local_process_group_ = nullptr;
    parallel_args_.tp_group_ = tp_pg_.get();
    refresh_ctx();
    return tp_pg_raw;
  }

  CountingProcessGroup* init_env(const CarrierCase& tc) {
    switch (tc.env) {
      case CarrierEnv::kBase:
        return nullptr;
      case CarrierEnv::kCountingTp:
        return set_counting_tp_ctx(tc.tp_world_size, tc.ep_size);
      case CarrierEnv::kTpDp:
        set_tp_dp_ctx(
            /*world_size=*/4, /*dp_size=*/2, /*tp_size=*/2, tc.ep_size);
        return nullptr;
      case CarrierEnv::kTpOnly:
        set_tp_ctx(tc.tp_world_size, tc.ep_size);
        return nullptr;
    }

    return nullptr;
  }

  void set_tp_full_tokens(const CarrierCase& tc) {
    if (tc.env != CarrierEnv::kTpDp || tc.tp_full_tokens.empty()) {
      return;
    }

    auto full_tokens = mat(/*rows=*/4, tc.tp_full_tokens);
    tp_pg_->set_allgather_outputs(
        {full_tokens.slice(0, 0, 2), full_tokens.slice(0, 2, 4)});
  }
};

TEST_P(DeepseekV2DecoderLayerParamTest,
       Constructor_WhenCreated_ThenRegistersExpectedSubmodules) {
  const auto decoder = make_decoder(GetParam().layer_id);

  const auto child_names = get_child_module_names(*decoder);
  EXPECT_TRUE(child_names.count("self_attn")) << "self_attn missing";
  EXPECT_TRUE(child_names.count("input_layernorm"))
      << "input_layernorm missing";
  EXPECT_TRUE(child_names.count("post_attention_layernorm"))
      << "post_attention_layernorm missing";
  EXPECT_TRUE(child_names.count("mlp")) << "mlp missing";
}

TEST_F(DeepseekV2DecoderLayerTest,
       ConstructorAllowsDpOneEpWorldWhenDeepEpEnabled) {
  ExpertDegreeGuard guard(/*value=*/2);
  parallel_args_.dp_size() = 1;
  parallel_args_.ep_size() = parallel_args_.world_size();
  refresh_ctx();

  EXPECT_NO_THROW((void)make_decoder(/*layer_id=*/0));
}

TEST_P(DeepseekV2DecoderLayerParamTest,
       LoadStateDict_WhenWeightsMatch_ThenSucceeds) {
  auto decoder = make_decoder(GetParam().layer_id);
  auto weight_dict = create_test_weights(GetParam().layer_id);
  StateDict state_dict(weight_dict);

  EXPECT_NO_THROW(decoder->load_state_dict(state_dict));
}

TEST_P(DeepseekV2DecoderCarrierTest,
       BuildPostAttnCarrier_WhenScenarioChanges_ThenMatchesExpectedState) {
  const auto& tc = GetParam();
  auto* tp_pg_raw = init_env(tc);
  auto decoder = make_decoder(/*layer_id=*/0);
  ModelInputParams input_params;
  input_params.parallel.dp_global_token_nums = tc.dp_global_token_nums;
  set_tp_full_tokens(tc);

  auto carrier = DeepseekV2DecoderLayerTestPeer::build_post_attn_carrier(
      *decoder,
      mat(tc.rows, tc.attn_out),
      mat(tc.rows, tc.residual),
      input_params,
      tc.attn_layout,
      tc.need_dp_gather,
      tc.enable_moe_all2all);

  EXPECT_EQ(carrier.mode, tc.mode);
  EXPECT_EQ(carrier.pad_info.active, tc.pad_active);
  EXPECT_EQ(carrier.pad_info.original_tokens, tc.pad_org_tokens);
  EXPECT_EQ(carrier.pad_info.padded_tokens, tc.pad_tokens);
  test::verify_tensor_close(
      carrier.ffn_in, mat(tc.expected_ffn_in.size() / 2, tc.expected_ffn_in));
  test::verify_tensor_close(
      carrier.skip_local,
      mat(tc.expected_skip_local.size() / 2, tc.expected_skip_local));
  if (tp_pg_raw != nullptr) {
    EXPECT_EQ(tp_pg_raw->allreduce_calls(), tc.expected_allreduce_calls);
    EXPECT_EQ(tp_pg_raw->rs_calls(), tc.expected_rs_calls);
  }
}

TEST_F(DeepseekV2DecoderLayerTest, BuildPostAttnCarrierPackedLocal) {
  auto decoder = make_loaded_decoder(/*layer_id=*/0);
  set_sp_ctx(decoder);

  auto attn_out =
      torch::full({2, model_args_.hidden_size()}, 1.0f, hidden_opts());
  auto residual =
      torch::full({2, model_args_.hidden_size()}, 2.0f, hidden_opts());
  auto expected_skip =
      torch::full({2, model_args_.hidden_size()}, 3.0f, hidden_opts());
  auto expected_local_norm = DeepseekV2DecoderLayerTestPeer::post_norm(
      *decoder, expected_skip.clone());
  auto remote_norm =
      torch::full({2, model_args_.hidden_size()}, 5.0f, hidden_opts());
  sp_pg_->set_allgather_outputs({expected_local_norm, remote_norm});

  ModelInputParams input_params;
  auto carrier = DeepseekV2DecoderLayerTestPeer::build_post_attn_carrier(
      *decoder,
      attn_out,
      residual,
      input_params,
      DeepseekV2AttentionImpl::PostAttnLayout::kPackedLocal,
      /*need_dp_gather=*/false,
      /*enable_moe_all2all=*/false);

  EXPECT_EQ(carrier.mode, DeepseekV2DecoderLayerTestPeer::Mode::kPackedLocal);
  EXPECT_FALSE(carrier.pad_info.active);
  test::verify_tensor_close(carrier.skip_local, expected_skip);
  test::verify_tensor_close(
      carrier.ffn_in,
      torch::cat({expected_local_norm, remote_norm.slice(0, 0, 1)}, 0));
}

TEST_F(DeepseekV2DecoderLayerTest, RestoreFfnOutputReplicated) {
  auto decoder = make_decoder(/*layer_id=*/0);
  ModelInputParams input_params;

  auto attn_out = torch::tensor(
      {{1.0f, 2.0f}, {3.0f, 4.0f}},
      torch::TensorOptions().dtype(torch::kFloat32).device(options_.device()));
  auto residual = torch::tensor(
      {{10.0f, 20.0f}, {30.0f, 40.0f}},
      torch::TensorOptions().dtype(torch::kFloat32).device(options_.device()));
  auto carrier = DeepseekV2DecoderLayerTestPeer::build_post_attn_carrier(
      *decoder,
      attn_out,
      residual,
      input_params,
      DeepseekV2AttentionImpl::PostAttnLayout::kTpShard,
      /*need_dp_gather=*/false,
      /*enable_moe_all2all=*/false);

  auto ffn_out = torch::tensor(
      {{100.0f, 200.0f}, {300.0f, 400.0f}},
      torch::TensorOptions().dtype(torch::kFloat32).device(options_.device()));
  auto restored = DeepseekV2DecoderLayerTestPeer::restore_ffn_output(
      *decoder, ffn_out, carrier, input_params);

  auto expected = torch::tensor(
      {{111.0f, 222.0f}, {333.0f, 444.0f}},
      torch::TensorOptions().dtype(torch::kFloat32).device(options_.device()));
  test::verify_tensor_close(restored, expected);
}

TEST_F(DeepseekV2DecoderLayerTest, RestoreFfnOutputPackedLocal) {
  auto decoder = make_loaded_decoder(/*layer_id=*/0);
  set_sp_ctx(decoder);

  auto attn_out =
      torch::full({2, model_args_.hidden_size()}, 1.0f, hidden_opts());
  auto residual =
      torch::full({2, model_args_.hidden_size()}, 2.0f, hidden_opts());
  auto expected_skip =
      torch::full({2, model_args_.hidden_size()}, 3.0f, hidden_opts());
  auto expected_local_norm = DeepseekV2DecoderLayerTestPeer::post_norm(
      *decoder, expected_skip.clone());
  auto remote_norm =
      torch::full({2, model_args_.hidden_size()}, 5.0f, hidden_opts());
  sp_pg_->set_allgather_outputs(
      std::vector<torch::Tensor>{expected_local_norm, remote_norm});

  ModelInputParams input_params;
  auto carrier = DeepseekV2DecoderLayerTestPeer::build_post_attn_carrier(
      *decoder,
      attn_out,
      residual,
      input_params,
      DeepseekV2AttentionImpl::PostAttnLayout::kPackedLocal,
      /*need_dp_gather=*/false,
      /*enable_moe_all2all=*/false);

  auto packed_ffn_out =
      torch::full({3, model_args_.hidden_size()}, 7.0f, hidden_opts());
  auto restored = DeepseekV2DecoderLayerTestPeer::restore_ffn_output(
      *decoder, packed_ffn_out, carrier, input_params);
  test::verify_tensor_close(
      restored,
      torch::full({2, model_args_.hidden_size()}, 10.0f, hidden_opts()));
}

TEST_F(DeepseekV2DecoderLayerTest, CanLocalOutPackedLocalNeedsEqualTokens) {
  global_pg_ = std::make_unique<test::MockProcessGroup>(
      options_.device(), /*rank=*/0, /*world_size=*/2);
  auto tp_pg = std::make_unique<CountingProcessGroup>(
      options_.device(), /*rank=*/0, /*world_size=*/2);
  auto* tp_pg_raw = tp_pg.get();
  tp_pg_ = std::move(tp_pg);

  parallel_args_ = ParallelArgs(
      /*rank=*/0, /*world_size=*/2, /*dp_size=*/1, global_pg_.get());
  parallel_args_.process_group_ = global_pg_.get();
  parallel_args_.tp_group_ = tp_pg_.get();
  refresh_ctx();

  auto decoder = make_decoder(/*layer_id=*/0);
  auto carrier = DeepseekV2DecoderLayerTestPeer::make_carrier(
      DeepseekV2DecoderLayerTestPeer::Mode::kPackedLocal);

  set_sp_ctx(decoder, tp_pg_.get(), {2, 2}, {2, 2});
  EXPECT_TRUE(DeepseekV2DecoderLayerTestPeer::can_keep_local_output(
      *decoder, carrier, tp_pg_.get()));

  set_sp_ctx(decoder, tp_pg_.get(), {2, 1}, {2, 2});
  EXPECT_FALSE(DeepseekV2DecoderLayerTestPeer::can_keep_local_output(
      *decoder, carrier, tp_pg_.get()));

  EXPECT_EQ(tp_pg_raw->allreduce_calls(), 0);
  EXPECT_EQ(tp_pg_raw->rs_calls(), 0);
}

TEST_F(DeepseekV2DecoderLayerTest, CanLocalOutPackedLocalAllowsGlobalAlias) {
  global_pg_ = std::make_unique<test::MockProcessGroup>(
      options_.device(), /*rank=*/0, /*world_size=*/2);
  tp_pg_ = std::make_unique<test::MockProcessGroup>(
      options_.device(), /*rank=*/0, /*world_size=*/2);

  parallel_args_ = ParallelArgs(
      /*rank=*/0, /*world_size=*/2, /*dp_size=*/1, global_pg_.get());
  parallel_args_.process_group_ = global_pg_.get();
  parallel_args_.tp_group_ = tp_pg_.get();
  refresh_ctx();

  auto decoder = make_decoder(/*layer_id=*/0);
  auto carrier = DeepseekV2DecoderLayerTestPeer::make_carrier(
      DeepseekV2DecoderLayerTestPeer::Mode::kPackedLocal);

  set_sp_ctx(decoder, tp_pg_.get(), {2, 2}, {2, 2});
  EXPECT_TRUE(DeepseekV2DecoderLayerTestPeer::can_keep_local_output(
      *decoder, carrier, global_pg_.get()));
}

TEST_F(DeepseekV2DecoderLayerTest, CommOutPackedLocalReduceScatters) {
  global_pg_ = std::make_unique<test::MockProcessGroup>(
      options_.device(), /*rank=*/0, /*world_size=*/2);
  auto tp_pg = std::make_unique<CountingProcessGroup>(
      options_.device(), /*rank=*/0, /*world_size=*/2);
  auto* tp_pg_raw = tp_pg.get();
  tp_pg_ = std::move(tp_pg);

  parallel_args_ = ParallelArgs(
      /*rank=*/0, /*world_size=*/2, /*dp_size=*/1, global_pg_.get());
  parallel_args_.process_group_ = global_pg_.get();
  parallel_args_.tp_group_ = tp_pg_.get();
  refresh_ctx();

  auto decoder = make_decoder(/*layer_id=*/0);
  set_sp_ctx(decoder, tp_pg_.get(), {2, 2}, {2, 2});
  auto carrier = DeepseekV2DecoderLayerTestPeer::make_carrier(
      DeepseekV2DecoderLayerTestPeer::Mode::kPackedLocal);
  auto packed_out = torch::tensor(
      {{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}, {7.0f, 8.0f}}, fp32_opts());

  auto local_out = DeepseekV2DecoderLayerTestPeer::comm_out(
      *decoder, packed_out, carrier, tp_pg_.get());

  test::verify_tensor_close(local_out, packed_out.slice(0, 0, 2));
  EXPECT_EQ(tp_pg_raw->allreduce_calls(), 0);
  EXPECT_EQ(tp_pg_raw->rs_calls(), 1);
}

TEST_F(DeepseekV2DecoderLayerTest, CommOutPackedLocalSlicesSingleRankOutput) {
  auto decoder = make_decoder(/*layer_id=*/0);
  set_sp_ctx(decoder, nullptr, {2, 2}, {2, 2});
  auto carrier = DeepseekV2DecoderLayerTestPeer::make_carrier(
      DeepseekV2DecoderLayerTestPeer::Mode::kPackedLocal);
  test::MockProcessGroup single_rank_pg(
      options_.device(), /*rank=*/0, /*world_size=*/1);
  auto packed_out = torch::tensor(
      {{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}, {7.0f, 8.0f}}, fp32_opts());

  auto local_out = DeepseekV2DecoderLayerTestPeer::comm_out(
      *decoder, packed_out, carrier, &single_rank_pg);

  test::verify_tensor_close(local_out, packed_out.slice(0, 0, 2));
}

TEST_F(DeepseekV2DecoderLayerTest, CommOutPackedLocalFallsBackOnPgMismatch) {
  global_pg_ = std::make_unique<test::MockProcessGroup>(
      options_.device(), /*rank=*/0, /*world_size=*/2);
  tp_pg_ = std::make_unique<test::MockProcessGroup>(
      options_.device(), /*rank=*/0, /*world_size=*/2);
  auto mismatch_pg = std::make_unique<CountingProcessGroup>(
      options_.device(), /*rank=*/0, /*world_size=*/4);
  auto* mismatch_pg_raw = mismatch_pg.get();

  parallel_args_ = ParallelArgs(
      /*rank=*/0, /*world_size=*/2, /*dp_size=*/1, global_pg_.get());
  parallel_args_.process_group_ = global_pg_.get();
  parallel_args_.tp_group_ = tp_pg_.get();
  refresh_ctx();

  auto decoder = make_decoder(/*layer_id=*/0);
  set_sp_ctx(decoder, tp_pg_.get(), {2, 2}, {2, 2});
  auto carrier = DeepseekV2DecoderLayerTestPeer::make_carrier(
      DeepseekV2DecoderLayerTestPeer::Mode::kPackedLocal);
  auto packed_out = torch::tensor(
      {{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}, {7.0f, 8.0f}}, fp32_opts());

  auto local_out = DeepseekV2DecoderLayerTestPeer::comm_out(
      *decoder, packed_out, carrier, mismatch_pg.get());

  EXPECT_FALSE(DeepseekV2DecoderLayerTestPeer::can_keep_local_output(
      *decoder, carrier, mismatch_pg.get()));
  test::verify_tensor_close(local_out, packed_out);
  EXPECT_EQ(mismatch_pg_raw->allreduce_calls(), 1);
  EXPECT_EQ(mismatch_pg_raw->rs_calls(), 0);
}

TEST_F(DeepseekV2DecoderLayerTest, PackedLocalFastAddsSkipLocally) {
  auto decoder = make_decoder(/*layer_id=*/0);
  auto carrier = DeepseekV2DecoderLayerTestPeer::make_carrier(
      DeepseekV2DecoderLayerTestPeer::Mode::kPackedLocal,
      torch::tensor({{10.0f, 20.0f}, {30.0f, 40.0f}}, fp32_opts()));
  auto local_out = torch::tensor({{1.0f, 2.0f}, {3.0f, 4.0f}}, fp32_opts());

  test::verify_tensor_close(
      local_out + carrier.skip_local,
      torch::tensor({{11.0f, 22.0f}, {33.0f, 44.0f}}, fp32_opts()));
}

TEST_F(DeepseekV2DecoderLayerTest, ForwardMixedDpMoEReturnsLocalSlice) {
  const int32_t layer_id =
      std::max<int32_t>(5, model_args_.first_k_dense_replace());
  ExpertDegreeGuard guard(/*value=*/0);
  set_mixed_dp_ctx();

  auto decoder = make_loaded_decoder(layer_id);

  auto hidden_states =
      xllm::layer::test::seeded_tensor("mixed_dp_moe.hidden_states",
                                       {2, model_args_.hidden_size()},
                                       torch::kBFloat16,
                                       options_.device());
  auto positions = torch::arange(
      0,
      2,
      torch::TensorOptions().dtype(torch::kInt32).device(options_.device()));

  ModelInputParams input_params;
  input_params.meta.batch_forward_type = BatchForwardType::PREFILL;
  input_params.meta.num_sequences = 2;
  input_params.meta.q_max_seq_len = 1;
  input_params.meta.kv_max_seq_len = 2;
  input_params.attention.device.q_seq_lens = torch::tensor(
      {0, 1, 2},
      torch::TensorOptions().dtype(torch::kInt32).device(options_.device()));
  input_params.attention.device.kv_seq_lens = torch::tensor(
      {0, 1, 2},
      torch::TensorOptions().dtype(torch::kInt32).device(options_.device()));
  input_params.attention.host.q_seq_lens = {1, 1};
  input_params.attention.host.kv_seq_lens = {1, 1};
  input_params.attention.device.new_cache_slots = torch::arange(
      1,
      3,
      torch::TensorOptions().dtype(torch::kInt32).device(options_.device()));
  input_params.attention.device.block_tables = torch::tensor(
      {{1, 0}, {2, 0}},
      torch::TensorOptions().dtype(torch::kInt32).device(options_.device()));
  input_params.parallel.dp_global_token_nums = {2, 1};
  input_params.parallel.dp_is_decode = {0, 0};
  input_params = input_params.to(options_.device());

  auto attn_metadata =
      AttentionMetadataBuilder::build(input_params, /*enable_mla=*/true);
  auto k_cache = torch::zeros(
      {2048, 1, 1, model_args_.qk_rope_head_dim() + model_args_.kv_lora_rank()},
      options_);
  auto index_cache =
      torch::zeros({2048, 1, 1, model_args_.index_head_dim()}, options_);
  KVCache kv_cache =
      build_indexed_cache(std::move(k_cache), std::move(index_cache));

  std::optional<torch::Tensor> residual = std::nullopt;
  auto output = decoder->forward(hidden_states,
                                 residual,
                                 positions,
                                 attn_metadata,
                                 kv_cache,
                                 input_params);

  sync_dev();

  ASSERT_EQ(output.dim(), 2);
  EXPECT_EQ(output.size(0), 2);
  EXPECT_EQ(output.size(1), model_args_.hidden_size());
  EXPECT_FALSE(residual.has_value());
}

TEST_F(DeepseekV2DecoderLayerTest, DenseMlpReductionMovesToDecoder) {
  global_pg_ = std::make_unique<test::MockProcessGroup>(
      options_.device(), /*rank=*/0, /*world_size=*/2);
  auto tp_pg = std::make_unique<CountingProcessGroup>(
      options_.device(), /*rank=*/0, /*world_size=*/2);
  auto* tp_pg_raw = tp_pg.get();
  tp_pg_ = std::move(tp_pg);

  parallel_args_ = ParallelArgs(
      /*rank=*/0, /*world_size=*/2, /*dp_size=*/1, global_pg_.get());
  parallel_args_.process_group_ = global_pg_.get();
  parallel_args_.tp_group_ = tp_pg_.get();
  refresh_ctx();

  auto decoder = make_loaded_decoder(/*layer_id=*/0);
  auto hidden_states = test::seeded_tensor("deepseek_v2_decoder.dense_ffn",
                                           {4, model_args_.hidden_size()},
                                           torch::kBFloat16,
                                           options_.device());

  auto local_out =
      DeepseekV2DecoderLayerTestPeer::mlp(*decoder)->forward(hidden_states);
  sync_dev();

  EXPECT_EQ(tp_pg_raw->allreduce_calls(), 0);
  ASSERT_EQ(local_out.size(0), hidden_states.size(0));
  ASSERT_EQ(local_out.size(1), model_args_.hidden_size());

  auto reduced_out = DeepseekV2DecoderLayerTestPeer::reduce_out(
      *decoder, local_out, parallel_args_.tp_group_);
  sync_dev();

  EXPECT_EQ(tp_pg_raw->allreduce_calls(), 1);
  EXPECT_EQ(reduced_out.sizes(), local_out.sizes());
}

TEST_P(DeepseekV2DecoderLayerParamTest,
       Forward_WhenPrefill_ThenMatchesReferencePrefix) {
  constexpr int64_t kBatchSize = 4;
  constexpr int64_t kSeqLen = 4;
  const int64_t block_num = kBatchSize * kSeqLen + 1;

  auto decoder = make_loaded_decoder(GetParam().layer_id);
  auto hidden_states = xllm::layer::test::seeded_tensor(
      "hidden_states",
      {kBatchSize * kSeqLen, model_args_.hidden_size()},
      torch::kBFloat16,
      options_.device());
  auto positions = torch::arange(
      0,
      kBatchSize * kSeqLen,
      torch::TensorOptions().dtype(torch::kInt32).device(options_.device()));
  auto input_params = build_prefill_params(kBatchSize, kSeqLen);
  auto attn_metadata =
      AttentionMetadataBuilder::build(input_params, /*enable_mla=*/true);
  auto kv_cache = build_cache(block_num, /*block_size=*/1);

  std::optional<torch::Tensor> residual = std::nullopt;
  auto output = decoder->forward(hidden_states,
                                 residual,
                                 positions,
                                 attn_metadata,
                                 kv_cache,
                                 input_params);

  sync_dev();

  ASSERT_EQ(output.dim(), 2);
  EXPECT_EQ(output.size(0), kBatchSize * kSeqLen);
  EXPECT_EQ(output.size(1), model_args_.hidden_size());
  verify_prefix(output, GetParam().expected_prefix);
}

class DeepseekV2DecoderKvCacheTest
    : public DeepseekV2DecoderLayerTest,
      public ::testing::WithParamInterface<QuantCase> {};

// TEST_P(DeepseekV2DecoderKvCacheTest,
//        KvCache_WhenKeyCacheQuantized_ThenPreservesMetadata) {
//   constexpr int64_t kBlockNum = 100;
//   constexpr int64_t kBlockSize = 16;
//   auto quant_kv_cache =
//       build_quant_cache(GetParam().seed_prefix, kBlockNum, kBlockSize);

//   const auto retrieved_k_scale = quant_kv_cache.get_k_cache_scale();
//   ASSERT_TRUE(retrieved_k_scale.has_value());
//   EXPECT_EQ(retrieved_k_scale.value().sizes(),
//             torch::IntArrayRef({kBlockNum, 1, kBlockSize}));
//   EXPECT_EQ(retrieved_k_scale.value().scalar_type(), torch::kFloat32);

//   const auto index_cache = quant_kv_cache.get_index_cache();
//   EXPECT_TRUE(index_cache.defined());
//   EXPECT_EQ(index_cache.scalar_type(), torch::kBFloat16);
//   EXPECT_NE(index_cache.scalar_type(), torch::kInt8);

//   if (GetParam().check_stats) {
//     test::expect_tensor_stats(retrieved_k_scale.value(),
//                               /*expected_min=*/0.000686,
//                               /*expected_max=*/0.999,
//                               /*expected_sum=*/803.8);
//   }

//   const auto retrieved_v_scale = quant_kv_cache.get_v_cache_scale();
//   EXPECT_TRUE(!retrieved_v_scale.has_value() ||
//               retrieved_v_scale.value().numel() == 0);
// }

INSTANTIATE_TEST_SUITE_P(
    CarrierCases,
    DeepseekV2DecoderCarrierTest,
    ::testing::Values(
        CarrierCase{"Replicated_NoShard_KeepReplicated",
                    CarrierEnv::kBase,
                    /*tp_world_size=*/1,
                    /*ep_size=*/1,
                    DeepseekV2AttentionImpl::PostAttnLayout::kReplicated,
                    /*need_dp_gather=*/false,
                    /*enable_moe_all2all=*/false,
                    /*rows=*/2,
                    {1.0f, 2.0f, 3.0f, 4.0f},
                    {10.0f, 20.0f, 30.0f, 40.0f},
                    {},
                    {},
                    DeepseekV2DecoderLayerTestPeer::Mode::kReplicated,
                    /*pad_active=*/false,
                    /*pad_org_tokens=*/0,
                    /*pad_tokens=*/0,
                    {11.0f, 22.0f, 33.0f, 44.0f},
                    {11.0f, 22.0f, 33.0f, 44.0f},
                    /*expected_allreduce_calls=*/0,
                    /*expected_rs_calls=*/0},
        CarrierCase{"Replicated_TpShardInput_ReducesOnce",
                    CarrierEnv::kCountingTp,
                    /*tp_world_size=*/2,
                    /*ep_size=*/2,
                    DeepseekV2AttentionImpl::PostAttnLayout::kTpShard,
                    /*need_dp_gather=*/false,
                    /*enable_moe_all2all=*/false,
                    /*rows=*/2,
                    {1.0f, 2.0f, 3.0f, 4.0f},
                    {10.0f, 20.0f, 30.0f, 40.0f},
                    {},
                    {},
                    DeepseekV2DecoderLayerTestPeer::Mode::kReplicated,
                    /*pad_active=*/false,
                    /*pad_org_tokens=*/0,
                    /*pad_tokens=*/0,
                    {11.0f, 22.0f, 33.0f, 44.0f},
                    {11.0f, 22.0f, 33.0f, 44.0f},
                    /*expected_allreduce_calls=*/1,
                    /*expected_rs_calls=*/0},
        CarrierCase{"Replicated_FullAttnOutput_SkipsReduce",
                    CarrierEnv::kCountingTp,
                    /*tp_world_size=*/2,
                    /*ep_size=*/2,
                    DeepseekV2AttentionImpl::PostAttnLayout::kReplicated,
                    /*need_dp_gather=*/false,
                    /*enable_moe_all2all=*/false,
                    /*rows=*/2,
                    {1.0f, 2.0f, 3.0f, 4.0f},
                    {10.0f, 20.0f, 30.0f, 40.0f},
                    {},
                    {},
                    DeepseekV2DecoderLayerTestPeer::Mode::kReplicated,
                    /*pad_active=*/false,
                    /*pad_org_tokens=*/0,
                    /*pad_tokens=*/0,
                    {11.0f, 22.0f, 33.0f, 44.0f},
                    {11.0f, 22.0f, 33.0f, 44.0f},
                    /*expected_allreduce_calls=*/0,
                    /*expected_rs_calls=*/0}),
    carrier_name);

INSTANTIATE_TEST_SUITE_P(
    DenseAndMoe,
    DeepseekV2DecoderLayerParamTest,
    ::testing::Values(
        LayerCase{"Dense", 0, {2352.0f, 2352.0f, 2352.0f, 2352.0f, 2352.0f}},
        LayerCase{"Moe", 5, {0.7773f, 0.7227f, 1.0938f, 1.1875f, 0.6367f}}),
    case_name);

namespace {

// Ground-truth indexer_types for GLM5.2-W4A8 (index_topk_freq=4,
// index_skip_topk_offset=3, 78 layers): 21 Full + 57 Shared. A Full layer owns
// indexer weights (layers 0/1 plus every 4th layer from 2); the rest are
// Shared and must reuse the previous Full layer's top-k.
bool is_full_indexer_layer_glm52(int32_t layer_id) {
  if (layer_id < 2) {
    return true;
  }
  return (layer_id - 2) % 4 == 0;
}

std::string build_fs_pattern_glm52(int32_t num_layers) {
  std::string pattern;
  pattern.reserve(num_layers);
  for (int32_t layer_id = 0; layer_id < num_layers; ++layer_id) {
    pattern.push_back(is_full_indexer_layer_glm52(layer_id) ? 'F' : 'S');
  }
  return pattern;
}

}  // namespace

class DeepseekV2DecoderTopkShareTest : public DeepseekV2DecoderLayerTest {
 protected:
  static constexpr int32_t kNumLayers = 78;

  void configure_glm5_indexer() {
    model_args_.model_type() = "glm_moe_dsa";
    model_args_.n_layers() = kNumLayers;
    model_args_.index_n_heads() = 4;
    model_args_.index_head_dim() = 128;
    model_args_.index_topk() = 64;
    // Keep every layer dense so construction only exercises attention/indexer.
    model_args_.first_k_dense_replace() = kNumLayers;
  }

  void use_cpu_constructor_context() {
    options_ = torch::TensorOptions()
                   .dtype(torch::kBFloat16)
                   .device(torch::kCPU)
                   .requires_grad(false);
    mock_process_group_ =
        std::make_unique<test::MockProcessGroup>(torch::Device(torch::kCPU));
    parallel_args_ = ParallelArgs(/*rank=*/0,
                                  /*world_size=*/1,
                                  /*dp_size=*/1,
                                  mock_process_group_.get());
    parallel_args_.tp_group_ = mock_process_group_.get();
    parallel_args_.single_rank_group_ = mock_process_group_.get();
    parallel_args_.cp_group_ = mock_process_group_.get();
    refresh_ctx();
  }
};

TEST_F(DeepseekV2DecoderTopkShareTest,
       DefaultConstructorWhenShareConfiguredKeepsOwnIndexer) {
  configure_glm5_indexer();
  model_args_.index_topk_freq() = 4;
  model_args_.index_skip_topk_offset() = 3;
  use_cpu_constructor_context();

  DecoderHolder decoder(DeepseekV2DecoderLayerImpl(context_, /*layer_id=*/3));

  const auto parameters = decoder->named_parameters(/*recurse=*/true);
  EXPECT_TRUE(parameters.contains("self_attn.indexer.wq_b.weight"));
  EXPECT_TRUE(parameters.contains("self_attn.indexer.wk.weight"));
  EXPECT_TRUE(parameters.contains("self_attn.indexer.weights_proj.weight"));
  EXPECT_TRUE(parameters.contains("self_attn.indexer.k_norm.weight"));
}

TEST_F(DeepseekV2DecoderTopkShareTest,
       FreqConfig_WhenGlm52_ThenPlanMatchesIndexerTypes) {
  configure_glm5_indexer();
  model_args_.index_topk_freq() = 4;
  model_args_.index_skip_topk_offset() = 3;
  model_args_.index_topk_pattern() = "";
  refresh_ctx();

  int32_t full_count = 0;
  int32_t shared_count = 0;
  for (int32_t layer_id = 0; layer_id < kNumLayers; ++layer_id) {
    auto decoder = make_decoder(layer_id);
    const DsaTopkShareDecision& decision =
        DeepseekV2DecoderLayerTestPeer::topk_share_decision(*decoder);
    const bool expect_full = is_full_indexer_layer_glm52(layer_id);
    const bool expect_output =
        expect_full && !is_full_indexer_layer_glm52(layer_id + 1);
    EXPECT_EQ(decision.reuse_topk, !expect_full) << "layer " << layer_id;
    EXPECT_EQ(decision.output_topk, expect_output) << "layer " << layer_id;
    if (expect_full) {
      ++full_count;
    } else {
      ++shared_count;
    }
  }
  EXPECT_EQ(full_count, 21);
  EXPECT_EQ(shared_count, 57);
}

TEST_F(DeepseekV2DecoderTopkShareTest,
       SharedLayerOwnsNoIndexerParametersOrConstants) {
  configure_glm5_indexer();
  model_args_.index_topk_freq() = 4;
  model_args_.index_skip_topk_offset() = 3;
  refresh_ctx();

  auto full_decoder = make_decoder(/*layer_id=*/2);
  auto shared_decoder = make_decoder(/*layer_id=*/3);

  EXPECT_TRUE(DeepseekV2DecoderLayerTestPeer::attention_has_child(*full_decoder,
                                                                  "indexer"));
  EXPECT_TRUE(DeepseekV2DecoderLayerTestPeer::attention_has_child(
      *full_decoder, "indexer_rotary_emb"));
  EXPECT_FALSE(DeepseekV2DecoderLayerTestPeer::attention_has_child(
      *shared_decoder, "indexer"));
  EXPECT_FALSE(DeepseekV2DecoderLayerTestPeer::attention_has_child(
      *shared_decoder, "indexer_rotary_emb"));

  const auto full_parameters = full_decoder->named_parameters(/*recurse=*/true);
  const auto shared_parameters =
      shared_decoder->named_parameters(/*recurse=*/true);
  EXPECT_TRUE(full_parameters.contains("self_attn.indexer.wq_b.weight"));
  EXPECT_TRUE(full_parameters.contains("self_attn.indexer.wk.weight"));
  EXPECT_TRUE(
      full_parameters.contains("self_attn.indexer.weights_proj.weight"));
  EXPECT_TRUE(full_parameters.contains("self_attn.indexer.k_norm.weight"));
  EXPECT_FALSE(shared_parameters.contains("self_attn.indexer.wq_b.weight"));
  EXPECT_FALSE(shared_parameters.contains("self_attn.indexer.wk.weight"));
  EXPECT_FALSE(
      shared_parameters.contains("self_attn.indexer.weights_proj.weight"));
  EXPECT_FALSE(shared_parameters.contains("self_attn.indexer.k_norm.weight"));
}

TEST_F(DeepseekV2DecoderTopkShareTest,
       Pattern_WhenEquivalentFsString_ThenPlanMatchesFreqConfig) {
  configure_glm5_indexer();
  model_args_.index_topk_freq() = 1;
  model_args_.index_skip_topk_offset() = 0;
  model_args_.index_topk_pattern() = build_fs_pattern_glm52(kNumLayers);
  refresh_ctx();

  for (int32_t layer_id = 0; layer_id < kNumLayers; ++layer_id) {
    auto decoder = make_decoder(layer_id);
    const DsaTopkShareDecision& decision =
        DeepseekV2DecoderLayerTestPeer::topk_share_decision(*decoder);
    const bool expect_full = is_full_indexer_layer_glm52(layer_id);
    const bool expect_output =
        expect_full && !is_full_indexer_layer_glm52(layer_id + 1);
    EXPECT_EQ(decision.reuse_topk, !expect_full) << "layer " << layer_id;
    EXPECT_EQ(decision.output_topk, expect_output) << "layer " << layer_id;
  }
}

TEST_F(DeepseekV2DecoderTopkShareTest,
       DefaultIndexer_WhenGlm51_ThenEveryLayerLoadsOwnIndexerWeights) {
  configure_glm5_indexer();
  // GLM5.1 has an indexer on every layer but no cross-layer share fields.
  model_args_.index_topk_freq() = 1;
  model_args_.index_skip_topk_offset() = 0;
  model_args_.index_topk_pattern() = "";
  refresh_ctx();

  const DsaTopkSharePlan topk_share_plan(model_args_);
  EXPECT_FALSE(topk_share_plan.has_reuse());
  EXPECT_EQ(topk_share_plan.num_indexer_layers(), kNumLayers);

  for (int32_t layer_id : {0, 3, 6, 7}) {
    auto decoder = make_loaded_decoder(layer_id);
    const DsaTopkShareDecision& decision =
        DeepseekV2DecoderLayerTestPeer::topk_share_decision(*decoder);
    EXPECT_FALSE(decision.reuse_topk) << "layer " << layer_id;
    EXPECT_FALSE(decision.output_topk) << "layer " << layer_id;
    EXPECT_TRUE(DeepseekV2DecoderLayerTestPeer::attention_has_child(*decoder,
                                                                    "indexer"))
        << "layer " << layer_id;
    EXPECT_TRUE(DeepseekV2DecoderLayerTestPeer::attention_has_child(
        *decoder, "indexer_rotary_emb"))
        << "layer " << layer_id;

    const auto parameters = decoder->named_parameters(/*recurse=*/true);
    EXPECT_TRUE(parameters.contains("self_attn.indexer.wq_b.weight"))
        << "layer " << layer_id;
    EXPECT_TRUE(parameters.contains("self_attn.indexer.wk.weight"))
        << "layer " << layer_id;
    EXPECT_TRUE(parameters.contains("self_attn.indexer.weights_proj.weight"))
        << "layer " << layer_id;
    EXPECT_TRUE(parameters.contains("self_attn.indexer.k_norm.weight"))
        << "layer " << layer_id;
  }
}

TEST_F(DeepseekV2DecoderTopkShareTest,
       MtpLayerAtMainModelBoundaryUsesOnlyCrossStepReuse) {
  configure_glm5_indexer();
  model_args_.model_type() = "glm_moe_dsa_mtp";
  model_args_.index_topk_freq() = 4;
  model_args_.index_skip_topk_offset() = 3;
  model_args_.index_share_for_mtp_iteration() = true;
  refresh_ctx();

  DecoderHolder decoder(
      DeepseekV2DecoderLayerImpl(context_, /*layer_id=*/kNumLayers));
  const DsaTopkShareDecision& decision =
      DeepseekV2DecoderLayerTestPeer::topk_share_decision(*decoder);
  EXPECT_FALSE(decision.reuse_topk);
  EXPECT_FALSE(decision.output_topk);
  EXPECT_TRUE(DeepseekV2DecoderLayerTestPeer::mtp_topk_reuse(*decoder));
  EXPECT_TRUE(
      DeepseekV2DecoderLayerTestPeer::attention_has_child(*decoder, "indexer"));
}

// Prefill CP combined with GLM5.2 cross-layer top-k sharing is not
// supported yet; the model entry must fatal on this combination. The guard is a
// pure decision so it can be verified without constructing the full model.
TEST_F(DeepseekV2DecoderTopkShareTest,
       CpGuard_WhenCpOffAndSharePlan_ThenNoConflict) {
  configure_glm5_indexer();
  model_args_.index_topk_freq() = 4;
  model_args_.index_skip_topk_offset() = 3;
  model_args_.index_topk_pattern() = "";

  const DsaTopkSharePlan topk_share_plan(model_args_);
  EXPECT_TRUE(topk_share_plan.has_reuse());
  EXPECT_FALSE(cp_conflicts_with_dsa_topk_share(/*enable_prefill_cp=*/false,
                                                topk_share_plan));
}

TEST_F(DeepseekV2DecoderTopkShareTest,
       CpGuard_WhenCpOnAndGlm52FreqPlan_ThenConflict) {
  configure_glm5_indexer();
  model_args_.index_topk_freq() = 4;
  model_args_.index_skip_topk_offset() = 3;
  model_args_.index_topk_pattern() = "";

  const DsaTopkSharePlan topk_share_plan(model_args_);
  EXPECT_TRUE(cp_conflicts_with_dsa_topk_share(/*enable_prefill_cp=*/true,
                                               topk_share_plan));
}

TEST_F(DeepseekV2DecoderTopkShareTest,
       CpGuard_WhenCpOnAndPatternPlan_ThenConflict) {
  configure_glm5_indexer();
  model_args_.index_topk_freq() = 1;
  model_args_.index_skip_topk_offset() = 0;
  model_args_.index_topk_pattern() = build_fs_pattern_glm52(kNumLayers);

  const DsaTopkSharePlan topk_share_plan(model_args_);
  EXPECT_TRUE(cp_conflicts_with_dsa_topk_share(/*enable_prefill_cp=*/true,
                                               topk_share_plan));
}

TEST_F(DeepseekV2DecoderTopkShareTest,
       CpGuard_WhenPatternHasOnlyFullLayers_ThenNoConflict) {
  configure_glm5_indexer();
  model_args_.n_layers() = 4;
  model_args_.index_topk_freq() = 1;
  model_args_.index_skip_topk_offset() = 0;
  model_args_.index_topk_pattern() = "FFFF";

  const DsaTopkSharePlan topk_share_plan(model_args_);
  EXPECT_FALSE(topk_share_plan.has_reuse());
  EXPECT_FALSE(cp_conflicts_with_dsa_topk_share(/*enable_prefill_cp=*/true,
                                                topk_share_plan));
  EXPECT_FALSE(topk_share_plan.decision_for(/*layer_id=*/3).reuse_topk);
  EXPECT_FALSE(topk_share_plan.decision_for(/*layer_id=*/3).output_topk);
}

TEST_F(DeepseekV2DecoderTopkShareTest,
       CpGuard_WhenShortFreqPlanHasNoSharedLayer_ThenNoConflict) {
  configure_glm5_indexer();
  model_args_.n_layers() = 3;
  model_args_.index_topk_freq() = 4;
  model_args_.index_skip_topk_offset() = 3;
  model_args_.index_topk_pattern() = "";

  const DsaTopkSharePlan topk_share_plan(model_args_);
  EXPECT_FALSE(topk_share_plan.has_reuse());
  EXPECT_FALSE(cp_conflicts_with_dsa_topk_share(/*enable_prefill_cp=*/true,
                                                topk_share_plan));
  EXPECT_FALSE(topk_share_plan.decision_for(/*layer_id=*/2).output_topk);
}

TEST_F(DeepseekV2DecoderTopkShareTest,
       CpGuard_WhenCpOnAndGlm51IndexerDoesNotShare_ThenNoConflict) {
  configure_glm5_indexer();
  // GLM5.1: every layer owns an indexer, so CP has no cross-layer carrier.
  model_args_.index_topk_freq() = 1;
  model_args_.index_skip_topk_offset() = 0;
  model_args_.index_topk_pattern() = "";

  const DsaTopkSharePlan topk_share_plan(model_args_);
  EXPECT_FALSE(topk_share_plan.has_reuse());
  EXPECT_EQ(topk_share_plan.num_indexer_layers(), kNumLayers);
  EXPECT_FALSE(cp_conflicts_with_dsa_topk_share(/*enable_prefill_cp=*/true,
                                                topk_share_plan));
}

TEST_F(DeepseekV2DecoderTopkShareTest,
       CpGuard_WhenCpOnAndDeepseekV32NoSharePlan_ThenNoConflict) {
  // DeepSeek-V3.2: has an indexer but every layer self-computes (freq=1, no
  // pattern), so the existing CP path is unaffected.
  configure_glm5_indexer();
  model_args_.index_topk_freq() = 1;
  model_args_.index_skip_topk_offset() = 0;
  model_args_.index_topk_pattern() = "";

  const DsaTopkSharePlan topk_share_plan(model_args_);
  EXPECT_FALSE(topk_share_plan.has_reuse());
  EXPECT_FALSE(cp_conflicts_with_dsa_topk_share(/*enable_prefill_cp=*/true,
                                                topk_share_plan));
}

}  // namespace layer
}  // namespace xllm
