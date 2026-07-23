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

#include "ops_api.h"

#if defined(USE_MLU)
#include "mlu/mlu_ops_api.h"
#elif defined(USE_NPU)
#include "core/kernels/npu/tilelang/tilelang_ops_api.h"
#include "npu/npu_ops_api.h"
#include "npu/xllm_ops/xllm_ops_api.h"
#include "triton_npu/torch_api/triton_ops_api.h"
#elif defined(USE_MUSA)
#include "musa/musa_ops_api.h"
#elif defined(USE_CUDA)
#include "cuda/attention_runner.h"
#include "cuda/cuda_ops_api.h"
#elif defined(USE_ILU)
#include "ilu/ilu_ops_api.h"
#elif defined(USE_DCU)
#include "cuda/cuda_ops_api.h"
#include "dcu/aiter_quant_adapter.h"
#include "dcu/dcu_ops_api.h"
#include "dcu/hipblaslt_fp8_adapter.h"
#endif

#include <cmath>
#include <numeric>

#include "common/macros.h"
#include "layers/common/attention_metadata.h"

namespace xllm::kernel {

namespace {
#if defined(USE_NPU)
bool is_supported_initial_state_dtype(torch::ScalarType dtype) {
  return dtype == torch::kBFloat16 || dtype == torch::kFloat32;
}
#endif

#if defined(USE_DCU)
torch::Tensor pack_2d_position_ids(const torch::Tensor& position_ids,
                                   const torch::Tensor& cu_query_lens,
                                   const torch::Tensor& query) {
  if (position_ids.numel() == query.size(0)) {
    return position_ids.reshape({-1}).contiguous();
  }

  torch::Tensor cu_cpu = cu_query_lens.to(torch::kCPU).to(torch::kInt64);
  CHECK_GE(cu_cpu.numel(), 2)
      << "apply_rotary: cu_query_lens must have at least 2 elements when "
         "packing 2D position_ids.";
  const int64_t num_seqs = cu_cpu.numel() - 1;
  CHECK_LE(num_seqs, position_ids.size(0))
      << "apply_rotary: position_ids batch is smaller than cu_query_lens, "
      << "position_ids: " << position_ids.sizes()
      << ", cu_query_lens: " << cu_query_lens.sizes();

  std::vector<torch::Tensor> packed_positions;
  packed_positions.reserve(static_cast<size_t>(num_seqs));
  int64_t total_tokens = 0;
  for (int64_t seq_idx = 0; seq_idx < num_seqs; ++seq_idx) {
    const int64_t start = cu_cpu[seq_idx].item<int64_t>();
    const int64_t end = cu_cpu[seq_idx + 1].item<int64_t>();
    const int64_t seq_len = end - start;
    CHECK_GE(seq_len, 0) << "apply_rotary: cu_query_lens must be monotonic.";
    CHECK_LE(seq_len, position_ids.size(1))
        << "apply_rotary: position_ids seq dimension is smaller than "
        << "q_seq_len, position_ids: " << position_ids.sizes()
        << ", seq_len: " << seq_len;
    if (seq_len > 0) {
      packed_positions.emplace_back(
          position_ids.select(/*dim=*/0, seq_idx)
              .slice(/*dim=*/0, /*start=*/0, /*end=*/seq_len));
    }
    total_tokens += seq_len;
  }
  CHECK_EQ(total_tokens, query.size(0))
      << "apply_rotary: packed position_ids token count mismatch, "
      << "position_ids: " << position_ids.sizes()
      << ", cu_query_lens: " << cu_query_lens.sizes()
      << ", query: " << query.sizes();
  if (packed_positions.empty()) {
    return torch::empty({0}, position_ids.options().dtype(torch::kInt64));
  }
  return torch::cat(packed_positions, /*dim=*/0).to(torch::kInt64).contiguous();
}

#endif
}  // namespace

void apply_rotary(RotaryParams& params) {
#if defined(USE_MLU)
  mlu::apply_rotary(params.q,
                    params.k,
                    params.sin,
                    params.cos,
                    params.position_ids,
                    params.cu_query_lens,
                    params.interleaved,
                    params.discrete,
                    params.dynamic_ntk,
                    params.max_query_len);
#elif defined(USE_NPU)
  npu::apply_rotary(
      params.q, params.k, params.cos_sin, params.position_ids.value());
#elif defined(USE_CUDA) || defined(USE_MUSA) || defined(USE_DCU)
  bool is_neox = !params.interleaved;
  torch::Tensor pos_ids;
  torch::Tensor cos_sin;

  if (params.position_ids.has_value()) {
    // positions is already int64 on CUDA/MUSA/DCU (pre-converted in
    // ForwardInput::to).
    pos_ids = params.position_ids.value().to(torch::kInt64);
#if defined(USE_DCU)
    if (pos_ids.dim() == 2 && params.q.dim() == 3) {
      if (pos_ids.numel() == params.q.size(0)) {
        pos_ids = pos_ids.reshape({-1}).contiguous();
      } else {
        CHECK(params.cu_query_lens.has_value())
            << "apply_rotary: cu_query_lens is required to pack 2D "
               "position_ids for packed query, position_ids: "
            << pos_ids.sizes() << ", query: " << params.q.sizes();
        pos_ids = pack_2d_position_ids(
            pos_ids, params.cu_query_lens.value(), params.q);
      }
    } else {
      pos_ids = pos_ids.contiguous();
    }
#endif
  } else if (params.cu_query_lens.has_value()) {
    auto cu = params.cu_query_lens.value().to(torch::kInt64);
    CHECK(cu.numel() >= 2)
        << "apply_rotary: cu_query_lens must have at least 2 elements when "
           "position_ids is not provided.";
    int64_t seq_len = cu[1].item<int64_t>() - cu[0].item<int64_t>();
    CHECK(seq_len > 0) << "apply_rotary: invalid sequence length inferred from "
                          "cu_query_lens when position_ids is not provided.";
    pos_ids = torch::arange(seq_len,
                            torch::TensorOptions()
                                .dtype(torch::kInt64)
                                .device(params.q.device()))
                  .contiguous();
  } else {
    // When neither position_ids nor cu_query_lens is provided,
    // infer sequence length from q tensor and create default position IDs.
    // This handles cases like LongCat-Image-Edit where rotary embedding
    // is applied uniformly across all sequence positions.
    int64_t seq_len = params.q.size(0);
    CHECK(seq_len > 0) << "apply_rotary: cannot infer valid sequence "
                          "length from q tensor.";
    pos_ids = torch::arange(seq_len,
                            torch::TensorOptions()
                                .dtype(torch::kInt64)
                                .device(params.q.device()))
                  .contiguous();
  }

  if (params.precomputed_cos_sin.defined()) {
    cos_sin = params.precomputed_cos_sin;
  } else if (params.cos.defined() && params.sin.defined()) {
    const int64_t head_dim = params.cos.size(-1);
    const int64_t rot_half = head_dim / 2;
    auto cos_sliced = params.cos.contiguous().slice(-1, 0, rot_half);
    auto sin_sliced = params.sin.contiguous().slice(-1, 0, rot_half);
    cos_sin = torch::cat({cos_sliced, sin_sliced}, -1);
  } else if (params.cos_sin.defined()) {
    auto cos_sin_vec = params.cos_sin.chunk(4, -1);
    auto cos = cos_sin_vec[0];
    auto sin = cos_sin_vec[2];
    cos_sin = torch::cat({cos, sin}, -1);
  } else {
    LOG(FATAL) << "apply_rotary: neither cos_sin nor cos/sin "
                  "provided; cannot infer cos_sin.";
  }
#if defined(USE_DCU)
  std::optional<torch::Tensor> key =
      params.k.defined() ? std::optional<torch::Tensor>(params.k)
                         : std::nullopt;
  cuda::rotary_embedding(pos_ids, params.q, key, cos_sin, is_neox);
#else
  cuda::rotary_embedding(pos_ids, params.q, params.k, cos_sin, is_neox);
#endif
#elif defined(USE_ILU)
  torch::Tensor ilu_cos_sin;
  if (params.precomputed_cos_sin.defined()) {
    ilu_cos_sin = params.precomputed_cos_sin;
  } else {
    auto cos_sin_vec = params.cos_sin.chunk(4, -1);
    ilu_cos_sin = torch::cat({cos_sin_vec[0], cos_sin_vec[2]}, -1);
  }
  // positions is already int64 on ILU (pre-converted in ForwardInput::to).
  torch::Tensor long_position_ids = params.position_ids.value().to(at::kLong);
  ilu::apply_rope_pos_ids_cos_sin_cache(
      params.q, params.k, ilu_cos_sin, long_position_ids, params.interleaved);
#else
  NOT_IMPLEMENTED();
#endif
}

void active(ActivationParams& params) {
#if defined(USE_MLU)
  mlu::active(params.input,
              params.output,
              params.bias,
              params.cusum_token_count,
              params.act_mode,
              params.is_gated,
              params.start_expert_id,
              params.expert_size);
#elif defined(USE_NPU)
  params.output = npu::active(params.input, params.act_mode);
#elif defined(USE_CUDA) || defined(USE_MUSA) || defined(USE_DCU)
  cuda::act_and_mul(params.output, params.input, params.act_mode);
#elif defined(USE_ILU)
  ilu::act_and_mul(params.output, params.input, params.act_mode);
#else
  NOT_IMPLEMENTED();
#endif
}

void reshape_paged_cache(ReshapePagedCacheParams& params) {
#if defined(USE_MLU)
  mlu::reshape_paged_cache(params.key,
                           params.value,
                           params.k_cache,
                           params.v_cache,
                           params.slot_mapping,
                           params.direction);
#elif defined(USE_NPU)
  npu::reshape_paged_cache(params.key,
                           params.value,
                           params.k_cache,
                           params.v_cache,
                           params.slot_mapping);
#elif defined(USE_CUDA) || defined(USE_MUSA) || defined(USE_DCU)
  cuda::reshape_paged_cache(params.slot_mapping,
                            params.key,
                            params.value.value_or(torch::Tensor()),
                            params.k_cache,
                            params.v_cache.value_or(torch::Tensor()));
#elif defined(USE_ILU)
  // auto v_cache = params.v_cache.value_or(torch::Tensor());
  ilu::reshape_paged_cache(params.key,
                           params.value,
                           params.k_cache,
                           params.v_cache,
                           params.slot_mapping);
#else
  NOT_IMPLEMENTED();
#endif
}

void reshape_from_cache(ReshapeFromCacheParams& params) {
#if defined(USE_MLU)
  mlu::reshape_from_cache(params.key,
                          params.value,
                          params.key_cache,
                          params.value_cache,
                          params.context_lengths,
                          params.max_context_len,
                          params.context_seq_offset,
                          params.block_tables,
                          params.cache_seq_offset);
#else
  NOT_IMPLEMENTED();
#endif
}

void quant_to_paged_cache(ReshapePagedCacheParams& params) {
#if defined(USE_MLU)
  CHECK(params.k_cache_scale.has_value())
      << "k_cache_scale is required for quant_to_paged_cache";
  mlu::quant_to_paged_cache(params.key,
                            params.value,
                            params.k_cache,
                            params.v_cache,
                            params.k_cache_scale.value(),
                            params.v_cache_scale,
                            params.slot_mapping);
#else
  NOT_IMPLEMENTED();
#endif
}

void dequant_from_paged_cache(ReshapeFromCacheParams& params) {
#if defined(USE_MLU)
  CHECK(params.key_cache_quant_scale.has_value())
      << "key_cache_quant_scale is required for dequant_from_paged_cache";
  mlu::dequant_from_paged_cache(params.key,
                                params.value,
                                params.key_cache,
                                params.value_cache,
                                params.key_cache_quant_scale.value(),
                                params.value_cache_quant_scale,
                                params.context_lengths,
                                params.max_context_len,
                                params.context_seq_offset,
                                params.block_tables.value(),
                                params.quant_mode,
                                params.quant_bit);
#else
  NOT_IMPLEMENTED();
#endif
}

void fused_layernorm(FusedLayerNormParams& params) {
#if defined(USE_MLU)
  mlu::fused_layernorm(params.input,
                       params.output,
                       params.residual,
                       params.weight,
                       params.beta,
                       params.bias,
                       params.quant_scale,
                       params.residual_out,
                       params.smooth_quant_scale,
                       params.normed_out,
                       params.mode,
                       params.eps,
                       params.store_output_before_norm,
                       params.store_output_after_norm,
                       params.dynamic_quant);
#elif defined(USE_MUSA)
  musa::fused_layernorm(params.input,
                        params.output,
                        params.residual,
                        params.weight,
                        params.beta,
                        params.bias,
                        params.quant_scale,
                        params.residual_out,
                        params.smooth_quant_scale,
                        params.normed_out,
                        params.mode,
                        params.eps,
                        params.store_output_before_norm,
                        params.store_output_after_norm,
                        params.dynamic_quant);
#elif defined(USE_NPU)
  if (params.residual.has_value()) {
    if (params.add_gamma_offset) {
      std::tie(params.output, std::ignore, params.residual_out) =
          npu::gamma_add_rms_norm(params.input,
                                  params.residual.value(),
                                  params.weight,
                                  params.eps,
                                  params.add_gamma_offset);
    } else {
      std::tie(params.output, std::ignore, params.residual_out) =
          npu::add_rms_norm(
              params.input, params.residual.value(), params.weight, params.eps);
    }
  } else {
    params.output =
        npu::rms_norm(params.input, params.weight, params.eps, params.mode);
  }
  if (params.beta.has_value()) {
    params.output += params.beta.value();
  }
#elif defined(USE_CUDA) || defined(USE_MUSA) || defined(USE_DCU)
  if (params.residual.has_value()) {
    cuda::fused_add_rms_norm(
        params.input, params.residual.value(), params.weight, params.eps);
    params.output = params.input;
    params.residual_out = params.residual;
  } else {
    cuda::rms_norm(params.output, params.input, params.weight, params.eps);
  }
#elif defined(USE_ILU)
  if (params.residual.has_value()) {
    ilu::residual_layer_norm(params.input,
                             params.output,
                             params.residual,
                             params.weight,
                             params.bias,  // residual_bias
                             params.residual_out,
                             params.eps);
  } else {
    ilu::rms_norm(params.output, params.input, params.weight, params.eps);
  }
#else
  NOT_IMPLEMENTED();
#endif
}

torch::Tensor fused_adalayer_norm(AdaLayerNormParams& params) {
#if defined(USE_NPU)
  params.output = npu::fused_adalayer_norm(params.input,
                                           params.scale,
                                           params.shift,
                                           params.weight,
                                           params.bias,
                                           params.eps);
#else
  NOT_IMPLEMENTED();
#endif
  return params.output;
}

std::tuple<torch::Tensor, torch::Tensor> rms_norm_dynamic_quant(
    RmsNormDynamicQuantParams& params) {
#if defined(USE_NPU)
  return npu::rms_norm_dynamic_quant(params.input, params.weight, params.eps);
#else
  NOT_IMPLEMENTED();
#endif
}

torch::Tensor matmul(MatmulParams& params) {
#if defined(USE_MLU)
  return mlu::matmul(
      params.a, params.b, params.bias, params.c, params.alpha, params.beta);
#elif defined(USE_NPU)
  return npu::matmul(params.a, params.b, params.bias);
#elif defined(USE_CUDA) || defined(USE_MUSA)
  return cuda::matmul(params.a, params.b, params.bias);
#elif defined(USE_ILU)
  return ilu::matmul(params.a, params.b, params.bias);
#elif defined(USE_DCU)
  return dcu::matmul(params.a, params.b, params.bias);
#else
  NOT_IMPLEMENTED();
#endif
}

torch::Tensor quant_matmul(QuantMatmulParams& params) {
#if defined(USE_NPU)
  return npu::quant_matmul(params.x1,
                           params.x2,
                           params.transpose2,
                           params.scale,
                           params.offset,
                           params.pertoken_scale,
                           params.bias,
                           params.output_dtype);
#else
  NOT_IMPLEMENTED();
#endif
}

torch::Tensor quantize(NpuQuantizeParams& params) {
#if defined(USE_NPU)
  CHECK(params.scale.has_value() && params.scale->defined())
      << "quantize requires params.scale.";
  return npu::quantize_per_tensor(params.input,
                                  params.scale.value(),
                                  params.zero_point.value_or(torch::Tensor()),
                                  params.output_dtype,
                                  params.axis);
#else
  NOT_IMPLEMENTED();
#endif
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>> dynamic_quant(
    NpuQuantizeParams& params) {
#if defined(USE_NPU)
  auto [output, scale] = npu::dynamic_quant(
      params.input, params.smooth_scales, params.group_index, params.dst_type);
  return std::make_tuple(output,
                         scale.has_value()
                             ? std::optional<torch::Tensor>(scale.value())
                             : std::nullopt);
#else
  NOT_IMPLEMENTED();
#endif
}

torch::Tensor group_gemm(GroupGemmParams& params) {
#if defined(USE_MLU)
  return mlu::group_gemm(params.a,
                         params.b,
                         params.token_count,
                         params.output,
                         params.a_scale,
                         params.b_scale,
                         params.quant_flag,
                         params.max_dim,
                         params.trans_a,
                         params.trans_b,
                         params.a_quant_bit);
#elif defined(USE_NPU)
  std::vector<torch::Tensor> x_list;
  std::vector<torch::Tensor> weight_list;
  torch::TensorList x_ref;
  torch::TensorList weight_ref;
  if (params.x_list.has_value()) {
    x_ref = params.x_list.value();
  } else {
    x_list = {params.a};
    x_ref = x_list;
  }
  if (params.weight_list.has_value()) {
    weight_ref = params.weight_list.value();
  } else {
    weight_list = {params.b};
    weight_ref = weight_list;
  }
  std::optional<torch::Tensor> group_list = params.group_list;
  if (!group_list.has_value()) {
    group_list = params.token_count;
  }

  auto outputs =
      npu::apply_npu_grouped_matmul(x_ref,
                                    weight_ref,
                                    params.bias_list,
                                    params.scale_list,
                                    params.offset_list,
                                    params.antiquant_scale_list,
                                    params.antiquant_offset_list,
                                    params.per_token_scale_list,
                                    group_list,
                                    params.activation_input_list,
                                    params.activation_quant_scale_list,
                                    params.activation_quant_offset_list,
                                    params.split_item,
                                    params.group_type,
                                    params.group_list_type,
                                    params.act_type,
                                    params.tuning_config,
                                    params.output_dtype);
  return outputs.back();
#elif defined(USE_ILU)
  return ilu::group_gemm(params.a,
                         params.b,
                         params.token_count,
                         params.combine_idx,
                         params.output);
#elif defined(USE_DCU)
  return dcu::group_gemm(params.a, params.b, params.token_count, params.output);
#else
  NOT_IMPLEMENTED();
#endif
}

std::tuple<torch::Tensor, torch::Tensor> dequant_swiglu_quant(
    DequantSwigluQuantParams& params) {
#if defined(USE_NPU)
  return npu::dequant_swiglu_quant(params.x,
                                   params.weight_scale,
                                   params.activation_scale,
                                   params.bias,
                                   params.quant_scale,
                                   params.quant_offset,
                                   params.group_index,
                                   params.activate_left,
                                   params.quant_mode,
                                   params.swiglu_mode,
                                   params.clamp_limit,
                                   params.glu_alpha,
                                   params.glu_bias);
#else
  NOT_IMPLEMENTED();
#endif
}

std::tuple<torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           std::optional<torch::Tensor>,
           std::optional<torch::Tensor>>
w4a8_dynamic_moe_preprocess(W4A8DynamicMoePreprocessParams& params) {
#if defined(USE_NPU)
  return npu::w4a8_dynamic_moe_preprocess(params.w13_weight,
                                          params.w2_weight,
                                          params.w13_weight_scale,
                                          params.w2_weight_scale,
                                          params.w13_weight_scale_second,
                                          params.w2_weight_scale_second,
                                          params.w13_scale_bias,
                                          params.w2_scale_bias,
                                          params.group_size,
                                          params.pack_weight_to_int32);
#else
  NOT_IMPLEMENTED();
#endif
}

std::tuple<torch::Tensor, torch::Tensor> moe_active_topk(
    MoeFusedTopkParams& params) {
#if defined(USE_MLU)
  return mlu::moe_active_topk(params.input,
                              params.topk,
                              params.num_expert_group,
                              params.topk_group,
                              params.normalize,
                              params.mask,
                              params.normed_by,
                              params.scoring_func,
                              params.route_scale,
                              params.e_score_correction_bias);
#elif defined(USE_NPU)
  CHECK_EQ(params.scoring_func, "softmax")
      << "Only softmax is supported for NPU";
  auto [topk_weights, topk_ids, row_ids] = npu::apply_moe_gating_topk_softmax(
      params.input, params.finished, params.topk);
  (void)row_ids;
  if (params.normalize) {
    topk_weights = topk_weights / topk_weights.sum(-1, true);
  }
  return std::make_tuple(topk_weights, topk_ids);
#elif defined(USE_ILU)
  return ilu::moe_active_topk(params.input,
                              params.topk,
                              params.num_expert_group,
                              params.topk_group,
                              params.normalize,
                              params.mask,
                              params.normed_by,
                              params.scoring_func,
                              params.route_scale,
                              params.e_score_correction_bias);
#elif defined(USE_DCU)
  return dcu::moe_active_topk(params.input,
                              params.topk,
                              params.num_expert_group,
                              params.topk_group,
                              params.normalize,
                              params.e_score_correction_bias,
                              params.scoring_func,
                              params.route_scale);
#elif defined(USE_CUDA) || defined(USE_MUSA)
  return cuda::moe_fused_topk(params.input,
                              params.topk,
                              params.normalize,
                              params.e_score_correction_bias,
                              params.scoring_func);
#else
  NOT_IMPLEMENTED();
#endif
}

std::vector<torch::Tensor> moe_gen_idx(MoeGenIdxParams& params) {
#if defined(USE_MLU)
  return mlu::moe_gen_idx(params.expert_id, params.expert_num);
#elif defined(USE_ILU)
  return ilu::moe_gen_idx(params.expert_id, params.expert_num);
#elif defined(USE_DCU)
  auto [src_dst, dst_src, expert_sizes] =
      cuda::moe_compute_index(params.expert_id, params.expert_num);
  return {src_dst, dst_src, expert_sizes};
#else
  NOT_IMPLEMENTED();
#endif
}

torch::Tensor moe_expand_input(MoeExpandInputParams& params) {
#if defined(USE_MLU)
  return mlu::moe_expand_input(params.input,
                               params.gather_index,
                               params.cusum_token_count,
                               params.start_expert_id,
                               params.expert_size);
#elif defined(USE_ILU)
  return ilu::moe_expand_input(
      params.input, params.gather_index, params.combine_idx, params.topk);
#else
  NOT_IMPLEMENTED();
#endif
}

torch::Tensor moe_combine_result(MoeCombineResultParams& params) {
#if defined(USE_MLU)
  return mlu::moe_combine_result(params.input,
                                 params.reduce_weight,
                                 params.gather_ids,
                                 params.residual,
                                 params.cusum_token_count,
                                 params.start_expert_id,
                                 params.expert_size,
                                 params.bias);
#elif defined(USE_NPU)
  std::optional<torch::Tensor> probes =
      params.probes.has_value()
          ? params.probes
          : std::optional<torch::Tensor>(params.reduce_weight);
  auto output = npu::apply_npu_moe_token_unpermute(params.input,
                                                   params.gather_ids,
                                                   probes,
                                                   params.padded_mode,
                                                   params.restore_shape);
  if (params.residual.has_value()) {
    output = output + params.residual.value();
  }
  return output;
#elif defined(USE_ILU)
  return ilu::moe_combine_result(params.input, params.reduce_weight);
#elif defined(USE_DCU)
  // N = params.reduce_weight.size(0), topk = params.reduce_weight.size(1)
  int64_t N = params.reduce_weight.size(0);
  int32_t topk = static_cast<int32_t>(params.reduce_weight.size(1));
  auto out =
      cuda::moe_combine_result(params.input, params.reduce_weight, N, topk);
  if (params.residual.has_value()) {
    out = out + params.residual.value();
  }
  return out;
#else
  NOT_IMPLEMENTED();
#endif
}

torch::Tensor moe_all2all_gen_send_layout(
    MoeAll2AllGenSendLayoutParams& params) {
#if defined(USE_MLU)
  return mlu::moe_all2all_gen_send_layout(params.token_count, params.nrank);
#else
  NOT_IMPLEMENTED();
#endif
}

std::vector<torch::Tensor> moe_all2all_gen_gather_index(
    MoeAll2AllGenGatherIndexParams& params) {
#if defined(USE_MLU)
  return mlu::moe_all2all_gen_gather_index(
      params.token_num, params.pad_num, params.return_cusum_token_count);
#else
  NOT_IMPLEMENTED();
#endif
}

std::vector<torch::Tensor> moe_all2all_create(MoeAll2AllCreateParams& params) {
#if defined(USE_MLU)
  return mlu::moe_all2all_create(params.dispatch_token_byte,
                                 params.combine_token_byte,
                                 params.max_expert_num,
                                 params.max_token_num,
                                 params.rank,
                                 params.nrank,
                                 params.device);
#else
  NOT_IMPLEMENTED();
#endif
}

void moe_all2all_init(MoeAll2AllInitParams& params) {
#if defined(USE_MLU)
  mlu::moe_all2all_init(params.handle, params.all_exchange_info, params.device);
#else
  NOT_IMPLEMENTED();
#endif
}

void moe_all2all_dispatch(MoeAll2AllDispatchParams& params) {
#if defined(USE_MLU)
  mlu::moe_all2all_dispatch(params.handle,
                            params.token_byte,
                            params.token_num,
                            params.send_layout,
                            params.send_token_num,
                            params.recv_layout,
                            params.recv_token_num,
                            params.send_token,
                            params.recv_token);
#else
  NOT_IMPLEMENTED();
#endif
}

void moe_all2all_combine(MoeAll2AllCombineParams& params) {
#if defined(USE_MLU)
  mlu::moe_all2all_combine(params.handle,
                           params.token_byte,
                           params.token_num,
                           params.send_src_layout,
                           params.send_dst_layout,
                           params.send_token,
                           params.recv_token);
#else
  NOT_IMPLEMENTED();
#endif
}

void moe_all2all_destroy(MoeAll2AllDestroyParams& params) {
#if defined(USE_MLU)
  mlu::moe_all2all_destroy(params.handle, params.device);
#else
  NOT_IMPLEMENTED();
#endif
}

std::tuple<torch::Tensor, torch::Tensor> scaled_quantize(
    ScaledQuantizeParams& params) {
#if defined(USE_MLU)
  return mlu::scaled_quantize(params.x,
                              params.smooth,
                              params.zero,
                              params.token_count,
                              params.gather_index,
                              params.gather_index_start_position,
                              params.output,
                              params.output_scale,
                              params.act_mode,
                              params.active_coef,
                              params.is_gated,
                              params.quant_type);
#elif defined(USE_DCU)
  return dcu::scaled_quantize(params.x,
                              params.smooth,
                              params.zero,
                              params.token_count,
                              params.gather_index,
                              params.gather_index_start_position,
                              params.output,
                              params.output_scale,
                              params.act_mode,
                              params.active_coef,
                              params.is_gated,
                              params.quant_type);
#else
  NOT_IMPLEMENTED();
#endif
}

torch::Tensor scaled_matmul(ScaledMatmulParams& params) {
#if defined(USE_MLU)
  return mlu::scaled_matmul(params.a,
                            params.b,
                            params.a_scale,
                            params.b_scale,
                            params.output_dtype,
                            params.bias,
                            params.c,
                            params.act_mode,
                            params.quant_bit_size,
                            params.alpha,
                            params.beta,
                            params.use_hp_active,
                            params.a_quant_bit_size,
                            params.a_calib,
                            params.b_calib,
                            params.output);
#elif defined(USE_DCU)
  return dcu::scaled_matmul(params.a,
                            params.b,
                            params.a_scale,
                            params.b_scale,
                            params.output_dtype,
                            params.bias,
                            params.c,
                            params.act_mode,
                            params.quant_bit_size,
                            params.alpha,
                            params.beta,
                            params.use_hp_active,
                            params.a_quant_bit_size,
                            params.a_calib,
                            params.b_calib,
                            params.output);
#else
  NOT_IMPLEMENTED();
#endif
}

torch::Tensor apply_top_k_top_p(TopKPParams& params) {
#if defined(USE_MLU)
  return mlu::apply_top_k_top_p(
      params.logits, params.temperatures, params.top_k, params.top_p);
#else
  NOT_IMPLEMENTED();
#endif
}

torch::Tensor random_sample(RandomSampleParams& params) {
#if defined(USE_MLU)
  return mlu::random_sample(params.logits);
#elif defined(USE_CUDA)
  return cuda::random_sample(params.logits);
#elif defined(USE_DCU)
  return dcu::random_sample(params.logits);
#else
  NOT_IMPLEMENTED();
#endif
}

torch::Tensor rejection_sample(RejectionSampleParams& params) {
#if defined(USE_MLU)
  return mlu::rejection_sample(params.draft_token_ids,
                               params.num_draft_tokens,
                               params.cu_num_draft_tokens,
                               params.draft_probs,
                               params.target_probs,
                               params.bonus_token_ids,
                               params.uniform_rand,
                               params.uniform_probs,
                               params.max_spec_len);
#elif defined(USE_DCU)
  return dcu::rejection_sample(params.draft_token_ids,
                               params.num_draft_tokens,
                               params.cu_num_draft_tokens,
                               params.draft_probs,
                               params.target_probs,
                               params.bonus_token_ids,
                               params.uniform_rand,
                               params.uniform_probs,
                               params.max_spec_len);
#else
  NOT_IMPLEMENTED();
#endif
}

void masked_indexer_select_paged_kv(MaskedIndexerSelectPagedKVParams& params) {
#if defined(USE_MLU)
  mlu::masked_indexer_select_paged_kv(params.query,
                                      params.k_cache,
                                      params.weights,
                                      params.kv_cache_block_table,
                                      params.cu_seq_q_lens,
                                      params.cu_seq_k_lens,
                                      params.k_context_lens,
                                      params.k_cache_block_table,
                                      params.is_prefill,
                                      params.index_topk,
                                      params.kv_cache_block_size,
                                      params.softmax_scale,
                                      params.q_scale,
                                      params.k_scale_cache,
                                      params.sparse_block_table,
                                      params.sparse_context_lens,
                                      params.is_score_float,
                                      params.compress_ratio,
                                      params.kv_cache_block_table_offset);
#else
  NOT_IMPLEMENTED();
#endif
}

void gather_split(GatherSplitParams& params) {
#if defined(USE_MLU)
  mlu::gather_split(params.input,
                    params.gather_index,
                    params.valid_token_num,
                    params.output_head,
                    params.output_tail);
#else
  NOT_IMPLEMENTED();
#endif
}

void fused_mla_q(FusedMlaQParams& params) {
#if defined(USE_MLU)
  mlu::fused_mla_q(params.q,
                   params.output,
                   params.output_scale,
                   params.output_norm,
                   params.gamma,
                   params.smooth_quant_scale,
                   params.weight_b,
                   params.weight_b_scale,
                   params.weight_c,
                   params.sin,
                   params.cos,
                   params.position_id,
                   params.quant_mode,
                   params.eps,
                   params.interleaved);
#else
  NOT_IMPLEMENTED();
#endif
}

void fused_mla_kv(FusedMlaKVParams& params) {
#if defined(USE_MLU)
  mlu::fused_mla_kv(params.input_kv,
                    params.sin,
                    params.cos,
                    params.position_id,
                    params.gamma,
                    params.kv_cache,
                    params.kv_cache_scale,
                    params.slot_mapping,
                    params.cache_bs_id,
                    params.cache_seq_offset,
                    params.quant_mode,
                    params.is_paged_cache,
                    params.eps,
                    params.interleaved);
#else
  NOT_IMPLEMENTED();
#endif
}

void fused_indexer_q(FusedIndexerQParams& params) {
#if defined(USE_MLU)
  mlu::fused_indexer_q(params.input_q,
                       params.output,
                       params.output_scale,
                       params.w_q,
                       params.w_q_scale,
                       params.hadamard_matrix,
                       params.sin,
                       params.cos,
                       params.position_id,
                       params.quant_mode,
                       params.interleaved,
                       params.rope_at_front);
#else
  NOT_IMPLEMENTED();
#endif
}

void fused_indexer_k(FusedIndexerKParams& params) {
#if defined(USE_MLU)
  mlu::fused_indexer_k(params.x,
                       params.wk,
                       params.wproj,
                       params.sin_table,
                       params.cos_table,
                       params.position_id,
                       params.slot_mapping,
                       params.head_weights,
                       params.k_cache,
                       params.k_cache_scale,
                       params.hadamard_matrix,
                       params.interleaved,
                       params.gamma,
                       params.beta,
                       params.eps);
#else
  NOT_IMPLEMENTED();
#endif
}

torch::Tensor l2_norm(torch::Tensor& x, double eps) {
#if defined(USE_NPU)
  return npu::npu_l2norm_last_dim(x, eps);
#else
  NOT_IMPLEMENTED();
#endif
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
moe_init_routing_v2(MoeInitRoutingV2Params& params) {
#if defined(USE_NPU)
  return npu::apply_npu_moe_init_routing_v2(params.x,
                                            params.expert_idx,
                                            params.scale,
                                            params.offset,
                                            params.active_num,
                                            params.expert_capacity,
                                            params.expert_num,
                                            params.drop_pad_mode,
                                            params.expert_tokens_num_type,
                                            params.expert_tokens_num_flag,
                                            params.quant_mode,
                                            params.active_expert_range,
                                            params.row_idx_type);
#else
  NOT_IMPLEMENTED();
#endif
}

std::tuple<torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor>
moe_distribute_dispatch_v2(MoeDistributeDispatchV2Params& params) {
#if defined(USE_NPU)
  return npu::apply_npu_moe_distribute_dispatch_v2(
      params.x,
      params.expert_ids,
      params.expert_scales,
      params.x_active_mask,
      params.scales,
      params.group_ep,
      params.ep_world_size,
      params.ep_rank_id,
      params.moe_expert_num,
      params.group_tp,
      params.tp_world_size,
      params.tp_rank_id,
      params.expert_shard_type,
      params.shared_expert_num,
      params.shared_expert_rank_num,
      params.quant_mode,
      params.global_bs,
      params.expert_token_nums_type,
      params.comm_alg);
#else
  NOT_IMPLEMENTED();
#endif
}

torch::Tensor moe_distribute_combine_v2(MoeDistributeCombineV2Params& params) {
#if defined(USE_NPU)
  return npu::apply_npu_moe_distribute_combine_v2(
      params.expand_x,
      params.expert_ids,
      params.assist_info_for_combine,
      params.ep_send_counts,
      params.expert_scales,
      params.tp_send_counts,
      params.x_active_mask,
      params.expand_scales,
      params.shared_expert_x,
      params.group_ep,
      params.ep_world_size,
      params.ep_rank_id,
      params.moe_expert_num,
      params.group_tp,
      params.tp_world_size,
      params.tp_rank_id,
      params.expert_shard_type,
      params.shared_expert_num,
      params.shared_expert_rank_num,
      params.global_bs,
      params.comm_quant_mode,
      params.comm_alg);
#else
  NOT_IMPLEMENTED();
#endif
}

bool has_moe_distribute_dispatch_combine_v2() {
#if defined(USE_NPU)
  return npu::has_moe_distribute_dispatch_combine_v2();
#else
  return false;
#endif
}

std::tuple<torch::Tensor, torch::Tensor> dispatch_ffn_combine(
    DispatchFFNCombineParams& params) {
#if defined(USE_NPU)
  return npu::apply_npu_dispatch_ffn_combine(params.x,
                                             params.weight1,
                                             params.weight2,
                                             params.expert_ids,
                                             params.scale1,
                                             params.scale2,
                                             params.probs,
                                             params.group,
                                             params.max_output_size,
                                             params.swiglu_limit,
                                             params.output,
                                             params.expert_token_nums);
#else
  NOT_IMPLEMENTED();
#endif
}

bool has_dispatch_ffn_combine() {
#if defined(USE_NPU)
  return npu::has_dispatch_ffn_combine();
#else
  return false;
#endif
}

std::tuple<torch::Tensor, torch::Tensor> dispatch_gmm_combine_decode(
    DispatchGmmCombineDecodeParams& params) {
#if defined(USE_NPU)
  return npu::apply_npu_dispatch_gmm_combine_decode(
      params.x,
      params.expert_ids,
      params.gmm1_permuted_weight,
      params.gmm1_permuted_weight_scale,
      params.gmm2_weight,
      params.gmm2_weight_scale,
      params.expert_scales,
      params.expert_smooth_scales,
      params.x_active_mask,
      params.group_ep,
      params.ep_rank_size,
      params.ep_rank_id,
      params.moe_expert_num,
      params.shared_expert_num,
      params.shared_expert_rank_num,
      params.quant_mode,
      params.global_bs);
#else
  NOT_IMPLEMENTED();
#endif
}

bool has_dispatch_gmm_combine_decode() {
#if defined(USE_NPU)
  return npu::has_dispatch_gmm_combine_decode();
#else
  return false;
#endif
}

torch::Tensor hc_post(HcPostParams& params) {
#if defined(USE_NPU)
  return npu::hc_post(params.x, params.residual, params.post, params.comb);
#else
  NOT_IMPLEMENTED();
#endif
}

std::tuple<torch::Tensor, torch::Tensor> fp8_scaled_quantize(
    Fp8ScaledQuantizeParams& params) {
#if defined(USE_CUDA)
  return cuda::fp8_scaled_quantize(params.input, params.output, params.scale);
#elif defined(USE_DCU)
  CHECK(!params.scale.has_value() || !params.scale.value().defined())
      << "DCU fp8_scaled_quantize currently supports only dynamic per-token "
         "quantization.";
  return dcu::aiter::per_token_quant_fp8(params.input, params.output);
#else
  NOT_IMPLEMENTED();
#endif
}

std::pair<torch::Tensor, torch::Tensor> fused_gdn_gating(
    FusedGdnGatingParams& params) {
#if defined(USE_MLU)
  return mlu::fused_gdn_gating(params.A_log,
                               params.a,
                               params.b,
                               params.dt_bias,
                               params.beta,
                               params.threshold);
#elif defined(USE_NPU)
  return npu::tilelang::fused_gdn_gating(params.A_log,
                                         params.a,
                                         params.b,
                                         params.dt_bias,
                                         params.beta,
                                         params.threshold);
  // return npu::npu_fused_gdn_gating(params.A_log,
  //                                  params.a,
  //                                  params.b,
  //                                  params.dt_bias,
  //                                  params.beta,
  //                                  params.threshold);
#else
  NOT_IMPLEMENTED();
#endif
}

std::tuple<torch::Tensor, torch::Tensor> quant_lightning_indexer(
    QuantLightningIndexerParams& params) {
#if defined(USE_NPU)
  return npu::quant_lightning_indexer(params.query,
                                      params.key,
                                      params.weights,
                                      params.query_dequant_scale,
                                      params.key_dequant_scale,
                                      params.query_quant_mode,
                                      params.key_quant_mode,
                                      params.actual_seq_lengths_query,
                                      params.actual_seq_lengths_key,
                                      params.block_table,
                                      params.metadata,
                                      params.layout_query,
                                      params.layout_key,
                                      params.sparse_count,
                                      params.sparse_mode,
                                      params.pre_tokens,
                                      params.next_tokens,
                                      params.cmp_ratio,
                                      params.return_value);
#else
  NOT_IMPLEMENTED();
#endif
}

std::pair<torch::Tensor, torch::Tensor> fused_recurrent_gated_delta_rule(
    FusedRecurrentGatedDeltaRuleParams& params) {
#if defined(USE_NPU)
  return npu::npu_fused_recurrent_gated_delta_rule(
      params.q,
      params.k,
      params.v,
      params.g,
      params.beta,
      params.scale,
      params.initial_state,
      params.inplace_final_state,
      params.cu_seqlens,
      params.ssm_state_indices,
      params.num_accepted_tokens,
      params.use_qk_l2norm_in_kernel);
#else
  NOT_IMPLEMENTED();
#endif
}

torch::Tensor hc_pre_inv_rms(HcPreInvRmsParams& params) {
#if defined(USE_NPU)
  return npu::hc_pre_inv_rms(params.x, params.epsilon);
#else
  NOT_IMPLEMENTED();
#endif
}

torch::Tensor fused_sigmoid_gating_delta_rule_update(
    FusedSigmoidGatingDeltaRuleUpdateParams& params) {
#if defined(USE_MLU)
  float scale = params.scale.has_value()
                    ? params.scale.value()
                    : 1.0f / std::sqrt(static_cast<float>(params.k.size(-1)));
  auto outputs = mlu::fused_sigmoid_gating_delta_rule_update(
      params.A_log,
      params.a,
      params.b,
      params.dt_bias,
      params.q,
      params.k,
      params.v,
      params.initial_state_source,
      params.initial_state_indices,
      params.cu_seqlens,
      scale,
      params.use_qk_l2norm_in_kernel,
      params.softplus_beta,
      params.softplus_threshold,
      params.num_accepted_tokens,
      /*inplace_final_state=*/true,
      params.is_kda);
  return outputs.first;
#elif defined(USE_NPU)
  return npu::npu_fused_sigmoid_gating_delta_rule_update(
      params.A_log,
      params.a,
      params.dt_bias,
      params.q,
      params.k,
      params.v,
      params.b,
      params.initial_state_source,
      params.initial_state_indices,
      params.cu_seqlens,
      params.scale,
      params.use_qk_l2norm_in_kernel,
      params.softplus_beta,
      params.softplus_threshold);
#else
  NOT_IMPLEMENTED();
#endif
}

torch::Tensor fp8_scaled_matmul(Fp8ScaledMatmulParams& params) {
#if defined(USE_CUDA)
  auto out_2d = cuda::fp8_scaled_matmul(params.a,
                                        params.b,
                                        params.a_scale,
                                        params.b_scale,
                                        params.output_dtype,
                                        params.bias,
                                        params.output);

  // Auto reshape output if original input shape is provided
  if (params.input_shape.has_value()) {
    auto out_shape = params.input_shape.value();
    out_shape.back() = params.b.size(0);
    return out_2d.view(out_shape);
  }
  return out_2d;
#elif defined(USE_DCU)
  torch::Tensor out_2d = dcu::hipblaslt_fp8::fp8_gemm_nt(params.a,
                                                         params.b,
                                                         params.a_scale,
                                                         params.b_scale,
                                                         params.output_dtype,
                                                         params.bias,
                                                         params.output);
  if (params.input_shape.has_value()) {
    std::vector<int64_t> out_shape = params.input_shape.value();
    out_shape.back() = params.b.size(0);
    return out_2d.view(out_shape);
  }
  return out_2d;
#else
  LOG(FATAL) << "fp8_scaled_matmul is only supported on CUDA/DCU";
  return torch::Tensor();
#endif
}

void static_scaled_fp8_quant(StaticScaledFp8QuantParams& params) {
#if defined(USE_CUDA)
  cuda::static_scaled_fp8_quant(params.output, params.input, params.scale);
#else
  LOG(FATAL) << "static_scaled_fp8_quant is only supported on CUDA";
#endif
}

// Fused RMSNorm + Static FP8 Quantization
torch::Tensor rms_norm_static_fp8_quant(RmsNormStaticFp8QuantParams& params) {
#if defined(USE_CUDA)
  auto org_shape = params.input.sizes().vec();
  auto hidden_size = params.input.size(-1);

  // Flatten input to 2D. Use reshape to support non-contiguous tensors.
  auto input_2d = params.input.reshape({-1, hidden_size});

  torch::Tensor output =
      torch::empty({input_2d.size(0), hidden_size},
                   input_2d.options().dtype(torch::kFloat8_e4m3fn));

  // Call fused kernel
  cuda::rms_norm_static_fp8_quant(
      output, input_2d, params.weight, params.scale, params.epsilon);

  return output.reshape(org_shape);
#else
  LOG(FATAL) << "rms_norm_static_fp8_quant is only supported on CUDA";
  return torch::Tensor();
#endif
}

std::tuple<torch::Tensor, torch::Tensor> fused_add_rms_norm_static_fp8_quant(
    FusedAddRmsNormStaticFp8QuantParams& params) {
#if defined(USE_CUDA)
  auto org_shape = params.input.sizes().vec();
  auto hidden_size = params.input.size(-1);

  // Flatten tensors to 2D. Use reshape to support non-contiguous tensors.
  auto input_2d = params.input.reshape({-1, hidden_size});
  auto residual_2d = params.residual.reshape({-1, hidden_size});

  torch::Tensor output =
      torch::empty({input_2d.size(0), hidden_size},
                   input_2d.options().dtype(torch::kFloat8_e4m3fn));

  // Call fused kernel (residual is updated in-place)
  cuda::fused_add_rms_norm_static_fp8_quant(output,
                                            input_2d,
                                            residual_2d,
                                            params.weight,
                                            params.scale,
                                            params.epsilon);

  // Reshape outputs
  auto output_reshaped = output.reshape(org_shape);
  auto residual_reshaped = residual_2d.reshape(org_shape);

  return std::make_tuple(output_reshaped, residual_reshaped);
#else
  LOG(FATAL) << "fused_add_rms_norm_static_fp8_quant is only supported on CUDA";
  return std::make_tuple(torch::Tensor(), torch::Tensor());
#endif
}

torch::Tensor causal_conv1d_update(CausalConv1dUpdateParams& params) {
#if defined(USE_MLU)
  return mlu::causal_conv1d_update_decode(params.x,
                                          params.conv_state,
                                          params.weight,
                                          params.bias,
                                          params.conv_state_indices,
                                          params.activation,
                                          params.pad_slot_id,
                                          params.query_start_loc,
                                          params.max_query_len,
                                          params.num_accepted_tokens,
                                          params.block_idx_last_scheduled_token,
                                          params.initial_state_idx);
#elif defined(USE_NPU)
  const bool has_silu = params.activation;

  auto x_work = params.x;
  auto weight_work = params.weight;
  auto conv_state_work = params.conv_state;

  const int32_t dim = static_cast<int32_t>(x_work.size(1));

  auto bias_work = params.bias.has_value() && params.bias.value().defined()
                       ? params.bias.value()
                       : torch::zeros({dim}, x_work.options());

  auto conv_state_t = conv_state_work;
  auto weight_t = weight_work;

  auto cu_seqlens =
      params.query_start_loc.has_value()
          ? params.query_start_loc.value().to(torch::kInt32)
          : torch::arange(0,
                          x_work.size(0) + 1,
                          std::max(params.max_query_len, int32_t{1}),
                          torch::TensorOptions()
                              .dtype(torch::kInt32)
                              .device(x_work.device()));

  int64_t batch = cu_seqlens.size(0) - 1;
  if (batch <= 0) {
    return x_work;
  }

  auto i32_opts =
      torch::TensorOptions().dtype(torch::kInt32).device(x_work.device());

  torch::Tensor init_indices;
  torch::Tensor current_indices;
  if (params.conv_state_indices.has_value()) {
    auto ci = params.conv_state_indices.value().to(torch::kInt32);
    if (ci.dim() == 1) {
      init_indices = ci;
      current_indices = ci;
    } else {
      auto ci_0 = ci.select(1, 0);
      auto ci_1 = ci.select(1, 1);
      if (params.initial_state_idx.has_value()) {
        auto isi = params.initial_state_idx.value().to(torch::kInt32);
        init_indices = torch::where(isi == 0, ci_0, ci_1);
      } else {
        init_indices = ci_0;
      }
      if (params.block_idx_last_scheduled_token.has_value()) {
        auto bilt =
            params.block_idx_last_scheduled_token.value().to(torch::kInt32);
        current_indices = torch::where(bilt == 0, ci_0, ci_1);
      } else {
        current_indices = ci_0;
      }
    }
  } else {
    init_indices = torch::arange(batch, i32_opts);
    current_indices = init_indices;
  }

  torch::Tensor initial_state_mode;
  if (params.initial_state_mode.has_value()) {
    initial_state_mode = params.initial_state_mode.value().to(torch::kInt32);
  } else {
    initial_state_mode = torch::ones({batch}, i32_opts);
  }

  const bool is_3d = (x_work.dim() == 3);
  auto x_flat = is_3d ? x_work.reshape({-1, dim}) : x_work;

  if (npu::tilelang::has_causal_conv1d_decode_specialization(
          batch, dim, has_silu)) {
    auto conv_state_t_nonconst = conv_state_t;
    auto y = npu::tilelang::causal_conv1d_decode(
        /*conv_state=*/conv_state_t_nonconst,
        /*x=*/x_flat,
        /*weight=*/weight_t,
        /*bias=*/bias_work,
        /*init_indices=*/init_indices,
        /*current_indices=*/current_indices,
        /*initial_state_mode=*/initial_state_mode,
        /*has_silu=*/has_silu);

    if (is_3d) {
      y = y.view(x_work.sizes());
    }
    return y;
  }

  // Fallback: per-batch loop using causal_conv1d (batch=1 kernel, fp16).
  auto original_dtype = x_work.scalar_type();
  bool need_cast = (original_dtype != torch::kFloat16);

  auto x_fp16 = need_cast ? x_flat.to(torch::kFloat16) : x_flat;
  auto weight_fp16 = need_cast ? weight_work.to(torch::kFloat16) : weight_work;
  auto conv_state_fp16 = need_cast ? conv_state_work.to(torch::kFloat16).clone()
                                   : conv_state_work.clone();
  auto bias_fp16 = need_cast ? bias_work.to(torch::kFloat16) : bias_work;

  auto y_fp16 = torch::empty({x_flat.size(0), dim}, x_fp16.options());
  auto cu_seqlens_cpu = cu_seqlens.to(torch::kCPU);
  const int32_t* cu_ptr = cu_seqlens_cpu.data_ptr<int32_t>();

  for (int64_t b = 0; b < batch; ++b) {
    int32_t seq_start_b = cu_ptr[b];
    int32_t seq_end_b = cu_ptr[b + 1];
    int32_t sb_len = seq_end_b - seq_start_b;
    if (sb_len <= 0) {
      continue;
    }

    auto x_b = x_fp16.slice(0, seq_start_b, seq_end_b);
    auto init_b = init_indices.slice(0, b, b + 1);
    auto curr_b = current_indices.slice(0, b, b + 1);
    auto ism_b = initial_state_mode.slice(0, b, b + 1);

    auto cu_b = torch::tensor(
        {0, sb_len},
        torch::TensorOptions().dtype(torch::kInt32).device(x_work.device()));

    auto y_b = npu::tilelang::causal_conv1d(conv_state_fp16,
                                            x_b,
                                            weight_fp16,
                                            bias_fp16,
                                            cu_b,
                                            init_b,
                                            curr_b,
                                            ism_b,
                                            has_silu);

    y_fp16.slice(0, seq_start_b, seq_end_b).copy_(y_b);
  }

  if (need_cast) {
    params.conv_state.copy_(conv_state_fp16.to(original_dtype));
  } else {
    params.conv_state.copy_(conv_state_fp16);
  }
  auto y = need_cast ? y_fp16.to(original_dtype) : y_fp16;

  if (is_3d) {
    y = y.view(x_work.sizes());
  }
  return y;
#else
  NOT_IMPLEMENTED();
#endif
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> hc_pre_sinkhorn(
    HcPreSinkhornParams& params) {
#if defined(USE_NPU)
  return npu::hc_pre_sinkhorn(params.mixes,
                              params.rsqrt,
                              params.hc_scale,
                              params.hc_base,
                              params.x,
                              params.hc_mult,
                              params.hc_sinkhorn_iters,
                              params.hc_eps);
#else
  NOT_IMPLEMENTED();
#endif
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> hc_pre(
    HcPreParams& params) {
#if defined(USE_NPU)
  return npu::hc_pre(params.x,
                     params.hc_fn,
                     params.hc_scale,
                     params.hc_base,
                     params.hc_mult,
                     params.hc_sinkhorn_iters,
                     params.norm_eps,
                     params.hc_eps);
#else
  NOT_IMPLEMENTED();
#endif
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> moe_gating_top_k_hash(
    MoeGatingTopKHashParams& params) {
#if defined(USE_NPU)
  return npu::moe_gating_top_k_hash(params.x,
                                    params.k,
                                    params.bias,
                                    params.input_ids,
                                    params.tid2eid,
                                    params.k_group,
                                    params.group_count,
                                    params.routed_scaling_factor,
                                    params.eps,
                                    params.group_select_mode,
                                    params.renorm,
                                    params.norm_type,
                                    params.out_flag);
#else
  NOT_IMPLEMENTED();
#endif
}

torch::Tensor gated_layer_norm(GatedLayerNormParams& params) {
#if defined(USE_NPU)
  return npu::layer_norm_fwd_aclnn(params.x,
                                   params.weight,
                                   params.bias,
                                   params.eps,
                                   params.z,
                                   params.group_size,
                                   params.norm_before_gate,
                                   params.is_rms_norm);
#elif defined(USE_MLU)
  return mlu::gated_layer_norm(params.x,
                               params.weight,
                               params.bias,
                               params.eps,
                               params.z,
                               params.group_size,
                               params.norm_before_gate);
#elif defined(USE_DCU)
  return dcu::gated_layer_norm(params.x,
                               params.weight,
                               params.bias,
                               params.eps,
                               params.z,
                               params.group_size,
                               params.norm_before_gate);
#else
  NOT_IMPLEMENTED();
#endif
}

std::tuple<torch::Tensor, torch::Tensor> sparse_attn_sharedkv(
    SparseAttnSharedkvParams& params) {
#if defined(USE_NPU)
  return npu::sparse_attn_sharedkv(params.q,
                                   params.ori_kv,
                                   params.cmp_kv,
                                   params.ori_sparse_indices,
                                   params.cmp_sparse_indices,
                                   params.ori_block_table,
                                   params.cmp_block_table,
                                   params.cu_seqlens_q,
                                   params.cu_seqlens_ori_kv,
                                   params.cu_seqlens_cmp_kv,
                                   params.seqused_q,
                                   params.seqused_kv,
                                   params.sinks,
                                   params.metadata,
                                   params.softmax_scale,
                                   params.cmp_ratio,
                                   params.ori_mask_mode,
                                   params.cmp_mask_mode,
                                   params.ori_win_left,
                                   params.ori_win_right,
                                   params.layout_q,
                                   params.layout_kv,
                                   params.return_softmax_lse);
#else
  NOT_IMPLEMENTED();
#endif
}

std::pair<torch::Tensor, torch::Tensor> partial_rotary_embedding(
    PartialRotaryEmbeddingParams& params) {
#if defined(USE_NPU)
  return npu::apply_npu_partial_rotary_embedding(params.positions,
                                                 params.query,
                                                 params.key,
                                                 params.head_size,
                                                 params.rotary_dim,
                                                 params.cos_sin_cache,
                                                 params.is_neox_style);
#else
  NOT_IMPLEMENTED();
#endif
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
fused_qkvzba_split_reshape_cat(FusedQkvzbaSplitReshapeParams& params) {
#if defined(USE_NPU)
  return npu::npu_fused_qkvzba_split_reshape_cat(params.mixed_qkvz,
                                                 params.mixed_ba,
                                                 params.num_heads_qk,
                                                 params.num_heads_v,
                                                 params.head_qk,
                                                 params.head_v);
#else
  NOT_IMPLEMENTED();
#endif
}

torch::Tensor sparse_flash_attention(SparseFlashAttentionParams& params) {
#if defined(USE_NPU)
  return npu::sparse_flash_attention(params.query,
                                     params.key,
                                     params.value,
                                     params.sparse_indices,
                                     params.block_table,
                                     params.actual_seq_lengths_query,
                                     params.actual_seq_lengths_kv,
                                     params.query_rope,
                                     params.key_rope,
                                     params.scale_value,
                                     params.sparse_block_size,
                                     params.layout_query,
                                     params.layout_kv,
                                     params.sparse_mode);
#else
  NOT_IMPLEMENTED();
#endif
}

void gemma_rms_norm(GemmaRMSNormParams& params) {
#if defined(USE_NPU)
  npu::npu_gemma_rms_norm(
      params.x, params.gamma, params.epsilon, params.rstd_out, params.norm_out);
#elif defined(USE_MLU)
  mlu::gemma_rms_norm(params.x,
                      params.gamma,
                      params.epsilon,
                      params.norm_out,
                      params.residual,
                      params.residual_out);
#elif defined(USE_DCU)
  dcu::gemma_rms_norm(params.x, params.gamma, params.epsilon, params.norm_out);
#else
  NOT_IMPLEMENTED();
#endif
}

std::tuple<torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor>
compressor(CompressorParams& params) {
#if defined(USE_NPU)
  return npu::compressor(params.x,
                         params.wkv,
                         params.wgate,
                         params.kv_state,
                         params.score_state,
                         params.ape,
                         params.norm_weight,
                         params.rope_sin,
                         params.rope_cos,
                         params.kv_block_table,
                         params.score_block_table,
                         params.cu_seqlens,
                         params.seqused,
                         params.start_pos,
                         params.rope_head_dim,
                         params.cmp_ratio,
                         params.coff,
                         params.norm_eps,
                         params.rotary_mode,
                         params.enable_grad);
#else
  NOT_IMPLEMENTED();
#endif
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
split_qkv_rmsnorm_mrope(SplitQkvRmsnormMropeParams& params) {
#if defined(USE_NPU)
  return npu::tilelang::split_qkv_rmsnorm_mrope(params.qkvg,
                                                params.q_weight,
                                                params.k_weight,
                                                params.cos_sin,
                                                params.gather_pattern,
                                                params.eps,
                                                params.num_q_heads,
                                                params.num_kv_heads,
                                                params.head_size);
#else
  NOT_IMPLEMENTED();
#endif
}

bool has_split_qkv_rmsnorm_mrope_specialization(int64_t num_q_heads,
                                                int64_t num_kv_heads,
                                                int64_t head_size) {
#if defined(USE_NPU)
  return npu::tilelang::has_split_qkv_rmsnorm_mrope_specialization(
      num_q_heads, num_kv_heads, head_size);
#else
  return false;
#endif
}

torch::Tensor build_split_qkv_rmsnorm_mrope_gather_pattern(
    int64_t rope_dim,
    const std::vector<int64_t>& mrope_section,
    bool is_interleaved,
    const torch::Device& device) {
#if defined(USE_NPU)
  return npu::tilelang::build_split_qkv_rmsnorm_mrope_gather_pattern(
      rope_dim, mrope_section, is_interleaved, device);
#else
  NOT_IMPLEMENTED();
#endif
}

torch::Tensor quant_lightning_indexer_metadata(
    QuantLightningIndexerMetadataParams& params) {
#if defined(USE_NPU)
  return npu::quant_lightning_indexer_metadata(params.num_heads_q,
                                               params.num_heads_k,
                                               params.head_dim,
                                               params.query_quant_mode,
                                               params.key_quant_mode,
                                               params.actual_seq_lengths_query,
                                               params.actual_seq_lengths_key,
                                               params.batch_size,
                                               params.max_seqlen_q,
                                               params.max_seqlen_k,
                                               params.layout_query,
                                               params.layout_key,
                                               params.sparse_count,
                                               params.sparse_mode,
                                               params.pre_tokens,
                                               params.next_tokens,
                                               params.cmp_ratio,
                                               params.device);
#else
  NOT_IMPLEMENTED();
#endif
}

torch::Tensor sparse_attn_sharedkv_metadata(
    SparseAttnSharedkvMetadataParams& params) {
#if defined(USE_NPU)
  return npu::sparse_attn_sharedkv_metadata(params.num_heads_q,
                                            params.num_heads_kv,
                                            params.head_dim,
                                            params.cu_seqlens_q,
                                            params.cu_seqlens_ori_kv,
                                            params.cu_seqlens_cmp_kv,
                                            params.seqused_q,
                                            params.seqused_kv,
                                            params.batch_size,
                                            params.max_seqlen_q,
                                            params.max_seqlen_kv,
                                            params.ori_topk,
                                            params.cmp_topk,
                                            params.cmp_ratio,
                                            params.ori_mask_mode,
                                            params.cmp_mask_mode,
                                            params.ori_win_left,
                                            params.ori_win_right,
                                            params.layout_q,
                                            params.layout_kv,
                                            params.has_ori_kv,
                                            params.has_cmp_kv);
#else
  NOT_IMPLEMENTED();
#endif
}

std::pair<torch::Tensor, torch::Tensor> chunk_gated_delta_rule(
    ChunkGatedDeltaRuleParams& params) {
#if defined(USE_NPU)
  CHECK(!params.head_first)
      << "chunk_gated_delta_rule only supports head_first=false.";
  CHECK(params.q.scalar_type() == torch::kBFloat16 &&
        params.k.scalar_type() == torch::kBFloat16 &&
        params.v.scalar_type() == torch::kBFloat16)
      << "chunk_gated_delta_rule expects q/k/v to be bfloat16.";
  if (params.initial_state.has_value()) {
    CHECK(is_supported_initial_state_dtype(
        params.initial_state.value().scalar_type()))
        << "chunk_gated_delta_rule expects initial_state to be bfloat16 or "
           "float32, got "
        << params.initial_state.value().scalar_type();
  }

  const torch::ScalarType input_dtype = params.q.scalar_type();
  const int64_t batch_size = params.q.size(0);
  const int64_t seq_len = params.q.size(1);
  const int64_t num_heads_qk = params.q.size(2);
  CHECK(params.q.sizes() == params.k.sizes())
      << "q and k must have the same shape.";
  CHECK(params.v.dim() == 4 && params.v.size(0) == batch_size &&
        params.v.size(1) == seq_len)
      << "v must have shape [B, T, Hv, V].";
  const int64_t num_heads_v = params.v.size(2);
  const int64_t head_dim = params.q.size(3);
  const int64_t chunk_size = params.chunk_size;
  CHECK(num_heads_v % num_heads_qk == 0)
      << "chunk_gated_delta_rule expects num_heads_v to be "
         "divisible by num_heads_qk, got "
      << num_heads_v << " and " << num_heads_qk;
  CHECK(params.beta.dim() == 3 && params.beta.size(0) == batch_size &&
        params.beta.size(1) == seq_len && params.beta.size(2) == num_heads_v)
      << "beta must have shape [B, T, H].";
  CHECK(params.g.dim() == 3 && params.g.size(0) == batch_size &&
        params.g.size(1) == seq_len && params.g.size(2) == num_heads_v)
      << "g must have shape [B, T, H].";

  auto q_prepared = params.use_qk_l2norm_in_kernel
                        ? npu::npu_l2norm_last_dim(params.q)
                        : params.q;
  auto k_prepared = params.use_qk_l2norm_in_kernel
                        ? npu::npu_l2norm_last_dim(params.k)
                        : params.k;
  auto cu_prepared =
      params.cu_seqlens.has_value()
          ? std::optional<torch::Tensor>(
                params.cu_seqlens.value().to(torch::kInt32).contiguous())
          : std::nullopt;
  auto g_cumsum =
      npu::npu_chunk_local_cumsum(params.g, chunk_size, cu_prepared);
  const float scale_value = params.scale.has_value()
                                ? params.scale.value()
                                : std::pow(static_cast<float>(head_dim), -0.5f);
  auto matrix_a = npu::npu_chunk_scaled_dot_kkt_fwd(
      k_prepared, params.beta, g_cumsum, chunk_size, cu_prepared);
  auto matrix_a_inv = npu::npu_solve_tril(
      matrix_a, chunk_size, cu_prepared, params.k.scalar_type());
  auto [w, u] = npu::npu_recompute_w_u_fwd(
      k_prepared, params.v, params.beta, g_cumsum, matrix_a_inv, cu_prepared);
  auto init_state_prepared =
      params.initial_state.has_value()
          ? std::optional<torch::Tensor>(
                params.initial_state.value().to(torch::kFloat32).contiguous())
          : std::nullopt;
  auto [h, v_new, final_state] = npu::tilelang::chunk_gated_delta_rule_fwd_h(
      k_prepared.squeeze(0),
      w.squeeze(0),
      u.squeeze(0),
      g_cumsum.squeeze(0),
      init_state_prepared,
      params.output_final_state,
      chunk_size,
      /*save_new_value=*/true,
      cu_prepared,
      /*chunk_offsets=*/std::nullopt);
  auto out = npu::npu_chunk_fwd_o(q_prepared,
                                  k_prepared,
                                  v_new.unsqueeze(0),
                                  h.unsqueeze(0),
                                  g_cumsum,
                                  scale_value,
                                  chunk_size,
                                  cu_prepared);

  return {out.to(input_dtype),
          params.output_final_state ? final_state : torch::Tensor()};
#else
  NOT_IMPLEMENTED();
#endif
}

std::pair<torch::Tensor, torch::Tensor> mega_chunk_gdn(
    MegaChunkGdnParams& params) {
#if defined(USE_NPU)
  return npu::npu_mega_chunk_gdn(params.q,
                                 params.k,
                                 params.v,
                                 params.g,
                                 params.beta,
                                 params.scale,
                                 params.initial_state,
                                 params.output_final_state,
                                 params.cu_seqlens,
                                 params.q_seq_lens,
                                 params.use_qk_l2norm_in_kernel);
#else
  NOT_IMPLEMENTED();
#endif
}

void npu_inplace_partial_rotary_mul(NpuInplacePartialRotaryMulParams& params) {
#if defined(USE_NPU)
  npu::npu_inplace_partial_rotary_mul(params.x,
                                      params.r1,
                                      params.r2,
                                      params.rotary_mode,
                                      at::IntArrayRef(params.partial_slice));
#else
  NOT_IMPLEMENTED();
#endif
}

torch::Tensor recurrent_gated_delta_rule(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    torch::Tensor& state,
    const std::optional<torch::Tensor>& beta,
    const std::optional<double> scale,
    const std::optional<torch::Tensor>& actual_seq_lengths,
    const std::optional<torch::Tensor>& ssm_state_indices,
    const std::optional<torch::Tensor>& num_accepted_tokens,
    const std::optional<torch::Tensor>& g,
    const std::optional<torch::Tensor>& gk) {
#if defined(USE_NPU)
  return npu::npu_recurrent_gated_delta_rule(query,
                                             key,
                                             value,
                                             state,
                                             beta,
                                             scale,
                                             actual_seq_lengths,
                                             ssm_state_indices,
                                             num_accepted_tokens,
                                             g,
                                             gk);
#else
  NOT_IMPLEMENTED();
#endif
}

torch::Tensor causal_conv1d(const torch::Tensor& x,
                            const torch::Tensor& weight,
                            const torch::Tensor& conv_state,
                            const std::optional<torch::Tensor>& bias_opt,
                            const torch::IntArrayRef query_start_loc_opt,
                            const torch::IntArrayRef cache_indices_opt,
                            const torch::IntArrayRef initial_state_mode_opt,
                            const torch::IntArrayRef num_accepted_tokens_opt,
                            int64_t activation_mode,
                            int64_t pad_slot_id,
                            int64_t run_mode) {
#if defined(USE_NPU)
  return npu::causal_conv1d(x,
                            weight,
                            conv_state,
                            bias_opt,
                            query_start_loc_opt,
                            cache_indices_opt,
                            initial_state_mode_opt,
                            num_accepted_tokens_opt,
                            activation_mode,
                            pad_slot_id,
                            run_mode);
#else
  NOT_IMPLEMENTED();
#endif
}

void causal_conv1d_out(const torch::Tensor& output,
                       const torch::Tensor& x,
                       const torch::Tensor& weight,
                       const torch::Tensor& conv_state,
                       const std::optional<torch::Tensor>& bias_opt,
                       const torch::IntArrayRef query_start_loc_opt,
                       const torch::IntArrayRef cache_indices_opt,
                       const torch::IntArrayRef initial_state_mode_opt,
                       const torch::IntArrayRef num_accepted_tokens_opt,
                       int64_t activation_mode,
                       int64_t pad_slot_id,
                       int64_t run_mode) {
#if defined(USE_NPU)
  npu::causal_conv1d_out(output,
                         x,
                         weight,
                         conv_state,
                         bias_opt,
                         query_start_loc_opt,
                         cache_indices_opt,
                         initial_state_mode_opt,
                         num_accepted_tokens_opt,
                         activation_mode,
                         pad_slot_id,
                         run_mode);
#else
  NOT_IMPLEMENTED();
#endif
}
}  // namespace xllm::kernel
