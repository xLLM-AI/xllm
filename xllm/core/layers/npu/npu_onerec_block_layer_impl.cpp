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

#include "npu_onerec_block_layer_impl.h"

#include <glog/logging.h>
#include <mstx/ms_tools_ext.h>

#include <algorithm>
#include <cstring>
#include <set>

#include "common/global_flags.h"
#include "core/util/rec_model_utils.h"
namespace xllm {
namespace layer {
namespace {

torch::Tensor EnsureNdFormat(torch::Tensor tensor) {
  if (!tensor.defined()) {
    return tensor;
  }
  if (!tensor.is_contiguous()) {
    tensor = tensor.contiguous();
  }
  if (tensor.device().type() == torch::DeviceType::PrivateUse1 &&
      at_npu::native::get_npu_format(tensor) != ACL_FORMAT_ND) {
    tensor =
        at_npu::native::npu_format_cast(tensor, ACL_FORMAT_ND).contiguous();
  }
  return tensor;
}

torch::Tensor PrepareOneRecAttentionMask(const at::Tensor& attn_mask,
                                         const at::Device& device,
                                         torch::Dtype dtype) {
  if (!attn_mask.defined()) {
    return torch::Tensor();
  }
  torch::Tensor result = attn_mask;
  if (result.device() != device) {
    result = result.to(device);
  }
  if (result.scalar_type() != dtype) {
    result = result.to(dtype);
  }
  return EnsureNdFormat(result);
}

int64_t ResolveOneRecBatchSize(const ModelInputParams& input_params) {
  const auto* onerec_params = input_params.onerec_xattention_params() != nullptr
                                  ? static_cast<const OneRecModelInputParams*>(
                                        input_params.onerec_xattention_params())
                                  : input_params.onerec_params();
  if (onerec_params != nullptr && onerec_params->bs > 0) {
    return onerec_params->bs;
  }
  return std::max<int64_t>(input_params.num_sequences, 1);
}

torch::Tensor NormalizeOneRecDecodeFasMask(torch::Tensor attn_mask,
                                           int64_t batch_size) {
  if (!attn_mask.defined() || attn_mask.dim() == 2 || attn_mask.dim() == 4) {
    return EnsureNdFormat(attn_mask);
  }
  if (attn_mask.dim() != 3) {
    return EnsureNdFormat(attn_mask);
  }

  batch_size = std::max<int64_t>(batch_size, 1);
  if (attn_mask.size(0) == batch_size) {
    return EnsureNdFormat(attn_mask.unsqueeze(1));
  }

  auto collapsed_mask = attn_mask.narrow(0, 0, 1);
  if (batch_size > 1) {
    collapsed_mask = collapsed_mask.expand(
        {batch_size, collapsed_mask.size(1), collapsed_mask.size(2)});
  }
  return EnsureNdFormat(collapsed_mask.unsqueeze(1));
}

// Decoder normal mode: self-attn(29) + cross-attn(28) + layer-norm(4) + mlp(18)
// = 79
static constexpr uint64_t kOneRecWeightCountPerLayer = 79;

// Decoder MoE mode weights count (exclude runtime tensors like expert_array).
static constexpr uint64_t kOneRecMoeWeightCountPerLayer = 97;

// OneRec attention linear best-practice defaults.
// Keep them local to avoid exposing extra user-facing flags.
static constexpr bool kEnableOneRecAclnnAttentionLinear = true;
static constexpr int32_t kOneRecAclnnAttentionLinearMinTokens = 128;

enum class OneRecBlockLayerTensorId : int32_t {
  // Self-attention layer norm
  IN_LAYER_NORM_WEIGHT = 0,
  IN_LAYER_NORM_BIAS,
  IN_INPUT_NORM_NEW_WEIGHT,
  IN_INPUT_NORM_NEW_BIAS,
  // Self-attention Q, K, V projections
  IN_Q_WEIGHT,
  IN_Q_BIAS,
  IN_Q_DEQSCALE,
  IN_Q_OFFSET,
  IN_Q_SCALE,
  IN_Q_COMPRESS_IDX,

  IN_K_WEIGHT,
  IN_K_BIAS,
  IN_K_DEQSCALE,
  IN_K_OFFSET,
  IN_K_SCALE,
  IN_K_COMPRESS_IDX,

  IN_V_WEIGHT,
  IN_V_BIAS,
  IN_V_DEQSCALE,
  IN_V_OFFSET,
  IN_V_SCALE,
  IN_V_COMPRESS_IDX,

  // Self-attention output projection
  IN_SELF_ATTN_OUT_WEIGHT,
  IN_SELF_ATTN_OUT_BIAS,
  IN_SELF_ATTN_OUT_DEQSCALE,
  IN_SELF_ATTN_OUT_OFFSET,
  IN_SELF_ATTN_OUT_SCALE,
  IN_SELF_ATTN_OUT_COMPRESS_IDX,

  // ONEREC relative attention bias (encoder only)
  IN_RELATIVE_ATTENTION_BIAS_WEIGHT,

  // Cross-attention layer norm (decoder only)
  IN_CROSS_LAYER_NORM_WEIGHT,
  IN_CROSS_LAYER_NORM_BIAS,
  IN_CROSS_LAYER_NORM_NEW_WEIGHT,
  IN_CROSS_LAYER_NORM_NEW_BIAS,

  // Cross-attention Q, K, V projections (decoder only)
  IN_CROSS_Q_WEIGHT,
  IN_CROSS_Q_BIAS,
  IN_CROSS_Q_DEQSCALE,
  IN_CROSS_Q_OFFSET,
  IN_CROSS_Q_SCALE,
  IN_CROSS_Q_COMPRESS_IDX,

  IN_CROSS_K_WEIGHT,
  IN_CROSS_K_BIAS,
  IN_CROSS_K_DEQSCALE,
  IN_CROSS_K_OFFSET,
  IN_CROSS_K_SCALE,
  IN_CROSS_K_COMPRESS_IDX,

  IN_CROSS_V_WEIGHT,
  IN_CROSS_V_BIAS,
  IN_CROSS_V_DEQSCALE,
  IN_CROSS_V_OFFSET,
  IN_CROSS_V_SCALE,
  IN_CROSS_V_COMPRESS_IDX,

  // Cross-attention output projection (decoder only)
  IN_CROSS_ATTN_OUT_WEIGHT,
  IN_CROSS_ATTN_OUT_BIAS,
  IN_CROSS_ATTN_OUT_DEQSCALE,
  IN_CROSS_ATTN_OUT_OFFSET,
  IN_CROSS_ATTN_OUT_SCALE,
  IN_CROSS_ATTN_OUT_COMPRESS_IDX,

  // Final layer norm
  IN_FINAL_LAYER_NORM_WEIGHT,
  IN_FINAL_LAYER_NORM_BIAS,
  IN_FINAL_LAYER_NORM_NEW_WEIGHT,
  IN_FINAL_LAYER_NORM_NEW_BIAS,

  // Feed-forward network (gated activation)
  IN_FFN_WI_0_WEIGHT = 61,  // wi_0 (gate projection)
  IN_FFN_WI_0_BIAS,
  IN_FFN_WI_0_DEQSCALE,
  IN_FFN_WI_0_OFFSET,
  IN_FFN_WI_0_SCALE,
  IN_FFN_WI_0_COMPRESS_IDX,

  IN_FFN_WI_1_WEIGHT,  // wi_1 (up projection)
  IN_FFN_WI_1_BIAS,
  IN_FFN_WI_1_DEQSCALE,
  IN_FFN_WI_1_OFFSET,
  IN_FFN_WI_1_SCALE,
  IN_FFN_WI_1_COMPRESS_IDX,

  IN_FFN_WO_WEIGHT,  // wo (down projection)
  IN_FFN_WO_BIAS,
  IN_FFN_WO_DEQSCALE,
  IN_FFN_WO_OFFSET,
  IN_FFN_WO_SCALE,
  IN_FFN_WO_COMPRESS_IDX,
};

constexpr int32_t kInLayerNormWeight =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_LAYER_NORM_WEIGHT);
constexpr int32_t kInLayerNormBias =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_LAYER_NORM_BIAS);
constexpr int32_t kInInputNormNewWeight =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_INPUT_NORM_NEW_WEIGHT);
constexpr int32_t kInInputNormNewBias =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_INPUT_NORM_NEW_BIAS);
constexpr int32_t kInQWeight =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_Q_WEIGHT);
constexpr int32_t kInQBias =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_Q_BIAS);
constexpr int32_t kInQDeqScale =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_Q_DEQSCALE);
constexpr int32_t kInQOffset =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_Q_OFFSET);
constexpr int32_t kInQScale =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_Q_SCALE);
constexpr int32_t kInQCompressIdx =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_Q_COMPRESS_IDX);
constexpr int32_t kInKWeight =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_K_WEIGHT);
constexpr int32_t kInKBias =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_K_BIAS);
constexpr int32_t kInKDeqScale =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_K_DEQSCALE);
constexpr int32_t kInKOffset =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_K_OFFSET);
constexpr int32_t kInKScale =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_K_SCALE);
constexpr int32_t kInKCompressIdx =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_K_COMPRESS_IDX);
constexpr int32_t kInVWeight =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_V_WEIGHT);
constexpr int32_t kInVBias =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_V_BIAS);
constexpr int32_t kInVDeqScale =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_V_DEQSCALE);
constexpr int32_t kInVOffset =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_V_OFFSET);
constexpr int32_t kInVScale =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_V_SCALE);
constexpr int32_t kInVCompressIdx =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_V_COMPRESS_IDX);
constexpr int32_t kInSelfAttnOutWeight =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_SELF_ATTN_OUT_WEIGHT);
constexpr int32_t kInSelfAttnOutBias =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_SELF_ATTN_OUT_BIAS);
constexpr int32_t kInSelfAttnOutDeqScale =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_SELF_ATTN_OUT_DEQSCALE);
constexpr int32_t kInSelfAttnOutOffset =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_SELF_ATTN_OUT_OFFSET);
constexpr int32_t kInSelfAttnOutScale =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_SELF_ATTN_OUT_SCALE);
constexpr int32_t kInSelfAttnOutCompressIdx = static_cast<int32_t>(
    OneRecBlockLayerTensorId::IN_SELF_ATTN_OUT_COMPRESS_IDX);
constexpr int32_t kInRelativeAttentionBiasWeight = static_cast<int32_t>(
    OneRecBlockLayerTensorId::IN_RELATIVE_ATTENTION_BIAS_WEIGHT);
constexpr int32_t kInCrossLayerNormWeight =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_CROSS_LAYER_NORM_WEIGHT);
constexpr int32_t kInCrossLayerNormBias =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_CROSS_LAYER_NORM_BIAS);
constexpr int32_t kInCrossLayerNormNewWeight = static_cast<int32_t>(
    OneRecBlockLayerTensorId::IN_CROSS_LAYER_NORM_NEW_WEIGHT);
constexpr int32_t kInCrossLayerNormNewBias = static_cast<int32_t>(
    OneRecBlockLayerTensorId::IN_CROSS_LAYER_NORM_NEW_BIAS);
constexpr int32_t kInCrossQWeight =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_CROSS_Q_WEIGHT);
constexpr int32_t kInCrossQBias =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_CROSS_Q_BIAS);
constexpr int32_t kInCrossQDeqScale =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_CROSS_Q_DEQSCALE);
constexpr int32_t kInCrossQOffset =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_CROSS_Q_OFFSET);
constexpr int32_t kInCrossQScale =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_CROSS_Q_SCALE);
constexpr int32_t kInCrossQCompressIdx =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_CROSS_Q_COMPRESS_IDX);
constexpr int32_t kInCrossKWeight =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_CROSS_K_WEIGHT);
constexpr int32_t kInCrossKBias =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_CROSS_K_BIAS);
constexpr int32_t kInCrossKDeqScale =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_CROSS_K_DEQSCALE);
constexpr int32_t kInCrossKOffset =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_CROSS_K_OFFSET);
constexpr int32_t kInCrossKScale =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_CROSS_K_SCALE);
constexpr int32_t kInCrossKCompressIdx =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_CROSS_K_COMPRESS_IDX);
constexpr int32_t kInCrossVWeight =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_CROSS_V_WEIGHT);
constexpr int32_t kInCrossVBias =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_CROSS_V_BIAS);
constexpr int32_t kInCrossVDeqScale =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_CROSS_V_DEQSCALE);
constexpr int32_t kInCrossVOffset =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_CROSS_V_OFFSET);
constexpr int32_t kInCrossVScale =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_CROSS_V_SCALE);
constexpr int32_t kInCrossVCompressIdx =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_CROSS_V_COMPRESS_IDX);
constexpr int32_t kInCrossAttnOutWeight =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_CROSS_ATTN_OUT_WEIGHT);
constexpr int32_t kInCrossAttnOutBias =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_CROSS_ATTN_OUT_BIAS);
constexpr int32_t kInCrossAttnOutDeqScale =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_CROSS_ATTN_OUT_DEQSCALE);
constexpr int32_t kInCrossAttnOutOffset =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_CROSS_ATTN_OUT_OFFSET);
constexpr int32_t kInCrossAttnOutScale =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_CROSS_ATTN_OUT_SCALE);
constexpr int32_t kInCrossAttnOutCompressIdx = static_cast<int32_t>(
    OneRecBlockLayerTensorId::IN_CROSS_ATTN_OUT_COMPRESS_IDX);
constexpr int32_t kInFinalLayerNormWeight =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_FINAL_LAYER_NORM_WEIGHT);
constexpr int32_t kInFinalLayerNormBias =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_FINAL_LAYER_NORM_BIAS);
constexpr int32_t kInFinalLayerNormNewWeight = static_cast<int32_t>(
    OneRecBlockLayerTensorId::IN_FINAL_LAYER_NORM_NEW_WEIGHT);
constexpr int32_t kInFinalLayerNormNewBias = static_cast<int32_t>(
    OneRecBlockLayerTensorId::IN_FINAL_LAYER_NORM_NEW_BIAS);
constexpr int32_t kInFfnWi0Weight =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_FFN_WI_0_WEIGHT);
constexpr int32_t kInFfnWi0Bias =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_FFN_WI_0_BIAS);
constexpr int32_t kInFfnWi0DeqScale =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_FFN_WI_0_DEQSCALE);
constexpr int32_t kInFfnWi0Offset =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_FFN_WI_0_OFFSET);
constexpr int32_t kInFfnWi0Scale =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_FFN_WI_0_SCALE);
constexpr int32_t kInFfnWi0CompressIdx =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_FFN_WI_0_COMPRESS_IDX);
constexpr int32_t kInFfnWi1Weight =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_FFN_WI_1_WEIGHT);
constexpr int32_t kInFfnWi1Bias =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_FFN_WI_1_BIAS);
constexpr int32_t kInFfnWi1DeqScale =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_FFN_WI_1_DEQSCALE);
constexpr int32_t kInFfnWi1Offset =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_FFN_WI_1_OFFSET);
constexpr int32_t kInFfnWi1Scale =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_FFN_WI_1_SCALE);
constexpr int32_t kInFfnWi1CompressIdx =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_FFN_WI_1_COMPRESS_IDX);
constexpr int32_t kInFfnWoWeight =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_FFN_WO_WEIGHT);
constexpr int32_t kInFfnWoBias =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_FFN_WO_BIAS);
constexpr int32_t kInFfnWoDeqScale =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_FFN_WO_DEQSCALE);
constexpr int32_t kInFfnWoOffset =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_FFN_WO_OFFSET);
constexpr int32_t kInFfnWoScale =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_FFN_WO_SCALE);
constexpr int32_t kInFfnWoCompressIdx =
    static_cast<int32_t>(OneRecBlockLayerTensorId::IN_FFN_WO_COMPRESS_IDX);

enum class OneRecMoeBlockLayerTensorId : int32_t {
  // MoE weights (only used when use_moe=true)
  IN_BLOCK_SPARSE_MOE_GATE_WEIGHT = 61,   // routing weights
  IN_BLOCK_SPARSE_MOE_GATE_BIAS = 62,     // routing bias
  IN_BLOCK_SPARSE_MOE_GATE_DESCALE,       // gate descale
  IN_BLOCK_SPARSE_MOE_GATE_OFFSET,        // gate offset
  IN_BLOCK_SPARSE_MOE_GATE_SCALE,         // gate scale
  IN_BLOCK_SPARSE_MOE_GATE_COMPRESS_IDX,  // gate compress index

  // Shared expert weights
  IN_MLP_GATEUP_WEIGHT_SHARED_EXPERT,
  IN_MLP_GATEUP_BIAS_SHARED_EXPERT,
  IN_MLP_GATEUP_DESCALE_SHARED_EXPERT,
  IN_MLP_GATEUP_OFFSET_SHARED_EXPERT,
  IN_MLP_GATEUP_SCALE_SHARED_EXPERT,
  IN_MLP_GATEUP_COMPRESS_IDX_SHARED_EXPERT,

  IN_MLP_DOWN_WEIGHT_SHARED_EXPERT,
  IN_MLP_DOWN_BIAS_SHARED_EXPERT,
  IN_MLP_DOWN_DESCALE_SHARED_EXPERT,
  IN_MLP_DOWN_OFFSET_SHARED_EXPERT,
  IN_MLP_DOWN_SCALE_SHARED_EXPERT,
  IN_MLP_DOWN_COMPRESS_IDX_SHARED_EXPERT,

  // Shared expert gate weights
  IN_SHARED_EXPERT_GATE_WEIGHT,
  IN_SHARED_EXPERT_GATE_BIAS,
  IN_SHARED_EXPERT_GATE_DESCALE,
  IN_SHARED_EXPERT_GATE_OFFSET,
  IN_SHARED_EXPERT_GATE_SCALE,
  IN_SHARED_EXPERT_GATE_COMPRESS_IDX,

  // Expert weights
  IN_MLP_GATEUP_WEIGHT_EXPERT,
  IN_MLP_GATEUP_BIAS_EXPERT,
  IN_MLP_GATEUP_DESCALE_EXPERT,
  IN_MLP_GATEUP_OFFSET_EXPERT,
  IN_MLP_GATEUP_SCALE_EXPERT,
  IN_MLP_GATEUP_COMPRESS_IDX_EXPERT,

  IN_MLP_DOWN_WEIGHT_EXPERT,
  IN_MLP_DOWN_BIAS_EXPERT,
  IN_MLP_DOWN_DESCALE_EXPERT,
  IN_MLP_DOWN_OFFSET_EXPERT,
  IN_MLP_DOWN_SCALE_EXPERT,
  IN_MLP_DOWN_COMPRESS_IDX_EXPERT = 96,

  // Runtime tensors (not part of weight tensor array)
  IN_EXPERT_ARRAY = 97,
  IN_EXPERT_GROUP = 98,
  IN_ONE_HOT = 99,
  IN_ZERO_HOT = 100,

  // Legacy aliases for backward compatibility
  IN_MOE_EXPERT_W1_WEIGHT = IN_MLP_GATEUP_WEIGHT_EXPERT,
  IN_MOE_EXPERT_W2_WEIGHT = IN_MLP_DOWN_WEIGHT_EXPERT,
  IN_MOE_EXPERT_W3_WEIGHT = IN_MLP_GATEUP_WEIGHT_EXPERT,
  IN_MOE_SHARED_W1_WEIGHT = IN_MLP_GATEUP_WEIGHT_SHARED_EXPERT,
  IN_MOE_SHARED_W2_WEIGHT = IN_MLP_DOWN_WEIGHT_SHARED_EXPERT,
};

constexpr int32_t kInBlockSparseMoeGateWeight = static_cast<int32_t>(
    OneRecMoeBlockLayerTensorId::IN_BLOCK_SPARSE_MOE_GATE_WEIGHT);
constexpr int32_t kInBlockSparseMoeGateBias = static_cast<int32_t>(
    OneRecMoeBlockLayerTensorId::IN_BLOCK_SPARSE_MOE_GATE_BIAS);
constexpr int32_t kInBlockSparseMoeGateDescale = static_cast<int32_t>(
    OneRecMoeBlockLayerTensorId::IN_BLOCK_SPARSE_MOE_GATE_DESCALE);
constexpr int32_t kInBlockSparseMoeGateOffset = static_cast<int32_t>(
    OneRecMoeBlockLayerTensorId::IN_BLOCK_SPARSE_MOE_GATE_OFFSET);
constexpr int32_t kInBlockSparseMoeGateScale = static_cast<int32_t>(
    OneRecMoeBlockLayerTensorId::IN_BLOCK_SPARSE_MOE_GATE_SCALE);
constexpr int32_t kInBlockSparseMoeGateCompressIdx = static_cast<int32_t>(
    OneRecMoeBlockLayerTensorId::IN_BLOCK_SPARSE_MOE_GATE_COMPRESS_IDX);
constexpr int32_t kInMlpGateUpWeightSharedExpert = static_cast<int32_t>(
    OneRecMoeBlockLayerTensorId::IN_MLP_GATEUP_WEIGHT_SHARED_EXPERT);
constexpr int32_t kInMlpGateUpBiasSharedExpert = static_cast<int32_t>(
    OneRecMoeBlockLayerTensorId::IN_MLP_GATEUP_BIAS_SHARED_EXPERT);
constexpr int32_t kInMlpGateUpDescaleSharedExpert = static_cast<int32_t>(
    OneRecMoeBlockLayerTensorId::IN_MLP_GATEUP_DESCALE_SHARED_EXPERT);
constexpr int32_t kInMlpGateUpOffsetSharedExpert = static_cast<int32_t>(
    OneRecMoeBlockLayerTensorId::IN_MLP_GATEUP_OFFSET_SHARED_EXPERT);
constexpr int32_t kInMlpGateUpScaleSharedExpert = static_cast<int32_t>(
    OneRecMoeBlockLayerTensorId::IN_MLP_GATEUP_SCALE_SHARED_EXPERT);
constexpr int32_t kInMlpGateUpCompressIdxSharedExpert = static_cast<int32_t>(
    OneRecMoeBlockLayerTensorId::IN_MLP_GATEUP_COMPRESS_IDX_SHARED_EXPERT);
constexpr int32_t kInMlpDownWeightSharedExpert = static_cast<int32_t>(
    OneRecMoeBlockLayerTensorId::IN_MLP_DOWN_WEIGHT_SHARED_EXPERT);
constexpr int32_t kInMlpDownBiasSharedExpert = static_cast<int32_t>(
    OneRecMoeBlockLayerTensorId::IN_MLP_DOWN_BIAS_SHARED_EXPERT);
constexpr int32_t kInMlpDownDescaleSharedExpert = static_cast<int32_t>(
    OneRecMoeBlockLayerTensorId::IN_MLP_DOWN_DESCALE_SHARED_EXPERT);
constexpr int32_t kInMlpDownOffsetSharedExpert = static_cast<int32_t>(
    OneRecMoeBlockLayerTensorId::IN_MLP_DOWN_OFFSET_SHARED_EXPERT);
constexpr int32_t kInMlpDownScaleSharedExpert = static_cast<int32_t>(
    OneRecMoeBlockLayerTensorId::IN_MLP_DOWN_SCALE_SHARED_EXPERT);
constexpr int32_t kInMlpDownCompressIdxSharedExpert = static_cast<int32_t>(
    OneRecMoeBlockLayerTensorId::IN_MLP_DOWN_COMPRESS_IDX_SHARED_EXPERT);
constexpr int32_t kInSharedExpertGateWeight = static_cast<int32_t>(
    OneRecMoeBlockLayerTensorId::IN_SHARED_EXPERT_GATE_WEIGHT);
constexpr int32_t kInSharedExpertGateBias = static_cast<int32_t>(
    OneRecMoeBlockLayerTensorId::IN_SHARED_EXPERT_GATE_BIAS);
constexpr int32_t kInSharedExpertGateDescale = static_cast<int32_t>(
    OneRecMoeBlockLayerTensorId::IN_SHARED_EXPERT_GATE_DESCALE);
constexpr int32_t kInSharedExpertGateOffset = static_cast<int32_t>(
    OneRecMoeBlockLayerTensorId::IN_SHARED_EXPERT_GATE_OFFSET);
constexpr int32_t kInSharedExpertGateScale = static_cast<int32_t>(
    OneRecMoeBlockLayerTensorId::IN_SHARED_EXPERT_GATE_SCALE);
constexpr int32_t kInSharedExpertGateCompressIdx = static_cast<int32_t>(
    OneRecMoeBlockLayerTensorId::IN_SHARED_EXPERT_GATE_COMPRESS_IDX);
constexpr int32_t kInMlpGateUpWeightExpert = static_cast<int32_t>(
    OneRecMoeBlockLayerTensorId::IN_MLP_GATEUP_WEIGHT_EXPERT);
constexpr int32_t kInMlpGateUpBiasExpert = static_cast<int32_t>(
    OneRecMoeBlockLayerTensorId::IN_MLP_GATEUP_BIAS_EXPERT);
constexpr int32_t kInMlpGateUpDescaleExpert = static_cast<int32_t>(
    OneRecMoeBlockLayerTensorId::IN_MLP_GATEUP_DESCALE_EXPERT);
constexpr int32_t kInMlpGateUpOffsetExpert = static_cast<int32_t>(
    OneRecMoeBlockLayerTensorId::IN_MLP_GATEUP_OFFSET_EXPERT);
constexpr int32_t kInMlpGateUpScaleExpert = static_cast<int32_t>(
    OneRecMoeBlockLayerTensorId::IN_MLP_GATEUP_SCALE_EXPERT);
constexpr int32_t kInMlpGateUpCompressIdxExpert = static_cast<int32_t>(
    OneRecMoeBlockLayerTensorId::IN_MLP_GATEUP_COMPRESS_IDX_EXPERT);
constexpr int32_t kInMlpDownWeightExpert = static_cast<int32_t>(
    OneRecMoeBlockLayerTensorId::IN_MLP_DOWN_WEIGHT_EXPERT);
constexpr int32_t kInMlpDownBiasExpert =
    static_cast<int32_t>(OneRecMoeBlockLayerTensorId::IN_MLP_DOWN_BIAS_EXPERT);
constexpr int32_t kInMlpDownDescaleExpert = static_cast<int32_t>(
    OneRecMoeBlockLayerTensorId::IN_MLP_DOWN_DESCALE_EXPERT);
constexpr int32_t kInMlpDownOffsetExpert = static_cast<int32_t>(
    OneRecMoeBlockLayerTensorId::IN_MLP_DOWN_OFFSET_EXPERT);
constexpr int32_t kInMlpDownScaleExpert =
    static_cast<int32_t>(OneRecMoeBlockLayerTensorId::IN_MLP_DOWN_SCALE_EXPERT);
constexpr int32_t kInMlpDownCompressIdxExpert = static_cast<int32_t>(
    OneRecMoeBlockLayerTensorId::IN_MLP_DOWN_COMPRESS_IDX_EXPERT);
constexpr int32_t kInExpertArray =
    static_cast<int32_t>(OneRecMoeBlockLayerTensorId::IN_EXPERT_ARRAY);
constexpr int32_t kInExpertGroup =
    static_cast<int32_t>(OneRecMoeBlockLayerTensorId::IN_EXPERT_GROUP);
constexpr int32_t kInOneHot =
    static_cast<int32_t>(OneRecMoeBlockLayerTensorId::IN_ONE_HOT);
constexpr int32_t kInZeroHot =
    static_cast<int32_t>(OneRecMoeBlockLayerTensorId::IN_ZERO_HOT);
constexpr int32_t kInMoeExpertW1Weight =
    static_cast<int32_t>(OneRecMoeBlockLayerTensorId::IN_MOE_EXPERT_W1_WEIGHT);
constexpr int32_t kInMoeExpertW2Weight =
    static_cast<int32_t>(OneRecMoeBlockLayerTensorId::IN_MOE_EXPERT_W2_WEIGHT);
constexpr int32_t kInMoeExpertW3Weight =
    static_cast<int32_t>(OneRecMoeBlockLayerTensorId::IN_MOE_EXPERT_W3_WEIGHT);
constexpr int32_t kInMoeSharedW1Weight =
    static_cast<int32_t>(OneRecMoeBlockLayerTensorId::IN_MOE_SHARED_W1_WEIGHT);
constexpr int32_t kInMoeSharedW2Weight =
    static_cast<int32_t>(OneRecMoeBlockLayerTensorId::IN_MOE_SHARED_W2_WEIGHT);

static const std::unordered_map<std::string, int32_t>
    kOneRecEncoderWeightMapping = {
        {"layer.0.layer_norm.weight", kInLayerNormWeight},
        {"layer.0.SelfAttention.q.weight", kInQWeight},
        {"layer.0.SelfAttention.k.weight", kInKWeight},
        {"layer.0.SelfAttention.v.weight", kInVWeight},
        {"layer.0.SelfAttention.o.weight", kInSelfAttnOutWeight},
        {"layer.0.SelfAttention.relative_attention_bias.weight",
         kInRelativeAttentionBiasWeight},
        {"layer.1.layer_norm.weight", kInFinalLayerNormWeight},
        {"layer.1.DenseReluDense.wi.weight", kInFfnWi1Weight},
        {"layer.1.DenseReluDense.wo.weight", kInFfnWoWeight},
        {"layer.1.DenseReluDense.gate_proj.weight", kInFfnWi0Weight},
        {"layer.1.ffn.wi.weight", kInFfnWi1Weight},
        {"layer.1.ffn.wo.weight", kInFfnWoWeight},
        {"layer.1.ffn.gate_proj.weight", kInFfnWi0Weight},
        // Alternative format
        {"0.layer_norm.weight", kInLayerNormWeight},
        {"0.SelfAttention.q.weight", kInQWeight},
        {"0.SelfAttention.k.weight", kInKWeight},
        {"0.SelfAttention.v.weight", kInVWeight},
        {"0.SelfAttention.o.weight", kInSelfAttnOutWeight},
        {"0.SelfAttention.relative_attention_bias.weight",
         kInRelativeAttentionBiasWeight},
        {"1.layer_norm.weight", kInFinalLayerNormWeight},
        {"1.DenseReluDense.wi.weight", kInFfnWi1Weight},
        {"1.DenseReluDense.wo.weight", kInFfnWoWeight},
        {"1.DenseReluDense.gate_proj.weight", kInFfnWi0Weight},
        {"1.ffn.wi.weight", kInFfnWi1Weight},
        {"1.ffn.wo.weight", kInFfnWoWeight},
        {"1.ffn.gate_proj.weight", kInFfnWi0Weight},
};

static const std::unordered_map<std::string, int32_t>
    kOneRecDecoderWeightMapping = {
        {"layer.0.layer_norm.weight", kInLayerNormWeight},
        {"layer.0.SelfAttention.q.weight", kInQWeight},
        {"layer.0.SelfAttention.k.weight", kInKWeight},
        {"layer.0.SelfAttention.v.weight", kInVWeight},
        {"layer.0.SelfAttention.o.weight", kInSelfAttnOutWeight},
        {"layer.0.SelfAttention.relative_attention_bias.weight",
         kInRelativeAttentionBiasWeight},
        {"layer.1.layer_norm.weight", kInCrossLayerNormWeight},
        {"layer.1.EncDecAttention.q.weight", kInCrossQWeight},
        {"layer.1.EncDecAttention.k.weight", kInCrossKWeight},
        {"layer.1.EncDecAttention.v.weight", kInCrossVWeight},
        {"layer.1.EncDecAttention.o.weight", kInCrossAttnOutWeight},
        {"layer.2.layer_norm.weight", kInFinalLayerNormWeight},
        {"layer.2.DenseReluDense.wi.weight", kInFfnWi1Weight},
        {"layer.2.DenseReluDense.wo.weight", kInFfnWoWeight},
        {"layer.2.DenseReluDense.gate_proj.weight", kInFfnWi0Weight},
        // Alternative format
        {"0.layer_norm.weight", kInLayerNormWeight},
        {"0.SelfAttention.q.weight", kInQWeight},
        {"0.SelfAttention.k.weight", kInKWeight},
        {"0.SelfAttention.v.weight", kInVWeight},
        {"0.SelfAttention.o.weight", kInSelfAttnOutWeight},
        {"0.SelfAttention.relative_attention_bias.weight",
         kInRelativeAttentionBiasWeight},
        {"1.layer_norm.weight", kInCrossLayerNormWeight},
        {"1.EncDecAttention.q.weight", kInCrossQWeight},
        {"1.EncDecAttention.k.weight", kInCrossKWeight},
        {"1.EncDecAttention.v.weight", kInCrossVWeight},
        {"1.EncDecAttention.o.weight", kInCrossAttnOutWeight},
        {"2.layer_norm.weight", kInFinalLayerNormWeight},
        {"2.DenseReluDense.wi.weight", kInFfnWi1Weight},
        {"2.DenseReluDense.wo.weight", kInFfnWoWeight},
        {"2.DenseReluDense.gate_proj.weight", kInFfnWi0Weight},
        {"2.ffn.wi.weight", kInFfnWi1Weight},
        {"2.ffn.wo.weight", kInFfnWoWeight},
        {"2.ffn.gate_proj.weight", kInFfnWi0Weight},
};

static std::unordered_map<std::string, int32_t>
get_onerec_decoder_moe_weight_mapping() {
  std::unordered_map<std::string, int32_t> mapping =
      kOneRecDecoderWeightMapping;

  mapping.emplace("layer.2.ffn.gate.weight", kInBlockSparseMoeGateWeight);
  mapping.emplace("layer.2.ffn.router.weight", kInBlockSparseMoeGateWeight);
  mapping.emplace("2.ffn.gate.weight", kInBlockSparseMoeGateWeight);
  mapping.emplace("2.ffn.router.weight", kInBlockSparseMoeGateWeight);

  mapping.emplace("layer.2.ffn.shared_experts.w1.weight",
                  kInMlpGateUpWeightSharedExpert);
  mapping.emplace("layer.2.ffn.shared_experts.w3.weight",
                  kInMlpGateUpWeightSharedExpert);
  mapping.emplace("layer.2.ffn.shared_experts.w2.weight",
                  kInMlpDownWeightSharedExpert);

  mapping.emplace("layer.2.ffn.shared_expert.gate.weight",
                  kInSharedExpertGateWeight);
  mapping.emplace("layer.2.ffn.shared_expert.gate.bias",
                  kInSharedExpertGateBias);
  mapping.emplace("layer.2.ffn.shared_expert.gate.weight_scale",
                  kInSharedExpertGateScale);
  mapping.emplace("layer.2.ffn.shared_expert.gate.weight_offset",
                  kInSharedExpertGateOffset);

  // Expert weights are handled by
  // process_expert_weights()/merge_experts_weights to avoid ambiguous suffix
  // matching and keep deterministic loading.

  return mapping;
}

static const std::unordered_map<std::string, int32_t>
    kOneRecDecoderMoeWeightMapping = get_onerec_decoder_moe_weight_mapping();

static const std::unordered_map<int32_t, int32_t> kOneRecWeightShard = {
    {kInQWeight, 0},
    {kInKWeight, 0},
    {kInVWeight, 0},
    {kInSelfAttnOutWeight, 1},
    {kInCrossQWeight, 0},
    {kInCrossKWeight, 0},
    {kInCrossVWeight, 0},
    {kInCrossAttnOutWeight, 1},
    {kInFfnWi0Weight, 0},
    {kInFfnWi1Weight, 0},
    {kInFfnWoWeight, 1},
    // MoE
    {kInBlockSparseMoeGateWeight, 0},
    {kInMlpGateUpWeightExpert, 0},
    {kInMlpDownWeightExpert, 1},
    // Shared experts
    {kInMlpGateUpWeightSharedExpert, 0},
    {kInMlpGateUpOffsetSharedExpert, 0},
    {kInMlpGateUpScaleSharedExpert, 0},
    {kInMlpDownWeightSharedExpert, 1},
    {kInMlpDownOffsetSharedExpert, 1},
    {kInMlpDownScaleSharedExpert, 1},
    {kInSharedExpertGateWeight, 0},
    {kInSharedExpertGateBias, 0},
    {kInSharedExpertGateScale, 0},
    {kInSharedExpertGateOffset, 0},
};

}  // namespace

NpuOneRecBlockLayerImpl::NpuOneRecBlockLayerImpl(const ModelContext& context,
                                                 bool is_decoder,
                                                 int32_t layer_id)
    : BaseLayer(context), is_decoder_(is_decoder), layer_id_(layer_id) {
  const auto& args = context.get_model_args();
  const auto& parallel_args = context.get_parallel_args();
  param_from_args(prefill_param_, args, parallel_args, /*is_prefill=*/true);
  param_from_args(prefill_param_atb_, args, parallel_args, /*is_prefill=*/true);
  prefill_param_atb_.matmulBackend = atb_speed::common::OpBackend::ATB;
  param_from_args(decode_param_, args, parallel_args, /*is_prefill=*/false);
  if (use_legacy_onerec_prefill_only_contract() && is_decoder_) {
    param_from_args(decoder_prefill_only_decode_param_,
                    args,
                    parallel_args,
                    /*is_prefill=*/true);
    decoder_prefill_only_decode_param_.emptyCrossAttn = false;
    param_from_args(decoder_prefill_only_decode_param_atb_,
                    args,
                    parallel_args,
                    /*is_prefill=*/true);
    decoder_prefill_only_decode_param_atb_.emptyCrossAttn = false;
    decoder_prefill_only_decode_param_atb_.matmulBackend =
        atb_speed::common::OpBackend::ATB;
  }

  const int32_t weight_count = prefill_param_.use_moe
                                   ? kOneRecMoeWeightCountPerLayer
                                   : kOneRecWeightCountPerLayer;
  at_weight_tensors_.resize(weight_count);
  atb_weight_tensors_.resize(weight_count);

  placeholder_vec_ = {1, 1};
  dtype_ = c10::typeMetaToScalarType(context.get_tensor_options().dtype());
  device_id_ = context.get_tensor_options().device().index();

  auto placeholder_tensor = torch::empty({1, 1}, torch::kInt32).to(device_);
  placeholder_ = atb_speed::Utils::AtTensor2Tensor(placeholder_tensor);
  at_placeholder_ = torch::empty({1, args.hidden_size()}, dtype_).to(device_);

  for (int32_t i = 0; i < weight_count; ++i) {
    at_weight_tensors_[i] =
        torch::zeros({1, args.hidden_size()}).to(context.get_tensor_options());
  }

  if (prefill_param_.use_moe) {
    auto device = context.get_tensor_options().device();
    one_hot_ = torch::tensor({1}, torch::kInt32).to(device);
    zero_hot_ = torch::tensor({0}, torch::kInt32).to(device);
    expert_group_ = torch::tensor({1}, torch::dtype(torch::kInt32)).to(device);
  }
}

void NpuOneRecBlockLayerImpl::param_from_args(
    atb_speed::onerec::BlockLayerParam& param,
    const ModelArgs& args,
    const ParallelArgs& parallel_args,
    bool is_prefill,
    const ModelInputParams* input_params) {
  (void)input_params;

  param.isFA = false;
  param.isPrefill = is_prefill;
  param.isBF16 = args.dtype() == "bfloat16";
  param.isPack = true;
  param.supportSwiGLU = true;
  // Shared experts in the current OneRec MoE path are loaded as bf16/fp
  // weights. Do not force the dedicated SwigluQuant scale contract unless the
  // shared expert path is explicitly wired for dynamic quant.
  param.enableSwiGLUQuantForSharedExperts = false;
  param.supportLcoc = is_prefill;
  param.supportSpeculate = false;
  param.enableSplitFuse = FLAGS_enable_chunked_prefill && is_prefill;
  param.supportLora = false;
  param.loraEnableGMM = false;
  param.enableLogN = false;
  param.kvQuant = false;
  param.enableIntraLayerAddNorm = false;
  param.enableInterLayerAddNorm = false;
  param.isDecoder = is_decoder_;
  param.isOneRecEncoder = !is_decoder_;
  param.use_xattn = is_decoder_ && is_onerec_xattention_mode();
  param.enableCrossAttentionKernel = true;
  param.enableOneRecPrefillOnly = use_legacy_onerec_prefill_only_contract();
  param.backend = FLAGS_communication_backend;
  param.matmulBackend = kEnableOneRecAclnnAttentionLinear
                            ? atb_speed::common::OpBackend::ACLNN
                            : atb_speed::common::OpBackend::ATB;
  param.rank = parallel_args.rank();
  param.worldSize = parallel_args.world_size();
  param.quantType = 0;
  param.quantGroupSize = 64;
  param.blockSize = FLAGS_block_size;

  const int64_t args_n_heads =
      is_decoder_ ? args.decoder_n_heads() : args.n_heads();
  const int64_t args_head_dim =
      is_decoder_ ? args.decoder_head_dim() : args.head_dim();
  param.numAttentionHeadsPerRank = args_n_heads / param.worldSize;
  param.hiddenSizePerAttentionHead = args_head_dim;
  // Existing OneRec configs use moe_use_shared_experts to identify the path
  // that needs attention scaling; do not add a generic ModelArgs field for it.
  param.useAttentionScaling = args.moe_use_shared_experts();

  const auto general_kv_heads = args.n_kv_heads();
  const auto decoder_kv_heads = args.decoder_n_kv_heads().has_value()
                                    ? args.decoder_n_kv_heads()
                                    : general_kv_heads;
  const int64_t args_kv_heads =
      is_decoder_ ? decoder_kv_heads.value_or(args.decoder_n_heads())
                  : general_kv_heads.value_or(args.n_heads());
  param.numKeyValueHeadsPerRank =
      static_cast<int>(args_kv_heads / param.worldSize);
  param.rmsNormEps = args.rms_norm_eps();

  param.seqLen = {};
  param.tokenOffset = {};
  param.packQuantType = {1, 1};
  param.linearQuantType = {0, -1, -1, 0, 0, -1, 0};
  param.layerId = layer_id_;
  param.linearTransposeType = {1, 1, 1, 1, 1, 1, 1};

  if (param.isBF16) {
    param.linearDescs = {
        static_cast<int>(atb_speed::common::LinearDesc::BFLOAT16_DESC),
        static_cast<int>(atb_speed::common::LinearDesc::BFLOAT16_DESC),
        static_cast<int>(atb_speed::common::LinearDesc::BFLOAT16_DESC),
        static_cast<int>(atb_speed::common::LinearDesc::BFLOAT16_DESC)};
  } else {
    param.linearDescs = {
        static_cast<int>(atb_speed::common::LinearDesc::FLOAT16_DESC),
        static_cast<int>(atb_speed::common::LinearDesc::FLOAT16_DESC),
        static_cast<int>(atb_speed::common::LinearDesc::FLOAT16_DESC),
        static_cast<int>(atb_speed::common::LinearDesc::FLOAT16_DESC)};
  }

  param.use_moe = args.use_moe() && is_decoder_;
  if (param.use_moe) {
    ep_size_ = 1;
    const int32_t ep_rank = 0;
    ep_local_tp_size_ = parallel_args.world_size() / ep_size_;
    CHECK_EQ(parallel_args.world_size(), ep_size_ * ep_local_tp_size_);
    ep_local_tp_rank_ = parallel_args.rank() % ep_local_tp_size_;

    num_experts_per_partition_ = args.n_routed_experts() / ep_size_;
    start_expert_id_ = ep_rank * num_experts_per_partition_;
    end_expert_id_ = start_expert_id_ + num_experts_per_partition_ - 1;

    resize_experts_weights(num_experts_per_partition_);

    param.moe_config = std::make_unique<atb_speed::onerec::OneRecMoEConfig>();
    param.moe_config->moe_topk = args.num_experts_per_tok();
    param.moe_config->moe_num_experts = args.n_routed_experts();
    param.moe_config->moe_score_func = "softmax";
    param.moe_config->moe_route_scale = args.moe_route_scale();
    param.moe_config->moe_inter_dim = args.moe_intermediate_size();
    param.moe_config->use_bf16 = param.isBF16;
    param.moe_config->hasSharedExpertGate = false;
    param.moe_config->moe_use_shared_experts = args.moe_use_shared_experts();
    param.moe_config->moe_num_shared_experts = args.n_shared_experts();
    param.moe_config->enable_integrated_softmax_topk = true;

    param.moeLinearQuantType = {atb_speed::common::LinearType::FP,
                                atb_speed::common::LinearType::FP,
                                atb_speed::common::LinearType::INVALID,
                                atb_speed::common::LinearType::FP};
  }
}

void NpuOneRecBlockLayerImpl::verify_loaded_weights(
    const std::string& prefix) const {
  std::unordered_map<std::string, int32_t> filtered_weight_mapping;
  const auto* weight_mapping = [this, &filtered_weight_mapping]()
      -> const std::unordered_map<std::string, int32_t>* {
    if (prefill_param_.use_moe) {
      filtered_weight_mapping.clear();
      const bool has_shared_experts =
          prefill_param_.moe_config != nullptr &&
          prefill_param_.moe_config->moe_use_shared_experts;
      const bool has_shared_expert_gate =
          prefill_param_.moe_config != nullptr &&
          prefill_param_.moe_config->hasSharedExpertGate;
      for (const auto& [name, index] : kOneRecDecoderMoeWeightMapping) {
        bool should_include = true;
        if (!has_shared_experts &&
            name.find("shared_expert") != std::string::npos) {
          should_include = false;
        }
        if (should_include && !has_shared_expert_gate &&
            (name.find("shared_expert.gate") != std::string::npos ||
             name.find("shared_expert_gate") != std::string::npos)) {
          should_include = false;
        }
        if (should_include) {
          filtered_weight_mapping.emplace(name, index);
        }
      }
      return &filtered_weight_mapping;
    }
    return is_decoder_ ? &kOneRecDecoderWeightMapping
                       : &kOneRecEncoderWeightMapping;
  }();

  // verify_loaded_weights() runs before merge_loaded_weights().
  // Only allow placeholders for tensors that are intentionally absent before
  // merge in the current mode.
  std::set<int32_t> allowed_placeholders;
  if (prefill_param_.use_moe) {
    // MoE decoder path does not consume dense FFN gate/up/down tensors.
    allowed_placeholders.insert(kInFfnWi0Weight);
    allowed_placeholders.insert(kInFfnWi1Weight);
    allowed_placeholders.insert(kInFfnWoWeight);
  }
  for (const auto& [name, index] : *weight_mapping) {
    const auto sizes = at_weight_tensors_[index].sizes();
    const bool is_placeholder = (sizes.size() == 2 && sizes[0] == 1);
    const bool expected_placeholder = allowed_placeholders.count(index) > 0;
    const bool is_relative_bias = (index == kInRelativeAttentionBiasWeight);
    if (is_placeholder && !expected_placeholder && !is_relative_bias) {
      CHECK(false) << "weight is not loaded for " << prefix << name;
    }
  }

  if (prefill_param_.use_moe) {
    CHECK(validate_decoder_moe_weights(prefix))
        << "OneRec MoE expert weights are incomplete for " << prefix;
  }
}

bool NpuOneRecBlockLayerImpl::validate_decoder_moe_weights(
    const std::string& prefix) const {
  const auto gate_it = experts_weights_.find("gate_proj.weight");
  const auto up_it = experts_weights_.find("up_proj.weight");
  const auto down_it = experts_weights_.find("down_proj.weight");
  if (gate_it == experts_weights_.end() || up_it == experts_weights_.end() ||
      down_it == experts_weights_.end()) {
    LOG(ERROR) << "Missing OneRec MoE expert tensors in " << prefix
               << " (layer " << layer_id_
               << ", gate/up/down map entry not found).";
    return false;
  }

  const auto& gate_weights = gate_it->second;
  const auto& up_weights = up_it->second;
  const auto& down_weights = down_it->second;

  if (gate_weights.size() != up_weights.size() ||
      gate_weights.size() != down_weights.size()) {
    LOG(ERROR) << "OneRec MoE expert vector size mismatch in " << prefix
               << ": gate=" << gate_weights.size()
               << ", up=" << up_weights.size()
               << ", down=" << down_weights.size() << ", layer " << layer_id_;
    return false;
  }

  for (size_t i = 0; i < gate_weights.size(); ++i) {
    const bool gate_defined = gate_weights[i].defined();
    const bool up_defined = up_weights[i].defined();
    const bool down_defined = down_weights[i].defined();
    if (gate_defined != up_defined || gate_defined != down_defined) {
      LOG(ERROR) << "OneRec MoE expert tensor mismatch in " << prefix
                 << " at local expert " << i << ": gate=" << gate_defined
                 << ", up=" << up_defined << ", down=" << down_defined
                 << ", layer " << layer_id_;
      return false;
    }
    if (!gate_defined) {
      LOG(ERROR) << "Missing OneRec MoE tensor for local expert " << i << " in "
                 << prefix << " (layer " << layer_id_ << ").";
      return false;
    }
  }
  return true;
}

void NpuOneRecBlockLayerImpl::merge_loaded_weights() {
  const bool q_loaded = !(at_weight_tensors_[kInQWeight].sizes().size() == 2 &&
                          at_weight_tensors_[kInQWeight].sizes()[0] == 1);
  const bool k_loaded = !(at_weight_tensors_[kInKWeight].sizes().size() == 2 &&
                          at_weight_tensors_[kInKWeight].sizes()[0] == 1);
  const bool v_loaded = !(at_weight_tensors_[kInVWeight].sizes().size() == 2 &&
                          at_weight_tensors_[kInVWeight].sizes()[0] == 1);
  CHECK(q_loaded && k_loaded && v_loaded)
      << "OneRec QKV weights are not properly loaded.";

  auto new_q_weight = torch::cat({at_weight_tensors_[kInQWeight],
                                  at_weight_tensors_[kInKWeight],
                                  at_weight_tensors_[kInVWeight]},
                                 0);
  at_weight_tensors_[kInQWeight] = new_q_weight;
  at_weight_tensors_[kInKWeight] =
      torch::zeros({1, at_weight_tensors_[kInQWeight].size(1)})
          .to(device_)
          .to(dtype_);
  at_weight_tensors_[kInVWeight] =
      torch::zeros({1, at_weight_tensors_[kInQWeight].size(1)})
          .to(device_)
          .to(dtype_);

  // Keep decoder cross-attention Q/K/V unpacked for current OneRec ATB
  // contract. Do not merge IN_CROSS_{Q,K,V}_WEIGHT here.

  if (!prefill_param_.use_moe) {
    const bool wi0_loaded =
        !(at_weight_tensors_[kInFfnWi0Weight].sizes().size() == 2 &&
          at_weight_tensors_[kInFfnWi0Weight].sizes()[0] == 1);
    const bool wi1_loaded =
        !(at_weight_tensors_[kInFfnWi1Weight].sizes().size() == 2 &&
          at_weight_tensors_[kInFfnWi1Weight].sizes()[0] == 1);
    CHECK(wi0_loaded && wi1_loaded)
        << "OneRec FFN gate/up weights are not properly loaded.";

    auto new_gate_up_weight = torch::cat({at_weight_tensors_[kInFfnWi0Weight],
                                          at_weight_tensors_[kInFfnWi1Weight]},
                                         0);
    at_weight_tensors_[kInFfnWi0Weight] = new_gate_up_weight;
    at_weight_tensors_[kInFfnWi1Weight] =
        torch::zeros({1, at_weight_tensors_[kInFfnWi0Weight].size(1)})
            .to(device_)
            .to(dtype_);
  } else {
    merge_experts_weights();
    merge_shared_experts_weights();
  }

  const uint64_t weight_count = prefill_param_.use_moe
                                    ? kOneRecMoeWeightCountPerLayer
                                    : kOneRecWeightCountPerLayer;
  for (int32_t i = 0; i < static_cast<int32_t>(weight_count); ++i) {
    if (!at_weight_tensors_[i].defined()) {
      at_weight_tensors_[i] = torch::zeros(
          {1, 1}, torch::TensorOptions().device(device_).dtype(dtype_));
    }
    if (!at_weight_tensors_[i].is_contiguous()) {
      at_weight_tensors_[i] = at_weight_tensors_[i].contiguous();
    }
  }

  for (int32_t i = 0; i < static_cast<int32_t>(weight_count); ++i) {
    atb_weight_tensors_[i] =
        atb_speed::Utils::AtTensor2Tensor(at_weight_tensors_[i]);
  }

  LOG(INFO) << "OneRec BlockLayer merge_loaded_weights calling init_layer"
            << ", layer_role=" << (is_decoder_ ? "decoder" : "encoder")
            << ", layer_id=" << layer_id_ << ", weight_count=" << weight_count;
  const int64_t init_status = init_layer();
  LOG(INFO) << "OneRec BlockLayer merge_loaded_weights init_layer returned"
            << ", layer_role=" << (is_decoder_ ? "decoder" : "encoder")
            << ", layer_id=" << layer_id_ << ", status=" << init_status;
  CHECK_EQ(init_status, atb::NO_ERROR)
      << "OneRec BlockLayer init_layer failed, layer_role="
      << (is_decoder_ ? "decoder" : "encoder") << ", layer_id=" << layer_id_;
}

void NpuOneRecBlockLayerImpl::load_state_dict(const StateDict& state_dict) {
  const auto target_weight_dtype = [this]() -> torch::ScalarType {
    if (torch_dtype_.empty()) {
      return dtype_;
    }
    if (torch_dtype_ == "float16") {
      return torch::kFloat16;
    }
    if (torch_dtype_ == "bfloat16") {
      return torch::kBFloat16;
    }
    if (torch_dtype_ == "float32") {
      return torch::kFloat32;
    }
    if (torch_dtype_ == "float64") {
      return torch::kFloat64;
    }
    if (torch_dtype_ == "int8") {
      return torch::kInt8;
    }
    if (torch_dtype_ == "int16") {
      return torch::kInt16;
    }
    if (torch_dtype_ == "int32") {
      return torch::kInt32;
    }
    if (torch_dtype_ == "int64") {
      return torch::kInt64;
    }
    if (torch_dtype_ == "uint8") {
      return torch::kUInt8;
    }
    if (torch_dtype_ == "bool") {
      return torch::kBool;
    }
    LOG(FATAL) << "Unsupported OneRec weight dtype " << torch_dtype_
               << ", layer_id=" << layer_id_;
    return dtype_;
  };
  const auto correct_tensor_dtype = [this, &target_weight_dtype](
                                        torch::Tensor& tensor,
                                        const std::string& tensor_name) {
    if (absl::EndsWith(tensor_name, "deq_scale") &&
        torch_dtype_ == "bfloat16") {
      return;
    }
    if (tensor.dtype() != torch::kInt8 && tensor.dtype() != torch::kInt32 &&
        tensor.dtype() != torch::kInt64) {
      tensor = tensor.to(target_weight_dtype());
    }
  };
  const auto load_weight = [this, &state_dict, &correct_tensor_dtype](
                               const std::string& tensor_name,
                               int32_t weight_position,
                               int32_t shard_dim = -1) {
    for (const auto& [name, tensor] : state_dict) {
      if (!absl::EndsWith(name, tensor_name)) {
        continue;
      }
      torch::Tensor mutable_tensor =
          (shard_dim >= 0 && parallel_args_.world_size() > 1)
              ? state_dict.get_sharded_tensor(name,
                                              shard_dim,
                                              parallel_args_.rank(),
                                              parallel_args_.world_size())
              : tensor;
      if (!mutable_tensor.defined()) {
        continue;
      }
      correct_tensor_dtype(mutable_tensor, tensor_name);
      at_weight_tensors_[weight_position] = mutable_tensor.to(device_);
      return;
    }
  };

  const auto& weight_mapping =
      [this]() -> const std::unordered_map<std::string, int32_t>& {
    if (prefill_param_.use_moe) {
      return kOneRecDecoderMoeWeightMapping;
    }
    return is_decoder_ ? kOneRecDecoderWeightMapping
                       : kOneRecEncoderWeightMapping;
  }();

  if (prefill_param_.use_moe) {
    for (auto& [key, tensors] : experts_weights_) {
      (void)key;
      for (auto& t : tensors) {
        t = torch::Tensor();
      }
    }
    shared_expert_weights_map_.clear();
    shared_expert_gate_weights_.clear();
    shared_expert_up_weights_.clear();
    shared_expert_down_weights_.clear();

    for (const auto& [state_key, tensor] : state_dict) {
      if (state_key.find(".ffn.experts.") != std::string::npos) {
        process_expert_weights(state_dict, state_key, tensor);
      }
    }

    for (const auto& [state_key, tensor] : state_dict) {
      const bool is_shared_expert =
          (state_key.find(".ffn.shared_experts.") != std::string::npos ||
           state_key.find(".ffn.shared_expert.") != std::string::npos);
      if (is_shared_expert) {
        process_shared_expert_weights(state_dict, state_key, tensor);
      }
    }
  }

  const auto load_dense_fused_ffn_weights =
      [this, &state_dict, &correct_tensor_dtype](const std::string& state_key,
                                                 const torch::Tensor& tensor) {
        if (absl::StrContains(state_key, ".ffn.experts.") ||
            absl::StrContains(state_key, ".ffn.shared_experts.") ||
            absl::StrContains(state_key, ".ffn.shared_expert.")) {
          return;
        }

        if (absl::StrContains(state_key, ".DenseReluDense.weight1") ||
            absl::StrContains(state_key, ".ffn.weight1")) {
          torch::Tensor fused_gate_up =
              (parallel_args_.world_size() > 1)
                  ? state_dict.get_sharded_tensor(state_key,
                                                  /*dim=*/0,
                                                  parallel_args_.rank(),
                                                  parallel_args_.world_size())
                  : tensor;
          if (!fused_gate_up.defined()) {
            return;
          }
          CHECK_EQ(fused_gate_up.dim(), 2)
              << "OneRec fused FFN weight1 must be 2D, got "
              << fused_gate_up.sizes() << " from " << state_key;
          CHECK_EQ(fused_gate_up.size(0) % 2, 0)
              << "OneRec fused FFN weight1 dim0 must be even, got "
              << fused_gate_up.sizes() << " from " << state_key;
          correct_tensor_dtype(fused_gate_up, state_key);
          auto chunks = fused_gate_up.chunk(2, 0);
          at_weight_tensors_[kInFfnWi0Weight] =
              chunks[0].contiguous().to(device_);
          at_weight_tensors_[kInFfnWi1Weight] =
              chunks[1].contiguous().to(device_);
          return;
        }

        if (absl::StrContains(state_key, ".DenseReluDense.weight2") ||
            absl::StrContains(state_key, ".ffn.weight2")) {
          torch::Tensor wo_weight =
              (parallel_args_.world_size() > 1)
                  ? state_dict.get_sharded_tensor(state_key,
                                                  /*dim=*/1,
                                                  parallel_args_.rank(),
                                                  parallel_args_.world_size())
                  : tensor;
          if (!wo_weight.defined()) {
            return;
          }
          correct_tensor_dtype(wo_weight, state_key);
          at_weight_tensors_[kInFfnWoWeight] = wo_weight.to(device_);
        }
      };

  for (const auto& [state_key, tensor] : state_dict) {
    load_dense_fused_ffn_weights(state_key, tensor);
  }

  std::vector<std::pair<std::string, int32_t>> ordered_mapping(
      weight_mapping.begin(), weight_mapping.end());
  std::sort(
      ordered_mapping.begin(),
      ordered_mapping.end(),
      [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });
  for (const auto& [name, index] : ordered_mapping) {
    const bool is_relative_bias = (index == kInRelativeAttentionBiasWeight);
    bool weight_exists = false;
    for (const auto& [state_key, tensor] : state_dict) {
      (void)tensor;
      if (absl::EndsWith(state_key, name)) {
        weight_exists = true;
        break;
      }
    }
    if (is_relative_bias && !weight_exists) {
      continue;
    }

    const auto it = kOneRecWeightShard.find(index);
    if (it != kOneRecWeightShard.end()) {
      load_weight(name, index, it->second);
    } else {
      load_weight(name, index);
    }
  }
}

int64_t NpuOneRecBlockLayerImpl::init_layer() {
  name_ =
      is_decoder_ ? "onerec_decoder_block_layer" : "onerec_encoder_block_layer";
  model_name_ = "onerec";
  if (decode_param_.use_xattn) {
    CHECK_OPERATION_STATUS_RETURN(init_attn_mask());
  }
  CHECK_OPERATION_STATUS_RETURN(init_node(prefill_node_, prefill_param_));
  if (kEnableOneRecAclnnAttentionLinear &&
      kOneRecAclnnAttentionLinearMinTokens > 0) {
    CHECK_OPERATION_STATUS_RETURN(
        init_node(prefill_node_atb_, prefill_param_atb_));
  }
  if (is_decoder_) {
    if (use_legacy_onerec_prefill_only_contract()) {
      CHECK_OPERATION_STATUS_RETURN(
          init_node(decoder_prefill_only_decode_node_,
                    decoder_prefill_only_decode_param_));
      if (kEnableOneRecAclnnAttentionLinear &&
          kOneRecAclnnAttentionLinearMinTokens > 0) {
        CHECK_OPERATION_STATUS_RETURN(
            init_node(decoder_prefill_only_decode_node_atb_,
                      decoder_prefill_only_decode_param_atb_));
      }
      LOG(INFO) << "OneRec BlockLayer init_layer success"
                << ", layer_role=" << (is_decoder_ ? "decoder" : "encoder")
                << ", layer_id=" << layer_id_ << ", status=" << atb::NO_ERROR;
      return atb::NO_ERROR;
    }
    const int64_t decode_status = init_node(decode_node_, decode_param_);
    LOG(INFO) << "OneRec BlockLayer init_layer node returned"
              << ", node=decoder-decode"
              << ", layer_id=" << layer_id_ << ", status=" << decode_status;
    CHECK_OPERATION_STATUS_RETURN(decode_status);
  } else {
    LOG(INFO) << "OneRec BlockLayer init_layer skip decode node"
              << ", layer_role=" << (is_decoder_ ? "decoder" : "encoder")
              << ", layer_id=" << layer_id_;
  }
  return atb::NO_ERROR;
}

int64_t NpuOneRecBlockLayerImpl::init_attn_mask() {
  torch::Dtype dtype =
      prefill_param_.isBF16 ? torch::kBFloat16 : torch::kFloat16;
  decode_attn_mask_ = EnsureNdFormat(
      torch::zeros({1}, torch::TensorOptions().device(device_).dtype(dtype)));
  prefill_attn_mask_ = EnsureNdFormat(torch::zeros(
      {1, 1, 1, 1}, torch::TensorOptions().device(device_).dtype(dtype)));
  return atb::NO_ERROR;
}

int64_t NpuOneRecBlockLayerImpl::init_node(
    atb_speed::Model::Node& node,
    atb_speed::onerec::BlockLayerParam& param) {
  atb::Operation* operation = nullptr;
  atb::Status status = atb_speed::onerec::BlockLayer(param, &operation);
  if (status != atb::NO_ERROR) {
    LOG(ERROR) << "Failed to create ONEREC BlockLayer operation, status: "
               << status;
    return status;
  }

  node.operation.reset(operation);
  if (node.operation == nullptr) {
    LOG(ERROR) << "node.operation is null after creation";
    return -1;
  }

  uint32_t input_num = node.operation->GetInputNum();
  uint32_t output_num = node.operation->GetOutputNum();

  node.inTensors.resize(input_num);
  node.outTensors.resize(output_num);

  const uint64_t weight_count = param.use_moe ? kOneRecMoeWeightCountPerLayer
                                              : kOneRecWeightCountPerLayer;
  for (size_t weight_tensor_id = 0; weight_tensor_id < weight_count;
       ++weight_tensor_id) {
    if (weight_tensor_id < input_num) {
      node.inTensors.at(weight_tensor_id) =
          &atb_weight_tensors_[weight_tensor_id];
    }
  }

  node.variantPack.inTensors.resize(input_num);
  node.variantPack.outTensors.resize(output_num);

  return atb::NO_ERROR;
}

void NpuOneRecBlockLayerImpl::set_cross_kv_cache_for_graph(
    const torch::Tensor& k_cache,
    const torch::Tensor& v_cache) {
  cross_k_cache_ = k_cache;
  cross_v_cache_ = v_cache;
}

torch::Tensor NpuOneRecBlockLayerImpl::forward(
    torch::Tensor& x,
    torch::Tensor& attn_mask,
    KVCache& kv_cache,
    ModelInputParams& input_params,
    torch::Tensor* encoder_output,
    int32_t node_id,
    aclrtEvent* event,
    std::atomic<bool>* event_flag,
    const torch::Tensor& expert_array) {
  const auto* onerec_params = input_params.onerec_params();
  CHECK(onerec_params != nullptr) << "OneRec requires rec_params.";

  const bool is_prefill =
      onerec_params->rec_stage == OneRecModelInputParams::RecStage::PREFILL;
  const bool is_first_prefill = onerec_params->is_first_prefill;
  const int64_t ntokens = x.dim() >= 1 ? x.size(0) : 1;
  const bool use_atb_small_tokens =
      kEnableOneRecAclnnAttentionLinear &&
      kOneRecAclnnAttentionLinearMinTokens > 0 && ntokens > 0 &&
      ntokens < kOneRecAclnnAttentionLinearMinTokens;

  atb::Status st;
  if (is_prefill) {
    if (is_decoder_) {
      if (is_first_prefill && encoder_output != nullptr &&
          use_legacy_onerec_prefill_only_contract()) {
        const int64_t bs = encoder_output->size(0);
        const int64_t seq_len = encoder_output->size(1);
        const int64_t kv_hidden_size =
            prefill_param_.numKeyValueHeadsPerRank *
            prefill_param_.hiddenSizePerAttentionHead;
        auto options = torch::TensorOptions()
                           .dtype(encoder_output->dtype())
                           .device(encoder_output->device());
        const std::vector<int64_t> expected_shape = {
            bs, seq_len, kv_hidden_size};
        if (!cross_k_cache_.defined() ||
            cross_k_cache_.sizes().vec() != expected_shape ||
            cross_k_cache_.dtype() != encoder_output->dtype() ||
            cross_k_cache_.device() != encoder_output->device()) {
          cross_k_cache_ = torch::empty(expected_shape, options);
        }
        if (!cross_v_cache_.defined() ||
            cross_v_cache_.sizes().vec() != expected_shape ||
            cross_v_cache_.dtype() != encoder_output->dtype() ||
            cross_v_cache_.device() != encoder_output->device()) {
          cross_v_cache_ = torch::empty(expected_shape, options);
        }
      }

      if (use_legacy_onerec_prefill_only_contract()) {
        atb_speed::Model::Node& target_node =
            is_first_prefill
                ? ((use_atb_small_tokens &&
                    prefill_node_atb_.operation != nullptr)
                       ? prefill_node_atb_
                       : prefill_node_)
                : ((use_atb_small_tokens &&
                    decoder_prefill_only_decode_node_atb_.operation != nullptr)
                       ? decoder_prefill_only_decode_node_atb_
                       : decoder_prefill_only_decode_node_);
        if (prefill_param_.use_moe) {
          build_decoder_moe_node_variant_pack(
              target_node,
              x,
              attn_mask,
              kv_cache,
              input_params,
              true,
              is_first_prefill,
              is_first_prefill ? encoder_output : nullptr,
              node_id,
              expert_array);
        } else {
          build_decoder_node_variant_pack(
              target_node,
              x,
              attn_mask,
              kv_cache,
              input_params,
              true,
              is_first_prefill,
              is_first_prefill ? encoder_output : nullptr,
              node_id);
        }
        st = execute_node(target_node, node_id, event, event_flag);
      } else if (prefill_param_.use_moe) {
        build_decoder_moe_node_variant_pack(prefill_node_,
                                            x,
                                            attn_mask,
                                            kv_cache,
                                            input_params,
                                            true,
                                            true,
                                            encoder_output,
                                            node_id,
                                            expert_array);
        st = execute_node(prefill_node_, node_id, event, event_flag);
      } else {
        build_decoder_node_variant_pack(prefill_node_,
                                        x,
                                        attn_mask,
                                        kv_cache,
                                        input_params,
                                        true,
                                        true,
                                        encoder_output,
                                        node_id);
        st = execute_node(prefill_node_, node_id, event, event_flag);
      }
      LOG_IF(FATAL, st != 0)
          << model_name_ << " execute prefill layer fail, error code: " << st;
    } else {
      build_encoder_node_variant_pack(
          prefill_node_, x, attn_mask, input_params, true, node_id);
      st = execute_node(prefill_node_, node_id, event, event_flag);
      LOG_IF(FATAL, st != 0)
          << model_name_
          << " execute encoder prefill layer fail, error code: " << st;
    }
  } else {
    if (!is_decoder_) {
      LOG(FATAL) << model_name_ << " encoder decode stage is not supported.";
    }

    if (decode_param_.use_moe) {
      build_decoder_moe_node_variant_pack(decode_node_,
                                          x,
                                          attn_mask,
                                          kv_cache,
                                          input_params,
                                          false,
                                          false,
                                          encoder_output,
                                          node_id,
                                          expert_array);
    } else {
      build_decoder_node_variant_pack(decode_node_,
                                      x,
                                      attn_mask,
                                      kv_cache,
                                      input_params,
                                      false,
                                      false,
                                      encoder_output,
                                      node_id);
    }
    st = execute_node(decode_node_, node_id + 1000, event, event_flag);
    LOG_IF(FATAL, st != 0) << model_name_
                           << " execute decode layer fail, error code: " << st;
  }

  return at_placeholder_;
}

void NpuOneRecBlockLayerImpl::build_encoder_node_variant_pack(
    atb_speed::Model::Node& node,
    torch::Tensor& x,
    at::Tensor& attn_mask,
    ModelInputParams& input_params,
    bool is_prefill,
    int32_t layer_id) {
  (void)is_prefill;
  (void)layer_id;

  internal_tensors_ = atb_speed::Utils::AtTensor2Tensor(x);

  for (size_t i = 0; i < kOneRecWeightCountPerLayer; ++i) {
    CHECK(node.inTensors.at(i) != nullptr)
        << model_name_ << " inTensor " << i << " is NULL";
    node.variantPack.inTensors.at(i) = *node.inTensors.at(i);
  }

  const int32_t input_tensor_idx =
      static_cast<int32_t>(kOneRecWeightCountPerLayer);
  const int32_t attention_mask_idx = input_tensor_idx + 1;
  const int32_t token_offset_idx = attention_mask_idx + 1;
  const int32_t layer_id_idx = token_offset_idx + 1;
  const int32_t seq_len_idx = layer_id_idx + 1;

  node.variantPack.inTensors.at(input_tensor_idx) = internal_tensors_;
  if (attn_mask.defined()) {
    auto mask_dtype =
        prefill_param_.isBF16 ? torch::kBFloat16 : torch::kFloat16;
    prefill_attn_mask_ =
        PrepareOneRecAttentionMask(attn_mask, device_, mask_dtype);
    node.variantPack.inTensors.at(attention_mask_idx) =
        atb_speed::Utils::AtTensor2Tensor(prefill_attn_mask_);
  } else {
    prefill_attn_mask_ = EnsureNdFormat(prefill_attn_mask_);
    node.variantPack.inTensors.at(attention_mask_idx) =
        atb_speed::Utils::AtTensor2Tensor(prefill_attn_mask_);
  }

  node.variantPack.inTensors.at(token_offset_idx) = placeholder_;
  node.variantPack.inTensors.at(token_offset_idx).hostData =
      placeholder_vec_.data();
  node.variantPack.inTensors.at(layer_id_idx) = placeholder_;
  node.variantPack.inTensors.at(layer_id_idx).hostData =
      placeholder_vec_.data();

  const auto* onerec_params = input_params.onerec_params();
  if (onerec_params != nullptr &&
      onerec_params->encoder_seq_lens_tensor.defined()) {
    node.variantPack.inTensors.at(seq_len_idx) =
        atb_speed::Utils::AtTensor2Tensor(
            onerec_params->encoder_seq_lens_tensor);
    node.variantPack.inTensors.at(seq_len_idx).hostData =
        const_cast<int32_t*>(onerec_params->encoder_seq_lens.data());
  } else {
    node.variantPack.inTensors.at(seq_len_idx) = placeholder_;
    node.variantPack.inTensors.at(seq_len_idx).hostData =
        placeholder_vec_.data();
  }

  node.variantPack.outTensors.at(0) = internal_tensors_;
}

void NpuOneRecBlockLayerImpl::build_decoder_moe_node_variant_pack(
    atb_speed::Model::Node& node,
    torch::Tensor& x,
    at::Tensor& attn_mask,
    KVCache& kv_cache,
    ModelInputParams& input_params,
    bool is_prefill,
    bool is_first_prefill,
    torch::Tensor* encoder_output,
    int32_t layer_id,
    const torch::Tensor& expert_array) {
  (void)is_prefill;
  (void)layer_id;

  for (size_t i = 0; i < kOneRecMoeWeightCountPerLayer; ++i) {
    CHECK(node.inTensors.at(i) != nullptr)
        << model_name_ << " inTensor " << i << " is NULL";
    node.variantPack.inTensors.at(i) = *node.inTensors.at(i);
  }

  const int32_t moe_tensor_start =
      static_cast<int32_t>(kOneRecMoeWeightCountPerLayer);
  if (expert_array.defined()) {
    node.variantPack.inTensors.at(moe_tensor_start) =
        atb_speed::Utils::AtTensor2Tensor(expert_array);
  } else {
    node.variantPack.inTensors.at(moe_tensor_start) = placeholder_;
  }

  node.variantPack.inTensors.at(moe_tensor_start + 1) =
      expert_group_.defined() ? atb_speed::Utils::AtTensor2Tensor(expert_group_)
                              : placeholder_;
  node.variantPack.inTensors.at(moe_tensor_start + 2) =
      one_hot_.defined() ? atb_speed::Utils::AtTensor2Tensor(one_hot_)
                         : placeholder_;
  node.variantPack.inTensors.at(moe_tensor_start + 3) =
      zero_hot_.defined() ? atb_speed::Utils::AtTensor2Tensor(zero_hot_)
                          : placeholder_;

  int32_t tensor_idx = setup_common_decoder_tensors(
      node,
      x,
      attn_mask,
      kv_cache,
      input_params,
      (use_legacy_onerec_prefill_only_contract() && is_prefill &&
       !is_first_prefill)
          ? decoder_prefill_only_decode_param_
          : (is_prefill ? prefill_param_ : decode_param_),
      is_first_prefill,
      encoder_output,
      moe_tensor_start + 4);

  while (tensor_idx < static_cast<int32_t>(node.variantPack.inTensors.size())) {
    node.variantPack.inTensors.at(tensor_idx) = placeholder_;
    node.variantPack.inTensors.at(tensor_idx).hostData =
        placeholder_vec_.data();
    ++tensor_idx;
  }
}

int32_t NpuOneRecBlockLayerImpl::setup_common_decoder_tensors(
    atb_speed::Model::Node& node,
    torch::Tensor& x,
    at::Tensor& attn_mask,
    KVCache& kv_cache,
    ModelInputParams& input_params,
    const atb_speed::onerec::BlockLayerParam& param,
    bool is_first_prefill,
    torch::Tensor* encoder_output,
    int32_t start_tensor_idx) {
  internal_tensors_ = atb_speed::Utils::AtTensor2Tensor(x);

  int32_t idx = start_tensor_idx;
  node.variantPack.inTensors.at(idx++) = internal_tensors_;
  if (attn_mask.defined()) {
    const bool keep_legacy_prefill_mask =
        use_legacy_onerec_prefill_only_contract() && param.isPrefill &&
        param.isDecoder &&
        (attn_mask.scalar_type() == torch::kUInt8 ||
         attn_mask.scalar_type() == torch::kBool);
    torch::Dtype mask_dtype =
        keep_legacy_prefill_mask
            ? attn_mask.scalar_type()
            : ((!param.isPrefill && param.isDecoder)
                   ? torch::kBool
                   : (param.isBF16 ? torch::kBFloat16 : torch::kFloat16));
    prefill_attn_mask_ =
        PrepareOneRecAttentionMask(attn_mask, device_, mask_dtype);
    if (!param.isPrefill && param.isDecoder) {
      prefill_attn_mask_ = NormalizeOneRecDecodeFasMask(
          prefill_attn_mask_, ResolveOneRecBatchSize(input_params));
    }
    node.variantPack.inTensors.at(idx++) =
        atb_speed::Utils::AtTensor2Tensor(prefill_attn_mask_);
  } else {
    decode_attn_mask_ = EnsureNdFormat(decode_attn_mask_);
    node.variantPack.inTensors.at(idx++) =
        atb_speed::Utils::AtTensor2Tensor(decode_attn_mask_);
  }

  auto k_cache = kv_cache.get_k_cache();
  auto v_cache = kv_cache.get_v_cache();
  const auto* onerec_xattn_params = input_params.onerec_xattention_params();
  if (param.use_xattn && onerec_xattn_params != nullptr &&
      static_cast<size_t>(param.layerId) <
          onerec_xattn_params->unshared_k_caches.size() &&
      static_cast<size_t>(param.layerId) <
          onerec_xattn_params->unshared_v_caches.size()) {
    if (onerec_xattn_params
            ->unshared_k_caches[static_cast<size_t>(param.layerId)]
            .defined()) {
      k_cache = onerec_xattn_params
                    ->unshared_k_caches[static_cast<size_t>(param.layerId)];
    }
    if (onerec_xattn_params
            ->unshared_v_caches[static_cast<size_t>(param.layerId)]
            .defined()) {
      v_cache = onerec_xattn_params
                    ->unshared_v_caches[static_cast<size_t>(param.layerId)];
    }
  }
  node.variantPack.inTensors.at(idx++) =
      k_cache.defined() ? atb_speed::Utils::AtTensor2Tensor(k_cache)
                        : placeholder_;
  node.variantPack.inTensors.at(idx++) =
      v_cache.defined() ? atb_speed::Utils::AtTensor2Tensor(v_cache)
                        : placeholder_;

  if (input_params.kv_seq_lens.defined()) {
    node.variantPack.inTensors.at(idx) =
        atb_speed::Utils::AtTensor2Tensor(input_params.kv_seq_lens);
    node.variantPack.inTensors.at(idx).hostData =
        input_params.kv_seq_lens_vec.data();
  } else {
    int32_t seq_len = std::max(static_cast<int32_t>(x.size(0)), 1);
    seq_lens_vec_ = {seq_len};
    auto seq_len_tensor = torch::tensor(
        seq_lens_vec_,
        torch::TensorOptions().dtype(torch::kInt32).device(device_));
    node.variantPack.inTensors.at(idx) =
        atb_speed::Utils::AtTensor2Tensor(seq_len_tensor);
    node.variantPack.inTensors.at(idx).hostData = seq_lens_vec_.data();
  }
  idx++;

  // Token offset and layer id placeholders.
  // ATB expects hostData to be valid for these scalar inputs. Keep them as
  // placeholders but always provide hostData to avoid undefined reads.
  node.variantPack.inTensors.at(idx) = placeholder_;
  node.variantPack.inTensors.at(idx++).hostData = placeholder_vec_.data();
  node.variantPack.inTensors.at(idx) = placeholder_;
  node.variantPack.inTensors.at(idx++).hostData = placeholder_vec_.data();

  // Align with xllm_rec T5 prefill-only path: self-attn block tables are not
  // consumed during decoder prefill-only execution, so do not forward the
  // runtime empty [bs, 0] tensor to ATB.
  if (!param.enableOneRecPrefillOnly && input_params.block_tables.defined()) {
    node.variantPack.inTensors.at(idx) =
        atb_speed::Utils::AtTensor2Tensor(input_params.block_tables);
  } else {
    node.variantPack.inTensors.at(idx) = placeholder_;
    node.variantPack.inTensors.at(idx).hostData = placeholder_vec_.data();
  }
  idx++;

  if (!param.enableOneRecPrefillOnly &&
      input_params.new_cache_slots.defined()) {
    node.variantPack.inTensors.at(idx) =
        atb_speed::Utils::AtTensor2Tensor(input_params.new_cache_slots);
  } else {
    node.variantPack.inTensors.at(idx) = placeholder_;
    node.variantPack.inTensors.at(idx).hostData = placeholder_vec_.data();
  }
  idx++;

  if (is_first_prefill && encoder_output != nullptr) {
    encoder_output_contiguous_ = encoder_output->is_contiguous()
                                     ? *encoder_output
                                     : encoder_output->contiguous();
    node.variantPack.inTensors.at(idx) =
        atb_speed::Utils::AtTensor2Tensor(encoder_output_contiguous_);
  } else {
    node.variantPack.inTensors.at(idx) = placeholder_;
  }
  idx++;

  const auto* onerec_params =
      onerec_xattn_params != nullptr
          ? static_cast<const OneRecModelInputParams*>(onerec_xattn_params)
          : input_params.onerec_params();
  const bool minimize_cross_attn_inputs =
      param.enableOneRecPrefillOnly && !param.enableSplitFuse && !param.isFA;
  const bool use_cross_attention_kernel =
      param.enableCrossAttentionKernel && param.use_xattn;
  if (!minimize_cross_attn_inputs) {
    if (use_cross_attention_kernel) {
      CHECK(onerec_params != nullptr)
          << "CrossAttention requires OneRec xattention parameters.";
      CHECK(onerec_params->cross_attn_kv_cu_seq_lens.defined())
          << "CrossAttention requires device actual_kv_lens.";
      CHECK(onerec_params->cross_attn_block_tables.defined())
          << "CrossAttention requires a device block_table.";
    }
    if (onerec_params != nullptr &&
        onerec_params->cross_attn_kv_cu_seq_lens.defined()) {
      if (use_cross_attention_kernel) {
        CHECK_EQ(onerec_params->cross_attn_kv_cu_seq_lens.device(), device_)
            << "CrossAttention actual_kv_lens must be on the NPU device.";
        CHECK_EQ(onerec_params->cross_attn_kv_cu_seq_lens.scalar_type(),
                 torch::kInt64)
            << "CrossAttention actual_kv_lens must be int64.";
      }
      node.variantPack.inTensors.at(idx) = atb_speed::Utils::AtTensor2Tensor(
          onerec_params->cross_attn_kv_cu_seq_lens);
      if (!use_cross_attention_kernel) {
        node.variantPack.inTensors.at(idx).hostData = const_cast<int32_t*>(
            onerec_params->cross_attn_kv_cu_seq_lens_vec.data());
      }
    } else {
      node.variantPack.inTensors.at(idx) = placeholder_;
      node.variantPack.inTensors.at(idx).hostData = placeholder_vec_.data();
    }
    idx++;

    if (onerec_params != nullptr &&
        onerec_params->cross_attn_block_tables.defined()) {
      if (use_cross_attention_kernel) {
        CHECK_EQ(onerec_params->cross_attn_block_tables.device(), device_)
            << "CrossAttention block_table must be on the NPU device.";
        CHECK_EQ(onerec_params->cross_attn_block_tables.scalar_type(),
                 torch::kInt32)
            << "CrossAttention block_table must be int32.";
      }
      node.variantPack.inTensors.at(idx) = atb_speed::Utils::AtTensor2Tensor(
          onerec_params->cross_attn_block_tables);
    } else {
      node.variantPack.inTensors.at(idx) = placeholder_;
      node.variantPack.inTensors.at(idx).hostData = placeholder_vec_.data();
    }
    idx++;

    if (is_first_prefill && onerec_params != nullptr &&
        onerec_params->cross_attn_new_cache_slots.defined()) {
      node.variantPack.inTensors.at(idx) = atb_speed::Utils::AtTensor2Tensor(
          onerec_params->cross_attn_new_cache_slots);
    } else {
      node.variantPack.inTensors.at(idx) = placeholder_;
      node.variantPack.inTensors.at(idx).hostData = placeholder_vec_.data();
    }
    idx++;
  }

  if (onerec_params != nullptr &&
      onerec_params->encoder_seq_lens_tensor.defined()) {
    node.variantPack.inTensors.at(idx) = atb_speed::Utils::AtTensor2Tensor(
        onerec_params->encoder_seq_lens_tensor);
    node.variantPack.inTensors.at(idx).hostData =
        const_cast<int32_t*>(onerec_params->encoder_seq_lens.data());
  } else {
    node.variantPack.inTensors.at(idx) = placeholder_;
    node.variantPack.inTensors.at(idx).hostData = placeholder_vec_.data();
  }
  idx++;

  if (param.use_xattn) {
    CHECK(onerec_xattn_params != nullptr)
        << "OneRec xattention requires onerec_xattention_params.";
    CHECK_LT(static_cast<size_t>(param.layerId),
             onerec_xattn_params->shared_k_caches.size())
        << "Missing OneRec shared_k_caches for layer " << param.layerId;
    CHECK_LT(static_cast<size_t>(param.layerId),
             onerec_xattn_params->shared_v_caches.size())
        << "Missing OneRec shared_v_caches for layer " << param.layerId;
    CHECK(onerec_xattn_params->beam_width_tensor.defined())
        << "OneRec xattention requires beam_width_tensor";
    CHECK(onerec_xattn_params->current_round_tensor.defined())
        << "OneRec xattention requires current_round_tensor";
    node.variantPack.inTensors.at(idx++) = atb_speed::Utils::AtTensor2Tensor(
        onerec_xattn_params
            ->shared_k_caches[static_cast<size_t>(param.layerId)]);
    node.variantPack.inTensors.at(idx++) = atb_speed::Utils::AtTensor2Tensor(
        onerec_xattn_params
            ->shared_v_caches[static_cast<size_t>(param.layerId)]);
    node.variantPack.inTensors.at(idx++) = atb_speed::Utils::AtTensor2Tensor(
        onerec_xattn_params->beam_width_tensor);
    node.variantPack.inTensors.at(idx++) = atb_speed::Utils::AtTensor2Tensor(
        onerec_xattn_params->current_round_tensor);
  }

  if (param.use_xattn) {
    auto cross_k_cache = kv_cache.get_k_cache();
    auto cross_v_cache = kv_cache.get_v_cache();
    CHECK(cross_k_cache.defined() && cross_v_cache.defined())
        << "OneRec xattention Cross KV block cache is not initialized.";
    CHECK_EQ(cross_k_cache.device(), device_)
        << "CrossAttention key cache must be on the NPU device.";
    CHECK_EQ(cross_v_cache.device(), device_)
        << "CrossAttention value cache must be on the NPU device.";
    node.variantPack.inTensors.at(idx++) =
        atb_speed::Utils::AtTensor2Tensor(cross_k_cache);
    node.variantPack.inTensors.at(idx++) =
        atb_speed::Utils::AtTensor2Tensor(cross_v_cache);
  } else if (param.enableOneRecPrefillOnly && cross_k_cache_.defined() &&
             cross_v_cache_.defined()) {
    if (is_first_prefill && node.variantPack.outTensors.size() >= 3) {
      node.variantPack.outTensors.at(1) =
          atb_speed::Utils::AtTensor2Tensor(cross_k_cache_);
      node.variantPack.outTensors.at(2) =
          atb_speed::Utils::AtTensor2Tensor(cross_v_cache_);
    } else if (!is_first_prefill) {
      node.variantPack.inTensors.at(idx++) =
          atb_speed::Utils::AtTensor2Tensor(cross_k_cache_);
      node.variantPack.inTensors.at(idx++) =
          atb_speed::Utils::AtTensor2Tensor(cross_v_cache_);
    }
  } else {
    node.variantPack.inTensors.at(idx) = placeholder_;
    node.variantPack.inTensors.at(idx++).hostData = placeholder_vec_.data();
    node.variantPack.inTensors.at(idx) = placeholder_;
    node.variantPack.inTensors.at(idx++).hostData = placeholder_vec_.data();
  }

  node.variantPack.outTensors.at(0) = internal_tensors_;
  return idx;
}

void NpuOneRecBlockLayerImpl::build_decoder_node_variant_pack(
    atb_speed::Model::Node& node,
    torch::Tensor& x,
    at::Tensor& attn_mask,
    KVCache& kv_cache,
    ModelInputParams& input_params,
    bool is_prefill,
    bool is_first_prefill,
    torch::Tensor* encoder_output,
    int32_t layer_id) {
  (void)is_prefill;
  (void)layer_id;

  for (size_t i = 0; i < kOneRecWeightCountPerLayer; ++i) {
    CHECK(node.inTensors.at(i) != nullptr)
        << model_name_ << " inTensor " << i << " is NULL";
    node.variantPack.inTensors.at(i) = *node.inTensors.at(i);
  }

  int32_t tensor_idx = setup_common_decoder_tensors(
      node,
      x,
      attn_mask,
      kv_cache,
      input_params,
      (use_legacy_onerec_prefill_only_contract() && is_prefill &&
       !is_first_prefill)
          ? decoder_prefill_only_decode_param_
          : (is_prefill ? prefill_param_ : decode_param_),
      is_first_prefill,
      encoder_output,
      static_cast<int32_t>(kOneRecWeightCountPerLayer));
  while (tensor_idx < static_cast<int32_t>(node.variantPack.inTensors.size())) {
    node.variantPack.inTensors.at(tensor_idx) = placeholder_;
    node.variantPack.inTensors.at(tensor_idx).hostData =
        placeholder_vec_.data();
    ++tensor_idx;
  }
}

void NpuOneRecBlockLayerImpl::resize_experts_weights(
    int32_t num_of_device_experts) {
  experts_weights_["gate_proj.weight"] =
      std::vector<torch::Tensor>(num_of_device_experts);
  experts_weights_["up_proj.weight"] =
      std::vector<torch::Tensor>(num_of_device_experts);
  experts_weights_["down_proj.weight"] =
      std::vector<torch::Tensor>(num_of_device_experts);
  experts_weights_["gate_proj.weight_offset"] =
      std::vector<torch::Tensor>(num_of_device_experts);
  experts_weights_["up_proj.weight_offset"] =
      std::vector<torch::Tensor>(num_of_device_experts);
  experts_weights_["down_proj.weight_offset"] =
      std::vector<torch::Tensor>(num_of_device_experts);
  experts_weights_["gate_proj.weight_scale"] =
      std::vector<torch::Tensor>(num_of_device_experts);
  experts_weights_["up_proj.weight_scale"] =
      std::vector<torch::Tensor>(num_of_device_experts);
  experts_weights_["down_proj.weight_scale"] =
      std::vector<torch::Tensor>(num_of_device_experts);
}

void NpuOneRecBlockLayerImpl::process_expert_weights(
    const StateDict& state_dict,
    const std::string& name,
    const torch::Tensor& tensor) {
  (void)state_dict;
  std::lock_guard<std::mutex> lock(experts_mutex_);

  auto unpack_fused_weight1 = [&](const torch::Tensor& fused_weight) {
    if (num_experts_per_partition_ <= 0 || fused_weight.dim() != 2) {
      LOG(WARNING) << "Invalid OneRec fused expert weight1 shape for " << name
                   << ": " << fused_weight.sizes();
      return;
    }
    torch::Tensor reshaped = fused_weight.contiguous().view(
        {num_experts_per_partition_, fused_weight.size(0), -1});
    CHECK_EQ(reshaped.size(2) % 2, 0)
        << "OneRec fused expert weight1 last dim must be even for " << name
        << ", got shape " << fused_weight.sizes();
    std::vector<torch::Tensor> gate_up_chunks =
        reshaped.chunk(/*chunks=*/2, /*dim=*/-1);
    for (int32_t i = 0; i < num_experts_per_partition_; ++i) {
      experts_weights_["gate_proj.weight"][i] =
          gate_up_chunks[0][i].transpose(0, 1).contiguous();
      experts_weights_["up_proj.weight"][i] =
          gate_up_chunks[1][i].transpose(0, 1).contiguous();
    }
    LOG(INFO) << "Unpacked OneRec fused routed expert weight1 into "
              << num_experts_per_partition_
              << " gate/up experts, source shape: [" << fused_weight.sizes()
              << "]";
  };

  auto unpack_fused_weight2 = [&](const torch::Tensor& fused_weight) {
    if (num_experts_per_partition_ <= 0 || fused_weight.dim() != 2) {
      LOG(WARNING) << "Invalid OneRec fused expert weight2 shape for " << name
                   << ": " << fused_weight.sizes();
      return;
    }
    auto reshaped = fused_weight.contiguous().view(
        {num_experts_per_partition_, -1, fused_weight.size(1)});
    for (int32_t i = 0; i < num_experts_per_partition_; ++i) {
      experts_weights_["down_proj.weight"][i] =
          reshaped[i].transpose(0, 1).contiguous();
    }
    LOG(INFO) << "Unpacked OneRec fused routed expert weight2 into "
              << num_experts_per_partition_ << " down experts, source shape: ["
              << fused_weight.sizes() << "]";
  };

  if (absl::StrContains(name, ".ffn.experts.weight1")) {
    unpack_fused_weight1(tensor);
    return;
  }
  if (absl::StrContains(name, ".ffn.experts.weight2")) {
    unpack_fused_weight2(tensor);
    return;
  }

  int32_t expert_id = extract_expert_index(name);
  if (expert_id < 0) {
    return;
  }

  const int32_t local_index = expert_id % num_experts_per_partition_;
  std::string weight_suffix = extract_endswith(name);

  std::string suffix;
  if (weight_suffix == "gate_proj.weight" || weight_suffix == "w1.weight") {
    suffix = "gate_proj.weight";
  } else if (weight_suffix == "up_proj.weight" ||
             weight_suffix == "w3.weight") {
    suffix = "up_proj.weight";
  } else if (weight_suffix == "down_proj.weight" ||
             weight_suffix == "w2.weight") {
    suffix = "down_proj.weight";
  } else if (weight_suffix == "gate_proj.weight_offset" ||
             weight_suffix == "w1.weight_offset") {
    suffix = "gate_proj.weight_offset";
  } else if (weight_suffix == "up_proj.weight_offset" ||
             weight_suffix == "w3.weight_offset") {
    suffix = "up_proj.weight_offset";
  } else if (weight_suffix == "down_proj.weight_offset" ||
             weight_suffix == "w2.weight_offset") {
    suffix = "down_proj.weight_offset";
  } else if (weight_suffix == "gate_proj.weight_scale" ||
             weight_suffix == "w1.weight_scale") {
    suffix = "gate_proj.weight_scale";
  } else if (weight_suffix == "up_proj.weight_scale" ||
             weight_suffix == "w3.weight_scale") {
    suffix = "up_proj.weight_scale";
  } else if (weight_suffix == "down_proj.weight_scale" ||
             weight_suffix == "w2.weight_scale") {
    suffix = "down_proj.weight_scale";
  } else {
    return;
  }

  auto it = experts_weights_.find(suffix);
  if (it == experts_weights_.end() || local_index < 0 ||
      local_index >= static_cast<int32_t>(it->second.size())) {
    LOG(ERROR) << "Invalid OneRec MoE local expert index " << local_index
               << " for " << suffix << " at layer " << layer_id_ << ".";
    return;
  }
  it->second[local_index] = tensor;
}

void NpuOneRecBlockLayerImpl::process_shared_expert_weights(
    const StateDict& state_dict,
    const std::string& name,
    const torch::Tensor& tensor) {
  (void)state_dict;
  torch::Tensor tmp_tensor = tensor.to(device_);

  std::string canonical_name;
  if (absl::StrContains(name, "gate_proj") || absl::StrContains(name, "w1")) {
    canonical_name = "gate_proj.weight";
  } else if (absl::StrContains(name, "up_proj") ||
             absl::StrContains(name, "w3")) {
    canonical_name = "up_proj.weight";
  } else if (absl::StrContains(name, "down_proj") ||
             absl::StrContains(name, "w2")) {
    canonical_name = "down_proj.weight";
  } else {
    return;
  }

  if (shared_expert_weights_map_.count(canonical_name) > 0) {
    LOG(WARNING) << "Duplicate OneRec shared expert tensor for "
                 << canonical_name << " at layer " << layer_id_
                 << ", overriding previous value.";
  }
  shared_expert_weights_map_[canonical_name] = tmp_tensor;
}

int32_t NpuOneRecBlockLayerImpl::extract_expert_index(const std::string& name) {
  size_t experts_pos = name.find(".experts.");
  if (experts_pos == std::string::npos) {
    return -1;
  }

  size_t start_pos = experts_pos + 9;
  size_t end_pos = name.find(".", start_pos);
  if (end_pos == std::string::npos) {
    return -1;
  }

  try {
    return std::stoi(name.substr(start_pos, end_pos - start_pos));
  } catch (const std::exception&) {
    return -1;
  }
}

std::string NpuOneRecBlockLayerImpl::extract_endswith(
    const std::string& input) {
  size_t experts_pos = input.find(".experts.");
  if (experts_pos == std::string::npos) {
    return "";
  }
  size_t start_pos = experts_pos + 9;
  size_t next_dot = input.find(".", start_pos);
  if (next_dot == std::string::npos) {
    return "";
  }
  return input.substr(next_dot + 1);
}

torch::Tensor NpuOneRecBlockLayerImpl::merge_experts_weights(
    std::vector<torch::Tensor>& experts,
    bool transpose) {
  std::vector<torch::Tensor> valid;
  valid.reserve(experts.size());
  for (auto& t : experts) {
    if (t.defined()) {
      valid.emplace_back(t.to(device_));
    }
  }
  if (valid.empty()) {
    LOG(ERROR) << "No expert weights to merge at layer " << layer_id_ << ".";
    return torch::Tensor();
  }
  torch::Tensor merged_tensor = torch::stack(valid, 0);
  if (transpose) {
    merged_tensor = merged_tensor.transpose(1, 2);
  }
  return merged_tensor.contiguous();
}

torch::Tensor NpuOneRecBlockLayerImpl::merge_experts_weights(
    std::vector<torch::Tensor>& experts_gate,
    std::vector<torch::Tensor>& experts_up,
    bool transpose) {
  if (experts_gate.size() != experts_up.size()) {
    LOG(ERROR) << "OneRec MoE gate/up expert size mismatch: gate="
               << experts_gate.size() << ", up=" << experts_up.size()
               << ", layer " << layer_id_;
    return torch::Tensor();
  }
  for (size_t i = 0; i < experts_gate.size(); ++i) {
    const bool gate_defined = experts_gate[i].defined();
    const bool up_defined = experts_up[i].defined();
    if (gate_defined != up_defined) {
      LOG(ERROR) << "OneRec MoE gate/up tensor mismatch at local expert " << i
                 << ": gate=" << gate_defined << ", up=" << up_defined
                 << ", layer " << layer_id_;
      return torch::Tensor();
    }
    if (gate_defined) {
      experts_gate[i] = torch::cat({experts_gate[i], experts_up[i]}, 0);
    }
  }
  return merge_experts_weights(experts_gate, transpose);
}

void NpuOneRecBlockLayerImpl::merge_experts_weights() {
  if (experts_weights_.count("gate_proj.weight") == 0 ||
      experts_weights_.count("up_proj.weight") == 0 ||
      experts_weights_.count("down_proj.weight") == 0) {
    return;
  }

  auto merged_gate_up =
      merge_experts_weights(experts_weights_["gate_proj.weight"],
                            experts_weights_["up_proj.weight"],
                            /*transpose=*/false);
  CHECK(merged_gate_up.defined()) << "OneRec MoE gate/up experts merge failed.";
  at_weight_tensors_[kInMoeExpertW1Weight] =
      at_npu::native::npu_format_cast(merged_gate_up, /*format=*/2)
          .contiguous();

  if (quantize_type_ == "w8a8_dynamic") {
    if (experts_weights_.count("gate_proj.weight_offset") > 0 &&
        experts_weights_.count("up_proj.weight_offset") > 0) {
      std::vector<torch::Tensor> gate_offset_1d;
      std::vector<torch::Tensor> up_offset_1d;
      gate_offset_1d.reserve(
          experts_weights_["gate_proj.weight_offset"].size());
      up_offset_1d.reserve(experts_weights_["up_proj.weight_offset"].size());
      for (const auto& tensor : experts_weights_["gate_proj.weight_offset"]) {
        if (tensor.defined()) {
          gate_offset_1d.emplace_back(tensor);
        }
      }
      for (const auto& tensor : experts_weights_["up_proj.weight_offset"]) {
        if (tensor.defined()) {
          up_offset_1d.emplace_back(tensor);
        }
      }
      if (!gate_offset_1d.empty() &&
          gate_offset_1d.size() == up_offset_1d.size()) {
        at_weight_tensors_[kInMlpGateUpOffsetExpert] =
            merge_experts_weights(gate_offset_1d,
                                  up_offset_1d,
                                  /*transpose=*/false);
      }
    }
    if (experts_weights_.count("gate_proj.weight_scale") > 0 &&
        experts_weights_.count("up_proj.weight_scale") > 0) {
      std::vector<torch::Tensor> gate_scale_1d;
      std::vector<torch::Tensor> up_scale_1d;
      gate_scale_1d.reserve(experts_weights_["gate_proj.weight_scale"].size());
      up_scale_1d.reserve(experts_weights_["up_proj.weight_scale"].size());
      for (const auto& tensor : experts_weights_["gate_proj.weight_scale"]) {
        if (tensor.defined()) {
          gate_scale_1d.emplace_back(tensor);
        }
      }
      for (const auto& tensor : experts_weights_["up_proj.weight_scale"]) {
        if (tensor.defined()) {
          up_scale_1d.emplace_back(tensor);
        }
      }
      if (!gate_scale_1d.empty() &&
          gate_scale_1d.size() == up_scale_1d.size()) {
        at_weight_tensors_[kInMlpGateUpScaleExpert] =
            merge_experts_weights(gate_scale_1d,
                                  up_scale_1d,
                                  /*transpose=*/false);
      }
    }
  }

  auto merged_down = merge_experts_weights(experts_weights_["down_proj.weight"],
                                           /*transpose=*/false);
  CHECK(merged_down.defined()) << "OneRec MoE down experts merge failed.";
  at_weight_tensors_[kInMoeExpertW2Weight] =
      at_npu::native::npu_format_cast(merged_down, /*format=*/2).contiguous();

  if (quantize_type_ == "w8a8_dynamic") {
    if (experts_weights_.count("down_proj.weight_offset") > 0) {
      std::vector<torch::Tensor> down_offset_1d;
      down_offset_1d.reserve(
          experts_weights_["down_proj.weight_offset"].size());
      for (const auto& tensor : experts_weights_["down_proj.weight_offset"]) {
        if (tensor.defined()) {
          down_offset_1d.emplace_back(tensor);
        }
      }
      if (!down_offset_1d.empty()) {
        at_weight_tensors_[kInMlpDownOffsetExpert] =
            merge_experts_weights(down_offset_1d, /*transpose=*/false);
      }
    }
    if (experts_weights_.count("down_proj.weight_scale") > 0) {
      std::vector<torch::Tensor> down_scale_1d;
      down_scale_1d.reserve(experts_weights_["down_proj.weight_scale"].size());
      for (const auto& tensor : experts_weights_["down_proj.weight_scale"]) {
        if (tensor.defined()) {
          down_scale_1d.emplace_back(tensor);
        }
      }
      if (!down_scale_1d.empty()) {
        at_weight_tensors_[kInMlpDownScaleExpert] =
            merge_experts_weights(down_scale_1d, /*transpose=*/false);
      }
    }
  }
}

void NpuOneRecBlockLayerImpl::merge_shared_experts_weights() {
  auto get_shared_weight = [this](const char* key) -> torch::Tensor {
    auto it = shared_expert_weights_map_.find(key);
    return it == shared_expert_weights_map_.end() ? torch::Tensor()
                                                  : it->second;
  };

  torch::Tensor gate_weight = get_shared_weight("gate_proj.weight");
  torch::Tensor up_weight = get_shared_weight("up_proj.weight");
  torch::Tensor down_weight = get_shared_weight("down_proj.weight");

  if (!gate_weight.defined() && !up_weight.defined() &&
      !down_weight.defined()) {
    return;
  }

  auto prepare_shared_weight_2d = [&](const torch::Tensor& weight,
                                      const char* tag) {
    CHECK(weight.defined()) << "No OneRec shared expert weights for " << tag;
    torch::Tensor merged = weight.to(device_).contiguous();
    CHECK_EQ(merged.dim(), 2) << "OneRec shared expert " << tag
                              << " must stay 2D, got " << merged.sizes();
    return merged;
  };

  if (gate_weight.defined() && up_weight.defined()) {
    auto merged_gate = prepare_shared_weight_2d(gate_weight, "gate");
    auto merged_up = prepare_shared_weight_2d(up_weight, "up");
    at_weight_tensors_[kInMlpGateUpWeightSharedExpert] =
        torch::cat({merged_gate, merged_up}, 0).contiguous();
  } else if (gate_weight.defined()) {
    at_weight_tensors_[kInMlpGateUpWeightSharedExpert] =
        prepare_shared_weight_2d(gate_weight, "gate_only");
  }

  if (down_weight.defined()) {
    at_weight_tensors_[kInMlpDownWeightSharedExpert] =
        prepare_shared_weight_2d(down_weight, "down");
  }

  shared_expert_weights_map_.clear();
}

}  // namespace layer
}  // namespace xllm
