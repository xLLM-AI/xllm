/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef LASER_ATTENTION_H_
#define LASER_ATTENTION_H_

#include "register/tilingdata_base.h"
namespace optiling {

struct LaserAttentionCompileInfo {
  uint64_t l0ASize;
  uint64_t l0BSize;
  uint64_t l0CSize;
  uint64_t l1Size;
  uint64_t ubSize;
  uint32_t maxAicCoresNum;
  uint32_t maxAivCoresNum;
  size_t defaultSysWorkspaceSize;
};

BEGIN_TILING_DATA_DEF(LaserAttentionTilingData)
TILING_DATA_FIELD_DEF(int32_t, batchSize);        // B
TILING_DATA_FIELD_DEF(int32_t, headNum);          // N
TILING_DATA_FIELD_DEF(int32_t, seqSize);          // S
TILING_DATA_FIELD_DEF(int32_t, headDim);          // D
TILING_DATA_FIELD_DEF(int32_t, coreNumPerGroup);  // Y
TILING_DATA_FIELD_DEF(int32_t, coreGroupNum);     // F

TILING_DATA_FIELD_DEF(int32_t, qSeqLength);     // qkv不等长
TILING_DATA_FIELD_DEF(int32_t, kSeqLength);     // qkv不等长
TILING_DATA_FIELD_DEF(int32_t, vSeqLength);     // qkv不等长
TILING_DATA_FIELD_DEF(int32_t, maskSeqLength);  // 预留
TILING_DATA_FIELD_DEF(float, scale);            // 预留
TILING_DATA_FIELD_DEF(float, keep_prob);        // 预留
TILING_DATA_FIELD_DEF(int32_t, pre_tokens);     // 预留
TILING_DATA_FIELD_DEF(int32_t, next_tokens);    // 预留

TILING_DATA_FIELD_DEF(bool, isTriangle);        // 是否倒三角
TILING_DATA_FIELD_DEF(int32_t, attenType);      // 0:MHA/1:GQA
TILING_DATA_FIELD_DEF(int32_t, sparseMode);     // 0:dense/1:sparse
TILING_DATA_FIELD_DEF(int32_t, headGroupSize);  // N/G
TILING_DATA_FIELD_DEF(int32_t, windowLen);      // sparse的滑动窗口
TILING_DATA_FIELD_DEF(bool, isHighPrecision);   // 高性能
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(XllmKernelLaserAttention, LaserAttentionTilingData)
}  // namespace optiling
#endif  // LASER_ATTENTION_H_
