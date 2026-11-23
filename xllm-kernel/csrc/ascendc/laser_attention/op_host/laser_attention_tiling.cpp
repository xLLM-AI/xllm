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

#include "laser_attention_tiling.h"

#include <cinttypes>
#include <string>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

using namespace std;

namespace {
constexpr size_t CONST_ONES_SIZE = 16384 * 2;     // half
constexpr size_t CONST_ZERO_SIZE = 32 * 128 * 4;  // float
constexpr size_t MAX_TOKEN = 2147483647;
const int ALIGNNUM = 16;
inline __attribute__((always_inline)) int32_t CeilDiv(int32_t num,
                                                      int32_t div) {
  if (div == 0) {
    return 0;
  }
  return (num + div - 1) / div;
}

inline __attribute__((always_inline)) int32_t Align(int32_t num,
                                                    int32_t alignNum) {
  if (alignNum != 0) {
    return (num + alignNum - 1) / alignNum * alignNum;
  } else {
    return num + alignNum - 1;
  }
}

}  // namespace

namespace optiling {
class LaserAttentionTiling {
 public:
  LaserAttentionTilingData tilingData;

  ge::graphStatus Tiling4LaserAttention(gert::TilingContext* context);

  ge::graphStatus DoTiling(gert::TilingContext* context);

  ge::graphStatus CheckTiling(gert::TilingContext* context);
};

ge::graphStatus LaserAttentionTiling::DoTiling(gert::TilingContext* context) {
  if (context == nullptr) {
    return ge::GRAPH_FAILED;
  }
  const auto platformInfo = context->GetPlatformInfo();
  if (platformInfo == nullptr) {
    return ge::GRAPH_FAILED;
  }
  const auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
  const auto aicNum = ascendcPlatform.GetCoreNumAic();

  auto col_size = tilingData.get_kSeqLength();
  if (tilingData.get_sparseMode() == 1) {
    col_size = tilingData.get_windowLen();
  }

  int32_t coreNumPerGroup = 1;
  int32_t factor = 2;
  if (col_size <= 8 * 1024 / factor) {  // value is 8 * 1024
    coreNumPerGroup = 1;
  } else if (col_size > 8 * 1024 / factor &&
             col_size <= 16 * 1024 / factor) {  // value is 8、16、1024
    coreNumPerGroup = 2;                        // 2 is coreNumPerGroup
  } else if (col_size > 16 * 1024 / factor &&
             col_size <= 32 * 1024 / factor) {  // value is 16、32、1024
    coreNumPerGroup = 4;                        // 4 is coreNumPerGroup
  } else {
    if (aicNum == 20) {     // 20 is aicNum
      coreNumPerGroup = 4;  // 4 is coreNumPerGroup
    } else {
      coreNumPerGroup = 8;  // 8 is coreNumPerGroup
    }
  }

  tilingData.set_coreNumPerGroup(coreNumPerGroup);
  tilingData.set_coreGroupNum(aicNum / coreNumPerGroup);

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus LaserAttentionTiling::CheckTiling(
    gert::TilingContext* context) {
  if (context == nullptr) {
    return ge::GRAPH_FAILED;
  }

  if (tilingData.get_batchSize() == 0) {
    cout << context->GetNodeName() << "op [TilingData]: batchSize is 0. "
         << endl;
    return ge::GRAPH_FAILED;
  }

  if (tilingData.get_headNum() == 0) {
    cout << context->GetNodeName() << "op [TilingData]: headNum is 0. " << endl;
    return ge::GRAPH_FAILED;
  }

  if (tilingData.get_qSeqLength() == 0) {
    cout << context->GetNodeName() << "op [TilingData]: qSeqLength is 0. "
         << endl;
    return ge::GRAPH_FAILED;
  }

  if (tilingData.get_kSeqLength() == 0) {
    cout << context->GetNodeName() << "op [TilingData]: kSeqLength is 0. "
         << endl;
    return ge::GRAPH_FAILED;
  }

  if (tilingData.get_vSeqLength() == 0) {
    cout << context->GetNodeName() << "op [TilingData]: vSeqLength is 0. "
         << endl;
    return ge::GRAPH_FAILED;
  }

  // sparse场景下windowsLength需要是256的倍数
  if (tilingData.get_sparseMode() == 1 &&
      (tilingData.get_windowLen() % 256 != 0)) {
    cout << context->GetNodeName()
         << "op [TilingData]: windowLen= " << tilingData.get_windowLen()
         << endl;
    return ge::GRAPH_FAILED;
  }

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus LaserAttentionTiling::Tiling4LaserAttention(
    gert::TilingContext* context) {
  if (context == nullptr) {
    return ge::GRAPH_FAILED;
  }
  int32_t inputNum = static_cast<int32_t>(context->GetComputeNodeInputNum());
  if (inputNum < 3) {  // 3 is the number of inputs
    return ge::GRAPH_FAILED;
  }
  const auto queryShape = context->GetInputShape(0);
  const auto keyShape = context->GetInputShape(1);
  const auto valueShape = context->GetInputShape(2);
  if (queryShape == nullptr || keyShape == nullptr || valueShape == nullptr) {
    return ge::GRAPH_FAILED;
  }
  const auto pScale = context->GetAttrs()->GetAttrPointer<float>(0);
  const auto pHeadNum = context->GetAttrs()->GetAttrPointer<int32_t>(1);
  const auto pre_tokens = context->GetAttrs()->GetAttrPointer<int32_t>(4);
  const auto isHighPrecision = context->GetAttrs()->GetAttrPointer<bool>(6);
  if (pScale == nullptr || pHeadNum == nullptr || pre_tokens == nullptr ||
      isHighPrecision == nullptr) {
    return ge::GRAPH_FAILED;
  }

  const auto qkvShape =
      queryShape->GetStorageShape();  // [batchSize, headNum, seqSize, headDim]
  int32_t batchSize = qkvShape.GetDim(0);
  int32_t headNum = *pHeadNum;
  int32_t seqSize = qkvShape.GetDim(2);
  int32_t headDim = qkvShape.GetDim(3);
  int32_t qSeqLength = queryShape->GetStorageShape().GetDim(2);
  int32_t kSeqLength = keyShape->GetStorageShape().GetDim(2);
  int32_t vSeqLength = valueShape->GetStorageShape().GetDim(2);

  int32_t maskSeqLength = 0;

  tilingData.set_batchSize(batchSize);
  tilingData.set_headNum(headNum);
  tilingData.set_seqSize(seqSize);
  tilingData.set_headDim(headDim);

  tilingData.set_qSeqLength(qSeqLength);
  tilingData.set_kSeqLength(kSeqLength);
  tilingData.set_vSeqLength(vSeqLength);
  tilingData.set_maskSeqLength(maskSeqLength);
  tilingData.set_scale(*pScale);

  bool isTriangle = false;
  tilingData.set_isTriangle(isTriangle);
  int32_t attenType = 0;
  int32_t sparseMode = 0;
  int32_t headGroupSize = 1;
  int32_t windowLen = 0;
  int32_t queryDim = queryShape->GetStorageShape().GetDim(1);
  int32_t keyDim = keyShape->GetStorageShape().GetDim(1);
  if (keyDim <= 0) {
    return ge::GRAPH_FAILED;
  }
  if (queryDim != keyDim) {
    attenType = 1;
    headGroupSize = queryDim / keyDim;
  }
  if (*pre_tokens == MAX_TOKEN) {
    windowLen = 0;
  } else {
    windowLen = *pre_tokens;
  }
  tilingData.set_attenType(attenType);
  tilingData.set_sparseMode(sparseMode);
  tilingData.set_headGroupSize(headGroupSize);
  tilingData.set_windowLen(windowLen);
  tilingData.set_isHighPrecision(*isHighPrecision);

  DoTiling(context);

  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus LaserAttentionTilingFunc(gert::TilingContext* context) {
  if (context == nullptr) {
    return ge::GRAPH_FAILED;
  }
  LaserAttentionTiling tiling;
  tiling.Tiling4LaserAttention(context);

  // set block_dim & workspace
  const auto platformInfo = context->GetPlatformInfo();
  if (platformInfo == nullptr) {
    return ge::GRAPH_FAILED;
  }
  const auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
  const auto aicNum = ascendcPlatform.GetCoreNumAic();
  if (aicNum == 25) {          // 25 is aicNum
    context->SetBlockDim(24);  // 24 is blockdim
  } else {
    context->SetBlockDim(aicNum);
  }
  const auto inputData = context->GetInputTensor(0);
  if (inputData == nullptr) {
    return ge::GRAPH_FAILED;
  }
  auto dtype = inputData->GetDataType();
  auto workspaces = context->GetWorkspaceSizes(1);
  if (workspaces == nullptr) {
    return ge::GRAPH_FAILED;
  }

  size_t coreGrouNum =
      static_cast<size_t>(tiling.tilingData.get_coreGroupNum());
  size_t coreNumPerGroup =
      static_cast<size_t>(tiling.tilingData.get_coreNumPerGroup());
  size_t batchSize = static_cast<size_t>(tiling.tilingData.get_batchSize());
  size_t headNum = static_cast<size_t>(tiling.tilingData.get_headNum());
  size_t seqSize = static_cast<size_t>(tiling.tilingData.get_seqSize());
  size_t workspaceSize =
      static_cast<size_t>(ascendcPlatform.GetLibApiWorkSpaceSize());

  auto groupNum = coreGrouNum * coreNumPerGroup;
  auto rowSumSize = batchSize * headNum * seqSize * sizeof(float);
  const size_t blocksPerCore = dtype == ge::DT_BF16 ? 8 : 64;
  const size_t scoreBytes = dtype == ge::DT_BF16
                                ? sizeof(float) + sizeof(uint16_t)
                                : sizeof(uint16_t);
  const size_t scoreWorkspace =
      groupNum * blocksPerCore * 128 * 128 * 2 * scoreBytes;
  workspaces[0] = scoreWorkspace + groupNum * 256 * 128 * 8 * 2 * 4 * 2 +
                  rowSumSize + workspaceSize;

  tiling.tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(),
                                 context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.tilingData.GetDataSize());

  if (dtype == ge::DT_BF16) {
    context->SetTilingKey(1);
  } else {
    context->SetTilingKey(0);  // FP16
  }

  return tiling.CheckTiling(context);
}

ge::graphStatus TilingPrepareForLaserAttention(
    gert::TilingParseContext* context) {
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(XllmKernelLaserAttention)
    .Tiling(LaserAttentionTilingFunc)
    .TilingParse<LaserAttentionCompileInfo>(TilingPrepareForLaserAttention);

}  // namespace optiling
