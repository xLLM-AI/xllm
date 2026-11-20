#include "mega_gdn_decode_tiling.h"

#include <algorithm>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

// This implementation include depends on the complete CANN host declarations.
// clang-format off
#include "mega_gdn_decode_tiling_a5.inc"
// clang-format on

namespace optiling {

namespace {

constexpr int64_t kHeadDim = 128;
constexpr int64_t kConvStateLen = 3;
constexpr int64_t kMaxNumCacheSlots = 1024;
constexpr int64_t kMaxBatchSize = 32;
constexpr size_t kFlaSsmStateLayoutAttr = 0;
constexpr uint64_t kNonFlaTilingKeyOffset = 10;

bool IsSupportedShape(int64_t numKHeads, int64_t numVHeads) {
  if (numKHeads < 1 || numKHeads > 16 || (numKHeads & (numKHeads - 1)) != 0 ||
      numVHeads % numKHeads != 0) {
    return false;
  }
  const int64_t vHeadsPerK = numVHeads / numKHeads;
  return vHeadsPerK >= 1 && vHeadsPerK <= 4;
}

ge::graphStatus GetShapeInfo(gert::TilingContext* context,
                             int64_t& batchSize,
                             int64_t& convTileCount,
                             int64_t& numVHeads,
                             int64_t& numKHeads,
                             bool& splitIndices) {
  const auto* qkvShape = context->GetInputShape(0);
  const auto* zShape = context->GetInputShape(1);
  const auto* convWeightShape = context->GetInputShape(4);
  const auto* convStateShape = context->GetInputShape(5);
  const auto* ssmStateShape = context->GetInputShape(8);
  const auto* readStateIndicesShape = context->GetInputShape(9);
  const auto* writeStateIndicesShape = context->GetInputShape(10);
  if (qkvShape == nullptr || zShape == nullptr || convWeightShape == nullptr ||
      convStateShape == nullptr || ssmStateShape == nullptr ||
      readStateIndicesShape == nullptr || writeStateIndicesShape == nullptr) {
    return ge::GRAPH_FAILED;
  }

  const gert::Shape& qkv = qkvShape->GetStorageShape();
  const gert::Shape& z = zShape->GetStorageShape();
  const gert::Shape& convWeight = convWeightShape->GetStorageShape();
  const gert::Shape& convState = convStateShape->GetStorageShape();
  const gert::Shape& ssmState = ssmStateShape->GetStorageShape();
  const gert::Shape& readStateIndices =
      readStateIndicesShape->GetStorageShape();
  const gert::Shape& writeStateIndices =
      writeStateIndicesShape->GetStorageShape();
  if (qkv.GetDimNum() != 2 || z.GetDimNum() != 3 ||
      convWeight.GetDimNum() != 2 || convState.GetDimNum() != 3 ||
      ssmState.GetDimNum() != 4) {
    return ge::GRAPH_FAILED;
  }

  batchSize = qkv.GetDim(0);
  splitIndices = readStateIndices.GetDimNum() == 2;
  const bool validIndices =
      splitIndices ? readStateIndices.GetDim(0) == batchSize &&
                         readStateIndices.GetDim(1) == 2 &&
                         writeStateIndices.GetDimNum() == 2 &&
                         writeStateIndices.GetDim(0) == batchSize &&
                         writeStateIndices.GetDim(1) == 2
                   : readStateIndices.GetDimNum() == 1 &&
                         writeStateIndices.GetDimNum() == 1 &&
                         readStateIndices.GetDim(0) == batchSize &&
                         writeStateIndices.GetDim(0) == batchSize;
  const int64_t convDim = qkv.GetDim(1);
  numVHeads = z.GetDim(1);
  const int64_t qkWidth = convDim - numVHeads * kHeadDim;
  if (batchSize < 1 || batchSize > kMaxBatchSize || convDim <= 0 ||
      convDim % kHeadDim != 0 || qkWidth <= 0 ||
      qkWidth % (2 * kHeadDim) != 0 || z.GetDim(0) != batchSize ||
      z.GetDim(2) != kHeadDim || convWeight.GetDim(0) != 4 ||
      convWeight.GetDim(1) != convDim || convState.GetDim(0) < 1 ||
      (!splitIndices && convState.GetDim(0) > kMaxNumCacheSlots) ||
      convState.GetDim(1) != kConvStateLen || convState.GetDim(2) != convDim ||
      ssmState.GetDim(0) < 1 ||
      (!splitIndices && ssmState.GetDim(0) != convState.GetDim(0)) ||
      ssmState.GetDim(1) != numVHeads || ssmState.GetDim(2) != kHeadDim ||
      ssmState.GetDim(3) != kHeadDim || !validIndices) {
    return ge::GRAPH_FAILED;
  }

  numKHeads = qkWidth / (2 * kHeadDim);
  if (!IsSupportedShape(numKHeads, numVHeads)) {
    return ge::GRAPH_FAILED;
  }
  convTileCount = convDim / kHeadDim;
  return ge::GRAPH_SUCCESS;
}

}  // namespace

static ge::graphStatus MegaGdnDecodeTiling(gert::TilingContext* context) {
  if (platform_ascendc::PlatformAscendC(context->GetPlatformInfo())
          .GetSocVersion() == platform_ascendc::SocVersion::ASCEND950) {
    return gdn_a5_backend::optiling::MegaGdnDecodeTiling(context);
  }

  int64_t batchSize = 0;
  int64_t convTileCount = 0;
  int64_t numVHeads = 0;
  int64_t numKHeads = 0;
  bool splitIndices = false;
  if (GetShapeInfo(context,
                   batchSize,
                   convTileCount,
                   numVHeads,
                   numKHeads,
                   splitIndices) != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }

  MegaGdnDecodeTilingData tiling;
  tiling.set_batch_size(batchSize);
  tiling.set_state_index_width(splitIndices ? 2 : 1);
  tiling.set_num_k_heads(numKHeads);
  tiling.set_num_v_heads(numVHeads);
  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                      context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
  const auto* attrs = context->GetAttrs();
  const bool* flaSsmStateLayout =
      attrs == nullptr ? nullptr
                       : attrs->GetAttrPointer<bool>(kFlaSsmStateLayoutAttr);
  if (flaSsmStateLayout == nullptr) {
    return ge::GRAPH_FAILED;
  }
  uint64_t tilingKey = batchSize == 1 ? 2 : 1;
  if (!*flaSsmStateLayout) {
    tilingKey += kNonFlaTilingKeyOffset;
  }
  if (splitIndices) {
    tilingKey = *flaSsmStateLayout ? 2001 : 2011;
  }
  context->SetTilingKey(tilingKey);

  auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  const uint32_t aicCoreNum = platform.GetCoreNumAic();
  const uint32_t aivCoreNum = platform.GetCoreNumAiv();
  const bool isAscend950 =
      platform.GetSocVersion() == platform_ascendc::SocVersion::ASCEND950;
  const uint32_t headCount = static_cast<uint32_t>(batchSize * numVHeads);
  const uint32_t convTaskCount = static_cast<uint32_t>(convTileCount);
  const uint32_t taskCount = batchSize == 1 && !isAscend950
                                 ? std::max((convTaskCount + 1) / 2, headCount)
                                 : std::max(convTaskCount, headCount);
  const uint32_t usedAivCoreNum = std::min(taskCount, aivCoreNum);
  const uint32_t blockDim =
      isAscend950
          ? usedAivCoreNum
          : platform.CalcTschBlockDim(usedAivCoreNum, aicCoreNum, aivCoreNum);
  if (blockDim == 0) {
    return ge::GRAPH_FAILED;
  }

  context->SetBlockDim(blockDim);
  if (context->SetScheduleMode(1) != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }

  size_t* workspaceSizes = context->GetWorkspaceSizes(1);
  workspaceSizes[0] = platform.GetLibApiWorkSpaceSize();
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(XllmKernelMegaGdnDecode).Tiling(MegaGdnDecodeTiling);

}  // namespace optiling
