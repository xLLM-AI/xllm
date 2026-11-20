/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under
the terms and conditions of  * CANN Open Software License Agreement Version 2.0
(the "License").  * Please refer to the License for details. You may not use
this file except in compliance with the License.  * THIS SOFTWARE IS PROVIDED ON
AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,  *
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
PARTICULAR PURPOSE.  * See LICENSE in the root of the software repository for
the full text of the License.  */

/*!
 * \file sparse_flash_attention_lse_tiling.cpp
 * \brief
 */

#include "sparse_flash_attention_lse_tiling.h"

#include <graph/utils/type_utils.h>

#include <algorithm>
#include <map>
#include <numeric>
#include <vector>

#include "err/ops_err.h"
#include "register/op_def_registry.h"
#include "sparse_flash_attention_lse/op_kernel/sparse_flash_attention_lse_template_tiling_key.h"

using std::map;
using std::pair;
using std::string;

using namespace ge;
using namespace AscendC;
namespace optiling {

constexpr uint32_t PRE_LOAD_NUM = 2;
constexpr uint32_t BLOCK_TABLE_ELEM_BYTE = 4;
constexpr int32_t SPARSE_MODE_BAND = 4;

static const std::string QUERY_NAME = "query";
static const std::string KEY_NAME = "key";
static const std::string VALUE_NAME = "value";
static const std::string BLOCK_TABLE_NAME = "block_table";
static const std::string SPARSE_INDICES_NAME = "sparse_indices";
static const std::string QUERY_ROPE_NAME = "query_rope";
static const std::string KEY_ROPE_NAME = "key_rope";
static const std::string ATTEN_OUT_NAME = "attention_out";
static const std::string SOFTMAX_MAX_NAME = "softmax_max";
static const std::string SOFTMAX_SUM_NAME = "softmax_sum";

const std::map<std::string, std::vector<ge::DataType>> DTYPE_SUPPORT_MAP = {
    {QUERY_NAME, {ge::DT_FLOAT16, ge::DT_BF16}},
    {KEY_NAME, {ge::DT_FLOAT16, ge::DT_BF16}},
    {VALUE_NAME, {ge::DT_FLOAT16, ge::DT_BF16}},
    {QUERY_ROPE_NAME, {ge::DT_FLOAT16, ge::DT_BF16}},
    {KEY_ROPE_NAME, {ge::DT_FLOAT16, ge::DT_BF16}},
    {ATTEN_OUT_NAME, {ge::DT_FLOAT16, ge::DT_BF16}},
    {SOFTMAX_MAX_NAME, {ge::DT_FLOAT}},
    {SOFTMAX_SUM_NAME, {ge::DT_FLOAT}},
    {SPARSE_INDICES_NAME, {ge::DT_INT32}},
    {BLOCK_TABLE_NAME, {ge::DT_INT32}},
};

const std::map<std::string, std::vector<SFALSELayout>> LAYOUT_SUPPORT_MAP = {
    {QUERY_NAME, {SFALSELayout::BSND, SFALSELayout::TND}},
    {KEY_NAME, {SFALSELayout::BSND, SFALSELayout::TND, SFALSELayout::PA_BSND}},
    {VALUE_NAME,
     {SFALSELayout::BSND, SFALSELayout::TND, SFALSELayout::PA_BSND}},
    {ATTEN_OUT_NAME, {SFALSELayout::BSND, SFALSELayout::TND}},
    {SOFTMAX_MAX_NAME, {SFALSELayout::BNSG, SFALSELayout::NTG}},
    {SOFTMAX_SUM_NAME, {SFALSELayout::BNSG, SFALSELayout::NTG}},
};

const std::map<ge::DataType, std::string> DATATYPE_TO_STRING_MAP = {
    {ge::DT_UNDEFINED,
     "DT_UNDEFINED"},  // Used to indicate a DataType field has not been set.
    {ge::DT_FLOAT, "DT_FLOAT"},                    // float type
    {ge::DT_FLOAT16, "DT_FLOAT16"},                // fp16 type
    {ge::DT_INT8, "DT_INT8"},                      // int8 type
    {ge::DT_INT16, "DT_INT16"},                    // int16 type
    {ge::DT_UINT16, "DT_UINT16"},                  // uint16 type
    {ge::DT_UINT8, "DT_UINT8"},                    // uint8 type
    {ge::DT_INT32, "DT_INT32"},                    // uint32 type
    {ge::DT_INT64, "DT_INT64"},                    // int64 type
    {ge::DT_UINT32, "DT_UINT32"},                  // unsigned int32
    {ge::DT_UINT64, "DT_UINT64"},                  // unsigned int64
    {ge::DT_BOOL, "DT_BOOL"},                      // bool type
    {ge::DT_DOUBLE, "DT_DOUBLE"},                  // double type
    {ge::DT_DUAL, "DT_DUAL"},                      // dual output type
    {ge::DT_DUAL_SUB_INT8, "DT_DUAL_SUB_INT8"},    // dual output int8 type
    {ge::DT_DUAL_SUB_UINT8, "DT_DUAL_SUB_UINT8"},  // dual output uint8 type
    {ge::DT_COMPLEX32, "DT_COMPLEX32"},            // complex32 type
    {ge::DT_COMPLEX64, "DT_COMPLEX64"},            // complex64 type
    {ge::DT_COMPLEX128, "DT_COMPLEX128"},          // complex128 type
    {ge::DT_QINT8, "DT_QINT8"},                    // qint8 type
    {ge::DT_QINT16, "DT_QINT16"},                  // qint16 type
    {ge::DT_QINT32, "DT_QINT32"},                  // qint32 type
    {ge::DT_QUINT8, "DT_QUINT8"},                  // quint8 type
    {ge::DT_QUINT16, "DT_QUINT16"},                // quint16 type
    {ge::DT_RESOURCE, "DT_RESOURCE"},              // resource type
    {ge::DT_STRING_REF, "DT_STRING_REF"},          // string ref type
    {ge::DT_STRING, "DT_STRING"},                  // string type
    {ge::DT_VARIANT, "DT_VARIANT"},                // dt_variant type
    {ge::DT_BF16, "DT_BFLOAT16"},                  // dt_bfloat16 type
    {ge::DT_INT4, "DT_INT4"},                      // dt_variant type
    {ge::DT_UINT1, "DT_UINT1"},                    // dt_variant type
    {ge::DT_INT2, "DT_INT2"},                      // dt_variant type
    {ge::DT_UINT2, "DT_UINT2"}                     // dt_variant type
};

struct SparseFlashAttentionLseCompileInfo {
  int64_t core_num;
};

static const std::map<SFALSELayout, std::vector<SFAAxis>> SFA_LAYOUT_AXIS_MAP =
    {
        {SFALSELayout::BSND, {SFAAxis::B, SFAAxis::S, SFAAxis::N, SFAAxis::D}},
        {SFALSELayout::TND, {SFAAxis::T, SFAAxis::N, SFAAxis::D}},
        {SFALSELayout::PA_BSND,
         {SFAAxis::Bn, SFAAxis::Bs, SFAAxis::N, SFAAxis::D}},
        {SFALSELayout::BNSG, {SFAAxis::B, SFAAxis::N, SFAAxis::S, SFAAxis::G}},
        {SFALSELayout::NTG, {SFAAxis::N, SFAAxis::T, SFAAxis::G}},
};

static const std::map<SFALSELayout, size_t> SFA_LAYOUT_DIM_MAP = {
    {SFALSELayout::BSND, DIM_NUM_FOUR},
    {SFALSELayout::TND, DIM_NUM_THREE},
    {SFALSELayout::PA_BSND, DIM_NUM_FOUR},
    {SFALSELayout::BNSG, DIM_NUM_FOUR},
    {SFALSELayout::NTG, DIM_NUM_THREE},
};

static std::string GetShapeStr(gert::Shape shape) {
  std::ostringstream oss;
  oss << "[";
  if (shape.GetDimNum() > 0) {
    for (size_t i = 0; i < shape.GetDimNum() - 1; ++i) {
      oss << shape.GetDim(i) << ", ";
    }
    oss << shape.GetDim(shape.GetDimNum() - 1);
  }
  oss << "]";
  return oss.str();
}

static std::string SFADataTypeToSerialString(ge::DataType type) {
  const auto it = DATATYPE_TO_STRING_MAP.find(type);
  if (it != DATATYPE_TO_STRING_MAP.end()) {
    return it->second;
  } else {
    OP_LOGE(
        "XllmKernelSparseFlashAttentionLse", "datatype %d not support", type);
    return "UNDEFINED";
  }
}

string SFALSETensorDesc2String(const gert::StorageShape* shape,
                               const gert::CompileTimeTensorDesc* tensor) {
  if (shape == nullptr || tensor == nullptr) {
    return "nil ";
  }

  std::ostringstream oss;
  oss << "(dtype: "
      << ge::TypeUtils::DataTypeToAscendString(tensor->GetDataType())
             .GetString()
      << "),";
  oss << "(shape:" << SFAShape2String(shape->GetStorageShape()) << "),";
  oss << "(ori_shape:" << SFAShape2String(shape->GetOriginShape()) << "),";
  oss << "(format: "
      << ge::TypeUtils::FormatToAscendString(
             static_cast<ge::Format>(
                 ge::GetPrimaryFormat(tensor->GetStorageFormat())))
             .GetString()
      << "),";
  oss << "(ori_format: "
      << ge::TypeUtils::FormatToAscendString(tensor->GetOriginFormat())
             .GetString()
      << ") ";

  return oss.str();
}

string SFALSEDebugTilingContext(const gert::TilingContext* context) {
  std::ostringstream oss;
  for (size_t i = 0; i < context->GetComputeNodeInfo()->GetInputsNum(); ++i) {
    oss << "input" << i << ": ";
    oss << SFALSETensorDesc2String(context->GetInputShape(i),
                                   context->GetInputDesc(i));
  }

  for (size_t i = 0; i < context->GetComputeNodeInfo()->GetOutputsNum(); ++i) {
    oss << "output" << i << ": ";
    oss << SFALSETensorDesc2String(context->GetOutputShape(i),
                                   context->GetOutputDesc(i));
  }
  return oss.str();
}

std::string SFALSELayoutToSerialString(SFALSELayout layout) {
  switch (layout) {
    case SFALSELayout::BSND:
      return "BSND";
    case SFALSELayout::TND:
      return "TND";
    case SFALSELayout::PA_BSND:
      return "PA_BSND";
    case SFALSELayout::BNSG:
      return "BNSG";
    case SFALSELayout::NTG:
      return "NTG";
    default:
      return "UNKNOWN";
  }
}

ge::graphStatus SFALSEMlaTiling::SetBlockDim(uint32_t blockDim) const {
  context_->SetBlockDim(blockDim);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFALSEMlaTiling::SetTilingKey(uint64_t tilingKey) const {
  context_->SetTilingKey(tilingKey);
  context_->SetScheduleMode(1);  // 1: batchmode模式
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFALSEMlaTiling::SetWorkspaceSize(
    uint64_t workspaceSize) const {
  OP_CHECK_IF(
      context_->GetWorkspaceSizes(1) == nullptr,
      OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(),
                                  "workSpaceSize got from ge is nullptr"),
      return ge::GRAPH_FAILED);
  size_t* workSpaces = context_->GetWorkspaceSizes(1);
  workSpaces[0] = workspaceSize;
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFALSEMlaTiling::SetTilingData(TilingDef& tilingData) const {
  OP_CHECK_IF(context_->GetRawTilingData() == nullptr,
              OPS_REPORT_VECTOR_INNER_ERR(
                  context_->GetNodeName(),
                  "RawTilingData got from GE context is nullptr."),
              return ge::GRAPH_FAILED);

  tilingData.SaveToBuffer(context_->GetRawTilingData()->GetData(),
                          context_->GetRawTilingData()->GetCapacity());
  context_->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFALSEMlaTiling::GetPlatformInfo() {
  OP_CHECK_IF(sfaInfo_->platformInfo == nullptr,
              OPS_REPORT_VECTOR_INNER_ERR(sfaInfo_->opName,
                                          "GetPlatformInfo is nullptr."),
              return ge::GRAPH_FAILED);

  auto ascendcPlatform =
      platform_ascendc::PlatformAscendC(sfaInfo_->platformInfo);
  libapiSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();
  aivNum_ = ascendcPlatform.GetCoreNumAiv();
  aicNum_ = ascendcPlatform.GetCoreNumAic();

  OP_CHECK_IF(aicNum_ == 0 || aivNum_ == 0,
              OPS_REPORT_VECTOR_INNER_ERR(sfaInfo_->opName,
                                          "num of core obtained is 0."),
              return GRAPH_FAILED);

  return ge::GRAPH_SUCCESS;
}

void SFALSEMlaTiling::GenTilingKey() {
  uint32_t inputQType = static_cast<uint32_t>(sfaInfo_->inputQType);
  uint32_t inputKvType = static_cast<uint32_t>(sfaInfo_->inputKvType);
  uint32_t outputType = static_cast<uint32_t>(sfaInfo_->outputType);
  uint32_t layoutQuery = static_cast<uint32_t>(sfaInfo_->qLayout);
  uint32_t layoutKV = static_cast<uint32_t>(sfaInfo_->kvLayout);
  uint32_t pageAttention = 0U;
  if (sfaInfo_->kvLayout == SFALSELayout::PA_BSND) {
    pageAttention = 1U;
  }

  const uint32_t hasRope =
      (sfaInfo_->opParamInfo.queryRope.tensor != nullptr) ? 1U : 0U;
  tilingKey_ = GET_TPL_TILING_KEY(0U,
                                  pageAttention,
                                  layoutQuery,
                                  layoutKV,
                                  perfMode_ == SFAPerfMode::V_TEMPLATE_MODE,
                                  static_cast<uint32_t>(sfaInfo_->gSize > 64),
                                  hasRope);  // N1 > 128时核间切G

  OP_LOGI(sfaInfo_->opName, "SFA tilingKey_: %lu.", tilingKey_);
}

void SFALSEMlaTiling::ZeroTensorProcess() const {
  if (sfaInfo_->s2Size == 0) {
    /*
     * 1024，空tensor场景下，作为默认值完成后续计算
     * 避免matmal tiling  softmax tiling异常
     * kernel计算使用真实的seqSize=0, 与actuseq_len流程归一
     */
    sfaInfo_->s2Size = 1024;
  }
}

void SFALSEMlaTiling::InitParams() {
  if (sfaInfo_->s2Size != 0 &&
      sfaInfo_->sparseBlockSize <= 4) {  // 4:当前支持范围
    perfMode_ = SFAPerfMode::V_TEMPLATE_MODE;
  } else {
    perfMode_ = SFAPerfMode::C_TEMPLATE_MODE;
  }

  coreNum_ = aicNum_;

  headDimAlign_ =
      Align(sfaInfo_->qkHeadDim, BYTE_BLOCK);  // 元素个数按照基本块大小对齐
  ZeroTensorProcess();
}

void SFALSEMlaTiling::CalcUbBmm() {
  uint32_t cubeMSize = sfaInfo_->gSize * sfaInfo_->s1Size;
  uint32_t maxMSize = mBaseSize_;
  if (cubeMSize > maxMSize) {
    cubeMSize = maxMSize;
  }
  mmResUbSize_ =
      sInnerSizeAlign_ *
      Align(cubeMSize,
            16U);  // kernel按照16对齐写出，tiling按照这个原则分配内存
  bmm2ResUbSize_ =
      headDimAlign_ *
      Align(cubeMSize,
            16U);  // kernel按照16对齐写出，tiling按照这个原则分配内存

  qPreSizeMla_ = sfaInfo_->gSize * (headDimAlign_ + 64U) * sfaInfo_->s1Size;
}

void SFALSEMlaTiling::CheckUbSpace() { CalcUbBmm(); }

void SFALSEMlaTiling::CalcInnerSize(uint32_t s2Size) {
  sInnerSize_ = 512;  // 512:s2默认切分大小
  // FlashDecode时，如果S2的计算量>=256(确保切分后不小于128)但又不足以分2次计算时，则修改sInnerSize_，均分为2份进行计算，确保Nbuffer=2
  if (splitKVFlag_ && sfaInfo_->qLayout != SFALSELayout::TND) {
    if (s2Size == 256) {  // 256:s2Size的阈值，判断sInnerSize_是否切分
      sInnerSize_ =
          128;  // 128:sInnerSize_值为s2Size的一半，均分为2份进行计算，
    } else if (s2Size > 256 &&
               s2Size <=
                   sInnerSize_) {  // 256:s2Size的阈值，判断sInnerSize_是否切分
      sInnerSize_ = (sInnerSize_ + 1) / 2;  // 2:减半
    }
  }

  sInnerLoopTimes_ = (s2Size + sInnerSize_ - 1) / sInnerSize_;
  sInnerSizeTail_ = s2Size - (sInnerLoopTimes_ - 1) * sInnerSize_;
  if (sInnerSize_ > s2Size) {
    sInnerSize_ = s2Size;
  }
  sInnerSizeAlign_ =
      Align(sInnerSize_, BYTE_BLOCK);  // 元素个数按照基本块大小对齐

  CheckUbSpace();
}

void SFALSEMlaTiling::SplitBalanced() {
  CalcInnerSize(sfaInfo_->s2Size);

  InnerSplitParams innerSplitParams;
  innerSplitParams.s1GBaseSize = sfaInfo_->gSize;
  innerSplitParams.s2BaseSize = sInnerSize_;
  tilingData_.innerSplitParams.set_mBaseSize(innerSplitParams.s1GBaseSize);
  tilingData_.innerSplitParams.set_s2BaseSize(innerSplitParams.s2BaseSize);

  usedCoreNum_ = aicNum_;
}

void SFALSEMlaTiling::Split() { SplitBalanced(); }

void SFALSEMlaTiling::FillTilingBaseParamsMla() {
  tilingData_.baseParams.set_batchSize(sfaInfo_->bSize);
  tilingData_.baseParams.set_seqSize(sfaInfo_->s2Size);
  tilingData_.baseParams.set_qSeqSize(sfaInfo_->s1Size);
  tilingData_.baseParams.set_blockSize(sfaInfo_->blockSize);
  tilingData_.baseParams.set_maxBlockNumPerBatch(sfaInfo_->maxBlockNumPerBatch);
  tilingData_.baseParams.set_scaleValue(sfaInfo_->scaleValue);
  tilingData_.baseParams.set_nNumOfQInOneGroup(sfaInfo_->n1Size /
                                               sfaInfo_->n2Size);
  tilingData_.baseParams.set_actualLenDimsQ(sfaInfo_->actualLenDimsQ);
  tilingData_.baseParams.set_actualLenDimsKV(sfaInfo_->actualLenDimsKV);
  tilingData_.baseParams.set_outputLayout(
      static_cast<uint32_t>(sfaInfo_->outLayout));
  tilingData_.baseParams.set_sparseMode(sfaInfo_->sparseMode);
  tilingData_.baseParams.set_preTokens(sfaInfo_->preTokens);
  tilingData_.baseParams.set_nextTokens(sfaInfo_->nextTokens);
  tilingData_.baseParams.set_sparseBlockSize(sfaInfo_->sparseBlockSize);
  tilingData_.baseParams.set_sparseBlockCount(sfaInfo_->sparseBlockCount);
  tilingData_.baseParams.set_attentionMode(sfaInfo_->attentionMode);
  tilingData_.baseParams.set_returnSoftmaxLse(sfaInfo_->returnSoftmaxLse);
  tilingData_.baseParams.set_isActualLenDimsNull(
      sfaInfo_->actualQSeqLenFlag ? 0U : 1U);
  tilingData_.baseParams.set_isActualLenDimsKVNull(
      sfaInfo_->actualSeqLenFlag ? 0U : 1U);
}

// for flash decode
void SFALSEMlaTiling::FillTilingSplitKVMla() {
  tilingData_.splitKVParams.set_s2(kvSplitPart_);

  tilingData_.splitKVParams.set_accumOutSize(
      aicNum_ * 2 * sfaInfo_->n2Size * mBaseSize_ *
      headDimAlign_);  // 2:每个核可能有头规约和尾规约，一共两份规约信息
  tilingData_.splitKVParams.set_logSumExpSize(
      2 * aicNum_ * 2 * sfaInfo_->n2Size *
      mBaseSize_ *  // 2:每个核可能有头规约和尾规约，一共两份规约信息;sum + max
      (BYTE_BLOCK / BLOCK_TABLE_ELEM_BYTE));

  if (!splitKVFlag_) {
    tilingData_.splitKVParams.set_s2(0);
  }
}

void SFALSEMlaTiling::FillTilingSingleCoreParamsMla() {
  tilingData_.singleCoreParams.set_usedCoreNum(usedCoreNum_);
}

void SFALSEMlaTiling::FillTilingSingleCoreTensorSizeMla() {
  tilingData_.singleCoreTensorSize.set_mmResUbSize(mmResUbSize_);
  tilingData_.singleCoreTensorSize.set_bmm2ResUbSize(bmm2ResUbSize_);
}

void SFALSEMlaTiling::FillTiling() {
  FillTilingBaseParamsMla();
  FillTilingSplitKVMla();
  FillTilingSingleCoreParamsMla();
  FillTilingSingleCoreTensorSizeMla();
}

uint32_t SFALSEMlaTiling::CalcBalanceFDParamNums(
    const uint32_t actCoreNum) const {
  return actCoreNum * 2 * sfaInfo_->n2Size *
         mBaseSize_;  // 2:每个核可能有头规约和尾规约，一共两份规约信息
}

void SFALSEMlaTiling::NormalCalcFDWorkSpace(const uint32_t actCoreNum) {
  if (splitKVFlag_) {
    uint32_t accumOutSize = 0;
    uint32_t logSumExpSize = 0;
    uint32_t FDParamNums = CalcBalanceFDParamNums(
        actCoreNum);  // balanceModeFlag_ ? CalcBalanceFDParamNums(actCoreNum) :
                      // CalcUnbalanceFDParamNums();
    accumOutSize = FDParamNums * headDimAlign_;
    logSumExpSize =
        2 * FDParamNums *
        (BYTE_BLOCK /
         sfaInfo_->blockTypeSize);  // log和sum的存储空间一致，共需要2份内存
    workspaceSize_ += (accumOutSize + logSumExpSize) * sfaInfo_->blockTypeSize;
  }
}

void SFALSEMlaTiling::CalcFDWorkSpace(const uint32_t actCoreNum) {
  NormalCalcFDWorkSpace(actCoreNum);
}

void SFALSEMlaTiling::GetWorkspaceSize() {
  uint32_t actCoreNum = coreNum_;
  if (sfaInfo_->isA5) {
    workspaceSize_ = libapiSize_;
    constexpr uint32_t TRIPLE_BUFFER_NUM = 3;
    constexpr uint32_t S2_BASE_SIZE = 128;  // S2轴基本块大小
    constexpr uint32_t D_SIZE = 576;
    auto ascendcPlatform =
        platform_ascendc::PlatformAscendC(sfaInfo_->platformInfo);
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    if (sfaInfo_->gSize >
        64) {  // N1大于64时切G，相邻两个cube核处理同一个s2Base
      aicNum = aicNum >> 1;
    }
    workspaceSize_ +=
        (S2_BASE_SIZE * D_SIZE * GetTypeSize(sfaInfo_->inputQType) *
         TRIPLE_BUFFER_NUM * aicNum);
  } else {
    uint32_t mmResElemSize = 4;        // 4:fp32
    uint32_t vec1ResElemSize = 2;      // 2:fp16/bf16
    uint32_t bmm2ResElemSize = 4;      // 4:fp32
    uint32_t qPreProcResElemSize = 0;  // 普通场景不涉及Q预处理
    uint32_t nUpdateElemSize = 4;      // 4:int32
    uint32_t softmaxSumElemSize = 4;   // 4:int32
    float kvDtypeRatio = 1.0;

    workspaceSize_ = libapiSize_;
    uint32_t preLoadNum = 1;
    preLoadNum = PRE_LOAD_NUM;

    workspaceSize_ += preLoadNum * (mmResUbSize_ * actCoreNum * mmResElemSize);
    workspaceSize_ +=
        preLoadNum *
        static_cast<size_t>(
            static_cast<float>(mmResUbSize_ * actCoreNum * vec1ResElemSize) *
            kvDtypeRatio);
    workspaceSize_ +=
        preLoadNum * bmm2ResUbSize_ * actCoreNum * bmm2ResElemSize;
    workspaceSize_ +=
        preLoadNum *
        static_cast<size_t>(static_cast<float>(qPreSizeMla_ * actCoreNum *
                                               qPreProcResElemSize) *
                            kvDtypeRatio);
    workspaceSize_ += preLoadNum * mBaseSize_ * actCoreNum * nUpdateElemSize;
    workspaceSize_ += preLoadNum * mBaseSize_ * actCoreNum * softmaxSumElemSize;
    // topk BlkSize == 1场景, 需要额外空间缓存离散聚合的值
    //              bufNum  s2Base   D   dRope  sizeOf(half)
    // 4:bufNum  512:s2Base  512:D  64:dRope  2:sizeOf(half)
    // kernel 侧 kvMerge 布局仍按 576 行宽寻址, NoPE 时 rope 半区闲置, 避免
    // host/device 不一致
    workspaceSize_ += 4 * 512 * (512 + 64) * 2 * actCoreNum;
    // 缓存有效mte2 size的长度 份数  512B对齐的长度  sizeof(int32_t)   aiv核数
    workspaceSize_ +=
        4 * 128 * 4 * (2 * actCoreNum);  // 4:缓存有效mte2 size的长度 128:份数
                                         // 4:512B对齐的长度  2:aiv核数
  }

  CalcFDWorkSpace(actCoreNum);
}

void SFALSEMlaTiling::CalcBlockDim() {
  auto ascendcPlatform =
      platform_ascendc::PlatformAscendC(sfaInfo_->platformInfo);
  auto aicNum = usedCoreNum_;
  auto aivNum = 2 * usedCoreNum_;

  blockDim_ = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);
  OP_LOGI(sfaInfo_->opName,
          "SFA block dim: %u aiv Num: %u aic Num: %u.",
          blockDim_,
          aivNum,
          aicNum);
}

ge::graphStatus SFALSEMlaTiling::DoOpTiling(SFALSETilingInfo* sfaInfo) {
  sfaInfo_ = sfaInfo;
  if (GetPlatformInfo() != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  InitParams();
  Split();
  FillTiling();
  CalcBlockDim();
  GetWorkspaceSize();
  GenTilingKey();

  if ((SetBlockDim(blockDim_) != ge::GRAPH_SUCCESS) ||
      (SetTilingKey(tilingKey_) != ge::GRAPH_SUCCESS) ||
      (SetWorkspaceSize(workspaceSize_) != ge::GRAPH_SUCCESS) ||
      (SetTilingData(tilingData_) != ge::GRAPH_SUCCESS)) {
    return ge::GRAPH_FAILED;
  }

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingSparseFlashAttentionLse(gert::TilingContext* context) {
  SFALSETilingInfo sfaInfo;
  SFALSEInfoParser sfaInfoParser(context);
  if (sfaInfoParser.Parse(sfaInfo) != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }

  SFALSETilingCheck tilingChecker(sfaInfo);
  if (tilingChecker.Process() != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }

  SFALSEMlaTiling tiling(context);
  return tiling.DoOpTiling(&sfaInfo);
}

ge::graphStatus TilingPrepareForSparseFlashAttentionLse(
    gert::TilingParseContext* const context) {
  (void)context;
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFALSETilingCheck::GetExpectedShape(
    gert::Shape& shapeExpected,
    const SFALSETilingShapeCompareParam& param,
    const SFALSELayout& layout) const {
  if (layout == SFALSELayout::BSND) {
    shapeExpected = gert::Shape({param.B, param.S, param.N, param.D});
  } else if (layout == SFALSELayout::TND) {
    shapeExpected = gert::Shape({param.T, param.N, param.D});
  } else if (layout == SFALSELayout::PA_BSND) {
    shapeExpected = gert::Shape({param.Bn, param.Bs, param.N, param.D});
  } else if (layout == SFALSELayout::BNSG) {
    shapeExpected = gert::Shape({param.B, param.N, param.S, param.G});
  } else if (layout == SFALSELayout::NTG) {
    shapeExpected = gert::Shape({param.N, param.T, param.G});
  } else {
    OP_LOGE(opName_,
            "layout %s is unsupported",
            SFALSELayoutToSerialString(layout).c_str());
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFALSETilingCheck::CompareShape(
    SFALSETilingShapeCompareParam& param,
    const gert::Shape& shape,
    const SFALSELayout& layout,
    const std::string& name) const {
  gert::Shape shapeExpected;
  if (GetExpectedShape(shapeExpected, param, layout) != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }

  if (shape.GetDimNum() != shapeExpected.GetDimNum()) {
    OP_LOGE(opName_,
            "%s dimension is %zu, expected dimension is %zu.",
            name.c_str(),
            shape.GetDimNum(),
            shapeExpected.GetDimNum());
    return ge::GRAPH_FAILED;
  }

  for (size_t i = 0; i < shape.GetDimNum(); i++) {
    if (shape.GetDim(i) != shapeExpected.GetDim(i)) {
      OP_LOGE(opName_,
              "%s layout is %s, shape is %s, expected shape is %s.",
              name.c_str(),
              SFALSELayoutToSerialString(layout).c_str(),
              GetShapeStr(shape).c_str(),
              GetShapeStr(shapeExpected).c_str());
      return ge::GRAPH_FAILED;
    }
  }

  return ge::GRAPH_SUCCESS;
}

void SFALSETilingCheck::LogErrorDtypeSupport(
    const std::vector<ge::DataType>& expectDtypeList,
    const ge::DataType& actualDtype,
    const std::string& name) const {
  std::ostringstream oss;
  for (size_t i = 0; i < expectDtypeList.size(); ++i) {
    oss << SFADataTypeToSerialString(expectDtypeList[i]);
    if (i < expectDtypeList.size() - 1) {
      oss << ", ";
    }
  }
  OP_LOGE(opName_,
          "Tensor %s only supports dtype %s, but got %s",
          name.c_str(),
          oss.str().c_str(),
          SFADataTypeToSerialString(actualDtype).c_str());
}

ge::graphStatus SFALSETilingCheck::CheckDtypeSupport(
    const gert::CompileTimeTensorDesc* desc,
    const std::string& name) const {
  if (desc != nullptr) {
    const auto& it = DTYPE_SUPPORT_MAP.find(name);
    OP_CHECK_IF(
        it == DTYPE_SUPPORT_MAP.end(),
        OP_LOGE(
            opName_,
            "%s datatype support list should be specify in DTYPE_SUPPORT_MAP",
            name.c_str()),
        return ge::GRAPH_FAILED);
    auto& expectDtypeList = it->second;
    OP_CHECK_IF(
        std::find(expectDtypeList.begin(),
                  expectDtypeList.end(),
                  desc->GetDataType()) == expectDtypeList.end(),
        LogErrorDtypeSupport(expectDtypeList, desc->GetDataType(), name),
        return ge::GRAPH_FAILED);
  }
  return ge::GRAPH_SUCCESS;
}

template <typename T>
void SFALSETilingCheck::LogErrorNumberSupport(
    const std::vector<T>& expectNumberList,
    const T& actualValue,
    const std::string& name,
    const std::string subName) const {
  std::ostringstream oss;
  for (size_t i = 0; i < expectNumberList.size(); ++i) {
    oss << std::to_string(expectNumberList[i]);
    if (i < expectNumberList.size() - 1) {
      oss << ", ";
    }
  }

  OP_LOGE(opName_,
          "%s %s only supports %s, but got %s",
          name.c_str(),
          subName.c_str(),
          oss.str().c_str(),
          std::to_string(actualValue).c_str());
}

template <typename T>
void SFALSETilingCheck::LogErrorDimNumSupport(
    const std::vector<T>& expectNumberList,
    const T& actualValue,
    const std::string& name) const {
  LogErrorNumberSupport(expectNumberList, actualValue, name, "dimension");
}

ge::graphStatus SFALSETilingCheck::CheckDimNumInLayoutSupport(
    const SFALSELayout& layout,
    const gert::StorageShape* shape,
    const std::string& name) const {
  const auto& dimIt = SFA_LAYOUT_DIM_MAP.find(layout);
  OP_CHECK_IF(
      shape->GetStorageShape().GetDimNum() != dimIt->second,
      OP_LOGE(opName_,
              "When layout is %s, %s dimension should be %zu, but it's %zu",
              SFALSELayoutToSerialString(layout).c_str(),
              name.c_str(),
              dimIt->second,
              shape->GetStorageShape().GetDimNum()),
      return ge::GRAPH_FAILED);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFALSETilingCheck::CheckDimNumSupport(
    const gert::StorageShape* shape,
    const std::vector<size_t>& expectDimNumList,
    const std::string& name) const {
  if (shape == nullptr) {
    return ge::GRAPH_SUCCESS;
  }

  if (std::find(expectDimNumList.begin(),
                expectDimNumList.end(),
                shape->GetStorageShape().GetDimNum()) ==
      expectDimNumList.end()) {
    LogErrorDimNumSupport(
        expectDimNumList, shape->GetStorageShape().GetDimNum(), name);
    return ge::GRAPH_FAILED;
  }

  return ge::GRAPH_SUCCESS;
}

void SFALSETilingCheck::LogErrorLayoutSupport(
    const std::vector<SFALSELayout>& expectLayoutList,
    const SFALSELayout& actualLayout,
    const std::string& name) const {
  std::ostringstream oss;
  for (size_t i = 0; i < expectLayoutList.size(); ++i) {
    oss << SFALSELayoutToSerialString(expectLayoutList[i]);
    if (i < expectLayoutList.size() - 1) {
      oss << ", ";
    }
  }
  OP_LOGE(opName_,
          "Tensor %s only supports layout %s, but got %s",
          name.c_str(),
          oss.str().c_str(),
          SFALSELayoutToSerialString(actualLayout).c_str());
}

ge::graphStatus SFALSETilingCheck::CheckLayoutSupport(
    const SFALSELayout& actualLayout,
    const std::string& name) const {
  const auto& it = LAYOUT_SUPPORT_MAP.find(name);
  OP_CHECK_IF(
      it == LAYOUT_SUPPORT_MAP.end(),
      OP_LOGE(opName_,
              "%s layout support list should be specify in LAYOUT_SUPPORT_MAP",
              name.c_str()),
      return ge::GRAPH_FAILED);
  auto& expectLayoutList = it->second;
  OP_CHECK_IF(std::find(expectLayoutList.begin(),
                        expectLayoutList.end(),
                        actualLayout) == expectLayoutList.end(),
              LogErrorLayoutSupport(expectLayoutList, actualLayout, name),
              return ge::GRAPH_FAILED);

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFALSETilingCheck::CheckSingleParaQuery() const {
  const std::vector<size_t> queryDimNumList = {DIM_NUM_THREE, DIM_NUM_FOUR};
  if (ge::GRAPH_SUCCESS !=
          CheckDtypeSupport(opParamInfo_.query.desc, QUERY_NAME) ||
      ge::GRAPH_SUCCESS != CheckLayoutSupport(qLayout_, QUERY_NAME) ||
      ge::GRAPH_SUCCESS != CheckDimNumSupport(opParamInfo_.query.shape,
                                              queryDimNumList,
                                              QUERY_NAME) ||
      ge::GRAPH_SUCCESS != CheckDimNumInLayoutSupport(qLayout_,
                                                      opParamInfo_.query.shape,
                                                      QUERY_NAME)) {
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFALSETilingCheck::CheckSingleParaKey() const {
  const std::vector<size_t> keyDimNumList = {DIM_NUM_FOUR, DIM_NUM_THREE};
  if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.key.desc, KEY_NAME) ||
      ge::GRAPH_SUCCESS != CheckLayoutSupport(kvLayout_, KEY_NAME) ||
      ge::GRAPH_SUCCESS !=
          CheckDimNumSupport(opParamInfo_.key.shape, keyDimNumList, KEY_NAME) ||
      ge::GRAPH_SUCCESS != CheckDimNumInLayoutSupport(
                               kvLayout_, opParamInfo_.key.shape, KEY_NAME)) {
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFALSETilingCheck::CheckSingleParaNumHeads() const {
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFALSETilingCheck::CheckSingleParaKvHeadNums() const {
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFALSETilingCheck::CheckSingleParaSparseMode() const {
  OP_CHECK_IF((*opParamInfo_.sparseMode != 3 && *opParamInfo_.sparseMode != 0),
              OP_LOGE(opName_,
                      "sparseMode must == 0/3, but got: %ld.",
                      *opParamInfo_.sparseMode),
              return ge::GRAPH_FAILED);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFALSETilingCheck::CheckSingleParaSparseBlockSize() const {
  OP_CHECK_IF(
      (*opParamInfo_.sparseBlockSize <= 0 ||
       *opParamInfo_.sparseBlockSize > 128 ||
       (static_cast<uint64_t>(*opParamInfo_.sparseBlockSize) &
        static_cast<uint64_t>(*opParamInfo_.sparseBlockSize - 1L)) != 0UL),
      OP_LOGE(opName_,
              "sparseBlockSize should be be in range [1, 128] and be a power "
              "of 2, but got: %ld.",
              *opParamInfo_.sparseBlockSize),
      return ge::GRAPH_FAILED);

  OP_CHECK_IF(
      (npuArch_ == NpuArch::DAV_3510 && *opParamInfo_.sparseBlockSize != 1),
      OP_LOGE(opName_,
              "when soc version is Ascend950, sparse_block_size only support "
              "1, but now is %d.",
              *opParamInfo_.sparseBlockSize),
      return ge::GRAPH_FAILED);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFALSETilingCheck::CheckSingleParaSparseIndices() const {
  if (ge::GRAPH_SUCCESS !=
      CheckDtypeSupport(opParamInfo_.sparseIndices.desc, SPARSE_INDICES_NAME)) {
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFALSETilingCheck::CheckSingleParaPreTokens() const {
  OP_CHECK_IF((*opParamInfo_.preTokens != INT64_MAX),
              OP_LOGE(opName_,
                      "preTokens should be 9223372036854775807, but got: %ld.",
                      *opParamInfo_.preTokens),
              return ge::GRAPH_FAILED);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFALSETilingCheck::CheckSingleParaNextTokens() const {
  OP_CHECK_IF((*opParamInfo_.nextTokens != INT64_MAX),
              OP_LOGE(opName_,
                      "nextTokens should be 9223372036854775807, but got: %ld.",
                      *opParamInfo_.nextTokens),
              return ge::GRAPH_FAILED);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFALSETilingCheck::CheckSinglePara() const {
  if (ge::GRAPH_SUCCESS != CheckSingleParaQuery() ||
      ge::GRAPH_SUCCESS != CheckSingleParaKey() ||
      ge::GRAPH_SUCCESS != CheckSingleParaSparseIndices() ||
      ge::GRAPH_SUCCESS != CheckSingleParaNumHeads() ||
      ge::GRAPH_SUCCESS != CheckSingleParaKvHeadNums() ||
      ge::GRAPH_SUCCESS != CheckSingleParaSparseMode() ||
      ge::GRAPH_SUCCESS != CheckSingleParaSparseBlockSize() ||
      ge::GRAPH_SUCCESS != CheckSingleParaPreTokens() ||
      ge::GRAPH_SUCCESS != CheckSingleParaNextTokens()) {
    return ge::GRAPH_FAILED;
  }

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFALSETilingCheck::CheckRopeExistence() {
  OP_CHECK_IF((opParamInfo_.queryRope.tensor != nullptr ||
               opParamInfo_.keyRope.tensor != nullptr) &&
                  *opParamInfo_.attentionMode == 0,
              OP_LOGE(opName_,
                      "In MHA/GQA situation(attentionMode=0), queryRope and "
                      "keyRope should be null."),
              return ge::GRAPH_FAILED);
  OP_CHECK_IF(*opParamInfo_.attentionMode != 2,
              OP_LOGE(opName_, "attentionMode only support 2."),
              return ge::GRAPH_FAILED);
  OP_CHECK_IF((opParamInfo_.queryRope.tensor != nullptr &&
               opParamInfo_.keyRope.tensor == nullptr),
              OP_LOGE(opName_,
                      "KeyRope is null, but queryRope exists, they should be "
                      "both null or exist."),
              return ge::GRAPH_FAILED);
  OP_CHECK_IF((opParamInfo_.queryRope.tensor == nullptr &&
               opParamInfo_.keyRope.tensor != nullptr),
              OP_LOGE(opName_,
                      "QueryRope is null, but keyRope exists, they should be "
                      "both null or exist."),
              return ge::GRAPH_FAILED);
  if (opParamInfo_.queryRope.tensor == nullptr) {
    // NoPE场景: queryRope/keyRope均为空, 走HAS_ROPE=0的kernel实例化,
    // 无需校验rope desc
    return ge::GRAPH_SUCCESS;
  }
  OP_CHECK_IF(
      opParamInfo_.keyRope.desc == nullptr ||
          opParamInfo_.queryRope.desc == nullptr,
      OP_LOGE(
          opName_,
          "In Mla situation, desc of keyRope and queryRope should not be null"),
      return ge::GRAPH_FAILED);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFALSETilingCheck::CheckExists(const void* pointer,
                                               const std::string& name) const {
  OP_CHECK_IF(pointer == nullptr,
              OP_LOGE(opName_, "%s should not be null", name.c_str()),
              return ge::GRAPH_FAILED);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFALSETilingCheck::CheckNotExists(
    const void* pointer,
    const std::string& name) const {
  OP_CHECK_IF(pointer != nullptr,
              OP_LOGE(opName_, "%s should be null", name.c_str()),
              return ge::GRAPH_FAILED);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFALSETilingCheck::CheckExistsByMap(
    const std::map<std::string, const void*>& paramMap) const {
  for (const auto& kv : paramMap) {
    if (CheckExists(kv.second, kv.first) != ge::GRAPH_SUCCESS) {
      return ge::GRAPH_FAILED;
    }
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFALSETilingCheck::CheckNotExistsByMap(
    const std::map<std::string, const void*>& paramMap) const {
  for (const auto& kv : paramMap) {
    if (CheckNotExists(kv.second, kv.first) != ge::GRAPH_SUCCESS) {
      return ge::GRAPH_FAILED;
    }
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFALSETilingCheck::CheckExistenceByMap(
    std::map<std::string, const void*>& existMap,
    std::map<std::string, const void*>& notExistMap) const {
  if (CheckExistsByMap(existMap) != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  if (CheckNotExistsByMap(notExistMap) != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

template <typename T>
ge::graphStatus SFALSETilingCheck::CheckAttrValueByMap(
    std::map<std::string, std::pair<const T*, T>>& attrMap) const {
  for (auto const& kv : attrMap) {
    const std::string& name = kv.first;
    const std::pair<const T*, T>& pointerValuePair = kv.second;
    if (pointerValuePair.first == nullptr) {
      OP_LOGE(opName_, "Attr %s should not be nullptr", name.c_str());
      return ge::GRAPH_FAILED;
    }

    if (*(pointerValuePair.first) != pointerValuePair.second) {
      std::ostringstream ossExpect;
      ossExpect << std::to_string(pointerValuePair.second);
      std::ostringstream ossActual;
      ossActual << std::to_string(*(pointerValuePair.first));
      OP_LOGE(opName_,
              "%s value should be %s, but got %s",
              name.c_str(),
              ossExpect.str().c_str(),
              ossActual.str().c_str());
      return ge::GRAPH_FAILED;
    }
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFALSETilingCheck::CheckParaExistenceMlaNoquant() const {
  if (kvStorageMode_ != KvStorageMode::PAGE_ATTENTION) {
    return ge::GRAPH_SUCCESS;
  }
  std::map<std::string, const void*> mlaNoquantParamExistMap = {
      {"actualSeqLengths", opParamInfo_.actualSeqLengths.tensor},
      {"blockTable", opParamInfo_.blockTable.tensor},
  };
  std::map<std::string, const void*> mlaNoquantParamNotExistMap = {};
  if (CheckExistenceByMap(mlaNoquantParamExistMap,
                          mlaNoquantParamNotExistMap) != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFALSETilingCheck::CheckParaExistenceMla() const {
  return CheckParaExistenceMlaNoquant();
}

ge::graphStatus SFALSETilingCheck::CheckParaExistence() {
  if (ge::GRAPH_SUCCESS != CheckRopeExistence()) {
    return ge::GRAPH_FAILED;
  }

  return CheckParaExistenceMla();
}

ge::graphStatus SFALSETilingCheck::GetActualSeqLenSize(
    uint32_t& size,
    const gert::Tensor* tensor,
    const SFALSELayout& layout,
    const std::string& name) const {
  if (tensor == nullptr) {
    OP_LOGE(opName_,
            "when layout of query is %s, %s must be provided.",
            SFALSELayoutToSerialString(layout).c_str(),
            name.c_str());
    return ge::GRAPH_FAILED;
  }
  int64_t shapeSize = tensor->GetShapeSize();
  if (shapeSize <= 0) {
    OP_LOGE(opName_,
            "the shape size of %s is %ld, it should be greater than 0.",
            name.c_str(),
            shapeSize);
    return ge::GRAPH_FAILED;
  }
  size = static_cast<uint32_t>(shapeSize);
  return ge::GRAPH_SUCCESS;
}

void SFALSETilingCheck::SetSFAShapeCompare() {
  queryShapeCmp_ = opParamInfo_.query.shape->GetStorageShape();
  topkShapeCmp_ = opParamInfo_.sparseIndices.shape->GetStorageShape();
  keyShapeCmp_ = opParamInfo_.key.shape->GetStorageShape();
  valueShapeCmp_ = opParamInfo_.value.shape->GetStorageShape();
  attenOutShapeCmp_ = opParamInfo_.attenOut.shape->GetStorageShape();
  if (opParamInfo_.queryRope.tensor != nullptr) {
    queryRopeShapeCmp_ = opParamInfo_.queryRope.tensor->GetStorageShape();
  }
  if (opParamInfo_.keyRope.tensor != nullptr) {
    keyRopeShapeCmp_ = opParamInfo_.keyRope.tensor->GetStorageShape();
  }
  softmaxMaxShapeCmp_ = opParamInfo_.softmaxMax.shape->GetStorageShape();
  softmaxSumShapeCmp_ = opParamInfo_.softmaxSum.shape->GetStorageShape();
}

ge::graphStatus SFALSETilingCheck::CheckBlockTable() const {
  if (kvStorageMode_ != KvStorageMode::PAGE_ATTENTION) {
    OP_CHECK_IF(opParamInfo_.blockTable.tensor != nullptr,
                OP_LOGE(opName_,
                        "when the layout_kv is %s, %s should be null",
                        SFALSELayoutToSerialString(kvLayout_).c_str(),
                        BLOCK_TABLE_NAME.c_str()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
  }
  if (ge::GRAPH_SUCCESS !=
      CheckDtypeSupport(opParamInfo_.blockTable.desc, BLOCK_TABLE_NAME)) {
    return ge::GRAPH_FAILED;
  }
  uint32_t blockTableBatch =
      opParamInfo_.blockTable.tensor->GetStorageShape().GetDim(0);
  OP_CHECK_IF(
      blockTableBatch != bSize_,
      OP_LOGE(opName_,
              "%s's first dimension(%u) should be equal to batch size(%u)",
              BLOCK_TABLE_NAME.c_str(),
              blockTableBatch,
              bSize_),
      return ge::GRAPH_FAILED);

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFALSETilingCheck::CheckDTypeConsistency(
    const ge::DataType& actualDtype,
    const ge::DataType& expectDtype,
    const std::string& name) const {
  if (actualDtype != expectDtype) {
    OP_LOGE(opName_,
            "%s dtype should be %s, but it's %s.",
            name.c_str(),
            SFADataTypeToSerialString(expectDtype).c_str(),
            SFADataTypeToSerialString(actualDtype).c_str());
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFALSETilingCheck::CheckQRopeShape() {
  SFALSETilingShapeCompareParam shapeParams;
  shapeParams.B = bSize_;
  shapeParams.N = n1Size_;
  shapeParams.S = s1Size_;
  shapeParams.D = ropeHeadDim_;
  shapeParams.T = qTSize_;
  return CompareShape(
      shapeParams, queryRopeShapeCmp_, qLayout_, QUERY_ROPE_NAME);
}

ge::graphStatus SFALSETilingCheck::CheckTopkShape() {
  SFALSETilingShapeCompareParam shapeParams;
  shapeParams.B = bSize_;
  shapeParams.N = n2Size_;
  shapeParams.S = s1Size_;
  shapeParams.D = sparseBlockCount_;
  shapeParams.T = qTSize_;
  return CompareShape(
      shapeParams, topkShapeCmp_, topkLayout_, SPARSE_INDICES_NAME);
}

ge::graphStatus SFALSETilingCheck::CheckAttenOutShape() {
  SFALSETilingShapeCompareParam shapeParams;
  shapeParams.B = bSize_;
  shapeParams.N = n1Size_;
  shapeParams.S = s1Size_;
  shapeParams.D = vHeadDim_;
  shapeParams.T = qTSize_;
  if (CompareShape(
          shapeParams, attenOutShapeCmp_, outLayout_, ATTEN_OUT_NAME) !=
      ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFALSETilingCheck::CheckSoftmaxMaxShape() {
  if (*opParamInfo_.returnSoftmaxLse) {
    SFALSETilingShapeCompareParam shapeParams;
    shapeParams.B = bSize_;
    shapeParams.N = n2Size_;
    shapeParams.S = s1Size_;
    shapeParams.T = qTSize_;
    shapeParams.G = n1Size_ / n2Size_;
    if (CompareShape(shapeParams,
                     softmaxMaxShapeCmp_,
                     softmaxMaxLayout_,
                     SOFTMAX_MAX_NAME) != ge::GRAPH_SUCCESS) {
      return ge::GRAPH_FAILED;
    }
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFALSETilingCheck::CheckSoftmaxSumShape() {
  if (*opParamInfo_.returnSoftmaxLse) {
    SFALSETilingShapeCompareParam shapeParams;
    shapeParams.B = bSize_;
    shapeParams.N = n2Size_;
    shapeParams.S = s1Size_;
    shapeParams.D = vHeadDim_;
    shapeParams.T = qTSize_;
    shapeParams.G = n1Size_ / n2Size_;
    if (CompareShape(shapeParams,
                     softmaxSumShapeCmp_,
                     softmaxSumLayout_,
                     SOFTMAX_SUM_NAME) != ge::GRAPH_SUCCESS) {
      return ge::GRAPH_FAILED;
    }
  }

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFALSETilingCheck::CheckAttenOut() {
  if (ge::GRAPH_SUCCESS !=
          CheckDTypeConsistency(opParamInfo_.attenOut.desc->GetDataType(),
                                inputQType_,
                                ATTEN_OUT_NAME) ||
      ge::GRAPH_SUCCESS != CheckAttenOutShape() ||
      ge::GRAPH_SUCCESS != CheckSoftmaxMaxShape() ||
      ge::GRAPH_SUCCESS != CheckSoftmaxSumShape()) {
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}
ge::graphStatus SFALSETilingCheck::CheckSoftmaxMax() {
  if (*opParamInfo_.returnSoftmaxLse) {
    OP_CHECK_IF(
        opParamInfo_.softmaxMax.shape->GetStorageShape().GetShapeSize() == 0,
        OP_LOGE(opName_,
                "When return_softmax_lse is true, SoftmaxMax tensor cannot be "
                "empty tensor."),
        return ge::GRAPH_FAILED);
    // type类型校验
    OP_CHECK_IF(opParamInfo_.softmaxMax.desc->GetDataType() != ge::DT_FLOAT,
                OP_LOGE(opName_, "SoftmaxMax's dtype must be FLOAT."),
                return ge::GRAPH_FAILED);
    // shape和维度校验
    if (ge::GRAPH_SUCCESS != CheckAttenOutShape()) {
      return ge::GRAPH_FAILED;
    }
  } else {
    if (opParamInfo_.softmaxMax.shape->GetStorageShape().GetShapeSize() != 0) {
      OP_LOGW(opName_,
              "When return_softmax_lse is false, SoftmaxMax tensor must be "
              "empty tensor.");
    }
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFALSETilingCheck::CheckSoftmaxSum() {
  if (*opParamInfo_.returnSoftmaxLse) {
    OP_CHECK_IF(
        opParamInfo_.softmaxSum.shape->GetStorageShape().GetShapeSize() == 0,
        OP_LOGE(opName_,
                "When return_softmax_lse is true, softmaxSum tensor cannot be "
                "empty tensor."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.softmaxSum.desc->GetDataType() != ge::DT_FLOAT,
                OP_LOGE(opName_, "softmaxSum's dtype must be FLOAT."),
                return ge::GRAPH_FAILED);
  } else {
    if (opParamInfo_.softmaxSum.shape->GetStorageShape().GetShapeSize() != 0) {
      OP_LOGW(opName_,
              "When return_softmax_lse is false, softmaxSum tensor must be "
              "empty tensor.");
    }
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFALSETilingCheck::CheckQRope() {
  if (opParamInfo_.queryRope.tensor == nullptr) {
    return ge::GRAPH_SUCCESS;
  }
  if (ge::GRAPH_SUCCESS !=
          CheckDTypeConsistency(opParamInfo_.queryRope.desc->GetDataType(),
                                inputQType_,
                                QUERY_ROPE_NAME) ||
      ge::GRAPH_SUCCESS != CheckQRopeShape()) {
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFALSETilingCheck::CheckTopK() {
  if (ge::GRAPH_SUCCESS != CheckTopkShape()) {
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFALSETilingCheck::CheckVAndKRopeShapeForBatchContinuous() {
  SFALSETilingShapeCompareParam shapeParams;
  shapeParams.B = bSize_;
  shapeParams.N = n2Size_;
  shapeParams.S = s2Size_;
  shapeParams.T = kvTSize_;
  shapeParams.D = qkHeadDim_;
  if (CompareShape(shapeParams, keyShapeCmp_, kvLayout_, KEY_NAME) !=
      ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }

  shapeParams.D = vHeadDim_;
  if (CompareShape(shapeParams, valueShapeCmp_, kvLayout_, VALUE_NAME) !=
      ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }

  if (opParamInfo_.keyRope.tensor != nullptr) {
    shapeParams.D = ropeHeadDim_;
    if (CompareShape(shapeParams, keyRopeShapeCmp_, kvLayout_, KEY_ROPE_NAME) !=
        ge::GRAPH_SUCCESS) {
      return ge::GRAPH_FAILED;
    }
  }
  return ge::GRAPH_SUCCESS;
}

uint32_t SFALSEMlaTiling::GetTypeSize(ge::DataType dtype) const {
  uint32_t typeSize = NUM_BYTES_FLOAT16;
  switch (dtype) {
    case ge::DT_FLOAT16:
      typeSize = NUM_BYTES_FLOAT16;
      break;
    case ge::DT_BF16:
      typeSize = NUM_BYTES_BF16;
      break;
    default:
      typeSize = NUM_BYTES_FLOAT16;
  }
  return typeSize;
}

ge::graphStatus SFALSETilingCheck::CheckVAndKRopeShapeForPageAttention() {
  int64_t blockNum = keyShapeCmp_.GetDim(0);
  OP_CHECK_IF(blockNum <= 0,
              OP_LOGE(opName_,
                      "The first dim(%ld) of key should be greater than 0",
                      blockNum),
              return ge::GRAPH_FAILED);
  SFALSETilingShapeCompareParam shapeParams;
  shapeParams.Bn = blockNum;
  shapeParams.N = n2Size_;
  shapeParams.Bs = blockSize_;
  shapeParams.D = vHeadDim_;
  shapeParams.T = kvTSize_;
  if (CompareShape(shapeParams, valueShapeCmp_, kvLayout_, VALUE_NAME) !=
      ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }

  if (opParamInfo_.keyRope.tensor != nullptr) {
    shapeParams.D = ropeHeadDim_;
    if (CompareShape(shapeParams, keyRopeShapeCmp_, kvLayout_, KEY_ROPE_NAME) !=
        ge::GRAPH_SUCCESS) {
      return ge::GRAPH_FAILED;
    }
  }

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFALSETilingCheck::CheckVAndKRopeShape() {
  if (kvStorageMode_ == KvStorageMode::BATCH_CONTINUOUS) {
    return CheckVAndKRopeShapeForBatchContinuous();
  }

  if (kvStorageMode_ == KvStorageMode::PAGE_ATTENTION) {
    return CheckVAndKRopeShapeForPageAttention();
  }

  OP_LOGE(opName_,
          "storage mode of key and value is %u, it is incorrect.",
          static_cast<uint32_t>(kvStorageMode_));
  return ge::GRAPH_FAILED;
}

// NoPE场景(queryRope/keyRope均为空)下仅校验key/value本身, 跳过rope相关校验
ge::graphStatus SFALSETilingCheck::CheckVAndKShapeWithoutRope() {
  if (kvStorageMode_ == KvStorageMode::BATCH_CONTINUOUS) {
    SFALSETilingShapeCompareParam shapeParams;
    shapeParams.B = bSize_;
    shapeParams.N = n2Size_;
    shapeParams.S = s2Size_;
    shapeParams.T = kvTSize_;
    shapeParams.D = qkHeadDim_;
    if (CompareShape(shapeParams, keyShapeCmp_, kvLayout_, KEY_NAME) !=
        ge::GRAPH_SUCCESS) {
      return ge::GRAPH_FAILED;
    }
    shapeParams.D = vHeadDim_;
    if (CompareShape(shapeParams, valueShapeCmp_, kvLayout_, VALUE_NAME) !=
        ge::GRAPH_SUCCESS) {
      return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
  }

  if (kvStorageMode_ == KvStorageMode::PAGE_ATTENTION) {
    int64_t blockNum = keyShapeCmp_.GetDim(0);
    OP_CHECK_IF(blockNum <= 0,
                OP_LOGE(opName_,
                        "The first dim(%ld) of key should be greater than 0",
                        blockNum),
                return ge::GRAPH_FAILED);
    SFALSETilingShapeCompareParam shapeParams;
    shapeParams.Bn = blockNum;
    shapeParams.N = n2Size_;
    shapeParams.Bs = blockSize_;
    shapeParams.D = qkHeadDim_;
    shapeParams.T = kvTSize_;
    if (CompareShape(shapeParams, keyShapeCmp_, kvLayout_, KEY_NAME) !=
        ge::GRAPH_SUCCESS) {
      return ge::GRAPH_FAILED;
    }
    shapeParams.D = vHeadDim_;
    if (CompareShape(shapeParams, valueShapeCmp_, kvLayout_, VALUE_NAME) !=
        ge::GRAPH_SUCCESS) {
      return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
  }

  OP_LOGE(opName_,
          "storage mode of key and value is %u, it is incorrect.",
          static_cast<uint32_t>(kvStorageMode_));
  return ge::GRAPH_FAILED;
}

ge::graphStatus SFALSETilingCheck::CheckVAndKRope() {
  if (ge::GRAPH_SUCCESS !=
          CheckDTypeConsistency(opParamInfo_.value.desc->GetDataType(),
                                inputKvType_,
                                VALUE_NAME) ||
      ge::GRAPH_SUCCESS !=
          CheckDTypeConsistency(opParamInfo_.keyRope.desc->GetDataType(),
                                inputKvType_,
                                KEY_ROPE_NAME) ||
      ge::GRAPH_SUCCESS != CheckVAndKRopeShape()) {
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFALSETilingCheck::CheckActualSeqLensQ() {
  if (ge::GRAPH_SUCCESS != CheckActualSeqLensQDType() ||
      ge::GRAPH_SUCCESS != CheckActualSeqLensQShape()) {
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFALSETilingCheck::CheckActualSeqLensQDType() {
  if (opParamInfo_.actualSeqLengthsQ.tensor == nullptr) {
    return ge::GRAPH_SUCCESS;
  }
  if (opParamInfo_.actualSeqLengthsQ.desc == nullptr) {
    OP_LOGE(opName_,
            "actualSeqLengthsQ is not empty,"
            "but actualSeqLengthsQ's dtype is nullptr.");
    return ge::GRAPH_FAILED;
  }
  if (opParamInfo_.actualSeqLengthsQ.desc->GetDataType() != ge::DT_INT32) {
    OP_LOGE(opName_,
            "actualSeqLengthsQ's dtype is %s, it should be DT_INT32.",
            SFADataTypeToSerialString(
                opParamInfo_.actualSeqLengthsQ.desc->GetDataType())
                .c_str());
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFALSETilingCheck::CheckActualSeqLensQShape() {
  if (opParamInfo_.actualSeqLengthsQ.tensor == nullptr) {
    return ge::GRAPH_SUCCESS;
  }
  uint32_t shapeSize = 0;
  if (GetActualSeqLenSize(shapeSize,
                          opParamInfo_.actualSeqLengthsQ.tensor,
                          qLayout_,
                          "actualSeqLengthsQ") != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  if (shapeSize != bSize_) {
    OP_LOGE(opName_,
            "actualSeqLengthsQ shape size is %u, it should be equal to batch "
            "size[%u]",
            shapeSize,
            bSize_);
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFALSETilingCheck::CheckActualSeqLens() {
  if (std::string(opParamInfo_.layoutKV) == "TND" &&
      opParamInfo_.actualSeqLengths.tensor == nullptr) {
    OP_LOGE(opName_,
            "when the layout of key and value is TND, "
            "the actualSeqLengths of key and value should not be empty.");
    return ge::GRAPH_PARAM_INVALID;
  }
  if (ge::GRAPH_SUCCESS != CheckActualSeqLensDType() ||
      ge::GRAPH_SUCCESS != CheckActualSeqLensShape()) {
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFALSETilingCheck::CheckActualSeqLensDType() {
  if (opParamInfo_.actualSeqLengths.tensor == nullptr) {
    return ge::GRAPH_SUCCESS;
  }
  if (opParamInfo_.actualSeqLengths.desc == nullptr) {
    OP_LOGE(opName_,
            "actualSeqLengths is not empty,"
            "but actualSeqLengths's dtype is nullptr.");
    return ge::GRAPH_FAILED;
  }
  if (opParamInfo_.actualSeqLengths.desc->GetDataType() != ge::DT_INT32) {
    OP_LOGE(opName_,
            "actualSeqLengths's dtype is %s, it should be DT_INT32.",
            SFADataTypeToSerialString(
                opParamInfo_.actualSeqLengths.desc->GetDataType())
                .c_str());
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFALSETilingCheck::CheckActualSeqLensShape() {
  if (opParamInfo_.actualSeqLengths.tensor == nullptr) {
    return ge::GRAPH_SUCCESS;
  }
  uint32_t shapeSize = 0;
  if (GetActualSeqLenSize(shapeSize,
                          opParamInfo_.actualSeqLengths.tensor,
                          kvLayout_,
                          "actualSeqLengths") != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  if (shapeSize != bSize_) {
    OP_LOGE(opName_,
            "actualSeqLengths shape size is %u, it should be equal to batch "
            "size[%u].",
            shapeSize,
            bSize_);
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFALSETilingCheck::CheckMultiParaConsistency() {
  SetSFAShapeCompare();
  // NoPE场景(queryRope/keyRope均为空)跳过rope相关的shape/dtype一致性校验,
  // 但仍需校验key/value
  bool hasRope = (opParamInfo_.queryRope.tensor != nullptr);
  if ((hasRope && (ge::GRAPH_SUCCESS != CheckVAndKRope() ||
                   ge::GRAPH_SUCCESS != CheckQRope())) ||
      (!hasRope &&
       (ge::GRAPH_SUCCESS !=
            CheckDTypeConsistency(opParamInfo_.value.desc->GetDataType(),
                                  inputKvType_,
                                  VALUE_NAME) ||
        ge::GRAPH_SUCCESS != CheckVAndKShapeWithoutRope())) ||
      ge::GRAPH_SUCCESS != CheckTopK() ||
      ge::GRAPH_SUCCESS != CheckAttenOut() ||
      ge::GRAPH_SUCCESS != CheckSoftmaxMax() ||
      ge::GRAPH_SUCCESS != CheckSoftmaxSum() ||
      ge::GRAPH_SUCCESS != CheckActualSeqLensQ() ||
      ge::GRAPH_SUCCESS != CheckActualSeqLens() ||
      ge::GRAPH_SUCCESS != CheckBlockTable()) {
    return ge::GRAPH_FAILED;
  }

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFALSETilingCheck::CheckFeatureMlaNoQuantShape() const {
  OP_CHECK_IF(
      bSize_ <= 0,
      OP_LOGE(
          opName_, "batch_size should be greater than 0, but got %u", bSize_),
      return ge::GRAPH_FAILED);

  OP_CHECK_IF(qTSize_ <= 0 && (qLayout_ == SFALSELayout::TND),
              OP_LOGE(opName_,
                      "T_size of query should be greater than 0, but got %u",
                      qTSize_),
              return ge::GRAPH_FAILED);

  OP_CHECK_IF(
      n1Size_ <= 0,
      OP_LOGE(
          opName_, "q_head_num should be greater than 0, but got %u", n1Size_),
      return ge::GRAPH_FAILED);

  OP_CHECK_IF(n2Size_ != 1,
              OP_LOGE(opName_, "kv_head_num should be 1, but got %u", n2Size_),
              return ge::GRAPH_FAILED);

  OP_CHECK_IF(n1Size_ % n2Size_ != 0,
              OP_LOGE(opName_,
                      "q_head_num(%u) must be divisible by kv_head_num(%u)",
                      n1Size_,
                      n2Size_),
              return ge::GRAPH_FAILED);

  OP_CHECK_IF(
      gSize_ < 1 || (gSize_ > 64 && gSize_ != 128),
      OP_LOGE(
          opName_, "group num should be in 1 ~ 64, 128, but got %u", gSize_),
      return ge::GRAPH_FAILED);

  OP_CHECK_IF(
      qkHeadDim_ != 512,
      OP_LOGE(opName_, "qk_head_dim only support 512, but got %u", qkHeadDim_),
      return ge::GRAPH_FAILED);

  OP_CHECK_IF(qkHeadDim_ != vHeadDim_,
              OP_LOGE(opName_,
                      "qk_head_dim[%u] should be equal to v_head_dim[%u]",
                      qkHeadDim_,
                      vHeadDim_),
              return ge::GRAPH_FAILED);

  // ropeHeadDim必须与rope tensor的存在性绑定: 有rope输入时只允许64维,
  // NoPE(无rope输入)时只允许0,
  // 防止"tensor非空但D=0"的非法输入通过校验后进入HAS_ROPE=1的kernel按64维越界读取
  const bool hasRope = (opParamInfo_.queryRope.tensor != nullptr);
  OP_CHECK_IF(
      hasRope && ropeHeadDim_ != 64,
      OP_LOGE(
          opName_,
          "rope_head_dim should be 64 when query_rope is provided, but got %u",
          ropeHeadDim_),
      return ge::GRAPH_FAILED);
  OP_CHECK_IF(!hasRope && ropeHeadDim_ != 0,
              OP_LOGE(opName_,
                      "rope_head_dim should be 0 when query_rope is not "
                      "provided, but got %u",
                      ropeHeadDim_),
              return ge::GRAPH_FAILED);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFALSETilingCheck::CheckFeatureMlaNoQuantLayout() const {
  const std::vector<std::string> layoutSupportList = {"BSND", "TND"};
  std::string layoutQuery = opParamInfo_.layoutQuery;
  OP_CHECK_IF(std::find(layoutSupportList.begin(),
                        layoutSupportList.end(),
                        layoutQuery) == layoutSupportList.end(),
              OP_LOGE(opName_,
                      "layoutQuery only supports BSND/TND, but got %s",
                      layoutQuery.c_str()),
              return ge::GRAPH_FAILED);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFALSETilingCheck::CheckFeatureMlaNoQuantDtype() const {
  OP_CHECK_IF(inputQType_ != ge::DT_BF16 && inputQType_ != ge::DT_FLOAT16,
              OP_LOGE(opName_,
                      "query dtype only support %s and %s, but got %s",
                      SFADataTypeToSerialString(ge::DT_BF16).c_str(),
                      SFADataTypeToSerialString(ge::DT_FLOAT16).c_str(),
                      SFADataTypeToSerialString(inputQType_).c_str()),
              return ge::GRAPH_FAILED);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFALSETilingCheck::CheckFeatureMlaNoquantPa() const {
  if (kvStorageMode_ != KvStorageMode::PAGE_ATTENTION) {
    return ge::GRAPH_SUCCESS;
  }

  OP_CHECK_IF(
      blockSize_ <= 0 || blockSize_ > static_cast<int32_t>(MAX_BLOCK_SIZE),
      OP_LOGE(opName_,
              "when page attention is enabled, block_size(%d) should be in "
              "range (0, %u].",
              blockSize_,
              MAX_BLOCK_SIZE),
      return ge::GRAPH_FAILED);

  OP_CHECK_IF(blockSize_ % 16 > 0,
              OP_LOGE(opName_,
                      "when page attention is enabled, block_size(%d) should "
                      "be 16-aligned.",
                      blockSize_),
              return ge::GRAPH_FAILED);

  OP_CHECK_IF(
      blockSize_ % sparseBlockSize_ > 0,
      OP_LOGE(opName_,
              "when page attention is enabled, block_size(%d) must be divided "
              "by sparse_block_size(%d), but now the remainder is %d.",
              blockSize_,
              sparseBlockSize_,
              blockSize_ % sparseBlockSize_),
      return ge::GRAPH_FAILED);

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFALSETilingCheck::CheckFeatureMlaNoquant() const {
  if (ge::GRAPH_SUCCESS != CheckFeatureMlaNoQuantShape() ||
      ge::GRAPH_SUCCESS != CheckFeatureMlaNoQuantLayout() ||
      ge::GRAPH_SUCCESS != CheckFeatureMlaNoQuantDtype() ||
      ge::GRAPH_SUCCESS != CheckFeatureMlaNoquantPa()) {
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFALSETilingCheck::CheckFeatureMla() const {
  return CheckFeatureMlaNoquant();
}

ge::graphStatus SFALSETilingCheck::CheckFeature() const {
  return CheckFeatureMla();
}

void SFALSETilingCheck::Init() {
  opName_ = sfaInfo_.opName;
  platformInfo_ = sfaInfo_.platformInfo;
  opParamInfo_ = sfaInfo_.opParamInfo;
  npuArch_ = sfaInfo_.npuArch;
  isA5_ = sfaInfo_.isA5;

  bSize_ = sfaInfo_.bSize;
  n1Size_ = sfaInfo_.n1Size;
  n2Size_ = sfaInfo_.n2Size;
  s1Size_ = sfaInfo_.s1Size;
  s2Size_ = sfaInfo_.s2Size;
  gSize_ = sfaInfo_.gSize;
  qkHeadDim_ = sfaInfo_.qkHeadDim;
  vHeadDim_ = sfaInfo_.vHeadDim;
  ropeHeadDim_ = sfaInfo_.ropeHeadDim;
  maxBlockNumPerBatch_ = sfaInfo_.maxBlockNumPerBatch;
  qTSize_ = sfaInfo_.qTSize;
  kvTSize_ = sfaInfo_.kvTSize;
  blockSize_ = sfaInfo_.blockSize;
  sparseBlockCount_ = sfaInfo_.sparseBlockCount;
  sparseBlockSize_ = sfaInfo_.sparseBlockSize;

  inputQType_ = sfaInfo_.inputQType;
  inputKvType_ = sfaInfo_.inputKvType;
  inputQRopeType_ = sfaInfo_.inputQRopeType;
  inputKRopeType_ = sfaInfo_.inputKRopeType;
  outputType_ = sfaInfo_.outputType;

  qLayout_ = sfaInfo_.qLayout;
  topkLayout_ = sfaInfo_.topkLayout;
  kvLayout_ = sfaInfo_.kvLayout;
  outLayout_ = sfaInfo_.outLayout;
  softmaxMaxLayout_ = sfaInfo_.softmaxMaxLayout;
  softmaxSumLayout_ = sfaInfo_.softmaxSumLayout;

  kvStorageMode_ = sfaInfo_.kvStorageMode;
  l2CacheSize_ = sfaInfo_.l2CacheSize;
}

ge::graphStatus SFALSETilingCheck::Process() {
  Init();
  if (CheckSinglePara() != ge::GRAPH_SUCCESS ||
      CheckParaExistence() != ge::GRAPH_SUCCESS ||
      CheckFeature() != ge::GRAPH_SUCCESS ||
      CheckMultiParaConsistency() != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

bool SFALSEInfoParser::HasAxis(const SFAAxis& axis,
                               const SFALSELayout& layout,
                               const gert::Shape& shape) const {
  const auto& layoutIt = SFA_LAYOUT_AXIS_MAP.find(layout);
  if (layoutIt == SFA_LAYOUT_AXIS_MAP.end()) {
    return false;
  }

  const std::vector<SFAAxis>& axes = layoutIt->second;
  const auto& axisIt = std::find(axes.begin(), axes.end(), axis);
  if (axisIt == axes.end()) {
    return false;
  }
  const auto& dimIt = SFA_LAYOUT_DIM_MAP.find(layout);
  if (dimIt == SFA_LAYOUT_DIM_MAP.end() || dimIt->second != shape.GetDimNum()) {
    return false;
  }
  return true;
}

size_t SFALSEInfoParser::GetAxisIdx(const SFAAxis& axis,
                                    const SFALSELayout& layout) const {
  const std::vector<SFAAxis>& axes = SFA_LAYOUT_AXIS_MAP.find(layout)->second;
  const auto& axisIt = std::find(axes.begin(), axes.end(), axis);
  return std::distance(axes.begin(), axisIt);
}

uint32_t SFALSEInfoParser::GetAxisNum(const gert::Shape& shape,
                                      const SFAAxis& axis,
                                      const SFALSELayout& layout) const {
  return HasAxis(axis, layout, shape) ? shape.GetDim(GetAxisIdx(axis, layout))
                                      : invalidDimValue_;
}

ge::graphStatus SFALSEInfoParser::CheckRequiredInOutExistence() const {
  OP_CHECK_IF(opParamInfo_.query.shape == nullptr,
              OP_LOGE(opName_, "Shape of tensor query is nullptr"),
              return ge::GRAPH_FAILED);
  OP_CHECK_IF(opParamInfo_.query.desc == nullptr,
              OP_LOGE(opName_, "Desc of tensor query is nullptr"),
              return ge::GRAPH_FAILED);
  OP_CHECK_IF(opParamInfo_.key.shape == nullptr,
              OP_LOGE(opName_, "Shape of tensor k is nullptr"),
              return ge::GRAPH_FAILED);
  OP_CHECK_IF(opParamInfo_.key.desc == nullptr,
              OP_LOGE(opName_, "Desc of tensor k is nullptr"),
              return ge::GRAPH_FAILED);
  OP_CHECK_IF(opParamInfo_.value.shape == nullptr,
              OP_LOGE(opName_, "Shape of tensor value is nullptr"),
              return ge::GRAPH_FAILED);
  OP_CHECK_IF(opParamInfo_.value.desc == nullptr,
              OP_LOGE(opName_, "Desc of tensor value is nullptr"),
              return ge::GRAPH_FAILED);
  OP_CHECK_IF(opParamInfo_.sparseIndices.shape == nullptr,
              OP_LOGE(opName_, "Shape of tensor sparseIndices is nullptr"),
              return ge::GRAPH_FAILED);
  OP_CHECK_IF(opParamInfo_.sparseIndices.desc == nullptr,
              OP_LOGE(opName_, "Desc of tensor sparseIndices is nullptr"),
              return ge::GRAPH_FAILED);
  OP_CHECK_IF(opParamInfo_.attenOut.shape == nullptr,
              OP_LOGE(opName_, "Shape of tensor output is nullptr"),
              return ge::GRAPH_FAILED);
  OP_CHECK_IF(opParamInfo_.attenOut.desc == nullptr,
              OP_LOGE(opName_, "Desc of tensor output is nullptr"),
              return ge::GRAPH_FAILED);
  OP_CHECK_IF(opParamInfo_.softmaxMax.shape == nullptr,
              OP_LOGE(opName_, "Shape of tensor softmaxMax is nullptr"),
              return ge::GRAPH_FAILED);
  OP_CHECK_IF(opParamInfo_.softmaxMax.desc == nullptr,
              OP_LOGE(opName_, "Desc of tensor softmaxMax is nullptr"),
              return ge::GRAPH_FAILED);
  OP_CHECK_IF(opParamInfo_.softmaxSum.shape == nullptr,
              OP_LOGE(opName_, "Shape of tensor softmaxSum is nullptr"),
              return ge::GRAPH_FAILED);
  OP_CHECK_IF(opParamInfo_.softmaxSum.desc == nullptr,
              OP_LOGE(opName_, "Desc of tensor softmaxSum is nullptr"),
              return ge::GRAPH_FAILED);
  // queryRope为可选输入(NoPE场景为nullptr), 其存在性在CheckRopeExistence中校验

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFALSEInfoParser::CheckRequiredAttrExistence() const {
  OP_CHECK_IF(opParamInfo_.layoutQuery == nullptr,
              OP_LOGE(opName_, "attr layoutQuery is nullptr"),
              return ge::GRAPH_FAILED);
  OP_CHECK_IF(opParamInfo_.layoutKV == nullptr,
              OP_LOGE(opName_, "attr layoutKV is nullptr"),
              return ge::GRAPH_FAILED);
  OP_CHECK_IF(opParamInfo_.sparseBlockSize == nullptr,
              OP_LOGE(opName_, "attr sparseBlockSize is nullptr"),
              return ge::GRAPH_FAILED);
  OP_CHECK_IF(opParamInfo_.scaleValue == nullptr,
              OP_LOGE(opName_, "attr scaleValue is nullptr"),
              return ge::GRAPH_FAILED);
  OP_CHECK_IF(opParamInfo_.sparseMode == nullptr,
              OP_LOGE(opName_, "attr sparseMode is nullptr"),
              return ge::GRAPH_FAILED);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFALSEInfoParser::CheckRequiredParaExistence() const {
  if (CheckRequiredInOutExistence() != ge::GRAPH_SUCCESS ||
      CheckRequiredAttrExistence() != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFALSEInfoParser::GetActualSeqLenSize(
    uint32_t& size,
    const gert::Tensor* tensor,
    SFALSELayout& layout,
    const std::string& name) const {
  if ((tensor == nullptr)) {
    OP_LOGE(opName_,
            "when layout of query is %s, %s must be provided.",
            SFALSELayoutToSerialString(layout).c_str(),
            name.c_str());
    return ge::GRAPH_FAILED;
  }
  int64_t shapeSize = tensor->GetShapeSize();
  if (shapeSize <= 0) {
    OP_LOGE(opName_,
            "the shape size of %s is %ld, it should be greater than 0.",
            name.c_str(),
            shapeSize);
    return ge::GRAPH_FAILED;
  }
  size = static_cast<uint32_t>(shapeSize);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFALSEInfoParser::GetActualSeqLenQSize(uint32_t& size) {
  return GetActualSeqLenSize(size,
                             opParamInfo_.actualSeqLengthsQ.tensor,
                             qLayout_,
                             "actualSeqLengthsQ");
}

ge::graphStatus SFALSEInfoParser::GetOpName() {
  if (context_->GetNodeName() == nullptr) {
    OP_LOGE("XllmKernelSparseFlashAttentionLse",
            "opName got from TilingContext is nullptr");
    return ge::GRAPH_FAILED;
  }
  opName_ = context_->GetNodeName();
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFALSEInfoParser::GetNpuInfo() {
  platformInfo_ = context_->GetPlatformInfo();
  OP_CHECK_IF(
      platformInfo_ == nullptr,
      OPS_REPORT_VECTOR_INNER_ERR(opName_, "GetPlatformInfo is nullptr."),
      return ge::GRAPH_FAILED);

  auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo_);
  uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
  uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
  OP_CHECK_IF(
      aicNum == 0 || aivNum == 0,
      OPS_REPORT_VECTOR_INNER_ERR(opName_, "num of core obtained is 0."),
      return GRAPH_FAILED);

  npuArch_ = ascendcPlatform.GetCurNpuArch();
  isA5_ = (npuArch_ == NpuArch::DAV_3510);
  if (npuArch_ != NpuArch::DAV_2201 && npuArch_ != NpuArch::DAV_3510) {
    OPS_REPORT_VECTOR_INNER_ERR(opName_,
                                "Npu Arch Version[%d] is not support.",
                                static_cast<int32_t>(npuArch_));
    return GRAPH_FAILED;
  }

  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2,
                                 l2CacheSize_);

  return ge::GRAPH_SUCCESS;
}

void SFALSEInfoParser::GetOptionalInputParaInfo() {
  opParamInfo_.blockTable.tensor =
      context_->GetOptionalInputTensor(BLOCK_TABLE_INPUT_INDEX);
  opParamInfo_.blockTable.desc =
      context_->GetOptionalInputDesc(BLOCK_TABLE_INPUT_INDEX);
  opParamInfo_.actualSeqLengthsQ.tensor =
      context_->GetOptionalInputTensor(ACT_SEQ_LEN_Q_INPUT_INDEX);
  opParamInfo_.actualSeqLengthsQ.desc =
      context_->GetOptionalInputDesc(ACT_SEQ_LEN_Q_INPUT_INDEX);
  opParamInfo_.actualSeqLengths.tensor =
      context_->GetOptionalInputTensor(ACT_SEQ_LEN_KV_INPUT_INDEX);
  opParamInfo_.actualSeqLengths.desc =
      context_->GetOptionalInputDesc(ACT_SEQ_LEN_KV_INPUT_INDEX);
  opParamInfo_.queryRope.tensor =
      context_->GetOptionalInputTensor(QUERY_ROPE_INPUT_INDEX);
  opParamInfo_.queryRope.desc =
      context_->GetOptionalInputDesc(QUERY_ROPE_INPUT_INDEX);
  opParamInfo_.keyRope.tensor =
      context_->GetOptionalInputTensor(KEY_ROPE_INPUT_INDEX);
  opParamInfo_.keyRope.desc =
      context_->GetOptionalInputDesc(KEY_ROPE_INPUT_INDEX);
}

void SFALSEInfoParser::GetInputParaInfo() {
  opParamInfo_.query.desc = context_->GetInputDesc(QUERY_INPUT_INDEX);
  opParamInfo_.query.shape = context_->GetInputShape(QUERY_INPUT_INDEX);
  opParamInfo_.key.desc = context_->GetInputDesc(KEY_INPUT_INDEX);
  opParamInfo_.key.shape = context_->GetInputShape(KEY_INPUT_INDEX);
  opParamInfo_.value.desc = context_->GetInputDesc(VALUE_INPUT_INDEX);
  opParamInfo_.value.shape = context_->GetInputShape(VALUE_INPUT_INDEX);
  opParamInfo_.sparseIndices.desc =
      context_->GetInputDesc(SPARSE_INDICES_INPUT_INDEX);
  opParamInfo_.sparseIndices.shape =
      context_->GetInputShape(SPARSE_INDICES_INPUT_INDEX);
  GetOptionalInputParaInfo();
}

void SFALSEInfoParser::GetOutputParaInfo() {
  opParamInfo_.attenOut.desc = context_->GetOutputDesc(OUTPUT_INDEX);
  opParamInfo_.attenOut.shape = context_->GetOutputShape(OUTPUT_INDEX);
  opParamInfo_.softmaxMax.desc = context_->GetOutputDesc(SOFTMAXMAX_INDEX);
  opParamInfo_.softmaxMax.shape = context_->GetOutputShape(SOFTMAXMAX_INDEX);
  opParamInfo_.softmaxSum.desc = context_->GetOutputDesc(SOFTMAXSUM_INDEX);
  opParamInfo_.softmaxSum.shape = context_->GetOutputShape(SOFTMAXSUM_INDEX);
}

ge::graphStatus SFALSEInfoParser::GetAttrParaInfo() {
  auto attrs = context_->GetAttrs();
  OP_CHECK_IF(attrs == nullptr,
              OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(),
                                          "attrs got from ge is nullptr"),
              return ge::GRAPH_FAILED);

  opParamInfo_.layoutQuery = attrs->GetStr(LAYOUT_QUERY_ATTR_INDEX);
  opParamInfo_.layoutKV = attrs->GetStr(LAYOUT_KV_ATTR_INDEX);
  opParamInfo_.sparseBlockSize =
      attrs->GetAttrPointer<int64_t>(SPARSE_BLOCK_SIZE_ATTR_INDEX);
  opParamInfo_.scaleValue =
      attrs->GetAttrPointer<float>(SCALE_VALUE_ATTR_INDEX);
  opParamInfo_.sparseMode =
      attrs->GetAttrPointer<int64_t>(SPARSE_MODE_ATTR_INDEX);
  opParamInfo_.preTokens =
      attrs->GetAttrPointer<int64_t>(PRE_TOKENS_ATTR_INDEX);
  opParamInfo_.nextTokens =
      attrs->GetAttrPointer<int64_t>(NEXT_TOKENS_ATTR_INDEX);
  opParamInfo_.attentionMode =
      attrs->GetAttrPointer<int64_t>(ATTENTION_MODE_ATTR_INDEX);
  opParamInfo_.returnSoftmaxLse =
      attrs->GetAttrPointer<bool>(RETURN_SOFTMAX_LSE_ATTR_INDEX);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFALSEInfoParser::GetOpParaInfo() {
  GetInputParaInfo();
  GetOutputParaInfo();
  if (ge::GRAPH_SUCCESS != GetAttrParaInfo()) {
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFALSEInfoParser::GetInOutDataType() {
  inputQType_ = opParamInfo_.query.desc->GetDataType();
  inputKvType_ = opParamInfo_.key.desc->GetDataType();
  outputType_ = opParamInfo_.attenOut.desc->GetDataType();
  if (opParamInfo_.queryRope.desc != nullptr) {
    inputQRopeType_ = opParamInfo_.queryRope.desc->GetDataType();
  }
  if (opParamInfo_.keyRope.desc != nullptr) {
    inputKRopeType_ = opParamInfo_.keyRope.desc->GetDataType();
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFALSEInfoParser::GetBatchSize() {
  // 获取B基准值
  // 1、非TND时, 以query的batch_size维度为基准;
  // 2、TND时, actual_seq_lens_q必须传入, 以actual_seq_lens_q数组的长度为B轴大小
  if (qLayout_ == SFALSELayout::TND) {
    return GetActualSeqLenQSize(bSize_);
  } else {  // BSND
    bSize_ = GetAxisNum(queryShape_, SFAAxis::B, qLayout_);
    return ge::GRAPH_SUCCESS;
  }
}

ge::graphStatus SFALSEInfoParser::GetQTSize() {
  // 获取query的T基准值
  // 1、非TND时, 以query的batch_size维度为基准;
  // 2、TND时, actual_seq_lens_q必须传入, 以actual_seq_lens_q数组的长度为B轴大小
  qTSize_ = (qLayout_ == SFALSELayout::TND)
                ? GetAxisNum(queryShape_, SFAAxis::T, qLayout_)
                : 0;
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFALSEInfoParser::GetKVTSize() {
  // 获取query的T基准值
  // 1、非TND时, 以key的batch_size维度为基准;
  // 2、TND时, actual_seq_lens_q必须传入, 以actual_seq_lens_q数组的长度为B轴大小
  kvTSize_ = (kvLayout_ == SFALSELayout::TND)
                 ? GetAxisNum(keyShape_, SFAAxis::T, kvLayout_)
                 : 0;
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFALSEInfoParser::GetQkHeadDim() {
  // 获取qkHeadDim基准值
  // 以query的D维度为基准
  qkHeadDim_ = GetAxisNum(queryShape_, SFAAxis::D, qLayout_);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFALSEInfoParser::GetS1Size() {
  // 获取S1基准值
  // 1、非TND时, 以query的S维度为基准;
  // 2、TND时, actual_seq_lens_q必须传入,
  // 以actual_seq_lens_q数组中的最大值为基准
  if (qLayout_ == SFALSELayout::TND) {
    s1Size_ = GetAxisNum(queryShape_, SFAAxis::T, qLayout_);
    return ge::GRAPH_SUCCESS;
  } else {  // BSND
    s1Size_ = GetAxisNum(queryShape_, SFAAxis::S, qLayout_);
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFALSEInfoParser::GetKvStorageMode() {
  if (kvLayout_ == SFALSELayout::PA_BSND) {
    kvStorageMode_ = KvStorageMode::PAGE_ATTENTION;
  } else {
    kvStorageMode_ = KvStorageMode::BATCH_CONTINUOUS;
  }
  // kv存储模式基准值
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFALSEInfoParser::GetKvLayout() {
  const map<string, SFALSELayout> layoutKVMap = {
      {"BSND", SFALSELayout::BSND},
      {"PA_BSND", SFALSELayout::PA_BSND},
      {"TND", SFALSELayout::TND}};

  std::string layout(opParamInfo_.layoutKV);
  auto it = layoutKVMap.find(layout);
  if (it != layoutKVMap.end()) {
    kvLayout_ = it->second;
  } else {
    OP_LOGE(opName_, "layoutKV is %s, it is unsupported.", layout.c_str());
    return ge::GRAPH_FAILED;
  }
  if (kvLayout_ != SFALSELayout::PA_BSND && qLayout_ != kvLayout_) {
    OP_LOGE(
        opName_,
        "When layoutKV is not PA_BSND, layoutKV must be the same as layoutQ.");
    return ge::GRAPH_FAILED;
  }
  uint32_t keyDimNum = opParamInfo_.key.shape->GetStorageShape().GetDimNum();
  if (kvLayout_ == SFALSELayout::PA_BSND && keyDimNum != 4U) {
    OP_LOGE(opName_,
            "When layoutKV is PA_BSND, kvDimNum must be 4, but now is %d.",
            keyDimNum);
    return ge::GRAPH_FAILED;
  }

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFALSEInfoParser::GetS2SizeForBatchContinuous() {
  if (kvLayout_ == SFALSELayout::BSND) {  // BSND
    s2Size_ = GetAxisNum(keyShape_, SFAAxis::S, kvLayout_);
  } else if (kvLayout_ == SFALSELayout::TND) {
    s2Size_ = GetAxisNum(keyShape_, SFAAxis::T, kvLayout_);
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFALSEInfoParser::GetMaxBlockNumPerBatch() {
  if (opParamInfo_.blockTable.tensor == nullptr) {
    OP_LOGE(opName_,
            "the layout_kv is %s, blockTable must be provided.",
            SFALSELayoutToSerialString(kvLayout_).c_str());
    return ge::GRAPH_FAILED;
  }
  uint32_t dimNum =
      opParamInfo_.blockTable.tensor->GetStorageShape().GetDimNum();
  if (dimNum != DIM_NUM_TWO) {
    OP_LOGE(opName_,
            "the dim num of block_table is %u, it should be %u.",
            dimNum,
            DIM_NUM_TWO);
    return ge::GRAPH_FAILED;
  }
  if (opParamInfo_.blockTable.tensor->GetStorageShape().GetDim(1) <= 0) {
    OP_LOGE(opName_,
            "%s's second dimension(%ld) should be greater than 0",
            BLOCK_TABLE_NAME.c_str(),
            opParamInfo_.blockTable.tensor->GetStorageShape().GetDim(1));
    return ge::GRAPH_FAILED;
  }
  maxBlockNumPerBatch_ =
      opParamInfo_.blockTable.tensor->GetStorageShape().GetDim(1);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFALSEInfoParser::GetBlockSize() {
  blockSize_ = GetAxisNum(keyShape_, SFAAxis::Bs, kvLayout_);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFALSEInfoParser::GetSparseBlockCount() {
  sparseBlockCount_ = GetAxisNum(sparseIndicesShape_, SFAAxis::K, qLayout_);

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFALSEInfoParser::GetS2SizeForPageAttention() {
  if (GetMaxBlockNumPerBatch() != ge::GRAPH_SUCCESS ||
      GetBlockSize() != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  s2Size_ = maxBlockNumPerBatch_ * blockSize_;
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFALSEInfoParser::GetS2Size() {
  // 获取S2基准值
  // 1、BATCH_CONTINUOUS时, 从key的S轴获取
  // 2、PAGE_ATTENTION时, S2 = block_table.dim1 * block_size
  if (kvStorageMode_ == KvStorageMode::BATCH_CONTINUOUS) {
    return GetS2SizeForBatchContinuous();
  }
  return GetS2SizeForPageAttention();
}

ge::graphStatus SFALSEInfoParser::GetValueHeadDim() {
  // 获取vHeadDim基准值
  // 以value的D维度为基准
  vHeadDim_ = GetAxisNum(valueShape_, SFAAxis::D, kvLayout_);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFALSEInfoParser::GetRopeHeadDim() {
  if (opParamInfo_.queryRope.tensor == nullptr) {
    // NoPE场景: 无rope输入, ropeHeadDim按0处理
    ropeHeadDim_ = 0;
    return ge::GRAPH_SUCCESS;
  }
  if (queryShape_.GetDimNum() != queryRopeShape_.GetDimNum()) {
    OP_LOGE(opName_,
            "The dimensions of query and query_rope should be equal, but query "
            "has dimension %zu while query_rope has dimension %zu.",
            queryShape_.GetDimNum(),
            queryRopeShape_.GetDimNum());
    return ge::GRAPH_PARAM_INVALID;
  }
  ropeHeadDim_ = GetAxisNum(queryRopeShape_, SFAAxis::D, qLayout_);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFALSEInfoParser::GetQueryAndOutLayout() {
  // 获取query和attentionOut的Layout基准值
  // layoutQuery: {qLayout, outLayout}
  const map<string, pair<SFALSELayout, SFALSELayout>> layoutMap = {
      {"BSND", {SFALSELayout::BSND, SFALSELayout::BSND}},
      {"TND", {SFALSELayout::TND, SFALSELayout::TND}},
  };

  std::string layout(opParamInfo_.layoutQuery);
  auto it = layoutMap.find(layout);
  if (it != layoutMap.end()) {
    qLayout_ = it->second.first;
    outLayout_ = it->second.second;
  } else {
    OP_LOGE(opName_, "layoutQuery is %s, it is unsupported.", layout.c_str());
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFALSEInfoParser::GetTopkLayout() {
  topkLayout_ = qLayout_;
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFALSEInfoParser::GetSoftmaxMaxAndSumLayout() {
  if (qLayout_ == SFALSELayout::BSND) {
    softmaxMaxLayout_ = SFALSELayout::BNSG;
    softmaxSumLayout_ = SFALSELayout::BNSG;
  } else if (qLayout_ == SFALSELayout::TND) {
    softmaxMaxLayout_ = SFALSELayout::NTG;
    softmaxSumLayout_ = SFALSELayout::NTG;
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFALSEInfoParser::GetN1Size() {
  n1Size_ = GetAxisNum(queryShape_, SFAAxis::N, qLayout_);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFALSEInfoParser::GetN2Size() {
  n2Size_ = GetAxisNum(keyShape_, SFAAxis::N, kvLayout_);
  return ge::GRAPH_SUCCESS;
}

void SFALSEInfoParser::SetSFAShape() {
  queryShape_ = opParamInfo_.query.shape->GetStorageShape();
  keyShape_ = opParamInfo_.key.shape->GetStorageShape();
  valueShape_ = opParamInfo_.value.shape->GetStorageShape();
  sparseIndicesShape_ = opParamInfo_.sparseIndices.shape->GetStorageShape();
  if (opParamInfo_.queryRope.tensor != nullptr) {
    queryRopeShape_ = opParamInfo_.queryRope.tensor->GetStorageShape();
  }
}

ge::graphStatus SFALSEInfoParser::GetGSize() {
  if (n2Size_ != 0) {
    gSize_ = n1Size_ / n2Size_;
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFALSEInfoParser::GetActualseqInfo() {
  maxActualseq_ = static_cast<uint32_t>(s2Size_);
  if (opParamInfo_.actualSeqLengths.tensor != nullptr) {
    actualLenDimsKV_ = opParamInfo_.actualSeqLengths.tensor->GetShapeSize();
  }
  if (opParamInfo_.actualSeqLengthsQ.tensor != nullptr) {
    actualLenDimsQ_ = opParamInfo_.actualSeqLengthsQ.tensor->GetShapeSize();
  }
  return ge::GRAPH_SUCCESS;
}

void SFALSEInfoParser::GenerateInfo(SFALSETilingInfo& sfaInfo) {
  sfaInfo.opName = opName_;
  sfaInfo.platformInfo = platformInfo_;
  sfaInfo.opParamInfo = opParamInfo_;
  sfaInfo.npuArch = npuArch_;
  sfaInfo.isA5 = isA5_;

  sfaInfo.bSize = bSize_;
  sfaInfo.n1Size = n1Size_;
  sfaInfo.n2Size = n2Size_;
  sfaInfo.s1Size = s1Size_;
  sfaInfo.s2Size = s2Size_;
  sfaInfo.gSize = gSize_;
  sfaInfo.qkHeadDim = qkHeadDim_;
  sfaInfo.vHeadDim = vHeadDim_;
  sfaInfo.ropeHeadDim = ropeHeadDim_;
  sfaInfo.qTSize = qTSize_;
  sfaInfo.kvTSize = kvTSize_;
  sfaInfo.sparseBlockSize = *opParamInfo_.sparseBlockSize;
  sfaInfo.sparseBlockCount = sparseBlockCount_;

  sfaInfo.inputQType = inputQType_;
  sfaInfo.inputKvType = inputKvType_;
  sfaInfo.inputQRopeType = inputQRopeType_;
  sfaInfo.inputKRopeType = inputKRopeType_;
  sfaInfo.outputType = outputType_;

  sfaInfo.kvStorageMode = kvStorageMode_;
  sfaInfo.l2CacheSize = l2CacheSize_;

  sfaInfo.totalBlockNum = opParamInfo_.key.shape->GetStorageShape().GetDim(0);
  sfaInfo.scaleValue = *opParamInfo_.scaleValue;
  sfaInfo.pageAttentionFlag = (kvStorageMode_ == KvStorageMode::PAGE_ATTENTION);
  sfaInfo.blockSize = blockSize_;
  sfaInfo.blockTypeSize = sizeof(float);
  sfaInfo.maxBlockNumPerBatch = maxBlockNumPerBatch_;

  sfaInfo.actualLenDimsQ = actualLenDimsQ_;
  sfaInfo.actualLenDimsKV = actualLenDimsKV_;
  sfaInfo.maxActualseq = maxActualseq_;
  sfaInfo.isSameSeqAllKVTensor = isSameSeqAllKVTensor_;
  sfaInfo.isSameActualseq = isSameActualseq_;

  sfaInfo.actualQSeqLenFlag =
      (opParamInfo_.actualSeqLengthsQ.tensor != nullptr);
  sfaInfo.actualSeqLenFlag = (opParamInfo_.actualSeqLengths.tensor != nullptr);

  sfaInfo.sparseMode = *opParamInfo_.sparseMode;
  sfaInfo.preTokens = *opParamInfo_.preTokens;
  sfaInfo.nextTokens = *opParamInfo_.nextTokens;
  sfaInfo.attentionMode = *opParamInfo_.attentionMode;
  sfaInfo.returnSoftmaxLse = *opParamInfo_.returnSoftmaxLse;

  sfaInfo.qLayout = qLayout_;
  sfaInfo.topkLayout = topkLayout_;
  sfaInfo.kvLayout = kvLayout_;
  sfaInfo.outLayout = outLayout_;
  sfaInfo.softmaxMaxLayout = softmaxMaxLayout_;
  sfaInfo.softmaxSumLayout = softmaxSumLayout_;
}

ge::graphStatus SFALSEInfoParser::Parse(SFALSETilingInfo& sfaInfo) {
  if (context_ == nullptr) {
    OP_LOGE("XllmKernelSparseFlashAttentionLse", "tiling context is nullptr!");
    return ge::GRAPH_FAILED;
  }

  if (ge::GRAPH_SUCCESS != GetOpName() || ge::GRAPH_SUCCESS != GetNpuInfo() ||
      ge::GRAPH_SUCCESS != GetOpParaInfo() ||
      ge::GRAPH_SUCCESS != CheckRequiredParaExistence()) {
    return ge::GRAPH_FAILED;
  }

  if (ge::GRAPH_SUCCESS != GetInOutDataType() ||
      ge::GRAPH_SUCCESS != GetQueryAndOutLayout() ||
      ge::GRAPH_SUCCESS != GetTopkLayout() ||
      ge::GRAPH_SUCCESS != GetSoftmaxMaxAndSumLayout() ||
      ge::GRAPH_SUCCESS != GetKvLayout() ||
      ge::GRAPH_SUCCESS != GetKvStorageMode()) {
    return ge::GRAPH_FAILED;
  }

  SetSFAShape();
  if (ge::GRAPH_SUCCESS != GetN1Size() || ge::GRAPH_SUCCESS != GetN2Size() ||
      ge::GRAPH_SUCCESS != GetGSize() || ge::GRAPH_SUCCESS != GetBatchSize() ||
      ge::GRAPH_SUCCESS != GetQTSize() || ge::GRAPH_SUCCESS != GetKVTSize() ||
      ge::GRAPH_SUCCESS != GetS1Size() || ge::GRAPH_SUCCESS != GetQkHeadDim() ||
      ge::GRAPH_SUCCESS != GetS2Size() ||
      ge::GRAPH_SUCCESS != GetValueHeadDim() ||
      ge::GRAPH_SUCCESS != GetRopeHeadDim() ||
      ge::GRAPH_SUCCESS != GetSparseBlockCount()) {
    return ge::GRAPH_FAILED;
  }

  if (ge::GRAPH_SUCCESS != GetActualseqInfo()) {
    return ge::GRAPH_FAILED;
  }

  GenerateInfo(sfaInfo);
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(XllmKernelSparseFlashAttentionLse)
    .Tiling(TilingSparseFlashAttentionLse)
    .TilingParse<SparseFlashAttentionLseCompileInfo>(
        TilingPrepareForSparseFlashAttentionLse);
}  // namespace optiling
