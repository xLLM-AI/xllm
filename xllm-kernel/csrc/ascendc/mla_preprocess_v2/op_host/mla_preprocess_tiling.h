/* Copyright 2025 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://gitcode.com/xLLM-AI/xllm_ops/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef OPTILING_PARAMS_PREPROCESS_TILING
#define OPTILING_PARAMS_PREPROCESS_TILING

#include <cstdint>
#include <sstream>
#include <string>

#include "mla_preprocess_tilingdata.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
namespace OpParam {
struct MlaPreprocessParam {
  enum class QuantMode : uint64_t {
    PER_TENSOR_ASYMM_QUANT = 0,
    PER_TOKEN_SYMM_QUANT,
    PER_TOKEN_ASYMM_QUANT,
    NO_QUANT,
  };
  uint64_t N = 128;
  uint64_t headNum = 0;
  uint64_t cacheMode = 0;
  QuantMode quantMode = QuantMode::PER_TENSOR_ASYMM_QUANT;
  bool operator==(const MlaPreprocessParam& other) const {
    return N == other.N && headNum == other.headNum &&
           cacheMode == other.cacheMode && quantMode == other.quantMode;
  }
};
}  // namespace OpParam

struct MlaPreProcessCompileInfo {};

class MlaPreprocessTiling {
 public:
  optiling::MlaTilingData mlaTilingData;

  ge::graphStatus Init(gert::TilingContext* context);

  void RmsNormQuantTiling(const uint64_t numTokens,
                          const uint64_t numVectorCore,
                          const uint64_t hiddtenState);
  void RopeConcatTiling(const OpParam::MlaPreprocessParam& param,
                        const uint64_t& aicNum);
  void EinSumQuantTiling(const OpParam::MlaPreprocessParam& param,
                         const uint64_t& aicNum,
                         const ge::DataType inDtype,
                         const bool doRmsQuant);
  void SetTilingKey(const ge::DataType inDtype,
                    const OpParam::MlaPreprocessParam& param,
                    const bool doRmsQuant,
                    gert::TilingContext* context);
  void SetMlapoWorkSpace(const ge::DataType inDtype,
                         const OpParam::MlaPreprocessParam& param,
                         uint32_t sysWorkSpaceSize,
                         gert::TilingContext* context);
  void PrintTilingData(gert::TilingContext* context);
  void PrintFirstTilingData(gert::TilingContext* context);
  void PrintLastTilingData(gert::TilingContext* context);
  OpParam::MlaPreprocessParam GetParam(gert::TilingContext* context);
};

}  // namespace optiling

#endif  // OPTILING_PARAMS_MLA_PRE_H