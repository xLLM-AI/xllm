/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

#pragma once

#include <vector>

#include "base_loader.h"

namespace xllm {
namespace layer {

enum Mistral3VisionTensorId : int32_t {
  kInputNormWeight = 0,
  kInputNormBias,
  kInputNormNewWeight,
  kInputNormNewBias,
  kQWeight,
  kKWeight,
  kVWeight,
  kQkvBias,
  kQkvDeqScale,
  kQkvOffset,
  kQkvScale,
  kQkvCompressIdx,
  kAttentionOutWeight,
  kAttentionOutBias,
  kAttentionOutDeqScale,
  kAttentionOutOffset,
  kAttentionOutScale,
  kAttentionOutCompressIdx,
  kPostNormWeight,
  kPostNormBias,
  kPostNormNewWeight,
  kPostNormNewBias,
  kMlpGateWeight,
  kMlpGateBias,
  kMlpGateDeqScale,
  kMlpGateOffset,
  kMlpGateScale,
  kMlpGateCompressIdx,
  kMlpUpWeight,
  kMlpUpBias,
  kMlpDownWeight,
  kMlpDownBias,
  kMlpDownDeqScale,
  kMlpDownOffset,
  kMlpDownScale,
  kMlpDownCompressIdx,
};

class Mistral3VisionEncoderLoader : public BaseLoader {
 public:
  Mistral3VisionEncoderLoader(uint64_t weight_count,
                              const ModelContext& context,
                              LoadMode mode = LoadMode::kEager);

  void load_state_dict(const StateDict& state_dict) override;
  void verify_loaded_weights() const override;

 protected:
  void merge_host_at_weights() override;
};

}  // namespace layer
}  // namespace xllm
