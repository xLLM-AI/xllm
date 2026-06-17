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

#include <string>
#include <vector>

#include "mm_handler.h"
#include "mm_type.h"

namespace xllm {

class MMEmbeddingHandler : public MMHandlerBase {
 public:
  MMEmbeddingHandler(MMType::Value mm_type);
  ~MMEmbeddingHandler() = default;

  virtual MMErrCode load(const MMContent& content,
                         MMInputItem& input,
                         MMPayload& payload) override;
  virtual MMErrCode decode(MMInputItem& input) override;

 private:
  MMType::Value mm_type_;
};

}  // namespace xllm