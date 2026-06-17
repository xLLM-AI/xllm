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

#include "mm_type.h"

namespace xllm {

std::optional<std::string> MMType::to_string() const {
  switch (value_) {
    case Value::NONE:
      return std::nullopt;
    case Value::IMAGE:
      return "image";
    case Value::VIDEO:
      return "video";
    case Value::AUDIO:
      return "audio";
    case Value::EMBEDDING:
      return "embedding";
    default:
      LOG(WARNING) << "Unknown mm type: " << static_cast<int>(value_);
  }
  return std::nullopt;
}

std::string get_embedding_key(MMType type) {
  if (type == MMType::IMAGE) {
    return "image|embedding";
  }
  if (type == MMType::VIDEO) {
    return "video|embedding";
  }
  if (type == MMType::AUDIO) {
    return "audio|embedding";
  }
  LOG(FATAL) << "unsupported modality type: " << type;
  return "";
}

}  // namespace xllm
