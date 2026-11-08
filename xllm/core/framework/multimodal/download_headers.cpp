/* Copyright 2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "core/framework/multimodal/download_headers.h"

#include "core/framework/config/model_config.h"
#include "core/util/http_downloader.h"

namespace xllm {

const std::unordered_map<std::string, std::string>&
multimodal_download_headers() {
  static const std::unordered_map<std::string, std::string> headers =
      parse_headers_json(ModelConfig::get_instance().mm_download_headers());
  return headers;
}

}  // namespace xllm
