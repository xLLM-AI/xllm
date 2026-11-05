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

#pragma once

#include <butil/iobuf.h>

#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "core/common/types.h"

namespace xllm::api_service {

struct MultipartPart {
  std::string name;
  std::string filename;
  std::string content_type;

  butil::IOBuf value;
};

struct MultipartFormData {
  std::vector<MultipartPart> parts;
};

std::optional<std::string> find_multipart_field_value(
    const MultipartFormData& form,
    std::string_view name);

Status parse_multipart_form_data(std::string_view content_type_header,
                                 const butil::IOBuf& body,
                                 size_t max_part_bytes,
                                 MultipartFormData& out);

}  // namespace xllm::api_service
