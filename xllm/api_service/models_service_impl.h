/* Copyright 2025-2026 The xLLM Authors.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include "core/common/macros.h"
#include "models.pb.h"

namespace xllm {

class ModelsServiceImpl final {
 public:
  ModelsServiceImpl(const std::vector<std::string>& model_names,
                    const std::vector<std::string>& model_repository_names,
                    const std::vector<std::string>& model_versions);

  bool list_models(const proto::ModelListRequest* request,
                   proto::ModelListResponse* response);
  std::string list_model_versions();

 private:
  DISALLOW_COPY_AND_ASSIGN(ModelsServiceImpl);

  std::vector<std::string> model_names_;
  std::vector<std::string> model_repository_names_;
  std::vector<std::string> model_versions_;
  uint32_t created_;
};

}  // namespace xllm
