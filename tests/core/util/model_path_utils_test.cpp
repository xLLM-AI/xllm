/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include <gtest/gtest.h>

#include <filesystem>

#include "core/util/utils.h"

namespace xllm::util {

TEST(ModelPathUtilsTest, ExtractsRepositoryNameFromVersionedPath) {
  const std::filesystem::path model_path =
      std::filesystem::path("/export/App/model_repository/glm-51-w8a8-npu/2/")
          .lexically_normal();

  EXPECT_EQ(get_model_name(model_path), "2");
  EXPECT_EQ(get_model_repository_name(model_path), "glm-51-w8a8-npu");
}

TEST(ModelPathUtilsTest, FallsBackToModelNameForNonVersionedPath) {
  const std::filesystem::path model_path = "/export/home/models/Qwen3-0.6B";

  EXPECT_EQ(get_model_repository_name(model_path), "Qwen3-0.6B");
}

}  // namespace xllm::util
