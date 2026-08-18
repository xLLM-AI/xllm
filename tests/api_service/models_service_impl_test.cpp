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

#include "xllm/api_service/models_service_impl.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace xllm {
namespace {

TEST(ModelsServiceImplTest, RepositoryIndexUsesRepositoryMetadata) {
  const std::vector<std::string> model_names = {"GLM-5.1", "Qwen3-8B"};
  const std::vector<std::string> model_repository_names = {"glm-51-w8a8-npu",
                                                           "qwen3"};
  const std::vector<std::string> model_versions = {"2", "3"};
  ModelsServiceImpl service(
      model_names, model_repository_names, model_versions);

  proto::ModelListRequest request;
  proto::ModelListResponse response;
  ASSERT_TRUE(service.list_models(&request, &response));
  ASSERT_EQ(response.data_size(), static_cast<int>(model_names.size()));

  const nlohmann::json repository_index =
      nlohmann::json::parse(service.list_model_versions());
  ASSERT_TRUE(repository_index.is_array());
  ASSERT_EQ(repository_index.size(), model_names.size());

  for (std::size_t i = 0; i < model_names.size(); ++i) {
    ASSERT_EQ(response.data(static_cast<int>(i)).id(), model_names[i]);
    ASSERT_TRUE(repository_index[i].is_object());
    ASSERT_TRUE(repository_index[i].contains("name"));
    ASSERT_TRUE(repository_index[i].contains("version"));
    ASSERT_TRUE(repository_index[i].contains("state"));
    ASSERT_TRUE(repository_index[i].contains("reason"));
    EXPECT_EQ(repository_index[i]["name"], model_repository_names[i]);
    EXPECT_EQ(repository_index[i]["version"], model_versions[i]);
    EXPECT_EQ(repository_index[i]["state"], "READY");
    EXPECT_EQ(repository_index[i]["reason"], "normal");
  }
}

}  // namespace
}  // namespace xllm
