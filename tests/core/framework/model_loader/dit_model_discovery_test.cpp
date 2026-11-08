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

#include "core/framework/model_loader/dit_model_discovery.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <system_error>

namespace xllm::util {
namespace {

class DitModelDiscoveryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    std::string directory_template =
        (std::filesystem::temp_directory_path() / "xllm_dit_discovery_XXXXXX")
            .string();
    ASSERT_NE(mkdtemp(directory_template.data()), nullptr);
    model_path_ = directory_template;
  }

  void TearDown() override {
    std::error_code error;
    std::filesystem::remove_all(model_path_, error);
    EXPECT_FALSE(error) << error.message();
  }

  std::filesystem::path create_component(std::string_view name,
                                         std::string_view config) {
    const std::filesystem::path component_path = model_path_ / name;
    std::filesystem::create_directories(component_path);
    std::ofstream(component_path / "config.json") << config;
    std::ofstream(component_path / "model.safetensors");
    return component_path;
  }

  std::filesystem::path model_path_;
};

TEST_F(DitModelDiscoveryTest, DiscoversDirectComponentWithConfiguredModelType) {
  const std::filesystem::path component_path = create_component(
      /*name=*/"transformer",
      /*config=*/R"({"model_type":"cola_transformer"})");

  auto components = discover_dit_components(model_path_);

  ASSERT_TRUE(components.has_value());
  ASSERT_EQ(components->size(), 1);
  EXPECT_EQ(components->front().name, "transformer");
  EXPECT_EQ(components->front().path, component_path);
  EXPECT_EQ(components->front().component_type, "cola_transformer");
}

TEST_F(DitModelDiscoveryTest, FallsBackToComponentDirectoryName) {
  create_component(/*name=*/"vae", /*config=*/"{}");

  auto components = discover_dit_components(model_path_);

  ASSERT_TRUE(components.has_value());
  ASSERT_EQ(components->size(), 1);
  EXPECT_EQ(components->front().name, "vae");
  EXPECT_EQ(components->front().component_type, "vae");
}

TEST_F(DitModelDiscoveryTest, DiscoversComponentsOneDirectoryBelowRepository) {
  const std::filesystem::path component_path = create_component(
      /*name=*/"pipeline/transformer",
      /*config=*/R"({"model_type":"cola_transformer"})");

  auto components = discover_dit_components(model_path_);

  ASSERT_TRUE(components.has_value());
  ASSERT_EQ(components->size(), 1);
  EXPECT_EQ(components->front().name, "transformer");
  EXPECT_EQ(components->front().path, component_path);
}

TEST_F(DitModelDiscoveryTest, PrefersDirectComponentsOverNestedComponents) {
  create_component(/*name=*/"transformer", /*config=*/"{}");
  create_component(/*name=*/"pipeline/vae", /*config=*/"{}");

  auto components = discover_dit_components(model_path_);

  ASSERT_TRUE(components.has_value());
  ASSERT_EQ(components->size(), 1);
  EXPECT_EQ(components->front().name, "transformer");
}

TEST_F(DitModelDiscoveryTest,
       IgnoresHiddenComponentsAndHiddenParentDirectories) {
  create_component(/*name=*/".hidden_transformer", /*config=*/"{}");
  create_component(/*name=*/".hidden_pipeline/vae", /*config=*/"{}");
  create_component(/*name=*/"pipeline/.hidden_vae", /*config=*/"{}");

  EXPECT_FALSE(discover_dit_components(model_path_).has_value());
}

TEST_F(DitModelDiscoveryTest, RequiresConfigurationAndSafetensorsWeights) {
  const std::filesystem::path no_config =
      create_component(/*name=*/"no_config", /*config=*/"{}");
  std::filesystem::remove(no_config / "config.json");
  const std::filesystem::path no_weights =
      create_component(/*name=*/"no_weights", /*config=*/"{}");
  std::filesystem::rename(no_weights / "model.safetensors",
                          no_weights / "model.bin");

  EXPECT_FALSE(discover_dit_components(model_path_).has_value());
}

TEST_F(DitModelDiscoveryTest, RejectsMissingPathAndNonDirectoryPath) {
  EXPECT_FALSE(discover_dit_components(model_path_ / "missing").has_value());
  const std::filesystem::path regular_file = model_path_ / "file";
  std::ofstream(regular_file) << "not a model directory";

  EXPECT_FALSE(discover_dit_components(regular_file).has_value());
}

}  // namespace
}  // namespace xllm::util
