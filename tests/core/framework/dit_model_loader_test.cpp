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

#include "core/framework/model_loader/dit_model_loader.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

#include "models/model_registry.h"

namespace xllm {
namespace {

class DiTModelLoaderTest : public ::testing::Test {
 protected:
  void SetUp() override {
    std::string path_template =
        (std::filesystem::temp_directory_path() / "xllm-dit-loader-XXXXXX")
            .string();
    const char* directory = mkdtemp(path_template.data());
    ASSERT_NE(directory, nullptr);
    root_ = directory;
  }

  void TearDown() override {
    if (!root_.empty()) {
      std::filesystem::remove_all(root_);
    }
  }

  void write_index(const nlohmann::json& index) const {
    std::ofstream stream(root_ / "model_index.json");
    ASSERT_TRUE(stream.is_open());
    stream << index.dump();
  }

  static void register_pipeline(const std::string& name) {
    ModelRegistry::register_dit_model_factory(
        name, [](const DiTModelContext& /*context*/) {
          return std::unique_ptr<DiTModel>();
        });
  }

  std::filesystem::path root_;
};

TEST_F(DiTModelLoaderTest, PrefersRegisteredPipelineWithoutRootWeights) {
  const std::string name = "dit_loader_registered_pipeline_test";
  register_pipeline(name);
  write_index({{"_class_name", name}});

  const auto loader = DiTModelLoader::create(
      root_.string(), /*fallback_model_type=*/"unknown_fallback");

  EXPECT_EQ(loader->get_model_type(), name);
  EXPECT_EQ(loader->get_registration_name(), name);
  EXPECT_TRUE(loader->get_model_args().empty());
}

TEST_F(DiTModelLoaderTest, UsesRegisteredModelIdForUnknownPipeline) {
  const std::string name = "dit_loader_fallback_pipeline_test";
  register_pipeline(name);
  write_index({{"_class_name", "unknown_checkpoint_pipeline"}});

  const auto loader = DiTModelLoader::create(root_.string(), name);

  EXPECT_EQ(loader->get_model_type(), "unknown_checkpoint_pipeline");
  EXPECT_EQ(loader->get_registration_name(), name);
}

TEST_F(DiTModelLoaderTest, PreservesParserOnlyComponentRegistrations) {
  const std::string pipeline = "dit_loader_component_pipeline_test";
  const std::string component = "dit_loader_parser_only_component_test";
  register_pipeline(pipeline);
  ModelRegistry::register_model_args_loader(
      component, [](const JsonReader& reader, ModelArgs* args) {
        args->shift() = reader.value<int64_t>("shift").value();
        return true;
      });
  write_index({{"_class_name", pipeline}, {"scheduler", {"test", component}}});
  std::filesystem::create_directory(root_ / "scheduler");
  {
    std::ofstream stream(root_ / "scheduler" / "scheduler_config.json");
    ASSERT_TRUE(stream.is_open());
    stream << R"({"shift": 2})";
  }

  auto loader = DiTModelLoader::create(root_.string());

  EXPECT_EQ(ModelRegistry::get_model_family(component), ModelFamily::UNKNOWN);
  EXPECT_EQ(loader->get_registration_name(), pipeline);
  ASSERT_TRUE(loader->has_component("scheduler"));
  EXPECT_EQ(loader->get_model_args().at("scheduler").shift(), 2);
  EXPECT_NE(loader->take_component_loader("scheduler"), nullptr);
  EXPECT_FALSE(loader->has_component("scheduler"));
}

TEST_F(DiTModelLoaderTest, RejectsFallbackWithoutDiTExecutionContract) {
  const std::string name = "dit_loader_non_dit_fallback_test";
  ModelRegistry::register_model_backend(name, "llm");
  write_index({{"_class_name", "unknown_checkpoint_pipeline"}});

  EXPECT_DEATH(DiTModelLoader::create(root_.string(), name),
               "does not support DiT pipelines");
}

}  // namespace
}  // namespace xllm
