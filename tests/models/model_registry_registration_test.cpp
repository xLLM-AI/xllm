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

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

#include "core/common/extension_registry.h"
#include "models/model_registry.h"

namespace xllm {
namespace {

template <int32_t Marker>
class RegistryTestModel final : public CausalLM, public DecoderModel {
 public:
  DecoderModel& model() override { return *this; }

  explicit RegistryTestModel(const ModelContext& /*context*/) {}

  ModelOutput forward(const torch::Tensor& /*tokens*/,
                      const torch::Tensor& /*positions*/,
                      std::vector<KVCache>& /*kv_caches*/,
                      const ModelParamsView& /*parameters*/) override {
    return {};
  }

  torch::Tensor logits(const torch::Tensor& /*hidden_states*/,
                       const torch::Tensor& /*selected_idxes*/) override {
    return {};
  }

  void load_model(std::unique_ptr<ModelLoader> /*loader*/) override {}

  torch::Device device() const override { return torch::Device(torch::kCPU); }

  const torch::TensorOptions& options() const override { return options_; }

 private:
  torch::TensorOptions options_ = torch::TensorOptions().device(torch::kCPU);
};

template <int32_t Marker>
ModelFactory make_factory() {
  return [](const ModelContext& context) {
    return std::make_unique<RegistryTestModel<Marker>>(context);
  };
}

ModelContext make_context(const std::string& model_name,
                          const std::string& model_impl,
                          const std::string& extension_impl = "") {
  ModelArgs args;
  args.model_type(model_name);
  const torch::TensorOptions options =
      torch::TensorOptions().device(torch::kCPU);
  ModelContext context(ParallelArgs(/*rank=*/0, /*world_size=*/1, nullptr),
                       args,
                       QuantArgs{},
                       options);
  context.set_model_impl(model_impl);
  context.set_extension_impl(extension_impl);
  return context;
}

TEST(ModelRegistryRegistrationTest, HigherPriorityReplacesFactory) {
  const std::string model_name = "registry_priority_limits_test";

  EXPECT_TRUE(ModelRegistry::register_model_factory(model_name,
                                                    /*implementation=*/"native",
                                                    make_factory<1>(),
                                                    /*priority=*/1));
  EXPECT_TRUE(ModelRegistry::register_model_factory(model_name,
                                                    /*implementation=*/"native",
                                                    make_factory<2>(),
                                                    /*priority=*/5));
  EXPECT_FALSE(
      ModelRegistry::register_model_factory(model_name,
                                            /*implementation=*/"native",
                                            make_factory<3>(),
                                            /*priority=*/4));
  EXPECT_FALSE(
      ModelRegistry::register_model_factory(model_name,
                                            /*implementation=*/"native",
                                            make_factory<4>(),
                                            /*priority=*/5));

  const ModelFactory selected_factory =
      ModelRegistry::get_model_factory(model_name, /*implementation=*/"native");
  ASSERT_TRUE(selected_factory);
  std::unique_ptr<CausalLM> model =
      std::get<std::unique_ptr<CausalLM>>(selected_factory(ModelContext{}));
  EXPECT_NE(dynamic_cast<RegistryTestModel<2>*>(model.get()), nullptr);
}

TEST(ModelRegistryRegistrationTest, NativeAndPythonAliasesShareRegistrations) {
  const std::string model_name = "registry_alias_test";
  EXPECT_TRUE(ModelRegistry::register_model_factory(
      model_name, "", make_factory<1>(), /*priority=*/-2));
  EXPECT_FALSE(ModelRegistry::register_model_factory(
      model_name, "cpp", make_factory<2>(), /*priority=*/-3));
  EXPECT_TRUE(ModelRegistry::register_model_factory(
      model_name, "py", make_factory<3>(), /*priority=*/-5));

  const ModelFactory native_factory =
      ModelRegistry::get_model_factory(model_name, /*implementation=*/"native");
  const ModelFactory python_factory =
      ModelRegistry::get_model_factory(model_name, /*implementation=*/"python");
  ASSERT_TRUE(native_factory);
  ASSERT_TRUE(python_factory);
  std::unique_ptr<CausalLM> native_model =
      std::get<std::unique_ptr<CausalLM>>(native_factory(ModelContext{}));
  std::unique_ptr<CausalLM> python_model =
      std::get<std::unique_ptr<CausalLM>>(python_factory(ModelContext{}));
  EXPECT_NE(dynamic_cast<RegistryTestModel<1>*>(native_model.get()), nullptr);
  EXPECT_NE(dynamic_cast<RegistryTestModel<3>*>(python_model.get()), nullptr);
}

TEST(ModelRegistryRegistrationTest,
     ExplicitExtensionImplementationPrecedesBackendResolution) {
  const std::string model_name = "registry_extension_impl_test";
  EXPECT_TRUE(ModelRegistry::register_model_factory(
      model_name, "custom", make_factory<8>(), /*priority=*/100));
  std::unique_ptr<CausalLM> model =
      std::get<std::unique_ptr<CausalLM>>(create_causal_model(make_context(
          model_name, /*model_impl=*/"native", /*extension_impl=*/"custom")));
  EXPECT_NE(dynamic_cast<RegistryTestModel<8>*>(model.get()), nullptr);
}

TEST(ModelRegistryRegistrationTest,
     LlmCreationPrefersRegisteredPythonOverride) {
  const std::string model_name = "registry_create_python_test";
  EXPECT_TRUE(ModelRegistry::register_model_factory(
      model_name, "native", make_factory<1>()));
  EXPECT_TRUE(ModelRegistry::register_model_factory(
      model_name, "python", make_factory<2>()));
  std::unique_ptr<CausalLM> model = std::get<std::unique_ptr<CausalLM>>(
      create_causal_model(make_context(model_name, /*model_impl=*/"py")));
  EXPECT_NE(dynamic_cast<RegistryTestModel<2>*>(model.get()), nullptr);
}

TEST(ModelRegistryRegistrationTest, LlmCreationRejectsUnregisteredCustomImpl) {
  const std::string model_name = "registry_create_missing_custom_test";
  EXPECT_TRUE(ModelRegistry::register_model_factory(
      model_name, "native", make_factory<1>()));
  EXPECT_EQ(std::get<std::unique_ptr<CausalLM>>(create_causal_model(
                make_context(model_name,
                             /*model_impl=*/"native",
                             /*extension_impl=*/"missing_custom"))),
            nullptr);
}

TEST(ModelRegistryRegistrationTest,
     ExtensionImplPreservesModelBackendSelection) {
  const std::string model_name = "registry_extension_backend_test";
  EXPECT_TRUE(ModelRegistry::register_model_factory(
      model_name, "custom", make_factory<8>(), /*priority=*/100));
  EXPECT_EQ(
      std::get<std::unique_ptr<CausalLM>>(create_causal_model(make_context(
          model_name, /*model_impl=*/"python", /*extension_impl=*/"custom"))),
      nullptr);
  EXPECT_EQ(std::get<std::unique_ptr<CausalLM>>(create_causal_model(
                make_context(model_name, /*model_impl=*/"custom"))),
            nullptr);
}

TEST(ModelRegistryRegistrationTest, ExecutorOnlyFamilyPreservesNativeModels) {
  const std::string family = "registry_executor_only_family";
  ExtensionRegistry::get_instance().register_implementation(family, "executor");
  const std::string model_name = "registry_executor_only_model_test";
  EXPECT_TRUE(ModelRegistry::register_model_factory(
      model_name, "native", make_factory<1>()));
  std::unique_ptr<CausalLM> model =
      std::get<std::unique_ptr<CausalLM>>(create_causal_model(
          make_context(model_name, /*model_impl=*/"native", family)));
  EXPECT_NE(dynamic_cast<RegistryTestModel<1>*>(model.get()), nullptr);
}

TEST(ModelRegistryRegistrationTest, MissingModelInSelectedFamilyRejectsNative) {
  const std::string model_name = "registry_family_missing_lm_test";
  EXPECT_TRUE(ModelRegistry::register_model_factory(
      model_name, "native", make_factory<1>()));
  EXPECT_TRUE(ModelRegistry::register_model_factory(
      "registry_family_other_lm", "registry_lm_family", make_factory<2>()));
  EXPECT_EQ(std::get<std::unique_ptr<CausalLM>>(create_causal_model(
                make_context(model_name,
                             /*model_impl=*/"native",
                             /*extension_impl=*/"registry_lm_family"))),
            nullptr);
}

}  // namespace
}  // namespace xllm
