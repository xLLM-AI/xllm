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
class RegistryTestModel final : public CausalVLM {
 public:
  explicit RegistryTestModel(const ModelContext& /*context*/) {
    register_module("dropout", torch::nn::Dropout(/*p=*/0.5));
  }

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

  void prepare_expert_weight(
      int32_t /*layer_id*/,
      const std::vector<int32_t>& /*expert_ids*/) override {}

  void update_expert_weight(int32_t /*layer_id*/) override {}

  MMDict encode(const ModelParamsView& /*parameters*/) override { return {}; }

  torch::Tensor get_input_embeddings(
      const torch::Tensor& /*input_ids*/,
      const ModelParamsView& /*input_params*/) override {
    return {};
  }

 private:
  torch::TensorOptions options_ = torch::TensorOptions().device(torch::kCPU);
};

using MacroRegisteredModel = RegistryTestModel<7>;
using MacroRegisteredVlm = RegistryTestModel<9>;

template <int32_t Marker>
CausalLMFactory make_factory() {
  return [](const ModelContext& context) {
    return std::make_unique<RegistryTestModel<Marker>>(context);
  };
}

template <int32_t Marker>
CausalVLMFactory make_vlm_factory() {
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
#if defined(USE_NPU)
  ModelContext context(ParallelArgs(/*rank=*/0, /*world_size=*/1, nullptr),
                       args,
                       QuantArgs{},
                       options,
                       /*context=*/nullptr);
#else
  ModelContext context(ParallelArgs(/*rank=*/0, /*world_size=*/1, nullptr),
                       args,
                       QuantArgs{},
                       options);
#endif
  context.set_model_impl(model_impl);
  context.set_extension_impl(extension_impl);
  return context;
}

REGISTER_CAUSAL_LM(registry_macro_model,
                   "native",
                   MacroRegisteredModel,
                   /*priority=*/7);

REGISTER_CAUSAL_VLM(registry_macro_vlm,
                    "custom",
                    MacroRegisteredVlm,
                    /*priority=*/9);
REGISTER_CAUSAL_LM(registry_macro_lm_alias,
                   "native",
                   xllm::MacroRegisteredModel);
REGISTER_CAUSAL_VLM(registry_macro_vlm_alias,
                    "custom",
                    xllm::MacroRegisteredVlm);

TEST(ModelRegistryRegistrationTest, HigherPriorityReplacesFactory) {
  const std::string model_name = "registry_priority_limits_test";

  EXPECT_TRUE(ModelRegistry::register_causallm_factory(model_name,
                                                       "native",
                                                       make_factory<1>(),
                                                       /*priority=*/1));
  EXPECT_TRUE(ModelRegistry::register_causallm_factory(model_name,
                                                       "native",
                                                       make_factory<2>(),
                                                       /*priority=*/5));
  EXPECT_FALSE(ModelRegistry::register_causallm_factory(model_name,
                                                        "native",
                                                        make_factory<3>(),
                                                        /*priority=*/4));
  EXPECT_FALSE(ModelRegistry::register_causallm_factory(model_name,
                                                        "native",
                                                        make_factory<4>(),
                                                        /*priority=*/5));

  const CausalLMFactory selected_factory =
      ModelRegistry::get_causallm_factory(model_name, "native");
  ASSERT_TRUE(selected_factory);
  std::unique_ptr<CausalLM> model = selected_factory(ModelContext{});
  EXPECT_NE(dynamic_cast<RegistryTestModel<2>*>(model.get()), nullptr);
}

TEST(ModelRegistryRegistrationTest, DirectSubclassMacroRegistersFactory) {
  const CausalLMFactory factory =
      ModelRegistry::get_causallm_factory("registry_macro_model");
  ASSERT_TRUE(factory);
  std::unique_ptr<CausalLM> model = factory(ModelContext{});
  EXPECT_NE(dynamic_cast<MacroRegisteredModel*>(model.get()), nullptr);
  EXPECT_FALSE(model->is_training());
  ASSERT_EQ(model->children().size(), 1);
  EXPECT_FALSE(model->children().front()->is_training());

  const CausalLMFactory alias_factory =
      ModelRegistry::get_causallm_factory("registry_macro_lm_alias");
  ASSERT_TRUE(alias_factory);
  std::unique_ptr<CausalLM> alias_model = alias_factory(ModelContext{});
  EXPECT_NE(dynamic_cast<MacroRegisteredModel*>(alias_model.get()), nullptr);
}

TEST(ModelRegistryRegistrationTest, VlmSubclassMacroRegistersFactory) {
  const CausalVLMFactory factory =
      ModelRegistry::get_causalvlm_factory("registry_macro_vlm", "custom");
  ASSERT_TRUE(factory);
  std::unique_ptr<CausalVLM> model = factory(ModelContext{});
  EXPECT_NE(dynamic_cast<MacroRegisteredVlm*>(model.get()), nullptr);
  EXPECT_FALSE(model->is_training());
  ASSERT_EQ(model->children().size(), 1);
  EXPECT_FALSE(model->children().front()->is_training());

  const CausalVLMFactory alias_factory = ModelRegistry::get_causalvlm_factory(
      "registry_macro_vlm_alias", "custom");
  ASSERT_TRUE(alias_factory);
  std::unique_ptr<CausalVLM> alias_model = alias_factory(ModelContext{});
  EXPECT_NE(dynamic_cast<MacroRegisteredVlm*>(alias_model.get()), nullptr);
}

TEST(ModelRegistryRegistrationTest, NativeAndPythonAliasesShareRegistrations) {
  const std::string model_name = "registry_alias_test";
  EXPECT_TRUE(ModelRegistry::register_causallm_factory(
      model_name, "", make_factory<1>(), /*priority=*/-2));
  EXPECT_FALSE(ModelRegistry::register_causallm_factory(
      model_name, "cpp", make_factory<2>(), /*priority=*/-3));
  EXPECT_TRUE(ModelRegistry::register_causallm_factory(
      model_name, "py", make_factory<3>(), /*priority=*/-5));

  const CausalLMFactory native_factory =
      ModelRegistry::get_causallm_factory(model_name, "native");
  const CausalLMFactory python_factory =
      ModelRegistry::get_causallm_factory(model_name, "python");
  ASSERT_TRUE(native_factory);
  ASSERT_TRUE(python_factory);
  std::unique_ptr<CausalLM> native_model = native_factory(ModelContext{});
  std::unique_ptr<CausalLM> python_model = python_factory(ModelContext{});
  EXPECT_NE(dynamic_cast<RegistryTestModel<1>*>(native_model.get()), nullptr);
  EXPECT_NE(dynamic_cast<RegistryTestModel<3>*>(python_model.get()), nullptr);
}

TEST(ModelRegistryRegistrationTest,
     ExplicitExtensionImplementationPrecedesBackendResolution) {
  const std::string model_name = "registry_extension_impl_test";
  EXPECT_TRUE(ModelRegistry::register_causallm_factory(
      model_name, "custom", make_factory<8>(), /*priority=*/100));
  std::unique_ptr<CausalLM> model =
      create_llm_model(make_context(model_name, "native", "custom"));
  EXPECT_NE(dynamic_cast<RegistryTestModel<8>*>(model.get()), nullptr);
}

TEST(ModelRegistryRegistrationTest, AutoModelImplUsesNativeRegistration) {
  const std::string lm_name = "registry_auto_lm_test";
  EXPECT_TRUE(ModelRegistry::register_causallm_factory(lm_name,
                                                       "native",
                                                       make_factory<1>(),
                                                       /*priority=*/3));

  std::unique_ptr<CausalLM> lm =
      create_llm_model(make_context(lm_name, "auto"));
  EXPECT_NE(dynamic_cast<RegistryTestModel<1>*>(lm.get()), nullptr);

  const std::string vlm_name = "registry_auto_vlm_test";
  EXPECT_TRUE(ModelRegistry::register_causalvlm_factory(vlm_name,
                                                        "native",
                                                        make_vlm_factory<2>(),
                                                        /*priority=*/4));

  std::unique_ptr<CausalVLM> vlm =
      create_vlm_model(make_context(vlm_name, "auto"));
  EXPECT_NE(dynamic_cast<RegistryTestModel<2>*>(vlm.get()), nullptr);
}

TEST(ModelRegistryRegistrationTest, VlmPrioritySelectsFactory) {
  const std::string model_name = "registry_vlm_priority_test";
  EXPECT_TRUE(ModelRegistry::register_causalvlm_factory(model_name,
                                                        "custom",
                                                        make_vlm_factory<1>(),
                                                        /*priority=*/1));
  EXPECT_TRUE(ModelRegistry::register_causalvlm_factory(model_name,
                                                        "custom",
                                                        make_vlm_factory<2>(),
                                                        /*priority=*/2));
  EXPECT_FALSE(ModelRegistry::register_causalvlm_factory(
      model_name, "custom", make_vlm_factory<3>(), /*priority=*/2));
  const CausalVLMFactory factory =
      ModelRegistry::get_causalvlm_factory(model_name, "custom");
  ASSERT_TRUE(factory);
  std::unique_ptr<CausalVLM> model = factory(ModelContext{});
  EXPECT_NE(dynamic_cast<RegistryTestModel<2>*>(model.get()), nullptr);
}

TEST(ModelRegistryRegistrationTest,
     LlmCreationPrefersRegisteredPythonOverride) {
  const std::string model_name = "registry_create_python_test";
  EXPECT_TRUE(ModelRegistry::register_causallm_factory(
      model_name, "native", make_factory<1>()));
  EXPECT_TRUE(ModelRegistry::register_causallm_factory(
      model_name, "python", make_factory<2>()));
  std::unique_ptr<CausalLM> model =
      create_llm_model(make_context(model_name, "py"));
  EXPECT_NE(dynamic_cast<RegistryTestModel<2>*>(model.get()), nullptr);
}

TEST(ModelRegistryRegistrationTest, LlmCreationRejectsUnregisteredCustomImpl) {
  const std::string model_name = "registry_create_missing_custom_test";
  EXPECT_TRUE(ModelRegistry::register_causallm_factory(
      model_name, "native", make_factory<1>()));
  EXPECT_EQ(
      create_llm_model(make_context(model_name, "native", "missing_custom")),
      nullptr);
}

TEST(ModelRegistryRegistrationTest,
     VlmCreationPrefersRegisteredPythonOverride) {
  const std::string model_name = "registry_create_vlm_python_test";
  EXPECT_TRUE(ModelRegistry::register_causalvlm_factory(
      model_name, "native", make_vlm_factory<1>()));
  EXPECT_TRUE(ModelRegistry::register_causalvlm_factory(
      model_name, "python", make_vlm_factory<2>()));
  std::unique_ptr<CausalVLM> model =
      create_vlm_model(make_context(model_name, "py"));
  EXPECT_NE(dynamic_cast<RegistryTestModel<2>*>(model.get()), nullptr);
}

TEST(ModelRegistryRegistrationTest, VlmCreationRejectsUnregisteredCustomImpl) {
  const std::string model_name = "registry_create_vlm_missing_custom_test";
  EXPECT_TRUE(ModelRegistry::register_causalvlm_factory(
      model_name, "native", make_vlm_factory<1>()));
  EXPECT_EQ(
      create_vlm_model(make_context(model_name, "native", "missing_custom")),
      nullptr);
}

TEST(ModelRegistryRegistrationTest,
     ExtensionImplPreservesModelBackendSelection) {
  const std::string model_name = "registry_extension_backend_test";
  EXPECT_TRUE(ModelRegistry::register_causallm_factory(
      model_name, "custom", make_factory<8>(), /*priority=*/100));
  EXPECT_EQ(create_llm_model(make_context(model_name, "python", "custom")),
            nullptr);
  EXPECT_EQ(create_llm_model(make_context(model_name, "custom")), nullptr);
}

TEST(ModelRegistryRegistrationTest, ExecutorOnlyFamilyPreservesNativeModels) {
  const std::string family = "registry_executor_only_family";
  ExtensionRegistry::get_instance().register_implementation(family, "executor");
  const std::string model_name = "registry_executor_only_model_test";
  EXPECT_TRUE(ModelRegistry::register_causallm_factory(
      model_name, "native", make_factory<1>()));
  std::unique_ptr<CausalLM> model =
      create_llm_model(make_context(model_name, "native", family));
  EXPECT_NE(dynamic_cast<RegistryTestModel<1>*>(model.get()), nullptr);
}

TEST(ModelRegistryRegistrationTest, MissingModelInSelectedFamilyRejectsNative) {
  const std::string model_name = "registry_family_missing_lm_test";
  EXPECT_TRUE(ModelRegistry::register_causallm_factory(
      model_name, "native", make_factory<1>()));
  EXPECT_TRUE(ModelRegistry::register_causallm_factory(
      "registry_family_other_lm", "registry_lm_family", make_factory<2>()));
  EXPECT_EQ(create_llm_model(
                make_context(model_name, "native", "registry_lm_family")),
            nullptr);
}

TEST(ModelRegistryRegistrationTest, VlmCreationUsesExtensionFamily) {
  const std::string model_name = "registry_create_custom_vlm_test";
  EXPECT_TRUE(ModelRegistry::register_causalvlm_factory(
      model_name, "custom", make_vlm_factory<9>()));
  std::unique_ptr<CausalVLM> model =
      create_vlm_model(make_context(model_name, "native", "custom"));
  EXPECT_NE(dynamic_cast<RegistryTestModel<9>*>(model.get()), nullptr);
}

TEST(ModelRegistryRegistrationTest, MissingVlmInSelectedFamilyRejectsNative) {
  const std::string model_name = "registry_family_missing_vlm_test";
  EXPECT_TRUE(ModelRegistry::register_causalvlm_factory(
      model_name, "native", make_vlm_factory<1>()));
  EXPECT_TRUE(
      ModelRegistry::register_causalvlm_factory("registry_family_other_vlm",
                                                "registry_vlm_family",
                                                make_vlm_factory<2>()));
  EXPECT_EQ(create_vlm_model(
                make_context(model_name, "native", "registry_vlm_family")),
            nullptr);
}

TEST(ModelRegistryRegistrationTest, DerivedContextsPreserveExtensionSelection) {
  const ModelContext context =
      make_context("registry_context_test", "native", "custom");
  const ModelContext parallel_context = context.with_parallel_args(
      ParallelArgs(/*rank=*/0, /*world_size=*/1, nullptr));
  const ModelContext quant_context = context.with_quant_args(QuantArgs{});
  EXPECT_EQ(parallel_context.get_extension_impl(), "custom");
  EXPECT_EQ(quant_context.get_extension_impl(), "custom");
}

}  // namespace
}  // namespace xllm
