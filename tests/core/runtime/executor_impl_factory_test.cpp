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

#include "core/runtime/executor_impl_factory.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/common/extension_registry.h"

namespace xllm {
namespace {

template <int32_t Priority>
class FactoryTestExecutorImpl final : public ExecutorImpl {
 public:
  FactoryTestExecutorImpl() = default;

  FactoryTestExecutorImpl(CausalLM* /*model*/,
                          const TransformerModelArgs& /*args*/,
                          const torch::Device& /*device*/,
                          const runtime::Options& /*options*/) {}

  ModelOutput run(const torch::Tensor& /*tokens*/,
                  const torch::Tensor& /*positions*/,
                  std::vector<KVCache>& /*kv_caches*/,
                  const ModelParamsView& /*params*/) override {
    return ModelOutput{};
  }
};

using DefaultFactoryTestExecutorImpl = FactoryTestExecutorImpl<0>;
using PriorityFactoryTestExecutorImpl = FactoryTestExecutorImpl<10>;
using EqualPriorityFactoryTestExecutorImpl = FactoryTestExecutorImpl<5>;

template <int32_t Priority>
ExecutorImplFactory::Creator create_test_creator() {
  return [](CausalLM* /*model*/,
            const TransformerModelArgs& /*args*/,
            const torch::Device& /*device*/,
            const runtime::Options& /*options*/) {
    return std::make_unique<FactoryTestExecutorImpl<Priority>>();
  };
}

std::unique_ptr<ExecutorImpl> create_executor(
    const std::string& executor_type,
    const std::string& extension_impl = "") {
  runtime::Options options;
  options.extension_impl(extension_impl);
  return ExecutorImplFactory::get_instance().create_executor_impl(
      /*model=*/nullptr,
      /*args=*/TransformerModelArgs{},
      /*device=*/torch::Device(torch::kCPU),
      options,
      /*executor_type=*/executor_type);
}

}  // namespace

namespace executor_factory_test {

class NamespacedExecutorImpl final : public ExecutorImpl {
 public:
  NamespacedExecutorImpl(CausalLM* /*model*/,
                         const ModelArgs& /*args*/,
                         const torch::Device& /*device*/,
                         const runtime::Options& /*options*/) {}

  ModelOutput run(const torch::Tensor& /*tokens*/,
                  const torch::Tensor& /*positions*/,
                  std::vector<KVCache>& /*kv_caches*/,
                  const ModelParamsView& /*params*/) override {
    return ModelOutput{};
  }
};

}  // namespace executor_factory_test

REGISTER_EXECUTOR("executor_factory_macro_default",
                  DefaultFactoryTestExecutorImpl);
REGISTER_EXECUTOR("executor_factory_macro_priority",
                  PriorityFactoryTestExecutorImpl,
                  10);
REGISTER_EXECUTOR("executor_factory_macro_namespaced_first",
                  executor_factory_test::NamespacedExecutorImpl);
REGISTER_EXECUTOR("executor_factory_macro_namespaced_second",
                  executor_factory_test::NamespacedExecutorImpl,
                  10);
REGISTER_EXECUTOR("executor_factory_macro_extension",
                  "executor_factory_macro_custom",
                  executor_factory_test::NamespacedExecutorImpl,
                  10);

namespace {

TEST(ExecutorImplFactoryTest, HigherPriorityReplacesLowerPriority) {
  auto& factory = ExecutorImplFactory::get_instance();
  EXPECT_TRUE(factory.register_creator("executor_factory_lower_then_higher",
                                       create_test_creator<0>(),
                                       /*priority=*/0));
  EXPECT_TRUE(factory.register_creator("executor_factory_lower_then_higher",
                                       create_test_creator<10>(),
                                       /*priority=*/10));

  auto executor = create_executor("executor_factory_lower_then_higher");
  EXPECT_NE(dynamic_cast<PriorityFactoryTestExecutorImpl*>(executor.get()),
            nullptr);
}

TEST(ExecutorImplFactoryTest, LowerPriorityDoesNotReplaceHigherPriority) {
  auto& factory = ExecutorImplFactory::get_instance();
  EXPECT_TRUE(factory.register_creator("executor_factory_higher_then_lower",
                                       create_test_creator<10>(),
                                       /*priority=*/10));
  EXPECT_FALSE(factory.register_creator("executor_factory_higher_then_lower",
                                        create_test_creator<0>(),
                                        /*priority=*/0));

  auto executor = create_executor("executor_factory_higher_then_lower");
  EXPECT_NE(dynamic_cast<PriorityFactoryTestExecutorImpl*>(executor.get()),
            nullptr);
}

TEST(ExecutorImplFactoryTest, EqualPriorityKeepsFirstRegistration) {
  auto& factory = ExecutorImplFactory::get_instance();
  EXPECT_TRUE(factory.register_creator("executor_factory_equal_priority",
                                       create_test_creator<5>(),
                                       /*priority=*/5));
  EXPECT_FALSE(factory.register_creator("executor_factory_equal_priority",
                                        create_test_creator<10>(),
                                        /*priority=*/5));

  auto executor = create_executor("executor_factory_equal_priority");
  EXPECT_NE(dynamic_cast<EqualPriorityFactoryTestExecutorImpl*>(executor.get()),
            nullptr);
}

TEST(ExecutorImplFactoryTest,
     RegisterExecutorSupportsDefaultAndExplicitPriority) {
  auto default_executor = create_executor("executor_factory_macro_default");
  EXPECT_NE(
      dynamic_cast<DefaultFactoryTestExecutorImpl*>(default_executor.get()),
      nullptr);

  auto& factory = ExecutorImplFactory::get_instance();
  EXPECT_FALSE(factory.register_creator("executor_factory_macro_priority",
                                        create_test_creator<0>(),
                                        /*priority=*/0));
  auto priority_executor = create_executor("executor_factory_macro_priority");
  EXPECT_NE(
      dynamic_cast<PriorityFactoryTestExecutorImpl*>(priority_executor.get()),
      nullptr);
}

TEST(ExecutorImplFactoryTest,
     RegisterExecutorSupportsNamespacedTypesAndRepeatedClass) {
  auto first = create_executor("executor_factory_macro_namespaced_first");
  auto second = create_executor("executor_factory_macro_namespaced_second");

  EXPECT_NE(
      dynamic_cast<executor_factory_test::NamespacedExecutorImpl*>(first.get()),
      nullptr);
  EXPECT_NE(dynamic_cast<executor_factory_test::NamespacedExecutorImpl*>(
                second.get()),
            nullptr);
}

TEST(ExecutorImplFactoryTest, UnknownExecutorTypeThrows) {
  EXPECT_THROW(create_executor("executor_factory_unknown_executor_type"),
               std::runtime_error);
}

TEST(ExecutorImplFactoryTest, ExtensionCanProvideExecutorWithoutModel) {
  auto& factory = ExecutorImplFactory::get_instance();
  EXPECT_TRUE(factory.register_creator("executor_factory_executor_only",
                                       "executor_factory_executor_only_custom",
                                       create_test_creator<10>()));

  auto executor = create_executor("executor_factory_executor_only",
                                  "executor_factory_executor_only_custom");
  EXPECT_NE(dynamic_cast<PriorityFactoryTestExecutorImpl*>(executor.get()),
            nullptr);
}

TEST(ExecutorImplFactoryTest, SelectedExtensionHasIndependentPriority) {
  auto& factory = ExecutorImplFactory::get_instance();
  EXPECT_TRUE(factory.register_creator("executor_factory_extension_priority",
                                       create_test_creator<0>(),
                                       /*priority=*/100));
  EXPECT_TRUE(factory.register_creator("executor_factory_extension_priority",
                                       "executor_factory_priority_custom",
                                       create_test_creator<5>(),
                                       /*priority=*/5));
  EXPECT_TRUE(factory.register_creator("executor_factory_extension_priority",
                                       "executor_factory_priority_custom",
                                       create_test_creator<10>(),
                                       /*priority=*/10));
  EXPECT_FALSE(factory.register_creator("executor_factory_extension_priority",
                                        "executor_factory_priority_custom",
                                        create_test_creator<5>(),
                                        /*priority=*/10));

  auto default_executor =
      create_executor("executor_factory_extension_priority");
  auto custom_executor = create_executor("executor_factory_extension_priority",
                                         "executor_factory_priority_custom");
  EXPECT_NE(
      dynamic_cast<DefaultFactoryTestExecutorImpl*>(default_executor.get()),
      nullptr);
  EXPECT_NE(
      dynamic_cast<PriorityFactoryTestExecutorImpl*>(custom_executor.get()),
      nullptr);
}

TEST(ExecutorImplFactoryTest, MissingSelectedExecutorDoesNotUseDefault) {
  auto& factory = ExecutorImplFactory::get_instance();
  EXPECT_TRUE(factory.register_creator("executor_factory_missing_extension",
                                       create_test_creator<0>()));
  EXPECT_TRUE(factory.register_creator("executor_factory_other_extension",
                                       "executor_factory_missing_custom",
                                       create_test_creator<10>()));

  EXPECT_DEATH(create_executor("executor_factory_missing_extension",
                               "executor_factory_missing_custom"),
               "No executor registered for extension implementation");
}

TEST(ExecutorImplFactoryTest, ModelOnlyExtensionKeepsDefaultExecutor) {
  auto& factory = ExecutorImplFactory::get_instance();
  EXPECT_TRUE(factory.register_creator("executor_factory_model_only",
                                       create_test_creator<0>()));
  ExtensionRegistry::get_instance().register_implementation(
      "executor_factory_model_only_custom", /*component=*/"causal_lm");

  auto executor = create_executor("executor_factory_model_only",
                                  "executor_factory_model_only_custom");
  EXPECT_NE(dynamic_cast<DefaultFactoryTestExecutorImpl*>(executor.get()),
            nullptr);
}

TEST(ExecutorImplFactoryTest, UnknownExtensionDoesNotUseDefault) {
  auto& factory = ExecutorImplFactory::get_instance();
  EXPECT_TRUE(factory.register_creator("executor_factory_unknown_extension",
                                       create_test_creator<0>()));

  EXPECT_DEATH(create_executor("executor_factory_unknown_extension",
                               "executor_factory_unregistered_custom"),
               "Unknown extension implementation");
}

TEST(ExecutorImplFactoryTest, RegisterExecutorSupportsExtensionFamily) {
  auto executor = create_executor("executor_factory_macro_extension",
                                  "executor_factory_macro_custom");
  EXPECT_NE(dynamic_cast<executor_factory_test::NamespacedExecutorImpl*>(
                executor.get()),
            nullptr);
}

}  // namespace
}  // namespace xllm
