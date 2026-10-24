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

#include <cstdlib>

#include "core/common/extension_registry.h"
#include "core/common/native_library.h"
#include "core/runtime/executor_impl_factory.h"
#include "models/model_registry.h"

namespace xllm {
namespace {

TEST(NativeLibraryIntegrationTest, RegistersInTheHostModelRegistry) {
  ASSERT_EQ(setenv("XLLM_NATIVE_LIBRARIES", NATIVE_LIBRARY_FIXTURE, 1), 0);
  load_native_libraries();
  EXPECT_EQ(ModelRegistry::get_model_backend("native_library_fixture"),
            "shared_object");
}

TEST(NativeLibraryIntegrationTest, CreatesTheRegisteredExecutorSubclass) {
  ASSERT_EQ(setenv("XLLM_NATIVE_LIBRARIES", NATIVE_LIBRARY_FIXTURE, 1), 0);
  load_native_libraries();
  runtime::Options options;
  options.extension_impl("custom");
  auto executor = ExecutorImplFactory::get_instance().create_executor_impl(
      /*model=*/nullptr,
      /*args=*/ModelArgs{},
      /*device=*/torch::Device(torch::kCPU),
      options,
      /*executor_type=*/"native_library_executor");
  ASSERT_NE(executor, nullptr);
  EXPECT_TRUE(executor->supports_prepared_attention_metadata());
  EXPECT_TRUE(ExtensionRegistry::get_instance().has_implementation("custom",
                                                                   "executor"));
}

}  // namespace
}  // namespace xllm
